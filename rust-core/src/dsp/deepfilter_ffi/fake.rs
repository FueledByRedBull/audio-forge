//! A stand-in for the DeepFilterNet C library whose behaviour belongs to each
//! instance, so a worker thread can drive it. A frame comes out halved, and the
//! instance can be told to fail to load, to report another frame length, to
//! fail a frame, to produce a non-finite frame or to hold a frame as a stalled
//! library would.

use super::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::OnceLock;
use std::time::Duration;

pub(crate) struct Fake {
    /// Creating the state fails, as it does for a model the library rejects.
    pub(crate) fail_create: AtomicBool,
    /// The frame length the library reports.
    pub(crate) frame_length: AtomicUsize,
    /// Frames from this call on return a NaN SNR.
    pub(crate) fail_from: AtomicU32,
    /// Frames from this call on carry a non-finite sample.
    pub(crate) non_finite_from: AtomicU32,
    /// While set, a frame waits inside the library.
    pub(crate) hold: AtomicBool,
    /// Milliseconds every frame takes, for a worker that cannot keep up.
    pub(crate) delay_ms: AtomicU32,
    pub(crate) created: AtomicU32,
    pub(crate) started: AtomicU32,
    pub(crate) finished: AtomicU32,
    pub(crate) freed: AtomicBool,
    /// A frame reached the library with a non-finite sample.
    pub(crate) non_finite_input_seen: AtomicBool,
}

impl Fake {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            fail_create: AtomicBool::new(false),
            frame_length: AtomicUsize::new(DEEPFILTER_FRAME_SIZE),
            fail_from: AtomicU32::new(u32::MAX),
            non_finite_from: AtomicU32::new(u32::MAX),
            hold: AtomicBool::new(false),
            delay_ms: AtomicU32::new(0),
            created: AtomicU32::new(0),
            started: AtomicU32::new(0),
            finished: AtomicU32::new(0),
            freed: AtomicBool::new(false),
            non_finite_input_seen: AtomicBool::new(false),
        })
    }
}

thread_local! {
    static NEXT: RefCell<Option<Arc<Fake>>> = const { RefCell::new(None) };
}

unsafe extern "C" fn create(
    _path: *const i8,
    _atten_lim: f32,
    _log_level: *const i8,
) -> *mut DFState {
    let fake = NEXT
        .with(|next| next.borrow_mut().take())
        .expect("a fake to create");
    fake.created.fetch_add(1, Ordering::AcqRel);
    if fake.fail_create.load(Ordering::Acquire) {
        return std::ptr::null_mut();
    }
    Arc::into_raw(fake) as *mut DFState
}

unsafe extern "C" fn free(state: *mut DFState) {
    let fake = Arc::from_raw(state as *const Fake);
    fake.freed.store(true, Ordering::Release);
}

unsafe extern "C" fn frame_length(state: *mut DFState) -> usize {
    (*(state as *const Fake))
        .frame_length
        .load(Ordering::Acquire)
}

unsafe extern "C" fn process(state: *mut DFState, input: *mut f32, output: *mut f32) -> f32 {
    let fake = &*(state as *const Fake);
    let call = fake.started.fetch_add(1, Ordering::AcqRel);
    while fake.hold.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let delay_ms = fake.delay_ms.load(Ordering::Acquire);
    if delay_ms > 0 {
        std::thread::sleep(Duration::from_millis(u64::from(delay_ms)));
    }
    let input = std::slice::from_raw_parts(input, DEEPFILTER_FRAME_SIZE);
    let output = std::slice::from_raw_parts_mut(output, DEEPFILTER_FRAME_SIZE);
    if input.iter().any(|sample| !sample.is_finite()) {
        fake.non_finite_input_seen.store(true, Ordering::Release);
    }
    for (out, sample) in output.iter_mut().zip(input) {
        *out = sample * 0.5;
    }
    if call >= fake.non_finite_from.load(Ordering::Acquire) {
        output[0] = f32::NAN;
    }
    fake.finished.fetch_add(1, Ordering::AcqRel);
    if call >= fake.fail_from.load(Ordering::Acquire) {
        f32::NAN
    } else {
        10.0
    }
}

unsafe extern "C" fn set(_state: *mut DFState, _value: f32) {}

fn library() -> DeepFilterLib {
    DeepFilterLib {
        _library: None,
        df_create: create,
        df_free: free,
        df_get_frame_length: frame_length,
        df_process_frame: process,
        df_set_atten_lim: set,
        df_set_post_filter_beta: set,
    }
}

/// A file with `contents` in the temp directory, one per `name` and process.
pub(crate) fn scratch_file(name: &str, contents: &[u8]) -> PathBuf {
    // Test fixture without privileged or secret data.
    // nosemgrep: rust.lang.security.temp-dir.temp-dir
    let path =
        std::env::temp_dir().join(format!("audioforge-{name}-{}.tar.gz", std::process::id()));
    std::fs::write(&path, contents).unwrap();
    path
}

/// A file the model lookup accepts; the stand-in never reads it.
fn model_file() -> &'static Path {
    static FILE: OnceLock<PathBuf> = OnceLock::new();
    FILE.get_or_init(|| scratch_file("fake-deepfilter-model", b"not a DeepFilterNet model"))
}

/// A processor on the stand-in library, answering inline, whatever the library
/// does. `bundled` hands it the stand-in model file as the app's own model,
/// which the identity check refuses; otherwise the file is an external
/// override, which skips the check.
pub(crate) fn processor_on(
    fake: &Arc<Fake>,
    model: DeepFilterModel,
    strength: f32,
    bundled: bool,
) -> DeepFilterProcessor {
    NEXT.with(|next| *next.borrow_mut() = Some(Arc::clone(fake)));
    let app_paths = AppOwnedDeepFilterPaths {
        library: None,
        model: bundled.then(|| model_file().to_path_buf()),
    };
    DeepFilterProcessor::from_library(
        Arc::new(AtomicU32::new(strength.to_bits())),
        model,
        DeepFilterRuntimeConfig::default(),
        Ok(library()),
        &app_paths,
        !bundled,
        (!bundled).then(model_file),
    )
}

/// A processor on the stand-in library that is expected to load, answering
/// inline.
pub(crate) fn inline_processor(
    fake: &Arc<Fake>,
    model: DeepFilterModel,
    strength: f32,
) -> DeepFilterProcessor {
    let processor = processor_on(fake, model, strength, false);
    assert!(
        processor.is_ffi_available(),
        "{:?}",
        processor.backend_error()
    );
    processor
}

/// The same, with the library at a worker thread.
pub(crate) fn worker_processor(
    fake: &Arc<Fake>,
    model: DeepFilterModel,
    strength: f32,
) -> DeepFilterProcessor {
    let processor = inline_processor(fake, model, strength).with_worker();
    assert!(
        processor.is_ffi_available(),
        "{:?}",
        processor.backend_error()
    );
    processor
}

/// A worker-backed engine for the live loop's tests.
pub(crate) fn worker_engine(
    fake: &Arc<Fake>,
    model: DeepFilterModel,
    strength: f32,
) -> crate::dsp::noise_suppressor::ControlledNoiseSuppressionEngine {
    Box::new(worker_processor(fake, model, strength))
}
