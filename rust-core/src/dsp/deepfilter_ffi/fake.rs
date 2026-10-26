//! A stand-in for the DeepFilterNet C library whose behaviour belongs to each
//! instance, so a worker thread can drive it: the library the other tests use
//! keeps its settings in a thread-local, which the worker would not see. A
//! frame comes out halved, and the instance can be told to fail, to produce a
//! non-finite frame or to hold a frame as a stalled library would.

use super::*;
use std::cell::RefCell;
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;
use std::time::Duration;

pub(crate) struct Fake {
    /// Frames from this call on return a NaN SNR.
    pub(crate) fail_from: AtomicU32,
    /// Frames from this call on carry a non-finite sample.
    pub(crate) non_finite_from: AtomicU32,
    /// While set, a frame waits inside the library.
    pub(crate) hold: AtomicBool,
    /// Milliseconds every frame takes, for a worker that cannot keep up.
    pub(crate) delay_ms: AtomicU32,
    pub(crate) started: AtomicU32,
    pub(crate) finished: AtomicU32,
    pub(crate) freed: AtomicBool,
}

impl Fake {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            fail_from: AtomicU32::new(u32::MAX),
            non_finite_from: AtomicU32::new(u32::MAX),
            hold: AtomicBool::new(false),
            delay_ms: AtomicU32::new(0),
            started: AtomicU32::new(0),
            finished: AtomicU32::new(0),
            freed: AtomicBool::new(false),
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
    Arc::into_raw(fake) as *mut DFState
}

unsafe extern "C" fn free(state: *mut DFState) {
    let fake = Arc::from_raw(state as *const Fake);
    fake.freed.store(true, Ordering::Release);
}

unsafe extern "C" fn frame_length(_state: *mut DFState) -> usize {
    DEEPFILTER_FRAME_SIZE
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

/// A file the model lookup accepts; the stand-in never reads it.
fn model_file() -> &'static Path {
    static FILE: OnceLock<PathBuf> = OnceLock::new();
    FILE.get_or_init(|| {
        // Test fixture without privileged or secret data.
        // nosemgrep: rust.lang.security.temp-dir.temp-dir
        let path = std::env::temp_dir().join(format!(
            "audioforge-fake-deepfilter-model-{}.tar.gz",
            std::process::id()
        ));
        std::fs::write(&path, b"not a DeepFilterNet model").unwrap();
        path
    })
}

/// A processor on the stand-in library, answering inline.
pub(crate) fn inline_processor(
    fake: &Arc<Fake>,
    model: DeepFilterModel,
    strength: f32,
) -> DeepFilterProcessor {
    NEXT.with(|next| *next.borrow_mut() = Some(Arc::clone(fake)));
    let processor = DeepFilterProcessor::from_library(
        Arc::new(AtomicU32::new(strength.to_bits())),
        model,
        DeepFilterRuntimeConfig::default(),
        Ok(library()),
        &AppOwnedDeepFilterPaths::default(),
        true,
        Some(model_file()),
    );
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
