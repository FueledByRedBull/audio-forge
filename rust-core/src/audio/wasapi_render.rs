//! Windows output stream on a short shared-mode engine period.
//!
//! cpal opens WASAPI through `IAudioClient::Initialize`, which gives every
//! client a 22 ms buffer on a 10 ms engine period however short a buffer it asks
//! for. `IAudioClient3` runs the endpoint's engine on a shorter period when the
//! driver offers one; on VB-Cable a 128-frame period took about 48 ms off the
//! time from writing a sample to reading it back from CABLE Output. A device
//! that offers nothing shorter, or refuses the stream, stays on cpal, and so does
//! one whose short-period stream fails later, until the app restarts. VB-Cable is
//! the only endpoint measured this way, so only VB-Audio's are tried: the period
//! belongs to the endpoint, and a hardware device has more to lose from it.

use super::input::AudioError;
use super::output::{OutputDeviceBuffer, OutputFailure, OutputRenderer};
use super::scheduling::ProAudioThread;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::{null, null_mut, NonNull};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

type Hresult = i32;
type Unused = usize;

#[repr(C)]
struct Guid(u32, u16, u16, [u8; 8]);

const CLSID_MM_DEVICE_ENUMERATOR: Guid = Guid(
    0xBCDE0395,
    0xE52F,
    0x467C,
    [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E],
);
const IID_MM_DEVICE_ENUMERATOR: Guid = Guid(
    0xA95664D2,
    0x9614,
    0x4F35,
    [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6],
);
const IID_AUDIO_CLIENT3: Guid = Guid(
    0x7ED4EE07,
    0x8E67,
    0x4CD4,
    [0x8C, 0x1A, 0x2B, 0x7A, 0x59, 0x87, 0xAD, 0x42],
);
const IID_AUDIO_RENDER_CLIENT: Guid = Guid(
    0xF294ACFC,
    0x3146,
    0x4483,
    [0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2],
);

const COINIT_MULTITHREADED: u32 = 0;
const CLSCTX_ALL: u32 = 23;
const AUDCLNT_SHAREMODE_SHARED: u32 = 0;
const AUDCLNT_STREAMFLAGS_EVENTCALLBACK: u32 = 0x0004_0000;
const AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY: u32 = 0x0800_0000;
const AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM: u32 = 0x8000_0000;
/// The flags cpal opens its shared-mode streams with.
const CPAL_STREAM_FLAGS: u32 = AUDCLNT_STREAMFLAGS_EVENTCALLBACK
    | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
    | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM;
/// 100 ns units in a second, the unit of buffer durations and device periods.
const HNS_PER_SECOND: i64 = 10_000_000;
const WAIT_OBJECT_0: u32 = 0;
const WAIT_OBJECT_1: u32 = 1;
const WAIT_TIMEOUT: u32 = 258;
const WAIT_MS: u32 = 500;
/// Waits with no event before the stream counts as dead: 2 s, under the watchdog's
/// limit, and the watchdog does not run during a recording.
const MAX_SILENT_WAITS: u32 = 4;
const E_FAIL: Hresult = 0x8000_4005_u32 as Hresult;
const E_POINTER: Hresult = 0x8000_4003_u32 as Hresult;
const AUDCLNT_E_DEVICE_INVALIDATED: Hresult = 0x8889_0004_u32 as Hresult;
const AUDCLNT_E_RESOURCES_INVALIDATED: Hresult = 0x8889_0026_u32 as Hresult;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
const WAVEFORMATEX_LEN: usize = 18;
const WAVEFORMATEXTENSIBLE_LEN: usize = 40;
/// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT as it lies in memory.
const FLOAT_SUBFORMAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

#[link(name = "ole32")]
unsafe extern "system" {
    fn CoInitializeEx(reserved: *const c_void, flags: u32) -> Hresult;
    fn CoUninitialize();
    fn CoCreateInstance(
        clsid: *const Guid,
        outer: *mut c_void,
        context: u32,
        iid: *const Guid,
        object: *mut *mut c_void,
    ) -> Hresult;
    fn CoTaskMemFree(memory: *mut c_void);
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateEventW(
        attributes: *const c_void,
        manual_reset: i32,
        initial_state: i32,
        name: *const u16,
    ) -> *mut c_void;
    fn SetEvent(event: *mut c_void) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
    fn WaitForMultipleObjects(
        count: u32,
        handles: *const *mut c_void,
        wait_all: i32,
        milliseconds: u32,
    ) -> u32;
}

#[repr(C)]
struct UnknownVtbl {
    query_interface: Unused,
    add_ref: Unused,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[repr(C)]
struct DeviceEnumeratorVtbl {
    unknown: UnknownVtbl,
    enum_audio_endpoints: Unused,
    get_default_audio_endpoint: Unused,
    get_device: unsafe extern "system" fn(*mut c_void, *const u16, *mut *mut c_void) -> Hresult,
}

#[repr(C)]
struct DeviceVtbl {
    unknown: UnknownVtbl,
    activate: unsafe extern "system" fn(
        *mut c_void,
        *const Guid,
        u32,
        *const c_void,
        *mut *mut c_void,
    ) -> Hresult,
}

#[repr(C)]
struct AudioClient3Vtbl {
    unknown: UnknownVtbl,
    initialize: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        i64,
        i64,
        *const u8,
        *const Guid,
    ) -> Hresult,
    get_buffer_size: unsafe extern "system" fn(*mut c_void, *mut u32) -> Hresult,
    get_stream_latency: Unused,
    get_current_padding: unsafe extern "system" fn(*mut c_void, *mut u32) -> Hresult,
    is_format_supported: Unused,
    get_mix_format: unsafe extern "system" fn(*mut c_void, *mut *mut u8) -> Hresult,
    get_device_period: unsafe extern "system" fn(*mut c_void, *mut i64, *mut i64) -> Hresult,
    start: unsafe extern "system" fn(*mut c_void) -> Hresult,
    stop: unsafe extern "system" fn(*mut c_void) -> Hresult,
    reset: Unused,
    set_event_handle: unsafe extern "system" fn(*mut c_void, *mut c_void) -> Hresult,
    get_service: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> Hresult,
    is_offload_capable: Unused,
    set_client_properties: Unused,
    get_buffer_size_limits: Unused,
    get_shared_mode_engine_period: unsafe extern "system" fn(
        *mut c_void,
        *const u8,
        *mut u32,
        *mut u32,
        *mut u32,
        *mut u32,
    ) -> Hresult,
    get_current_shared_mode_engine_period: Unused,
    initialize_shared_audio_stream:
        unsafe extern "system" fn(*mut c_void, u32, u32, *const u8, *const Guid) -> Hresult,
}

#[repr(C)]
struct RenderClientVtbl {
    unknown: UnknownVtbl,
    get_buffer: unsafe extern "system" fn(*mut c_void, u32, *mut *mut u8) -> Hresult,
    release_buffer: unsafe extern "system" fn(*mut c_void, u32, u32) -> Hresult,
}

/// An owned COM interface whose vtable is `V`, which starts with `UnknownVtbl`.
struct Com<V> {
    object: NonNull<c_void>,
    vtable: PhantomData<*const V>,
}

impl<V> Com<V> {
    /// Takes the reference a COM call just returned through `object`.
    ///
    /// # Safety
    /// On success `object` must be an interface whose vtable is `V`.
    unsafe fn own(hresult: Hresult, object: *mut c_void) -> Result<Self, Hresult> {
        check(hresult)?;
        let object = NonNull::new(object).ok_or(hresult)?;
        Ok(Self {
            object,
            vtable: PhantomData,
        })
    }

    fn raw(&self) -> *mut c_void {
        self.object.as_ptr()
    }

    fn vtbl(&self) -> &V {
        // SAFETY: A COM object starts with its vtable pointer, and `own` requires
        // the interface `V` describes, which lives as long as `self`.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe { &**self.object.as_ptr().cast::<*const V>() }
    }
}

impl<V> Drop for Com<V> {
    fn drop(&mut self) {
        // SAFETY: Every vtable starts with IUnknown's and `self` holds one reference.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let vtable = *self.object.as_ptr().cast::<*const UnknownVtbl>();
            ((*vtable).release)(self.object.as_ptr());
        }
    }
}

fn check(hresult: Hresult) -> Result<(), Hresult> {
    if hresult >= 0 {
        Ok(())
    } else {
        Err(hresult)
    }
}

/// Whether this endpoint's short period has been measured. See the module note.
pub(crate) fn short_period_is_verified(device_name: &str) -> bool {
    device_name.contains("VB-Audio")
}

/// The shared-mode periods an endpoint offers, in frames.
#[derive(Clone, Copy, Default)]
struct EnginePeriods {
    default: u32,
    fundamental: u32,
    min: u32,
    max: u32,
}

/// The shortest offered period not under `wanted`, if it beats the default.
/// Every offered period is a multiple of the fundamental one.
fn short_period_frames(periods: EnginePeriods, wanted: u32) -> Option<u32> {
    if periods.fundamental == 0 {
        return None;
    }
    let frames = wanted
        .max(periods.min)
        .div_ceil(periods.fundamental)
        .checked_mul(periods.fundamental)?;
    (frames < periods.default && frames <= periods.max).then_some(frames)
}

/// Whether a mix format is 32-bit float at this rate and channel count, the only
/// form the renderer writes here. The rest go to cpal, which converts.
fn is_float32_mix(format: &[u8], sample_rate: u32, channels: u16) -> bool {
    let u16_at = |at: usize| u16::from_le_bytes([format[at], format[at + 1]]);
    if format.len() < WAVEFORMATEX_LEN {
        return false;
    }
    let is_float = match u16_at(0) {
        WAVE_FORMAT_IEEE_FLOAT => true,
        WAVE_FORMAT_EXTENSIBLE => {
            format.len() >= WAVEFORMATEXTENSIBLE_LEN && format[24..40] == FLOAT_SUBFORMAT
        }
        _ => false,
    };
    is_float
        && u16_at(2) == channels
        && u32::from_le_bytes([format[4], format[5], format[6], format[7]]) == sample_rate
        && u16_at(12) == channels.saturating_mul(4)
        && u16_at(14) == 32
}

/// Whether a failure is the device going away, which a restart answers, and not
/// the short period failing on it.
fn device_is_gone(hresult: Hresult) -> bool {
    matches!(
        hresult,
        AUDCLNT_E_DEVICE_INVALIDATED | AUDCLNT_E_RESOURCES_INVALIDATED
    )
}

/// Endpoints whose short-period stream failed, with why; cpal serves them until
/// the app restarts. Only setup and a dying stream's thread touch this.
static REFUSED_ENDPOINTS: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());

fn refused_code(endpoint_id: &str) -> Option<u32> {
    let refused = REFUSED_ENDPOINTS.lock().ok()?;
    refused
        .iter()
        .find(|(id, _)| id == endpoint_id)
        .map(|&(_, code)| code)
}

fn remember_refused(endpoint_id: &str, hresult: Hresult) {
    if let Ok(mut refused) = REFUSED_ENDPOINTS.lock() {
        if !refused.iter().any(|(id, _)| id == endpoint_id) {
            refused.push((endpoint_id.to_string(), hresult as u32));
        }
    }
}

fn win32_hresult(error: &std::io::Error) -> Hresult {
    error.raw_os_error().map_or(E_FAIL, |code| {
        (0x8007_0000_u32 | (code as u32 & 0xFFFF)) as Hresult
    })
}

struct MixFormat(NonNull<u8>);

impl MixFormat {
    fn of(client: &Com<AudioClient3Vtbl>) -> Result<Self, Hresult> {
        let mut format = null_mut();
        // SAFETY: The client is live and `format` receives a CoTaskMem allocation.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        check(unsafe { (client.vtbl().get_mix_format)(client.raw(), &mut format) })?;
        NonNull::new(format).map(Self).ok_or(E_POINTER)
    }

    fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: A mix format is a WAVEFORMATEX followed by its cbSize extra bytes.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let extra = u16::from_le_bytes([*self.as_ptr().add(16), *self.as_ptr().add(17)]);
            std::slice::from_raw_parts(self.as_ptr(), WAVEFORMATEX_LEN + usize::from(extra))
        }
    }
}

impl Drop for MixFormat {
    fn drop(&mut self) {
        // SAFETY: GetMixFormat allocated this block with CoTaskMemAlloc.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe { CoTaskMemFree(self.0.as_ptr().cast()) };
    }
}

struct Event(NonNull<c_void>);

// SAFETY: A kernel event handle may be used from any thread.
// nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
unsafe impl Send for Event {}
// SAFETY: As above; SetEvent and the waits are thread-safe.
// nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
unsafe impl Sync for Event {}

impl Event {
    fn new() -> Result<Self, AudioError> {
        // SAFETY: All pointers may be null: default security, unnamed, auto-reset.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        let handle = unsafe { CreateEventW(null(), 0, 0, null()) };
        NonNull::new(handle).map(Self).ok_or_else(|| {
            AudioError::BuildStream(format!("CreateEvent: {}", std::io::Error::last_os_error()))
        })
    }

    fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    fn set(&self) {
        // SAFETY: The handle stays open until this value drops.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe { SetEvent(self.raw()) };
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: This value owns the handle.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe { CloseHandle(self.raw()) };
    }
}

struct Apartment;

impl Apartment {
    fn enter() -> Result<Self, Hresult> {
        // SAFETY: A new thread has no apartment yet; `Drop` leaves it on this thread.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        check(unsafe { CoInitializeEx(null(), COINIT_MULTITHREADED) })?;
        Ok(Self)
    }
}

impl Drop for Apartment {
    fn drop(&mut self) {
        // SAFETY: Paired with the successful CoInitializeEx in `enter`.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe { CoUninitialize() };
    }
}

/// An initialized render client and what its loop needs.
struct Client {
    audio: Com<AudioClient3Vtbl>,
    render: Com<RenderClientVtbl>,
    audio_event: Event,
    buffer_frames: u32,
    channels: usize,
    period_frames: u32,
}

impl Client {
    /// `None` when the endpoint offers no short period or an unsuitable format.
    fn open(endpoint_id: &str, sample_rate: u32, channels: u16) -> Result<Option<Self>, Hresult> {
        let audio = Self::activate(endpoint_id)?;
        let format = MixFormat::of(&audio)?;
        if !is_float32_mix(format.bytes(), sample_rate, channels) {
            return Ok(None);
        }
        let mut periods = EnginePeriods::default();
        // SAFETY: The client and format are live and the out pointers are valid.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        check(unsafe {
            (audio.vtbl().get_shared_mode_engine_period)(
                audio.raw(),
                format.as_ptr(),
                &mut periods.default,
                &mut periods.fundamental,
                &mut periods.min,
                &mut periods.max,
            )
        })?;
        // 2.67 ms; below it the engine's own wakeups start to cost more than they save.
        let Some(period_frames) = short_period_frames(periods, sample_rate.div_ceil(375)) else {
            return Ok(None);
        };
        // SAFETY: As above; the format outlives the call, which copies it.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        check(unsafe {
            (audio.vtbl().initialize_shared_audio_stream)(
                audio.raw(),
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                period_frames,
                format.as_ptr(),
                null(),
            )
        })?;
        let audio_event = Event::new().map_err(|_| E_FAIL)?;
        let mut buffer_frames = 0;
        let mut render = null_mut();
        // SAFETY: The client is initialized and the out pointers are valid.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        let render = unsafe {
            check((audio.vtbl().set_event_handle)(
                audio.raw(),
                audio_event.raw(),
            ))?;
            check((audio.vtbl().get_buffer_size)(
                audio.raw(),
                &mut buffer_frames,
            ))?;
            Com::own(
                (audio.vtbl().get_service)(audio.raw(), &IID_AUDIO_RENDER_CLIENT, &mut render),
                render,
            )?
        };
        Ok(Some(Self {
            audio,
            render,
            audio_event,
            buffer_frames,
            channels: usize::from(channels),
            period_frames,
        }))
    }

    fn activate(endpoint_id: &str) -> Result<Com<AudioClient3Vtbl>, Hresult> {
        let wide: Vec<u16> = endpoint_id.encode_utf16().chain([0]).collect();
        let (mut enumerator, mut device, mut client) = (null_mut(), null_mut(), null_mut());
        // SAFETY: The GUIDs and the NUL-terminated ID outlive these calls, and
        // each call fills the out pointer `Com::own` then takes.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let hresult = CoCreateInstance(
                &CLSID_MM_DEVICE_ENUMERATOR,
                null_mut(),
                CLSCTX_ALL,
                &IID_MM_DEVICE_ENUMERATOR,
                &mut enumerator,
            );
            let enumerator = Com::<DeviceEnumeratorVtbl>::own(hresult, enumerator)?;
            let hresult =
                (enumerator.vtbl().get_device)(enumerator.raw(), wide.as_ptr(), &mut device);
            let device = Com::<DeviceVtbl>::own(hresult, device)?;
            let hresult = (device.vtbl().activate)(
                device.raw(),
                &IID_AUDIO_CLIENT3,
                CLSCTX_ALL,
                null(),
                &mut client,
            );
            Com::own(hresult, client)
        }
    }

    // RT_REGION_START: wasapi_render_step
    /// Hand the device everything it has room for.
    fn fill(&self, renderer: &mut OutputRenderer) -> Result<(), Hresult> {
        let mut padding = 0;
        // SAFETY: The client is live; GetBuffer returns room for `room` frames of
        // the float mix format, and the slice ends before ReleaseBuffer.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            check((self.audio.vtbl().get_current_padding)(
                self.audio.raw(),
                &mut padding,
            ))?;
            let room = self.buffer_frames.saturating_sub(padding);
            if room == 0 {
                return Ok(());
            }
            let mut data = null_mut();
            check((self.render.vtbl().get_buffer)(
                self.render.raw(),
                room,
                &mut data,
            ))?;
            if data.is_null() {
                return Err(E_POINTER);
            }
            let samples = room as usize * self.channels;
            renderer.render(std::slice::from_raw_parts_mut(data.cast::<f32>(), samples));
            check((self.render.vtbl().release_buffer)(
                self.render.raw(),
                room,
                0,
            ))
        }
    }
    // RT_REGION_END: wasapi_render_step

    fn run(
        &self,
        renderer: &mut OutputRenderer,
        started: &Event,
        stop: &Event,
    ) -> Result<(), Hresult> {
        let waits = [stop.raw(), started.raw()];
        // SAFETY: Both handles stay open for the wait.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        if unsafe { WaitForMultipleObjects(2, waits.as_ptr(), 0, u32::MAX) } == WAIT_OBJECT_0 {
            return Ok(());
        }
        // SAFETY: The client is initialized and has its event handle.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        check(unsafe { (self.audio.vtbl().start)(self.audio.raw()) })?;
        let waits = [stop.raw(), self.audio_event.raw()];
        let mut silent_waits = 0;
        let result = loop {
            // SAFETY: Both handles stay open for the wait.
            // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
            match unsafe { WaitForMultipleObjects(2, waits.as_ptr(), 0, WAIT_MS) } {
                WAIT_OBJECT_0 => break Ok(()),
                WAIT_OBJECT_1 => {
                    silent_waits = 0;
                    if let Err(hresult) = self.fill(renderer) {
                        break Err(hresult);
                    }
                }
                WAIT_TIMEOUT => {
                    // A removed device may just stop signalling; asking finds out.
                    let mut padding = 0;
                    // SAFETY: The client is live and `padding` is valid.
                    // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
                    let asked = check(unsafe {
                        (self.audio.vtbl().get_current_padding)(self.audio.raw(), &mut padding)
                    });
                    silent_waits += 1;
                    if let Err(hresult) = asked {
                        break Err(hresult);
                    }
                    if silent_waits >= MAX_SILENT_WAITS {
                        break Err(E_FAIL);
                    }
                }
                _ => break Err(E_FAIL),
            }
        };
        // SAFETY: The client was started above.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe { (self.audio.vtbl().stop)(self.audio.raw()) };
        result
    }
}

/// Frames of a duration in 100 ns units at `sample_rate`.
fn hns_to_frames(hns: i64, sample_rate: u32) -> u32 {
    let frames = (hns.max(0) as u64 * u64::from(sample_rate) + 5_000_000) / HNS_PER_SECOND as u64;
    u32::try_from(frames).unwrap_or(u32::MAX)
}

/// The buffer duration cpal asks for: its fixed size, or 0 for the engine's default.
fn buffer_duration_hns(fixed_frames: Option<u32>, sample_rate: u32) -> i64 {
    fixed_frames.map_or(0, |frames| {
        i64::from(frames) * HNS_PER_SECOND / i64::from(sample_rate.max(1))
    })
}

/// The WAVEFORMATEXTENSIBLE cpal builds for a 32-bit float stream.
fn float32_format(sample_rate: u32, channels: u16) -> [u8; WAVEFORMATEXTENSIBLE_LEN] {
    let block_align = channels.saturating_mul(4);
    let mut format = [0; WAVEFORMATEXTENSIBLE_LEN];
    format[0..2].copy_from_slice(&WAVE_FORMAT_EXTENSIBLE.to_le_bytes());
    format[2..4].copy_from_slice(&channels.to_le_bytes());
    format[4..8].copy_from_slice(&sample_rate.to_le_bytes());
    format[8..12].copy_from_slice(
        &sample_rate
            .saturating_mul(u32::from(block_align))
            .to_le_bytes(),
    );
    format[12..14].copy_from_slice(&block_align.to_le_bytes());
    format[14..16].copy_from_slice(&32_u16.to_le_bytes());
    format[16..18]
        .copy_from_slice(&((WAVEFORMATEXTENSIBLE_LEN - WAVEFORMATEX_LEN) as u16).to_le_bytes());
    // Valid bits per sample; the channel mask stays 0, the direct-out layout.
    format[18..20].copy_from_slice(&32_u16.to_le_bytes());
    format[24..40].copy_from_slice(&FLOAT_SUBFORMAT);
    format
}

/// The buffer and callback period a standard shared-mode stream gets on this
/// endpoint, read from a client opened the way cpal opens its own and dropped
/// at once. It runs on a thread of its own, which brings its own COM apartment.
/// On failure the buffer stays 0 and `failure_code` holds the HRESULT.
pub(crate) fn probe_standard_buffer(
    endpoint_id: Option<&str>,
    sample_rate: u32,
    channels: u16,
    fixed_frames: Option<u32>,
) -> OutputDeviceBuffer {
    let read = match endpoint_id {
        Some(endpoint_id) => {
            let endpoint_id = endpoint_id.to_owned();
            std::thread::Builder::new()
                .name("audioforge-output-probe".into())
                .spawn(move || {
                    read_standard_buffer(&endpoint_id, sample_rate, channels, fixed_frames)
                })
                .map_err(|error| win32_hresult(&error))
                .and_then(|thread| thread.join().unwrap_or(Err(E_FAIL)))
        }
        None => Err(E_POINTER),
    };
    read.unwrap_or_else(|hresult| OutputDeviceBuffer {
        failure_code: hresult as u32,
        ..OutputDeviceBuffer::default()
    })
}

fn read_standard_buffer(
    endpoint_id: &str,
    sample_rate: u32,
    channels: u16,
    fixed_frames: Option<u32>,
) -> Result<OutputDeviceBuffer, Hresult> {
    let _apartment = Apartment::enter()?;
    let audio = Client::activate(endpoint_id)?;
    let format = float32_format(sample_rate, channels);
    let (mut frames, mut default_period, mut minimum_period) = (0, 0, 0);
    // SAFETY: The client is live, the format outlives the call, which copies it,
    // and the out pointers are valid.
    // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
    unsafe {
        check((audio.vtbl().initialize)(
            audio.raw(),
            AUDCLNT_SHAREMODE_SHARED,
            CPAL_STREAM_FLAGS,
            buffer_duration_hns(fixed_frames, sample_rate),
            0,
            format.as_ptr(),
            null(),
        ))?;
        check((audio.vtbl().get_buffer_size)(audio.raw(), &mut frames))?;
        check((audio.vtbl().get_device_period)(
            audio.raw(),
            &mut default_period,
            &mut minimum_period,
        ))?;
    }
    if frames == 0 {
        return Err(E_FAIL);
    }
    Ok(OutputDeviceBuffer {
        frames,
        period_frames: hns_to_frames(default_period, sample_rate),
        failure_code: 0,
    })
}

/// What the attempt came to; a refusal hands the renderer back for cpal.
pub(crate) enum Opened {
    Running(LowPeriodStream),
    /// `failure_code` is an HRESULT, 0 when the device simply offers nothing shorter.
    Declined {
        renderer: Box<OutputRenderer>,
        failure_code: u32,
    },
}

pub(crate) struct LowPeriodStream {
    started: Arc<Event>,
    stop: Arc<Event>,
    thread: Option<JoinHandle<()>>,
    device_buffer: OutputDeviceBuffer,
}

type Ready = Result<OutputDeviceBuffer, (OutputRenderer, u32)>;

impl LowPeriodStream {
    pub(crate) fn open(
        endpoint_id: String,
        sample_rate: u32,
        channels: u16,
        renderer: OutputRenderer,
        failure: OutputFailure,
    ) -> Result<Opened, AudioError> {
        if let Some(failure_code) = refused_code(&endpoint_id) {
            return Ok(Opened::Declined {
                renderer: Box::new(renderer),
                failure_code,
            });
        }
        let started = Arc::new(Event::new()?);
        let stop = Arc::new(Event::new()?);
        let (ready_tx, ready_rx) = sync_channel::<Ready>(1);
        let thread = {
            let (started, stop) = (Arc::clone(&started), Arc::clone(&stop));
            std::thread::Builder::new()
                .name("audioforge-output".into())
                .spawn(move || {
                    render_thread(
                        &endpoint_id,
                        sample_rate,
                        channels,
                        renderer,
                        failure,
                        &started,
                        &stop,
                        ready_tx,
                    );
                })
                .map_err(|error| AudioError::BuildStream(error.to_string()))?
        };
        match ready_rx.recv() {
            Ok(Ok(device_buffer)) => Ok(Opened::Running(Self {
                started,
                stop,
                thread: Some(thread),
                device_buffer,
            })),
            Ok(Err((renderer, failure_code))) => {
                let _ = thread.join();
                Ok(Opened::Declined {
                    renderer: Box::new(renderer),
                    failure_code,
                })
            }
            Err(_) => Err(AudioError::BuildStream(
                "the output thread ended during setup".to_string(),
            )),
        }
    }

    pub(crate) fn start(&self) {
        self.started.set();
    }

    /// Ends the stream for good; the processor only pauses to drop it.
    pub(crate) fn pause(&self) {
        self.stop.set();
    }

    pub(crate) fn period_frames(&self) -> u32 {
        self.device_buffer.period_frames
    }

    pub(crate) fn device_buffer(&self) -> OutputDeviceBuffer {
        self.device_buffer
    }
}

impl Drop for LowPeriodStream {
    fn drop(&mut self) {
        self.stop.set();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_thread(
    endpoint_id: &str,
    sample_rate: u32,
    channels: u16,
    mut renderer: OutputRenderer,
    failure: OutputFailure,
    started: &Event,
    stop: &Event,
    ready: SyncSender<Ready>,
) {
    let decline = |renderer, code: Hresult| {
        let _ = ready.send(Err((renderer, code as u32)));
    };
    let _apartment = match Apartment::enter() {
        Ok(apartment) => apartment,
        Err(hresult) => return decline(renderer, hresult),
    };
    let client = match Client::open(endpoint_id, sample_rate, channels) {
        Ok(Some(client)) => client,
        Ok(None) => return decline(renderer, 0),
        Err(hresult) => return decline(renderer, hresult),
    };
    // A period this short does not hold without the priority.
    let pro_audio = match ProAudioThread::enter() {
        Ok(pro_audio) => pro_audio,
        Err(error) => return decline(renderer, win32_hresult(&error)),
    };
    let device_buffer = OutputDeviceBuffer {
        frames: client.buffer_frames,
        period_frames: client.period_frames,
        failure_code: 0,
    };
    if ready.send(Ok(device_buffer)).is_err() {
        return;
    }
    if let Err(hresult) = client.run(&mut renderer, started, stop) {
        if !device_is_gone(hresult) {
            remember_refused(endpoint_id, hresult);
        }
        failure.report(hresult as u32);
    }
    let _ = pro_audio.finish();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn periods(default: u32, fundamental: u32, min: u32, max: u32) -> EnginePeriods {
        EnginePeriods {
            default,
            fundamental,
            min,
            max,
        }
    }

    #[test]
    fn durations_in_100_ns_units_become_frames_at_the_stream_rate() {
        assert_eq!(hns_to_frames(100_000, 48_000), 480);
        assert_eq!(hns_to_frames(220_000, 48_000), 1056);
        assert_eq!(hns_to_frames(100_000, 44_100), 441);
        assert_eq!(hns_to_frames(-1, 48_000), 0);
    }

    #[test]
    fn cpal_asks_for_its_fixed_buffer_or_the_engine_default() {
        assert_eq!(buffer_duration_hns(None, 48_000), 0);
        assert_eq!(buffer_duration_hns(Some(480), 48_000), 100_000);
        assert_eq!(buffer_duration_hns(Some(480), 0), 4_800_000_000);
    }

    #[test]
    fn the_probe_format_is_cpals_float_format() {
        let format = float32_format(48_000, 2);
        assert!(is_float32_mix(&format, 48_000, 2));
        assert_eq!(&format[8..12], &(48_000_u32 * 8).to_le_bytes());
        assert_eq!(&format[16..18], &22_u16.to_le_bytes());
        assert_eq!(&format[20..24], &[0; 4]);
    }

    #[test]
    fn a_probe_that_cannot_read_leaves_the_buffer_unknown_with_its_code() {
        let no_endpoint = probe_standard_buffer(None, 48_000, 2, None);
        assert_eq!(no_endpoint.frames, 0);
        assert_eq!(no_endpoint.failure_code, E_POINTER as u32);

        let missing = probe_standard_buffer(Some("no-such-endpoint"), 48_000, 2, None);
        assert_eq!(missing.frames, 0);
        assert_eq!(missing.period_frames, 0);
        assert_ne!(missing.failure_code, 0);
    }

    #[test]
    fn short_period_is_tried_only_on_vb_audio() {
        assert!(short_period_is_verified(
            "CABLE Input (VB-Audio Virtual Cable)"
        ));
        assert!(!short_period_is_verified(
            "Realtek Digital Output (Realtek USB Audio)"
        ));
        assert!(!short_period_is_verified(
            "Speakers (High Definition Audio)"
        ));
    }

    #[test]
    fn short_period_takes_what_vb_cable_offers() {
        assert_eq!(
            short_period_frames(periods(480, 1, 128, 480), 128),
            Some(128)
        );
    }

    #[test]
    fn short_period_is_refused_when_only_the_default_is_offered() {
        assert_eq!(short_period_frames(periods(480, 480, 480, 480), 128), None);
    }

    #[test]
    fn short_period_rounds_up_to_the_fundamental_and_never_under_the_minimum() {
        assert_eq!(
            short_period_frames(periods(480, 48, 96, 480), 128),
            Some(144)
        );
        assert_eq!(
            short_period_frames(periods(480, 1, 240, 480), 128),
            Some(240)
        );
    }

    #[test]
    fn short_period_that_would_not_beat_the_default_is_refused() {
        assert_eq!(short_period_frames(periods(480, 400, 400, 800), 450), None);
        assert_eq!(short_period_frames(periods(480, 0, 0, 0), 128), None);
    }

    #[test]
    fn only_a_short_period_failure_refuses_the_endpoint() {
        assert!(device_is_gone(AUDCLNT_E_DEVICE_INVALIDATED));
        assert!(device_is_gone(AUDCLNT_E_RESOURCES_INVALIDATED));
        assert!(!device_is_gone(E_FAIL));
    }

    #[test]
    fn refused_endpoint_keeps_its_first_code() {
        let id = "test-refused-endpoint";
        assert_eq!(refused_code(id), None);
        remember_refused(id, E_FAIL);
        remember_refused(id, E_POINTER);
        assert_eq!(refused_code(id), Some(E_FAIL as u32));
        assert_eq!(refused_code("another-endpoint"), None);
    }

    #[test]
    fn win32_error_becomes_an_hresult() {
        let error = std::io::Error::from_raw_os_error(1552);
        assert_eq!(win32_hresult(&error) as u32, 0x8007_0610);
    }

    fn extensible_format(
        sample_rate: u32,
        channels: u16,
        bits: u16,
        subformat: [u8; 16],
    ) -> Vec<u8> {
        let mut format = vec![0; WAVEFORMATEXTENSIBLE_LEN];
        format[..2].copy_from_slice(&WAVE_FORMAT_EXTENSIBLE.to_le_bytes());
        format[2..4].copy_from_slice(&channels.to_le_bytes());
        format[4..8].copy_from_slice(&sample_rate.to_le_bytes());
        format[12..14].copy_from_slice(&(channels * 4).to_le_bytes());
        format[14..16].copy_from_slice(&bits.to_le_bytes());
        format[16..18].copy_from_slice(&22_u16.to_le_bytes());
        format[24..40].copy_from_slice(&subformat);
        format
    }

    #[test]
    fn mix_format_must_be_float32_at_the_opened_rate_and_channels() {
        let good = extensible_format(48_000, 2, 32, FLOAT_SUBFORMAT);
        assert!(is_float32_mix(&good, 48_000, 2));
        assert!(!is_float32_mix(&good, 44_100, 2));
        assert!(!is_float32_mix(&good, 48_000, 1));
        assert!(!is_float32_mix(
            &extensible_format(48_000, 2, 16, FLOAT_SUBFORMAT),
            48_000,
            2
        ));
        let mut pcm = FLOAT_SUBFORMAT;
        pcm[0] = 1;
        assert!(!is_float32_mix(
            &extensible_format(48_000, 2, 32, pcm),
            48_000,
            2
        ));
    }

    #[test]
    fn mix_format_with_an_unexpected_frame_size_is_refused() {
        let mut format = extensible_format(48_000, 2, 32, FLOAT_SUBFORMAT);
        format[12..14].copy_from_slice(&16_u16.to_le_bytes());
        assert!(!is_float32_mix(&format, 48_000, 2));
    }

    #[test]
    fn mix_format_too_short_to_read_is_refused() {
        assert!(!is_float32_mix(&[0; 4], 48_000, 2));
        let good = extensible_format(48_000, 2, 32, FLOAT_SUBFORMAT);
        assert!(!is_float32_mix(&good[..WAVEFORMATEX_LEN], 48_000, 2));
    }
}
