//! PipeWire virtual microphone: AudioForge plays into a sink and other apps
//! record the matching source, so Linux needs no separate virtual cable.
//!
//! PipeWire's loopback module, loaded into this process, creates the pair. It
//! lives exactly as long as AudioForge does: a crash takes it down with the
//! process instead of leaving a stale device behind. It also ends with the
//! connection to the PipeWire daemon: the nodes vanish with a daemon restart,
//! so the thread that owns them exits and `create` builds them again.
#![allow(clippy::useless_conversion)] // PyO3 proc-macro wrappers trigger false positives.

use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Mutex, TryLockError};
use std::thread::JoinHandle;
use std::time::Duration;

use pipewire as pw;
use pw::context::ContextRc;
use pw::main_loop::MainLoopRc;
use pw::metadata::{Metadata, MetadataListener};
use pw::types::ObjectType;
use pyo3::prelude::*;

/// The sink AudioForge's output stream plays into.
pub const SINK_NODE_NAME: &str = "audioforge_mic_input";
/// The source Discord, OBS and browsers record from.
pub const SOURCE_NODE_NAME: &str = "audioforge_mic";

/// Mono at the DSP rate. `Audio/Source` (not `Audio/Source/Virtual`) is the
/// class PipeWire's loopback documentation gives a virtual microphone; apps
/// list it like a hardware mic.
const LOOPBACK_ARGS: &str = r#"{
    node.description = "AudioForge Mic"
    audio.rate = 48000
    audio.position = [ MONO ]
    capture.props = {
        node.name = "audioforge_mic_input"
        node.description = "AudioForge Mic Input"
        media.class = "Audio/Sink"
    }
    playback.props = {
        node.name = "audioforge_mic"
        node.description = "AudioForge Mic"
        media.class = "Audio/Source"
    }
}"#;

const READY_TIMEOUT: Duration = Duration::from_secs(5);

struct VirtualMic {
    quit: pw::channel::Sender<()>,
    thread: JoinHandle<()>,
}

static VIRTUAL_MIC: Mutex<Option<VirtualMic>> = Mutex::new(None);

/// Whether the session manager made AudioForge Mic the system default
/// microphone, as PipeWire's `default` metadata last said. The thread that owns
/// the microphone keeps it current and clears it when it ends, so a read never
/// waits and is false while there is no microphone.
static DEFAULT_SOURCE_IS_OWN: AtomicBool = AtomicBool::new(false);

/// Whether the thread that owns the microphone's nodes is still alive.
fn alive(slot: &Option<VirtualMic>) -> bool {
    slot.as_ref().is_some_and(|mic| !mic.thread.is_finished())
}

/// Whether the virtual microphone exists: false before `create`, after
/// `destroy` and after PipeWire dropped the connection. Never waits, so it is
/// false while a `create` is still building the microphone.
pub fn is_running() -> bool {
    match VIRTUAL_MIC.try_lock() {
        Ok(slot) => alive(&slot),
        Err(TryLockError::Poisoned(error)) => alive(&error.into_inner()),
        Err(TryLockError::WouldBlock) => false,
    }
}

/// Create the virtual microphone unless it already exists, and wait until
/// PipeWire announces its source node. A microphone whose connection to
/// PipeWire was lost is built again.
pub fn create() -> Result<(), String> {
    let mut slot = VIRTUAL_MIC
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if alive(&slot) {
        return Ok(());
    }
    let (ready_tx, ready_rx) = mpsc::channel();
    let (quit_tx, quit_rx) = pw::channel::channel();
    let thread = std::thread::Builder::new()
        .name("audioforge-virtual-mic".to_string())
        .spawn(move || run(quit_rx, ready_tx))
        .map_err(|error| format!("could not start the virtual microphone thread: {error}"))?;
    let result = match ready_rx.recv_timeout(READY_TIMEOUT) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            Err("PipeWire did not announce the virtual microphone in time".to_string())
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(
            "PipeWire closed the connection before it announced the virtual microphone".to_string(),
        ),
    };
    match result {
        Ok(()) => {
            *slot = Some(VirtualMic {
                quit: quit_tx,
                thread,
            });
            Ok(())
        }
        Err(error) => {
            let _ = quit_tx.send(());
            let _ = thread.join();
            Err(error)
        }
    }
}

/// Remove the virtual microphone; apps recording from it lose the device.
pub fn destroy() {
    let mic = VIRTUAL_MIC
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take();
    if let Some(mic) = mic {
        let _ = mic.quit.send(());
        let _ = mic.thread.join();
    }
}

fn run(quit_rx: pw::channel::Receiver<()>, ready_tx: mpsc::Sender<Result<(), String>>) {
    // A reference of our own for the life of the process: libpipewire counts
    // pw_init calls, and cpal drops its references whenever its work ends.
    // SAFETY: pw_init accepts null argc/argv.
    // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
    unsafe { pw::sys::pw_init(std::ptr::null_mut(), std::ptr::null_mut()) };
    if let Err(error) = serve(quit_rx, &ready_tx) {
        let _ = ready_tx.send(Err(error));
    }
    DEFAULT_SOURCE_IS_OWN.store(false, Ordering::Relaxed);
}

fn serve(
    quit_rx: pw::channel::Receiver<()>,
    ready_tx: &mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let pipewire_error = |error: pw::Error| format!("PipeWire is unavailable: {error}");
    let mainloop = MainLoopRc::new(None).map_err(pipewire_error)?;
    let context = ContextRc::new(&mainloop, None).map_err(pipewire_error)?;
    let core = context.connect_rc(None).map_err(pipewire_error)?;
    let registry = core.get_registry_rc().map_err(pipewire_error)?;

    let module_name = c"libpipewire-module-loopback";
    let args = CString::new(LOOPBACK_ARGS).map_err(|error| error.to_string())?;
    // SAFETY: the context pointer is valid while `context` lives, both strings
    // are NUL-terminated and outlive the call, and null properties are allowed.
    // The context owns the module and destroys it when it is dropped below.
    // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
    let module = unsafe {
        pw::sys::pw_context_load_module(
            context.as_raw_ptr(),
            module_name.as_ptr(),
            args.as_ptr(),
            std::ptr::null_mut(),
        )
    };
    if module.is_null() {
        return Err(format!(
            "PipeWire could not load its loopback module: {}",
            std::io::Error::last_os_error()
        ));
    }

    let ready = Rc::new(Cell::new(Some(ready_tx.clone())));
    // The `default` metadata object, where the session manager records the
    // default devices. Listener first: it must go before the proxy it listens to.
    let default_metadata: RefCell<Option<(MetadataListener, Metadata)>> = RefCell::new(None);
    let _registry_listener = registry
        .add_listener_local()
        .global({
            let ready = ready.clone();
            let registry = registry.clone();
            move |global| {
                let is_default_metadata = global.type_ == ObjectType::Metadata
                    && global
                        .props
                        .is_some_and(|props| props.get("metadata.name") == Some("default"));
                if is_default_metadata {
                    if let Ok(metadata) = registry.bind::<Metadata, _>(global) {
                        let listener = metadata
                            .add_listener_local()
                            .property(|_subject, key, _type, value| {
                                let own = super::default_source_is_node(
                                    key,
                                    value,
                                    SOURCE_NODE_NAME,
                                    DEFAULT_SOURCE_IS_OWN.load(Ordering::Relaxed),
                                );
                                DEFAULT_SOURCE_IS_OWN.store(own, Ordering::Relaxed);
                                0
                            })
                            .register();
                        *default_metadata.borrow_mut() = Some((listener, metadata));
                    }
                    return;
                }
                let is_source = global.type_ == ObjectType::Node
                    && global
                        .props
                        .is_some_and(|props| props.get("node.name") == Some(SOURCE_NODE_NAME));
                if is_source {
                    if let Some(sender) = ready.take() {
                        let _ = sender.send(Ok(()));
                    }
                }
            }
        })
        .register();
    // A core error with the core's own ID is the connection failing: the daemon
    // restarted or went away and took the nodes with it. Leave the loop so the
    // thread ends and `create` can build the microphone again. Errors of other
    // objects are theirs to handle.
    let _core_listener = core
        .add_listener_local()
        .error({
            let mainloop = mainloop.clone();
            move |id, _seq, _res, _message| {
                if connection_lost(id) {
                    mainloop.quit();
                }
            }
        })
        .register();
    let _quit = quit_rx.attach(mainloop.loop_(), {
        let mainloop = mainloop.clone();
        move |()| mainloop.quit()
    });
    mainloop.run();
    Ok(())
}

/// Whether a core error is the connection to the daemon failing.
fn connection_lost(id: u32) -> bool {
    id == pw::core::PW_ID_CORE
}

/// Create AudioForge's PipeWire virtual microphone ("AudioForge Mic").
///
/// Its input sink, "AudioForge Mic Input", then appears among the output
/// devices. Raises RuntimeError when PipeWire is unavailable.
#[pyfunction]
pub fn create_virtual_microphone(py: Python<'_>) -> PyResult<()> {
    // Waits on the PipeWire daemon; keep the GIL free.
    py.detach(|| crate::catch_panic(|| create().map_err(pyo3::exceptions::PyRuntimeError::new_err)))
}

/// Whether the virtual microphone exists right now.
///
/// False before `create_virtual_microphone`, after `destroy_virtual_microphone`
/// and once PipeWire dropped the connection (the daemon restarted or went
/// away). Then `create_virtual_microphone` builds it again, and raises
/// RuntimeError while PipeWire is still unavailable. Never waits for PipeWire,
/// so it is also false while a `create_virtual_microphone` is still running.
#[pyfunction]
pub fn virtual_microphone_running() -> bool {
    is_running()
}

/// Whether the system default microphone is AudioForge Mic, so that following
/// the default would record AudioForge itself.
///
/// The session manager makes it the default when no other microphone exists.
/// The answer is the last one PipeWire's `default` metadata gave to the thread
/// that owns the microphone, a cached flag that never waits for PipeWire. It
/// is false while the microphone does not exist, and for a moment after it is
/// created, until the metadata has been read.
#[pyfunction]
pub fn default_source_is_virtual_microphone() -> bool {
    DEFAULT_SOURCE_IS_OWN.load(Ordering::Relaxed)
}

/// Remove the virtual microphone created by `create_virtual_microphone`.
#[pyfunction]
pub fn destroy_virtual_microphone(py: Python<'_>) -> PyResult<()> {
    py.detach(|| {
        crate::catch_panic(|| {
            destroy();
            Ok(())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn microphone(thread: JoinHandle<()>) -> Option<VirtualMic> {
        let (quit, _receiver) = pw::channel::channel();
        Some(VirtualMic { quit, thread })
    }

    #[test]
    fn only_a_core_error_with_the_cores_own_id_is_a_lost_connection() {
        assert!(connection_lost(pw::core::PW_ID_CORE));
        assert!(!connection_lost(pw::core::PW_ID_CORE + 1));
    }

    #[test]
    fn a_microphone_is_running_until_its_thread_ends() {
        assert!(!alive(&None));
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let _ = release_rx.recv();
        });
        let slot = microphone(thread);
        assert!(alive(&slot));
        release_tx.send(()).unwrap();
        while !slot.as_ref().unwrap().thread.is_finished() {
            std::thread::yield_now();
        }
        assert!(!alive(&slot));
    }
}
