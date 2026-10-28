//! PipeWire virtual microphone: AudioForge plays into a sink and other apps
//! record the matching source, so Linux needs no separate virtual cable.
//!
//! PipeWire's loopback module, loaded into this process, creates the pair. It
//! lives exactly as long as AudioForge does: a crash takes it down with the
//! process instead of leaving a stale device behind.
#![allow(clippy::useless_conversion)] // PyO3 proc-macro wrappers trigger false positives.

use std::cell::Cell;
use std::ffi::CString;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Duration;

use pipewire as pw;
use pw::context::ContextRc;
use pw::main_loop::MainLoopRc;
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

/// Create the virtual microphone unless it already exists, and wait until
/// PipeWire announces its source node.
pub fn create() -> Result<(), String> {
    let mut slot = VIRTUAL_MIC
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if slot.as_ref().is_some_and(|mic| !mic.thread.is_finished()) {
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
        Err(_) => Err("PipeWire did not announce the virtual microphone in time".to_string()),
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
    unsafe { pw::sys::pw_init(std::ptr::null_mut(), std::ptr::null_mut()) };
    if let Err(error) = serve(quit_rx, &ready_tx) {
        let _ = ready_tx.send(Err(error));
    }
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
    let _registry_listener = registry
        .add_listener_local()
        .global({
            let ready = ready.clone();
            move |global| {
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
    let _quit = quit_rx.attach(mainloop.loop_(), {
        let mainloop = mainloop.clone();
        move |()| mainloop.quit()
    });
    mainloop.run();
    Ok(())
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
