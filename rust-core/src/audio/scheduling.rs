//! Calling-thread MMCSS registration for the Windows DSP worker.

use std::{ffi::c_void, io, marker::PhantomData, ptr::NonNull, rc::Rc};

#[link(name = "Avrt")]
unsafe extern "system" {
    fn AvSetMmThreadCharacteristicsW(task_name: *const u16, task_index: *mut u32) -> *mut c_void;
    fn AvRevertMmThreadCharacteristics(handle: *mut c_void) -> i32;
}

const PRO_AUDIO_TASK: [u16; 10] = [80, 114, 111, 32, 65, 117, 100, 105, 111, 0];

pub(super) struct ProAudioThread {
    handle: Option<NonNull<c_void>>,
    // MMCSS must be reverted by the registering thread, including during unwind.
    _same_thread: PhantomData<Rc<()>>,
}

impl ProAudioThread {
    pub(super) fn enter() -> io::Result<Self> {
        Self::enter_task(&PRO_AUDIO_TASK)
    }

    fn enter_task(task: &[u16]) -> io::Result<Self> {
        if task.last() != Some(&0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unterminated MMCSS task",
            ));
        }
        let mut task_index = 0;
        // SAFETY: The task is NUL-terminated and both pointers remain valid for this call.
        let handle = unsafe { AvSetMmThreadCharacteristicsW(task.as_ptr(), &mut task_index) };
        let handle = NonNull::new(handle).ok_or_else(io::Error::last_os_error)?;
        Ok(Self {
            handle: Some(handle),
            _same_thread: PhantomData,
        })
    }

    pub(super) fn finish(mut self) -> io::Result<()> {
        self.revert()
    }

    fn revert(&mut self) -> io::Result<()> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        // SAFETY: This uniquely owned handle cannot leave its registering thread.
        // Take it first so a reported failure cannot cause another attempt in Drop.
        let success = unsafe { AvRevertMmThreadCharacteristics(handle.as_ptr()) };
        let result = if success == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        };
        #[cfg(test)]
        tests::record_revert(success != 0);
        result
    }
}

impl Drop for ProAudioThread {
    fn drop(&mut self) {
        // Normal shutdown checks finish(); unwinding must remain best-effort and nonpanicking.
        let _ = self.revert();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThreadId() -> u32;
    }

    thread_local! {
        static REVERT: Cell<(usize, bool, u32)> = const { Cell::new((0, false, 0)) };
    }

    pub(super) fn record_revert(success: bool) {
        REVERT.with(|result| {
            let count = result.get().0;
            // SAFETY: This getter takes no arguments and only reads the calling thread ID.
            result.set((count + 1, success, unsafe { GetCurrentThreadId() }));
        });
    }

    #[test]
    fn pro_audio_finish_reverts_once_on_the_registering_thread() {
        std::thread::spawn(|| {
            // SAFETY: This getter takes no arguments and only reads the calling thread ID.
            let thread_id = unsafe { GetCurrentThreadId() };
            ProAudioThread::enter().unwrap().finish().unwrap();
            assert_eq!(REVERT.get(), (1, true, thread_id));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn invalid_task_returns_the_win32_error_without_reverting() {
        std::thread::spawn(|| {
            let task: Vec<u16> = "AudioForge-Missing-Task-74DA\0".encode_utf16().collect();
            let error = ProAudioThread::enter_task(&task).err().unwrap();
            assert!(error.raw_os_error().is_some_and(|code| code != 0));
            eprintln!("invalid task: {error}");
            assert_eq!(REVERT.get().0, 0);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn pro_audio_drop_reverts_once_on_unwind() {
        std::thread::spawn(|| {
            // SAFETY: This getter takes no arguments and only reads the calling thread ID.
            let thread_id = unsafe { GetCurrentThreadId() };
            assert!(std::panic::catch_unwind(|| {
                let _guard = ProAudioThread::enter().unwrap();
                panic!("exercise guard cleanup");
            })
            .is_err());
            assert_eq!(REVERT.get(), (1, true, thread_id));
        })
        .join()
        .unwrap();
    }
}
