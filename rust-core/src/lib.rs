#![cfg_attr(test, allow(clippy::float_cmp))]

//! MicEq Core - High-performance DSP engine for real-time audio processing
//!
//! Processing chain: Mic Input -> Pre-Filter -> Noise Gate -> Noise Suppression
//! -> De-Esser -> 10-Band EQ -> Compressor -> Limiter -> Output

use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod audio;
pub mod dsp;

#[cfg(test)]
pub(crate) mod test_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct CountingAllocator;

    static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static COUNTING_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count_allocation();
            System.alloc(layout)
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count_allocation();
            System.alloc_zeroed(layout)
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            count_allocation();
            System.realloc(ptr, layout, new_size)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
        }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

    fn count_allocation() {
        COUNTING_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        });
    }

    struct AllocationScope {
        previous: bool,
    }

    impl AllocationScope {
        fn enter() -> Self {
            ALLOCATION_COUNT.store(0, Ordering::SeqCst);
            let previous = COUNTING_ALLOCATIONS.with(|enabled| {
                let previous = enabled.get();
                enabled.set(true);
                previous
            });
            Self { previous }
        }
    }

    impl Drop for AllocationScope {
        fn drop(&mut self) {
            COUNTING_ALLOCATIONS.with(|enabled| enabled.set(self.previous));
        }
    }

    pub fn allocation_count_during(function: impl FnOnce()) -> usize {
        let _scope = AllocationScope::enter();
        function();
        ALLOCATION_COUNT.load(Ordering::SeqCst)
    }

    pub fn assert_no_allocations(label: &str, function: impl FnOnce()) {
        let allocations = allocation_count_during(function);
        assert_eq!(allocations, 0, "{label} allocated {allocations} time(s)");
    }
}

// Re-export main types
pub use audio::{AudioProcessor, PyAudioProcessor};
pub use dsp::{Biquad, Compressor, DeEsser, Limiter, NoiseGate, ParametricEQ, RNNoiseProcessor};

/// Python module initialization
#[pymodule]
fn mic_eq_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main audio processor
    m.add_class::<audio::PyAudioProcessor>()?;

    // VAD Gate Mode enum (VAD feature only)
    #[cfg(feature = "vad")]
    m.add_class::<audio::PyGateMode>()?;

    // Device enumeration
    m.add_class::<audio::DeviceInfo>()?;
    m.add_function(wrap_pyfunction!(audio::list_input_devices, m)?)?;
    m.add_function(wrap_pyfunction!(audio::list_output_devices, m)?)?;

    Ok(())
}
