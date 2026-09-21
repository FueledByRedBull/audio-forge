//! Small bounded helpers for AudioForge-owned real-time regions.
//!
//! RT regions are the CPAL input callback, the CPAL output callback, and the
//! DSP processing loop after its startup allocation phase. Those regions must
//! not grow repo-owned buffers, block on locks, format/log, or call convenience
//! APIs that allocate returned vectors.

use ringbuf::{traits::Split, HeapCons, HeapProd, HeapRb};
use std::sync::atomic::{AtomicU32, Ordering};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtErrorCode {
    None = 0,
    InputStreamError = 1,
    OutputStreamError = 2,
    OutputQueueOverflow = 3,
    InputBacklogDropped = 4,
    SuppressorBackendFailed = 5,
    SuppressorNonFinite = 6,
    FixedBufferOverflow = 7,
}

impl RtErrorCode {
    pub fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::InputStreamError,
            2 => Self::OutputStreamError,
            3 => Self::OutputQueueOverflow,
            4 => Self::InputBacklogDropped,
            5 => Self::SuppressorBackendFailed,
            6 => Self::SuppressorNonFinite,
            7 => Self::FixedBufferOverflow,
            _ => Self::None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::InputStreamError => "input stream error",
            Self::OutputStreamError => "output stream error",
            Self::OutputQueueOverflow => "output queue overflow",
            Self::InputBacklogDropped => "input backlog dropped",
            Self::SuppressorBackendFailed => "suppressor backend failed",
            Self::SuppressorNonFinite => "suppressor produced non-finite output",
            Self::FixedBufferOverflow => "fixed real-time buffer overflow",
        }
    }
}

pub fn store_rt_error(error: &AtomicU32, code: RtErrorCode) {
    error.store(code as u32, Ordering::Relaxed);
}

pub struct FixedAudioBuffer<T, const N: usize> {
    data: Vec<T>,
    len: usize,
}

impl<T: Copy + Default, const N: usize> FixedAudioBuffer<T, N> {
    pub fn new() -> Self {
        let mut data = Vec::with_capacity(N);
        data.resize(N, T::default());
        Self { data, len: 0 }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        N.saturating_sub(self.len)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }

    #[inline]
    pub fn as_mut_capacity_slice(&mut self) -> &mut [T] {
        &mut self.data[..]
    }

    #[inline]
    pub fn push(&mut self, value: T) -> bool {
        if self.len == N {
            return false;
        }
        self.data[self.len] = value;
        self.len += 1;
        true
    }

    pub fn extend_from_slice(&mut self, values: &[T]) -> usize {
        let written = values.len().min(self.remaining());
        if written > 0 {
            let end = self.len + written;
            self.data[self.len..end].copy_from_slice(&values[..written]);
            self.len = end;
        }
        written
    }

    pub fn set_len_zeroed(&mut self, len: usize) -> bool {
        if len > N {
            return false;
        }
        if len > self.len {
            self.data[self.len..len].fill(T::default());
        }
        self.len = len;
        true
    }
}

impl<T: Copy + Default, const N: usize> Default for FixedAudioBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FixedAudioRing<T, const N: usize> {
    data: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Copy + Default, const N: usize> FixedAudioRing<T, N> {
    pub fn new() -> Self {
        let mut data = Vec::with_capacity(N);
        data.resize(N, T::default());
        Self {
            data,
            head: 0,
            len: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        N.saturating_sub(self.len)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    pub fn push_slice(&mut self, values: &[T]) -> usize {
        let written = values.len().min(self.remaining());
        for (offset, value) in values.iter().take(written).copied().enumerate() {
            let idx = (self.head + self.len + offset) % N;
            self.data[idx] = value;
        }
        self.len += written;
        written
    }

    pub fn push(&mut self, value: T) -> bool {
        if self.len == N {
            return false;
        }
        let idx = (self.head + self.len) % N;
        self.data[idx] = value;
        self.len += 1;
        true
    }

    pub fn pop_into(&mut self, output: &mut [T]) -> usize {
        let count = output.len().min(self.len);
        for (offset, sample) in output.iter_mut().take(count).enumerate() {
            let idx = (self.head + offset) % N;
            *sample = self.data[idx];
        }
        self.head = (self.head + count) % N;
        self.len -= count;
        if self.len == 0 {
            self.head = 0;
        }
        count
    }

    pub fn move_into<const M: usize>(&mut self, output: &mut FixedAudioRing<T, M>) -> usize {
        let mut moved = 0usize;
        let mut scratch = [T::default(); 64];
        while self.len > 0 && output.remaining() > 0 {
            let count = self.len.min(output.remaining()).min(scratch.len());
            let popped = self.pop_into(&mut scratch[..count]);
            if popped == 0 {
                break;
            }
            moved += output.push_slice(&scratch[..popped]);
        }
        moved
    }
}

impl<T: Copy + Default, const N: usize> Default for FixedAudioRing<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RtCommandQueue<T, const N: usize> {
    rb: HeapRb<T>,
}

impl<T, const N: usize> RtCommandQueue<T, N> {
    pub fn new() -> Self {
        Self { rb: HeapRb::new(N) }
    }

    pub fn split(self) -> (HeapProd<T>, HeapCons<T>) {
        self.rb.split()
    }
}

impl<T, const N: usize> Default for RtCommandQueue<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ringbuf::traits::{Consumer, Observer, Producer};
    use std::cell::Cell;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn command_queue_drops_panicking_items_once() {
        struct DropProbe<'a> {
            id: usize,
            panic_at: usize,
            drops: &'a [Cell<usize>; 4],
        }

        impl Drop for DropProbe<'_> {
            fn drop(&mut self) {
                let count = self.drops[self.id].get() + 1;
                self.drops[self.id].set(count);
                if self.id == self.panic_at && count == 1 {
                    panic!("queue element destructor");
                }
            }
        }

        for clear in [false, true] {
            for panic_at in 0..3 {
                let drops = [const { Cell::new(0) }; 4];
                let (mut producer, mut consumer) =
                    RtCommandQueue::<DropProbe<'_>, 3>::new().split();
                for id in 0..3 {
                    assert!(producer
                        .try_push(DropProbe {
                            id,
                            panic_at,
                            drops: &drops
                        })
                        .is_ok());
                }

                assert!(catch_unwind(AssertUnwindSafe(|| {
                    if clear {
                        consumer.clear();
                    } else {
                        consumer.skip(3);
                    }
                }))
                .is_err());
                assert_eq!(consumer.occupied_len(), 2 - panic_at);
                for id in panic_at + 1..3 {
                    assert_eq!(consumer.try_pop().unwrap().id, id);
                }
                assert!(producer
                    .try_push(DropProbe {
                        id: 3,
                        panic_at,
                        drops: &drops
                    })
                    .is_ok());
                drop(producer);
                drop(consumer);
                assert_eq!(drops.map(|count| count.get()), [1; 4]);
            }
        }
    }
}
