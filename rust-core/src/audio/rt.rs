//! Small bounded helpers for AudioForge-owned real-time regions.
//!
//! RT regions are the CPAL input callback, the CPAL output callback, and the
//! DSP processing loop after its startup allocation phase. Those regions must
//! not grow repo-owned buffers, block on locks, format/log, or call convenience
//! APIs that allocate returned vectors.

use ringbuf::{HeapConsumer, HeapProducer, HeapRb};
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

    pub fn pop_all_vec(&mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        while self.len > 0 {
            let idx = self.head;
            out.push(self.data[idx]);
            self.head = (self.head + 1) % N;
            self.len -= 1;
        }
        self.head = 0;
        out
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

    pub fn split(self) -> (HeapProducer<T>, HeapConsumer<T>) {
        self.rb.split()
    }
}

impl<T, const N: usize> Default for RtCommandQueue<T, N> {
    fn default() -> Self {
        Self::new()
    }
}
