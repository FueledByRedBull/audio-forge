//! Lock-free ring buffer for audio data
//!
//! Thread-safe circular buffer for passing audio between threads.
//! Adapted from Spectral Workbench project.

use ringbuf::{HeapConsumer, HeapProducer, HeapRb};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Thread-safe audio ring buffer
pub struct AudioRingBuffer {
    producer: HeapProducer<f32>,
    consumer: HeapConsumer<f32>,
    capacity: usize,
}

impl AudioRingBuffer {
    /// Create new ring buffer with given capacity
    ///
    /// # Arguments
    /// * `capacity` - Buffer capacity in samples
    pub fn new(capacity: usize) -> Self {
        let rb = HeapRb::<f32>::new(capacity);
        let (producer, consumer) = rb.split();

        Self {
            producer,
            consumer,
            capacity,
        }
    }

    /// Split into producer and consumer ends
    pub fn split(self) -> (AudioProducer, AudioConsumer) {
        let dropped_count = Arc::new(AtomicU64::new(0));

        (
            AudioProducer {
                producer: self.producer,
                capacity: self.capacity,
                dropped_count: Arc::clone(&dropped_count),
            },
            AudioConsumer {
                consumer: self.consumer,
                capacity: self.capacity,
                last_sample: 0.0,  // Start at silence
            },
        )
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Producer end of audio ring buffer (for writing)
pub struct AudioProducer {
    producer: HeapProducer<f32>,
    capacity: usize,
    dropped_count: Arc<AtomicU64>,
}

impl AudioProducer {
    /// Write samples to buffer
    ///
    /// # Arguments
    /// * `samples` - Samples to write
    ///
    /// # Returns
    /// Number of samples actually written (may be less if buffer is full)
    pub fn write(&mut self, samples: &[f32]) -> usize {
        let written = self.producer.push_slice(samples);
        let dropped = samples.len() - written;
        if dropped > 0 {
            self.dropped_count.fetch_add(dropped as u64, Ordering::Relaxed);
        }
        written
    }

    /// Check if buffer has space for n samples
    pub fn has_space(&self, n: usize) -> bool {
        self.free_len() >= n
    }

    /// Get number of free slots
    pub fn free_len(&self) -> usize {
        self.capacity - self.producer.len()
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get dropped sample count
    pub fn get_dropped_count(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Reset dropped sample counter
    pub fn reset_dropped_count(&self) {
        self.dropped_count.store(0, Ordering::Relaxed);
    }

    /// Get reference to dropped counter (for sharing with processor)
    pub fn dropped_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.dropped_count)
    }
}

/// Consumer end of audio ring buffer (for reading)
pub struct AudioConsumer {
    consumer: HeapConsumer<f32>,
    capacity: usize,
    last_sample: f32,  // Track last sample for interpolation during underrun
}

impl AudioConsumer {
    /// Read samples from buffer
    ///
    /// # Arguments
    /// * `buffer` - Output buffer to read into
    ///
    /// # Returns
    /// Number of samples actually read (may be less if buffer doesn't have enough)
    pub fn read(&mut self, buffer: &mut [f32]) -> usize {
        let count = self.consumer.pop_slice(buffer);
        if count > 0 {
            self.last_sample = buffer[count - 1];
        }
        count
    }

    /// Check if buffer has n samples available
    pub fn has_data(&self, n: usize) -> bool {
        self.consumer.len() >= n
    }

    /// Get number of available samples
    pub fn len(&self) -> usize {
        self.consumer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.consumer.is_empty()
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the last sample read (for interpolation during underrun)
    pub fn last_sample(&self) -> f32 {
        self.last_sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_write_read() {
        let rb = AudioRingBuffer::new(1024);
        let (mut producer, mut consumer) = rb.split();

        // Write some data
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let written = producer.write(&data);
        assert_eq!(written, 5);

        // Read it back
        let mut output = vec![0.0f32; 5];
        let read = consumer.read(&mut output);
        assert_eq!(read, 5);
        assert_eq!(output, data);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let rb = AudioRingBuffer::new(10);
        let (mut producer, mut consumer) = rb.split();

        // Try to write more than capacity
        let data = vec![1.0f32; 20];
        let written = producer.write(&data);

        // Should only write up to capacity
        assert!(written <= 10);

        // Read back
        let mut output = vec![0.0f32; 20];
        let read = consumer.read(&mut output);
        assert_eq!(read, written);
    }

    #[test]
    fn test_ring_buffer_underflow() {
        let rb = AudioRingBuffer::new(1024);
        let (mut _producer, mut consumer) = rb.split();

        // Try to read from empty buffer
        let mut output = vec![0.0f32; 10];
        let read = consumer.read(&mut output);

        // Should read 0
        assert_eq!(read, 0);
    }

    #[test]
    fn test_ring_buffer_dropped_samples() {
        let rb = AudioRingBuffer::new(10);
        let (mut producer, _consumer) = rb.split();

        // Initially no drops
        assert_eq!(producer.get_dropped_count(), 0);

        // Write more than capacity
        let data = vec![1.0f32; 20];
        let written = producer.write(&data);

        // Should only write up to capacity (10)
        assert!(written <= 10);

        // Should have dropped samples
        let dropped = producer.get_dropped_count();
        assert!(dropped > 0);
        assert_eq!(dropped, (20 - written) as u64);

        // Reset counter
        producer.reset_dropped_count();
        assert_eq!(producer.get_dropped_count(), 0);
    }
}
