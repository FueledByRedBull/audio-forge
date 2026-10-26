#![cfg_attr(test, allow(clippy::float_cmp))]

//! AudioForge core - high-performance DSP for real-time audio processing
//!
//! Processing chain: input cleanup -> noise gate -> noise suppression ->
//! de-esser -> 10-band EQ -> compressor -> sample/true-peak limiter -> output.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::time::Instant;

pub mod audio;
pub mod dsp;

#[cfg(test)]
pub(crate) mod test_alloc {
    // This counts Rust global allocations and frees on the calling thread, so
    // dropping a buffer on the RT path is caught like allocating one. Calls
    // made inside foreign libraries are outside this instrumentation.
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::Barrier;
    use std::thread;

    pub struct CountingAllocator;

    thread_local! {
        static COUNTING_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
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
            count_allocation();
            System.dealloc(ptr, layout);
        }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

    fn count_allocation() {
        COUNTING_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                ALLOCATION_COUNT.with(|count| count.set(count.get().saturating_add(1)));
            }
        });
    }

    struct AllocationScope {
        previous: bool,
        previous_count: usize,
    }

    impl AllocationScope {
        fn enter() -> Self {
            let previous = COUNTING_ALLOCATIONS.with(|enabled| {
                let previous = enabled.get();
                enabled.set(true);
                previous
            });
            let previous_count = ALLOCATION_COUNT.with(|count| {
                let previous_count = count.get();
                count.set(0);
                previous_count
            });
            Self {
                previous,
                previous_count,
            }
        }
    }

    impl Drop for AllocationScope {
        fn drop(&mut self) {
            let nested_count = ALLOCATION_COUNT.with(Cell::get);
            COUNTING_ALLOCATIONS.with(|enabled| enabled.set(self.previous));
            ALLOCATION_COUNT.with(|count| {
                count.set(self.previous_count.saturating_add(nested_count));
            });
        }
    }

    pub fn allocation_count_during(function: impl FnOnce()) -> usize {
        let _scope = AllocationScope::enter();
        function();
        ALLOCATION_COUNT.with(Cell::get)
    }

    pub fn assert_no_allocations(label: &str, function: impl FnOnce()) {
        let allocations = allocation_count_during(function);
        assert_eq!(allocations, 0, "{label} allocated {allocations} time(s)");
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn concurrent_scope_cannot_reset_another_thread() {
            let allocated = std::sync::Arc::new(Barrier::new(2));
            let release = std::sync::Arc::new(Barrier::new(2));
            let thread_allocated = allocated.clone();
            let thread_release = release.clone();
            let handle = thread::spawn(move || {
                let _scope = AllocationScope::enter();
                std::hint::black_box(vec![0_u8; 17]);
                thread_allocated.wait();
                thread_release.wait();
                ALLOCATION_COUNT.with(Cell::get)
            });

            allocated.wait();
            let _other_scope = AllocationScope::enter();
            release.wait();
            assert!(handle.join().unwrap() > 0);
        }

        #[test]
        fn nonallocating_scope_ignores_concurrent_thread_allocations() {
            let start = std::sync::Arc::new(Barrier::new(2));
            let finished = std::sync::Arc::new(Barrier::new(2));
            let thread_start = start.clone();
            let thread_finished = finished.clone();
            let handle = thread::spawn(move || {
                thread_start.wait();
                let count = allocation_count_during(|| {
                    std::hint::black_box(vec![0_u8; 23]);
                });
                thread_finished.wait();
                count
            });

            let main_count = allocation_count_during(|| {
                start.wait();
                finished.wait();
            });

            assert_eq!(main_count, 0);
            assert!(handle.join().unwrap() > 0);
        }

        #[test]
        fn freeing_a_buffer_counts() {
            let buffer = std::hint::black_box(vec![0_u8; 19]);
            let count = allocation_count_during(move || drop(buffer));
            assert!(count > 0);
        }

        #[test]
        fn nested_scope_count_isolated_and_included_by_outer_scope() {
            let outer = allocation_count_during(|| {
                let inner = allocation_count_during(|| {
                    std::hint::black_box(vec![0_u8; 11]);
                });
                assert!(inner > 0);
                std::hint::black_box(vec![0_u8; 13]);
            });

            assert!(outer >= 2);
        }
    }
}

/// Click measurement for transition tests on tone inputs.
#[cfg(test)]
pub(crate) mod test_signal {
    /// What is left of `signal` after the FIR `1 - 2cos(w)z^-1 + z^-2` for each
    /// tone, which cancels a steady sinusoid exactly. A slowly moving gain
    /// leaves a small residual; a step or a restart leaves a large one.
    pub(crate) fn tone_residual(signal: &[f32], tones_hz: &[f64], sample_rate: f64) -> Vec<f64> {
        let mut residual: Vec<f64> = signal.iter().map(|&sample| f64::from(sample)).collect();
        for &tone_hz in tones_hz {
            let c = 2.0 * (std::f64::consts::TAU * tone_hz / sample_rate).cos();
            residual = (2..residual.len())
                .map(|n| residual[n] - c * residual[n - 1] + residual[n - 2])
                .collect();
        }
        residual
    }

    /// Largest residual magnitude among signal samples `start..end`.
    pub(crate) fn max_residual(
        signal: &[f32],
        tones_hz: &[f64],
        sample_rate: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let offset = 2 * tones_hz.len();
        tone_residual(signal, tones_hz, sample_rate)[start.saturating_sub(offset)..end - offset]
            .iter()
            .fold(0.0, |peak, value| peak.max(value.abs()))
    }

    pub(crate) fn sine(frequency_hz: f64, amplitude: f64, sample_rate: f64, n: usize) -> f32 {
        (amplitude * (std::f64::consts::TAU * frequency_hz * n as f64 / sample_rate).sin()) as f32
    }
}

/// Source scans of the realtime code: the `RT_REGION_START: name` ..
/// `RT_REGION_END: name` regions, the helpers cut out of them by name, and the
/// function bodies of the DSP block sources the loop calls.
///
/// `.push(` is not listed: the fixed buffers and rings the loop uses push
/// without allocating, so only the counting-allocator tests can tell a `Vec`
/// push apart.
#[cfg(test)]
pub(crate) mod rt_scan {
    /// Blocking, allocating, freeing, formatting and logging APIs the RT path
    /// must not reach.
    const FORBIDDEN: [&str; 41] = [
        ".lock()",
        "try_lock",
        "Mutex",
        "RwLock",
        ".read()",
        ".write()",
        ".recv(",
        ".send(",
        ".join(",
        "thread::spawn",
        "std::fs",
        ".reserve(",
        ".resize(",
        ".to_vec(",
        ".drain(",
        "drop(",
        "format!",
        "println!",
        "eprintln!",
        "processor_debug_log!",
        "std::mem::forget",
        "vec!",
        "Vec::",
        "Box::new",
        "Arc::new",
        "Rc::new",
        "String::",
        ".to_string(",
        ".to_owned(",
        ".push_str(",
        ".collect",
        ".extend(",
        ".insert(",
        ".entry(",
        ".clone()",
        ".into_boxed",
        ".sort(",
        ".sort_by(",
        ".sort_by_key(",
        "HashMap",
        "VecDeque",
    ];

    /// A marked region shorter than this has been emptied or mis-marked.
    const MIN_REGION_LINES: usize = 10;

    const START: &str = "RT_REGION_START: ";

    /// `source` with comments and the insides of string and char literals
    /// blanked, so scans and brace matching see code only. Raw strings and
    /// nested block comments are not handled; the scanned sources use neither.
    fn code_only(source: &str) -> String {
        let chars: Vec<char> = source.chars().collect();
        let mut out = String::with_capacity(source.len());
        let mut i = 0;
        while i < chars.len() {
            let blank_to = match (chars[i], chars.get(i + 1)) {
                ('/', Some('/')) => chars[i..]
                    .iter()
                    .position(|&c| c == '\n')
                    .map_or(chars.len(), |n| i + n),
                ('/', Some('*')) => (i + 2..chars.len().saturating_sub(1))
                    .find(|&j| chars[j] == '*' && chars[j + 1] == '/')
                    .map_or(chars.len(), |j| j + 2),
                ('"', _) => {
                    let mut j = i + 1;
                    while j < chars.len() && chars[j] != '"' {
                        j += if chars[j] == '\\' { 2 } else { 1 };
                    }
                    (j + 1).min(chars.len())
                }
                ('\'', Some('\\')) => (i + 3..chars.len())
                    .find(|&j| chars[j] == '\'')
                    .map_or(chars.len(), |j| j + 1),
                ('\'', Some(_)) if chars.get(i + 2) == Some(&'\'') => i + 3,
                _ => {
                    out.push(chars[i]);
                    i += 1;
                    continue;
                }
            };
            out.extend(
                chars[i..blank_to]
                    .iter()
                    .map(|&c| if c == '\n' { c } else { ' ' }),
            );
            i = blank_to;
        }
        assert!(out.is_ascii(), "non-ASCII outside comments and literals");
        out
    }

    /// Index of the `}` closing the `{` at `open`.
    fn matching_brace(code: &str, open: usize) -> usize {
        let mut depth = 0_usize;
        for (index, byte) in code.bytes().enumerate().skip(open) {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return index;
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced braces");
    }

    /// Blank every `#[cfg(test)]` item: up to the first `;`, or through the
    /// first `{ .. }` block, after the attribute.
    fn without_test_items(mut code: String) -> String {
        const ATTRIBUTE: &str = "#[cfg(test)]";
        while let Some(start) = code.find(ATTRIBUTE) {
            let after = start + ATTRIBUTE.len();
            let end = match code[after..].find(['{', ';']) {
                Some(offset) if code.as_bytes()[after + offset] == b'{' => {
                    matching_brace(&code, after + offset)
                }
                Some(offset) => after + offset,
                None => code.len() - 1,
            };
            code.replace_range(start..=end, &" ".repeat(end + 1 - start));
        }
        code
    }

    /// The type an `impl` header is for: `<T> Trait for Foo<T> where ..` is
    /// `Foo`.
    fn impl_type(header: &str) -> &str {
        let header = header.split(" where ").next().unwrap_or(header).trim();
        let mut ty = header.rsplit(" for ").next().unwrap_or(header).trim();
        if ty.starts_with('<') {
            // Skip the generic parameters; the `>` of `->` does not close them.
            let mut depth = 0_usize;
            let mut previous = ' ';
            for (index, c) in ty.char_indices() {
                match c {
                    '<' => depth += 1,
                    '>' if previous != '-' => {
                        depth -= 1;
                        if depth == 0 {
                            ty = ty[index + 1..].trim();
                            break;
                        }
                    }
                    _ => {}
                }
                previous = c;
            }
        }
        ty.split(|c: char| c == '<' || c.is_whitespace())
            .next()
            .unwrap_or(ty)
    }

    /// Byte range of the body and the type of every `impl` block in `code`.
    fn impl_blocks(code: &str) -> Vec<(usize, usize, &str)> {
        let is_word = |c: char| c.is_alphanumeric() || c == '_';
        let mut found = Vec::new();
        for (at, _) in code.match_indices("impl") {
            let before = code[..at].trim_end();
            let before = before.strip_suffix("unsafe").unwrap_or(before).trim_end();
            let starts_item = before.is_empty() || before.ends_with(['}', ';', '{', ']']);
            let after = &code[at + "impl".len()..];
            if !starts_item || after.starts_with(is_word) {
                continue;
            }
            let Some(open) = after.find('{') else {
                continue;
            };
            let open = at + "impl".len() + open;
            let header = &code[at + "impl".len()..open];
            found.push((open, matching_brace(code, open), impl_type(header)));
        }
        found
    }

    /// Qualified name (`Type::name` inside an `impl`) and body (`{ .. }`) of
    /// every function in `code` that has one.
    fn functions(code: &str) -> Vec<(String, &str)> {
        let is_word = |c: char| c.is_alphanumeric() || c == '_';
        let impls = impl_blocks(code);
        let mut found = Vec::new();
        for (at, _) in code.match_indices("fn ") {
            if code[..at].chars().next_back().is_some_and(is_word) {
                continue;
            }
            let name_start = at + "fn ".len();
            let name_end = code[name_start..]
                .find(|c: char| !is_word(c))
                .map_or(code.len(), |n| name_start + n);
            let mut depth = 0_i32;
            for (offset, byte) in code.bytes().enumerate().skip(name_end) {
                match byte {
                    b'(' | b'[' => depth += 1,
                    b')' | b']' => depth -= 1,
                    b'{' if depth == 0 => {
                        let close = matching_brace(code, offset);
                        let name = &code[name_start..name_end];
                        let owner = impls
                            .iter()
                            .filter(|(open, close, _)| (*open..=*close).contains(&at))
                            .max_by_key(|(open, ..)| *open);
                        let qualified = match owner {
                            Some((.., ty)) => format!("{ty}::{name}"),
                            None => name.to_string(),
                        };
                        found.push((qualified, &code[offset..=close]));
                        break;
                    }
                    b';' if depth == 0 => break,
                    _ => {}
                }
            }
        }
        found
    }

    /// Check every function body of `source` outside `#[cfg(test)]` items.
    /// `non_rt` lists the functions (`Type::name`, or `name` outside an `impl`)
    /// that never run on a callback or in the DSP loop (setup, parameter
    /// validation, offline analysis), each with the reason; an entry that no
    /// longer needs to be listed fails too.
    pub(crate) fn assert_functions_rt_clean(file: &str, source: &str, non_rt: &[(&str, &str)]) {
        let code = without_test_items(code_only(source));
        let bodies = functions(&code);
        assert!(
            bodies.len() >= 3,
            "{file}: found {} functions",
            bodies.len()
        );
        let mut needed = vec![false; non_rt.len()];
        let mut offenders = Vec::new();
        for (name, body) in &bodies {
            let Some(pattern) = FORBIDDEN.iter().find(|pattern| body.contains(**pattern)) else {
                continue;
            };
            match non_rt.iter().position(|(listed, _)| listed == name) {
                Some(entry) => needed[entry] = true,
                None => offenders.push(format!("{name} ({pattern})")),
            }
        }
        assert!(
            offenders.is_empty(),
            "{file}: forbidden RT patterns in {}; remove them, or list the function as non-RT with a reason",
            offenders.join(", ")
        );
        for ((name, reason), needed) in non_rt.iter().zip(needed) {
            assert!(!reason.is_empty(), "{file}: {name} needs a reason");
            assert!(needed, "{file}: {name} no longer needs its non-RT entry");
        }
    }

    /// The body of the region `name` (between its marker lines).
    pub(crate) fn marked_region<'a>(source: &'a str, name: &str) -> &'a str {
        let start_marker = format!("{START}{name}");
        let end_marker = format!("RT_REGION_END: {name}");
        let start = source
            .find(&start_marker)
            .unwrap_or_else(|| panic!("missing {start_marker}"));
        let body_start = source[start..]
            .find('\n')
            .map_or(start, |offset| start + offset + 1);
        let end = source[body_start..]
            .find(&end_marker)
            .map(|offset| body_start + offset)
            .unwrap_or_else(|| panic!("missing {end_marker}"));
        &source[body_start..end]
    }

    pub(crate) fn assert_rt_clean(label: &str, text: &str) {
        let code = code_only(text);
        for pattern in FORBIDDEN {
            assert!(
                !code.contains(pattern),
                "{label} contains forbidden RT pattern {pattern}"
            );
        }
    }

    /// Check every marked region of `source`; returns their names.
    pub(crate) fn assert_marked_regions_clean<'a>(file: &str, source: &'a str) -> Vec<&'a str> {
        let names: Vec<&str> = source
            .match_indices(START)
            .map(|(index, _)| {
                source[index + START.len()..]
                    .split_whitespace()
                    .next()
                    .unwrap_or_else(|| panic!("{file}: unnamed RT region"))
            })
            .collect();
        for name in &names {
            let region = marked_region(source, name);
            let lines = code_only(region)
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count();
            assert!(
                lines >= MIN_REGION_LINES,
                "{file}:{name} has only {lines} lines of code"
            );
            assert_rt_clean(&format!("{file}:{name}"), region);
        }
        names
    }

    const SCANNED: &str = r#"
        // format! in a comment, and "vec![]" in a string, are not code.
        fn quiet(buffer: &mut [f32]) -> &'static str { buffer[0] = 'x' as u8 as f32; "vec![]" }
        fn also_quiet<const N: usize>(a: [f32; N]) -> f32 where f32: Copy { a[0] }
        fn third() {}
        #[cfg(test)]
        mod tests { fn fixture() { let _ = vec![1]; } }
        struct First;
        struct Second<T>(T);
        impl First { fn new() -> Vec<f32> { Vec::new() } }
        impl<T: Fn(f32) -> f32> Clone for Second<T> where T: Copy { fn clone(&self) -> Self { loop {} } }
    "#;

    #[test]
    fn the_function_scan_ignores_comments_strings_and_test_items() {
        assert_functions_rt_clean("sample", SCANNED, &[("First::new", SETUP)]);
    }

    #[test]
    #[should_panic(expected = "offender (.to_vec()")]
    fn the_function_scan_flags_an_allocating_function() {
        let source = format!("{SCANNED} fn offender(x: &[f32]) -> usize {{ x.to_vec().len() }}");
        assert_functions_rt_clean("sample", &source, &[("First::new", SETUP)]);
    }

    #[test]
    #[should_panic(expected = "offender no longer needs its non-RT entry")]
    fn the_function_scan_rejects_a_stale_non_rt_entry() {
        assert_functions_rt_clean(
            "sample",
            SCANNED,
            &[("First::new", SETUP), ("offender", SETUP)],
        );
    }

    /// A same-named function of another type is not covered by an entry for
    /// the first one.
    #[test]
    #[should_panic(expected = "Second::new (Vec::)")]
    fn the_function_scan_matches_the_owning_type_not_just_the_name() {
        let source =
            format!("{SCANNED} impl<T> Second<T> {{ fn new() -> Vec<f32> {{ Vec::new() }} }}");
        assert_functions_rt_clean("sample", &source, &[("First::new", SETUP)]);
    }

    /// Setup, validation and offline code in the scanned sources, by function
    /// name (all functions of that name in the file), with why the loop never
    /// reaches it.
    const SETUP: &str = "built once on the control thread before the stream starts";
    const VALIDATION: &str = "setter validation; the loop applies snapshots that already passed it";
    const OFFLINE: &str = "offline analysis or UI curve, never called from the loop";

    #[test]
    fn dsp_block_sources_hold_no_blocking_or_allocating_apis() {
        for (file, source, non_rt) in [
            ("dsp/biquad.rs", include_str!("dsp/biquad.rs"), &[][..]),
            ("dsp/compressor.rs", include_str!("dsp/compressor.rs"), &[][..]),
            (
                "dsp/compressor/makeup.rs",
                include_str!("dsp/compressor/makeup.rs"),
                &[][..],
            ),
            ("dsp/deesser.rs", include_str!("dsp/deesser.rs"), &[][..]),
            (
                "dsp/eq.rs",
                include_str!("dsp/eq.rs"),
                &[
                    ("EqBandConfig::validate", VALIDATION),
                    ("eq_max_frequency_hz", VALIDATION),
                    ("validate_eq_frequency_hz", VALIDATION),
                    ("validate_eq_gain_db", VALIDATION),
                    ("validate_eq_q", VALIDATION),
                    ("validate_eq_slope", VALIDATION),
                    ("ParametricEQ::magnitude_response_db", OFFLINE),
                ][..],
            ),
            ("dsp/gate.rs", include_str!("dsp/gate.rs"), &[][..]),
            #[cfg(feature = "deepfilter")]
            (
                "dsp/deepfilter_ffi.rs",
                include_str!("dsp/deepfilter_ffi.rs"),
                &[
                    (
                        "DeepFilterRuntimeConfig::try_new",
                        "runtime config validation; builds its error text on rejection",
                    ),
                    ("app_owned_paths", SETUP),
                    ("canonical_app_owned_path", SETUP),
                    ("configure_app_owned_paths", SETUP),
                    ("configured_app_owned_paths", SETUP),
                    ("find_model_path", SETUP),
                    ("DeepFilterLib::trusted_library_candidates", SETUP),
                    ("DeepFilterLib::load_symbols_from_path", SETUP),
                    ("DeepFilterLib::try_load", SETUP),
                    ("DeepFilterFFI::new", SETUP),
                    ("DeepFilterFFI::open", SETUP),
                    ("DeepFilterProcessor::new_with_config", SETUP),
                    ("DeepFilterProcessor::from_library", SETUP),
                    ("DeepFilterProcessor::default", SETUP),
                ][..],
            ),
            (
                "dsp/limiter.rs",
                include_str!("dsp/limiter.rs"),
                &[
                    ("Limiter::new_with_lookahead_ms", SETUP),
                    (
                        "Limiter::set_lookahead_ms",
                        "resizes the delay line and is documented as not realtime-safe; only the Python setup calls it",
                    ),
                ][..],
            ),
            (
                "dsp/loudness.rs",
                include_str!("dsp/loudness.rs"),
                &[
                    ("LoudnessMeter::new", SETUP),
                    ("integrated_loudness_lufs", OFFLINE),
                ][..],
            ),
            (
                "dsp/noise_suppressor.rs",
                include_str!("dsp/noise_suppressor.rs"),
                &[(
                    "new_controlled_noise_suppression_engine",
                    "engine factory; engines are built on the control thread and queued to the loop",
                )][..],
            ),
            (
                "dsp/rnnoise.rs",
                include_str!("dsp/rnnoise.rs"),
                &[("RNNoiseProcessor::default", SETUP)][..],
            ),
            (
                "dsp/true_peak.rs",
                include_str!("dsp/true_peak.rs"),
                &[("generate_fir", SETUP)][..],
            ),
            ("dsp/util.rs", include_str!("dsp/util.rs"), &[][..]),
            (
                "audio/rt.rs",
                include_str!("audio/rt.rs"),
                &[
                    ("FixedAudioBuffer::new", SETUP),
                    ("FixedAudioRing::new", SETUP),
                ][..],
            ),
            (
                "audio/buffer.rs",
                include_str!("audio/buffer.rs"),
                &[("AudioRingBuffer::split", SETUP)][..],
            ),
            (
                "audio/processor/output_writer.rs",
                include_str!("audio/processor/output_writer.rs"),
                &[][..],
            ),
            (
                "audio/processor/resampling.rs",
                include_str!("audio/processor/resampling.rs"),
                &[
                    ("build_sinc_resampler_with_quality", SETUP),
                    ("product_resampler_configuration", SETUP),
                    ("simulate_product_resampler", OFFLINE),
                ][..],
            ),
            (
                "audio/processor/routing.rs",
                include_str!("audio/processor/routing.rs"),
                &[("static_input_pre_filter_response_db", OFFLINE)][..],
            ),
        ] {
            assert_functions_rt_clean(file, source, non_rt);
        }
    }
}

// Re-export main types
pub use audio::{AudioProcessor, PyAudioProcessor};
pub use dsp::{Biquad, Compressor, DeEsser, Limiter, NoiseGate, ParametricEQ, RNNoiseProcessor};

/// Run a pyfunction body so a Rust panic becomes a `RuntimeError`. PyO3's own
/// `PanicException` derives from `BaseException`, which the app's
/// `except Exception` handlers do not catch.
pub(crate) fn catch_panic<T>(body: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_or_else(|payload| {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "internal error: the native audio core panicked: {}",
            panic_message(payload.as_ref())
        )))
    })
}

/// The text a panic carried (`panic!` with a literal or a formatted message).
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

const MIN_SAMPLE_RATE: f64 = 8_000.0;
const MAX_SAMPLE_RATE: f64 = 768_000.0;

/// The sample-rate range the offline entry points accept (EQ responses,
/// simulators, the resampler).
pub(crate) fn validate_sample_rate(sample_rate: f64) -> PyResult<()> {
    if !sample_rate.is_finite() || !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sample_rate must be finite and between {MIN_SAMPLE_RATE:.0} and {MAX_SAMPLE_RATE:.0} Hz"
        )));
    }
    Ok(())
}

#[pyfunction]
fn eq_magnitude_response(
    frequencies_hz: Vec<f64>,
    bands: Vec<(f64, f64, f64)>,
    sample_rate: f64,
) -> PyResult<Vec<f64>> {
    catch_panic(|| {
        validate_sample_rate(sample_rate)?;
        if bands.len() != dsp::eq::NUM_BANDS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} EQ bands, got {}",
                dsp::eq::NUM_BANDS,
                bands.len()
            )));
        }
        let nyquist = sample_rate / 2.0;
        for (index, (frequency_hz, gain_db, q)) in bands.iter().copied().enumerate() {
            dsp::eq::validate_eq_frequency_hz(frequency_hz, sample_rate)
                .and_then(|()| dsp::eq::validate_eq_gain_db(gain_db))
                .and_then(|()| dsp::eq::validate_eq_q(q))
                .map_err(|message| {
                    pyo3::exceptions::PyValueError::new_err(format!("band {index}: {message}"))
                })?;
        }
        if frequencies_hz.iter().any(|frequency_hz| {
            !frequency_hz.is_finite() || *frequency_hz < 0.0 || *frequency_hz > nyquist
        }) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "response frequencies must be finite and between 0 Hz and Nyquist",
            ));
        }

        let mut eq = ParametricEQ::new(sample_rate);
        for (index, (frequency_hz, gain_db, q)) in bands.into_iter().enumerate() {
            eq.set_band_frequency(index, frequency_hz);
            eq.set_band_gain(index, gain_db);
            eq.set_band_q(index, q);
        }
        Ok(eq.magnitude_response_db(&frequencies_hz))
    })
}

type PyEqBandV2 = (String, f64, f64, f64, u8, bool);

fn parse_eq_v2_bands(bands: &[PyEqBandV2], sample_rate: f64) -> PyResult<Vec<dsp::EqBandConfig>> {
    validate_sample_rate(sample_rate)?;
    if bands.len() != dsp::eq::NUM_BANDS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {} EQ bands, got {}",
            dsp::eq::NUM_BANDS,
            bands.len()
        )));
    }
    let mut configs = Vec::with_capacity(dsp::eq::NUM_BANDS);
    for (index, (filter_type, frequency_hz, gain_db, q, slope, enabled)) in bands.iter().enumerate()
    {
        let filter_type = dsp::EqFilterType::from_name(filter_type).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "band {index} has unsupported EQ filter type: {filter_type}"
            ))
        })?;
        let config = dsp::EqBandConfig {
            filter_type,
            frequency_hz: *frequency_hz,
            gain_db: *gain_db,
            q: *q,
            slope_db_per_octave: *slope,
            enabled: *enabled,
        };
        config
            .validate(index, sample_rate)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        configs.push(config);
    }
    Ok(configs)
}

#[pyfunction]
fn eq_magnitude_response_v2(
    frequencies_hz: Vec<f64>,
    bands: Vec<PyEqBandV2>,
    sample_rate: f64,
) -> PyResult<Vec<f64>> {
    catch_panic(|| {
        let configs = parse_eq_v2_bands(&bands, sample_rate)?;
        let nyquist = sample_rate / 2.0;
        if frequencies_hz.iter().any(|frequency_hz| {
            !frequency_hz.is_finite() || *frequency_hz < 0.0 || *frequency_hz > nyquist
        }) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "response frequencies must be finite and between 0 Hz and Nyquist",
            ));
        }

        let mut eq = ParametricEQ::new(sample_rate);
        for (index, config) in configs.into_iter().enumerate() {
            eq.set_band_config(index, config);
        }
        Ok(eq.magnitude_response_db(&frequencies_hz))
    })
}

#[pyfunction]
#[pyo3(signature = (audio, sample_rate, bands, return_output_audio=false))]
fn simulate_eq_v2(
    py: Python<'_>,
    audio: numpy::PyReadonlyArray1<'_, f32>,
    sample_rate: f64,
    bands: Vec<PyEqBandV2>,
    return_output_audio: bool,
) -> PyResult<Py<PyAny>> {
    crate::catch_panic(|| {
        let configs = parse_eq_v2_bands(&bands, sample_rate)?;
        let input = audio.as_slice()?.to_vec();
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "audio must contain only finite samples",
            ));
        }
        let sample_count = input.len();

        let (
            output,
            input_peak,
            output_peak,
            input_true_peak,
            output_true_peak,
            input_rms,
            output_rms,
            max_response_db,
            runtime_ms,
            non_finite_output,
        ) = py.detach(move || {
            let mut eq = ParametricEQ::new(sample_rate);
            for (index, config) in configs.into_iter().enumerate() {
                eq.set_band_config(index, config);
            }
            eq.reset();

            let mut output = input.clone();
            let started = Instant::now();
            eq.process_block_inplace(&mut output);
            let runtime_ms = started.elapsed().as_secs_f64() * 1000.0;

            let input_square_sum = input
                .iter()
                .map(|sample| f64::from(*sample) * f64::from(*sample))
                .sum::<f64>();
            let output_square_sum = output
                .iter()
                .map(|sample| f64::from(*sample) * f64::from(*sample))
                .sum::<f64>();
            let divisor = input.len().max(1) as f64;
            let input_peak = input
                .iter()
                .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
            let output_peak = output
                .iter()
                .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
            let mut input_true_peak_detector = dsp::TruePeakDetector::new();
            let mut output_true_peak_detector = dsp::TruePeakDetector::new();
            let input_true_peak = input_true_peak_detector.process_block(&input);
            let output_true_peak = output_true_peak_detector.process_block(&output);
            let response_frequencies = (0..512)
                .map(|index| 20.0 * (20_000.0_f64 / 20.0).powf(index as f64 / 511.0))
                .collect::<Vec<_>>();
            let max_response_db = eq
                .magnitude_response_db(&response_frequencies)
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            let input_rms = (input_square_sum / divisor).sqrt();
            let output_rms = (output_square_sum / divisor).sqrt();
            let non_finite_output = output.iter().any(|sample| !sample.is_finite());

            (
                output,
                input_peak,
                output_peak,
                input_true_peak,
                output_true_peak,
                input_rms,
                output_rms,
                max_response_db,
                runtime_ms,
                non_finite_output,
            )
        });

        let diagnostics = pyo3::types::PyDict::new(py);
        diagnostics.set_item("input_sample_peak", input_peak)?;
        diagnostics.set_item("output_sample_peak", output_peak)?;
        diagnostics.set_item("input_true_peak", input_true_peak)?;
        diagnostics.set_item("output_true_peak", output_true_peak)?;
        diagnostics.set_item("input_rms", input_rms)?;
        diagnostics.set_item("output_rms", output_rms)?;
        diagnostics.set_item("max_response_db", max_response_db)?;
        diagnostics.set_item("runtime_ms", runtime_ms)?;
        diagnostics.set_item("sample_count", sample_count)?;
        diagnostics.set_item("algorithmic_latency_samples", 0)?;
        diagnostics.set_item("non_finite_output", non_finite_output)?;
        if return_output_audio {
            diagnostics.set_item("output_audio", numpy::PyArray1::from_vec(py, output))?;
        }
        Ok(diagnostics.into_any().unbind())
    })
}

#[pyfunction]
fn measure_integrated_loudness(
    py: Python<'_>,
    audio: numpy::PyReadonlyArray1<'_, f32>,
    sample_rate: u32,
) -> PyResult<f64> {
    let samples = audio.as_slice()?.to_vec();
    catch_panic(|| {
        py.detach(move || dsp::loudness::integrated_loudness_lufs(&samples, sample_rate))
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    })
}

#[pyfunction]
fn _k_weighted_48k<'py>(
    py: Python<'py>,
    audio: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    catch_panic(|| {
        let mut samples: Vec<f64> = audio.as_array().iter().copied().collect();
        let output = py.detach(move || {
            dsp::loudness::k_weighted_48k_inplace(&mut samples);
            samples
        });
        Ok(numpy::PyArray1::from_vec(py, output))
    })
}

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
    m.add_class::<audio::InputFormats>()?;
    m.add_function(wrap_pyfunction!(audio::list_input_devices, m)?)?;
    m.add_function(wrap_pyfunction!(audio::list_input_device_formats, m)?)?;
    m.add_function(wrap_pyfunction!(audio::list_output_devices, m)?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::simulate_auto_eq_chain,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::simulate_auto_makeup_control,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::simulate_product_resampler_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::product_resampler_configuration,
        m
    )?)?;
    #[cfg(feature = "vad")]
    m.add_function(wrap_pyfunction!(
        audio::processor::simulate_gate_suppressor_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(eq_magnitude_response, m)?)?;
    m.add_function(wrap_pyfunction!(eq_magnitude_response_v2, m)?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::input_prefilter_response_db,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(simulate_eq_v2, m)?)?;
    m.add_function(wrap_pyfunction!(measure_integrated_loudness, m)?)?;
    m.add_function(wrap_pyfunction!(_k_weighted_48k, m)?)?;

    #[cfg(feature = "vad")]
    m.add_function(wrap_pyfunction!(
        audio::processor::analyze_vad_probabilities,
        m
    )?)?;

    #[cfg(feature = "deepfilter")]
    m.add_function(wrap_pyfunction!(configure_deepfilter_runtime_paths, m)?)?;

    Ok(())
}

#[cfg(feature = "deepfilter")]
#[pyfunction]
#[pyo3(signature = (library_path=None, model_path=None))]
fn configure_deepfilter_runtime_paths(
    library_path: Option<&str>,
    model_path: Option<&str>,
) -> PyResult<()> {
    catch_panic(|| {
        dsp::deepfilter_ffi::configure_app_owned_paths(library_path, model_path)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    fn default_bands() -> Vec<(f64, f64, f64)> {
        (0..dsp::eq::NUM_BANDS)
            .map(|index| (100.0 * (index + 1) as f64, 0.0, 1.0))
            .collect()
    }

    fn eq_response_error(bands: Vec<(f64, f64, f64)>, sample_rate: f64) -> String {
        eq_magnitude_response(vec![1000.0], bands, sample_rate)
            .expect_err("out-of-range input must be rejected")
            .to_string()
    }

    #[test]
    fn eq_response_v1_rejects_values_outside_the_live_eq_ranges() {
        assert!(eq_magnitude_response(vec![1000.0], default_bands(), 48_000.0).is_ok());

        let mut bands = default_bands();
        bands[3].1 = 1e300;
        assert!(eq_response_error(bands, 48_000.0).contains("band 3"));

        let mut bands = default_bands();
        bands[4].2 = 1e300;
        assert!(eq_response_error(bands, 48_000.0).contains("band 4"));

        let mut bands = default_bands();
        bands[5].0 = 1e-3;
        assert!(eq_response_error(bands, 48_000.0).contains("band 5"));

        assert!(eq_response_error(default_bands(), 1e300).contains("sample_rate"));
    }

    #[test]
    fn eq_response_v2_bounds_sample_rate_like_simulate_eq_v2() {
        let bands = (0..dsp::eq::NUM_BANDS)
            .map(|index| {
                (
                    "bell".to_string(),
                    100.0 * (index + 1) as f64,
                    0.0,
                    1.0,
                    12,
                    true,
                )
            })
            .collect::<Vec<_>>();
        for sample_rate in [1e9, 1_000.0] {
            let error = eq_magnitude_response_v2(vec![1000.0], bands.clone(), sample_rate)
                .expect_err("sample rate outside 8-768 kHz must be rejected");
            assert!(error.to_string().contains("sample_rate"));
        }
    }

    #[pyfunction]
    fn guarded_panic() -> PyResult<()> {
        catch_panic(|| panic!("test panic"))
    }

    /// Every module-level `#[pyfunction]` must run its body under `catch_panic`,
    /// or a Rust panic reaches Python as a `BaseException` the app does not catch.
    #[test]
    fn every_pyfunction_runs_under_catch_panic() {
        let lib = include_str!("lib.rs");
        let production = &lib[..lib
            .find(
                "#[cfg(test)]
mod boundary_tests",
            )
            .unwrap()];
        let api = include_str!("audio/processor/python_api.rs");
        let device = include_str!("audio/device.rs");
        let resampling = include_str!("audio/processor/resampling.rs");
        let mut checked = 0;
        for source in [production, api, device, resampling] {
            // Also `#[pyfunction(name = ...)]`.
            for chunk in source.split("#[pyfunction").skip(1) {
                let body = &chunk[..chunk
                    .find(
                        "
}
",
                    )
                    .unwrap()];
                let name = body.split("fn ").nth(1).unwrap().split('(').next().unwrap();
                assert!(
                    body.contains("catch_panic"),
                    "{name} is not under catch_panic"
                );
                checked += 1;
            }
        }
        assert!(checked >= 15, "only {checked} pyfunctions found");
    }

    #[test]
    fn pyfunction_panic_becomes_a_runtime_error() {
        Python::initialize();
        Python::attach(|py| {
            let guarded = wrap_pyfunction!(guarded_panic, py).unwrap();
            let error = guarded.call0().unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            assert!(error.is_instance_of::<pyo3::exceptions::PyException>(py));
            assert!(
                error.to_string().contains("test panic"),
                "the panic message is lost: {error}"
            );
        });
    }
}
