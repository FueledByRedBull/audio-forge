//! Source scans of the realtime code: the `RT_REGION_START: name` ..
//! `RT_REGION_END: name` regions, the helpers cut out of them by name, and the
//! function bodies of the DSP block sources the loop calls.
//!
//! `.push(` is not listed: the fixed buffers and rings the loop uses push
//! without allocating, so only the counting-allocator tests can tell a `Vec`
//! push apart.

/// Blocking, allocating, freeing, formatting and logging APIs the RT path
/// must not reach.
const FORBIDDEN: &[&str] = &[
    ".lock()",
    "try_lock",
    "Mutex",
    "RwLock",
    ".read()",
    ".write()",
    ".recv(",
    ".send(",
    ".join(",
    ".unpark(",
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

/// The text from the first `start` up to, not including, the next `end`.
pub(crate) fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let from = source
        .find(start)
        .unwrap_or_else(|| panic!("missing {start}"));
    let to = source[from..]
        .find(end)
        .map(|offset| from + offset)
        .unwrap_or_else(|| panic!("missing {end}"));
    &source[from..to]
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
    let source = format!("{SCANNED} impl<T> Second<T> {{ fn new() -> Vec<f32> {{ Vec::new() }} }}");
    assert_functions_rt_clean("sample", &source, &[("First::new", SETUP)]);
}

/// A `#[cfg(test)]` on a field or statement blanks up to the next `{`, which
/// can swallow a whole impl; the loop's worker calls must stay scanned.
#[cfg(feature = "deepfilter")]
#[test]
fn the_function_scan_reaches_the_deepfilter_worker_link() {
    let code = without_test_items(code_only(include_str!("dsp/deepfilter_ffi/worker.rs")));
    let names: Vec<String> = functions(&code).into_iter().map(|(name, _)| name).collect();
    for needed in [
        "WorkerLink::offer",
        "WorkerLink::late_frames",
        "WorkerLink::take_head",
    ] {
        assert!(
            names.iter().any(|name| name == needed),
            "{needed} not scanned: {names:?}"
        );
    }
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
                ("DeepFilterProcessor::with_worker", SETUP),
                ("DeepFilterProcessor::default", SETUP),
            ][..],
        ),
        #[cfg(feature = "deepfilter")]
        (
            "dsp/deepfilter_ffi/worker.rs",
            include_str!("dsp/deepfilter_ffi/worker.rs"),
            &[
                ("InferenceWorker::spawn", SETUP),
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
                "new_engine",
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
            "audio/processor/dsp_loop.rs",
            include_str!("audio/processor/dsp_loop.rs"),
            &[
                (
                    "DspLoop::new",
                    "runs once on the DSP thread before it reports ready, so before the streams start",
                ),
                (
                    "ensure_resampler_scratch_capacity",
                    "start-up validation on the control thread; builds its error text on rejection",
                ),
                (
                    "record_dsp_thread_panic",
                    "runs once, after the loop has unwound, to record why it died",
                ),
            ][..],
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
