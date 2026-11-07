use super::*;

#[test]
fn excess_is_the_term_that_fails_most() {
    // Headroom 0.25 dB short of the target, limiter 0.5 dB under its warning, true peak 1 dB over.
    assert_eq!(headroom_excess_db(0.75, 0.5, 1.5), 1.0);
    assert_eq!(headroom_excess_db(3.0, 0.0, 0.0), -0.5);
    assert_eq!(headroom_excess_db(2.0, 3.0, 0.0), 2.0);
}

#[test]
fn a_tie_keeps_the_first_term_like_python_max() {
    // 1.0 - 1.0 is +0.0; a later term that ties it at -0.0 does not replace it.
    assert_eq!(
        headroom_excess_db(1.0, 0.0, 0.5).to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn the_rule_fits_at_or_below_zero() {
    assert!(headroom_excess_db(1.0, 1.0, 0.5) <= 0.0);
    assert!(headroom_excess_db(0.99, 0.0, 0.0) > 0.0);
}
