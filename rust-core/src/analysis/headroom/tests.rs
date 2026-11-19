use super::*;

#[test]
fn excess_is_the_term_that_fails_most() {
    // Headroom 0.25 dB short of the target, limiter 0.5 dB under its warning, true peak 1 dB over.
    assert_eq!(headroom_excess_db(0.75, 0.5, 1.5), 1.0);
    assert_eq!(headroom_excess_db(3.0, 0.0, 0.0), -0.5);
    assert_eq!(headroom_excess_db(2.0, 3.0, 0.0), 2.0);
}

#[test]
fn tied_terms_give_their_shared_value() {
    // Which tied term wins cannot be observed: each term is a difference
    // with a nonzero constant, which is never -0.0 (an exact zero difference
    // is +0.0), so equal terms have equal bits and `>` acts as `>=` would.
    assert_eq!(headroom_excess_db(0.5, 1.5, 1.0), 0.5);
    assert_eq!(
        headroom_excess_db(1.0, 1.0, 0.5).to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(headroom_excess_db(2.0, 0.0, -0.5), -1.0);
}

#[test]
fn the_rule_fits_at_or_below_zero() {
    assert!(headroom_excess_db(1.0, 1.0, 0.5) <= 0.0);
    assert!(headroom_excess_db(0.99, 0.0, 0.0) > 0.0);
}
