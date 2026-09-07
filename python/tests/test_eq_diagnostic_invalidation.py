from unittest.mock import Mock

from mic_eq.ui.eq_panel import EQPanel


def _bands() -> list[tuple[float, float, float]]:
    return [
        (80.0 * (1.6**index), 1.0 if index == 4 else 0.0, 1.41)
        for index in range(10)
    ]


def _diagnostics() -> dict[str, float | str]:
    return {
        "analysis_confidence": 0.82,
        "eq_confidence": 0.76,
        "capture_confidence": 0.88,
        "validation_confidence": 0.79,
        "validation_before_error_db": 4.2,
        "validation_after_error_db": 2.1,
        "validation_gain_scale": 0.85,
        "target_profile": "broadcast:adaptive",
        "recommendation_status": "apply",
    }


def test_manual_eq_edits_invalidate_diagnostics_and_fresh_results_restore_them(
    qapp,
):
    panel = EQPanel(Mock())
    diagnostics = _diagnostics()

    def apply_candidate() -> None:
        panel.apply_auto_eq_results(_bands(), diagnostics=diagnostics)
        assert panel._auto_eq_diagnostics == diagnostics

    def assert_cleared() -> None:
        assert panel._auto_eq_diagnostics is None
        assert panel.auto_eq_diag_label.text() == (
            "Auto-EQ: no calibration diagnostics"
        )

    band = panel.band_sliders[4]

    def edit_gain() -> None:
        band.slider.setValue(band.slider.value() + 1)
        band._rate_limiter.flush()

    def edit_q() -> None:
        band.q_spinbox.setValue(1.5)
        band._rate_limiter.flush()

    def edit_slope() -> None:
        band.filter_type_combo.setCurrentIndex(
            band.filter_type_combo.findData("high_pass")
        )
        panel.set_auto_eq_diagnostics(diagnostics)
        band.slope_combo.setCurrentIndex(band.slope_combo.findData(24))

    def edit_frequency() -> None:
        band.frequency_spinbox.setValue(1100.0)
        band._frequency_rate_limiter.flush()

    def edit_graph() -> None:
        panel._on_curve_band_dragged(4, 2100.0, 2.0)
        panel._curve_rate_limiter.flush()

    edits = (
        edit_gain,
        edit_q,
        lambda: band.filter_type_combo.setCurrentIndex(
            band.filter_type_combo.findData("notch")
        ),
        edit_slope,
        lambda: band.band_enabled_checkbox.setChecked(False),
        edit_frequency,
        edit_graph,
        lambda: panel.enabled_checkbox.setChecked(False),
    )

    try:
        for edit in edits:
            apply_candidate()
            edit()
            qapp.processEvents()
            assert_cleared()

        apply_candidate()
        assert panel.auto_eq_diag_label.text().startswith("Auto-EQ: ")
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()
