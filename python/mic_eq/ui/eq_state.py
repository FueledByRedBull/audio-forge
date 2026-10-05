"""Exact EQ layers and edit transactions shared by both presentation layers."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
import math
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..config import (
    BUILTIN_PRESETS,
    EQBandSettings,
    EQSettings,
    EQ_FREQUENCIES,
    EQ_SCHEMA_VERSION,
)
from ..config_parts.settings import _default_filter_type
from .auto_eq_explanation import explain_auto_eq
from .rate_limiter import RateLimiter


EQ_FILTER_OPTIONS = (
    ("Bell", "bell"),
    ("Notch", "notch"),
    ("Low shelf", "low_shelf"),
    ("High shelf", "high_shelf"),
    ("High-pass", "high_pass"),
    ("Low-pass", "low_pass"),
)
PASS_FILTER_TYPES = frozenset({"high_pass", "low_pass"})
GAIN_FILTER_TYPES = frozenset({"bell", "low_shelf", "high_shelf"})
Q_FILTER_TYPES = frozenset({"bell", "notch", "low_shelf", "high_shelf"})


def _format_frequency_label(freq_hz: float) -> str:
    if freq_hz >= 1000.0:
        value = f"{freq_hz / 1000.0:.1f}".rstrip("0").rstrip(".")
        return f"{value}k"
    return f"{freq_hz:.0f}"


def _bands_from_arrays(
    gains: list, qs: list | None, freqs: list | None
) -> tuple[EQBandSettings, ...]:
    qs = qs if qs is not None else []
    freqs = freqs if freqs is not None else []
    return tuple(
        EQBandSettings(
            _default_filter_type(index),
            float(freqs[index] if index < len(freqs) else EQ_FREQUENCIES[index]),
            float(gain),
            float(qs[index] if index < len(qs) else 1.41),
        )
        for index, gain in enumerate(gains[:10])
    )


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.0f}%"
    except (TypeError, ValueError):
        return "--"


def _db_value(value: Any) -> str:
    try:
        return f"{float(value):.1f} dB"
    except (TypeError, ValueError):
        return "--"


def _format_auto_eq_diagnostics(diagnostics: dict | None) -> tuple[str, str, str]:
    if not diagnostics:
        return "Auto-EQ: no calibration diagnostics", "idle", ""

    explanation = explain_auto_eq(diagnostics)
    confidence = float(diagnostics.get("analysis_confidence", 0.0) or 0.0)
    eq_confidence = float(diagnostics.get("eq_confidence", confidence) or 0.0)
    capture_confidence = float(diagnostics.get("capture_confidence", confidence) or 0.0)
    validation_confidence = float(diagnostics.get("validation_confidence", 0.0) or 0.0)
    before = diagnostics.get("validation_before_error_db")
    after = diagnostics.get("validation_after_error_db")
    scale = diagnostics.get("validation_gain_scale")
    used_fallback = bool(diagnostics.get("used_spectrum_fallback", False))
    low_confidence = diagnostics.get("low_confidence_active_bands", 0)
    headroom = diagnostics.get("headroom_validation") or {}
    headroom_after = headroom.get("after") if isinstance(headroom, dict) else None
    headroom_safe = (
        bool(headroom.get("safe", True)) if isinstance(headroom, dict) else True
    )
    headroom_advisory = (
        bool(headroom.get("advisory", False)) if isinstance(headroom, dict) else False
    )
    headroom_scale = headroom.get("gain_scale") if isinstance(headroom, dict) else None
    if not isinstance(low_confidence, int) or isinstance(low_confidence, bool):
        low_confidence = 0

    state = explanation.state
    if not headroom_safe:
        state = "warn" if state == "ok" else state

    text = (
        f"Auto-EQ: {explanation.summary} | "
        f"overall {_percent(confidence)} | "
        f"EQ {_percent(eq_confidence)} | "
        f"capture {_percent(capture_confidence)} | "
        f"validation {_percent(validation_confidence)} | "
        f"target error {_db_value(before)} -> {_db_value(after)} | "
        f"gain scale {_percent(scale)}"
    )
    if used_fallback:
        text += " | fallback spectrum"
    if low_confidence:
        text += f" | active low-confidence bands {low_confidence}"
    if isinstance(headroom_after, dict):
        pre_tp_headroom = headroom_after.get("pre_limiter_true_peak_headroom_db")
        limiter_gr = headroom_after.get("limiter_gain_reduction_db")
        true_peak_gr = headroom_after.get("true_peak_limiter_gain_reduction_db")
        headroom_status = (
            "advisory" if headroom_advisory else "safe" if headroom_safe else "risk"
        )
        text += " | headroom " + headroom_status + f" TP {_db_value(pre_tp_headroom)}"
        if headroom_scale is not None and float(headroom_scale) < 1.0:
            text += f" scale {_percent(headroom_scale)}"
        tooltip_status = (
            "advisory only (Rust simulator unavailable)"
            if headroom_advisory
            else "safe correction"
            if headroom_safe
            else "headroom risk"
        )
        tooltip_extra = (
            f"\nHeadroom status: {tooltip_status}"
            f"\nPre-limiter true-peak headroom: {_db_value(pre_tp_headroom)}"
            f"\nLimiter gain reduction: {_db_value(limiter_gr)}"
            f"\nTrue-peak limiter gain reduction: {_db_value(true_peak_gr)}"
            f"\nHeadroom gain scale: {_percent(headroom_scale)}"
        )
    else:
        tooltip_extra = ""

    explanation_details = "\n".join(f"- {detail}" for detail in explanation.details)
    tooltip = (
        "Auto-EQ calibration diagnostics\n"
        f"Result: {explanation.summary}\n"
        f"Reason code: {explanation.outcome_code}\n"
        f"{explanation_details}\n"
        f"Raw recommendation status: "
        f"{diagnostics.get('recommendation_status', '--')}\n"
        f"Overall confidence: {_percent(confidence)}\n"
        f"EQ confidence: {_percent(eq_confidence)}\n"
        f"Capture confidence: {_percent(capture_confidence)}\n"
        f"Validation confidence: {_percent(validation_confidence)}\n"
        f"Weighted target error before: {_db_value(before)}\n"
        f"Weighted target error after: {_db_value(after)}\n"
        f"Validation gain scale: {_percent(scale)}\n"
        f"Target profile: {diagnostics.get('target_profile', '--')}\n"
        f"Fallback analysis: {'yes' if used_fallback else 'no'}"
        f"{tooltip_extra}"
    )
    return text, state, tooltip


class EQState(QObject):
    """Own one typed settings snapshot; band editors are commands and views."""

    changed = Signal()
    presentationChanged = Signal()
    configurationEditStarted = Signal()
    configurationEditFinished = Signal(str)
    configurationEdited = Signal(str)
    writeFailed = Signal(str)
    recoveryFailed = Signal(str)

    def __init__(self, processor, parent: QObject | None = None):
        super().__init__(parent)
        self.processor = processor
        self._settings = EQSettings()
        self._applied_settings = copy(self._settings)
        self._rate_limiter = RateLimiter(interval_ms=33)
        self._rate_limiter._timer.setParent(self)
        self._curve_rate_limiter = self._rate_limiter
        self.selected_band: int | None = None
        self.show_markers = False
        self._diagnostics: dict | None = None

    def get_eq_settings(self) -> EQSettings:
        return copy(self._settings)

    def get_settings(self) -> dict:
        settings = self._settings
        return settings.to_dict() | {
            "band_freqs": settings.band_freqs,
            "band_gains": settings.band_gains,
            "band_qs": settings.band_qs,
        }

    def get_band(self, index: int) -> EQBandSettings:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 10:
            raise ValueError(f"Invalid EQ band index: {index}")
        return self._settings.bands[index]

    def _with_bands(
        self, bands: tuple[EQBandSettings, ...], *, correction=None
    ) -> EQSettings:
        if correction is None:
            correction = self._settings.correction_bands
        return EQSettings(
            enabled=self._settings.enabled,
            bands=bands,
            correction_bands=correction,
            tone_bands=bands if correction is not None else None,
        )

    def set_band(
        self, index: int, band: EQBandSettings, *, edited: bool = True
    ) -> None:
        self.get_band(index)
        bands = list(self._settings.bands)
        bands[index] = band
        candidate = self._with_bands(tuple(bands))
        self._request(candidate, label="EQ band edit" if edited else None)

    def set_band_value(
        self, index: int, name: str, value: object, *, edited: bool = True
    ) -> None:
        band = self.get_band(index)
        if name not in {
            "filter_type",
            "frequency_hz",
            "gain_db",
            "q",
            "slope_db_per_octave",
            "enabled",
        }:
            raise ValueError(f"Unsupported EQ band setting: {name}")
        changes = {name: value}
        if name in {"frequency_hz", "q", "filter_type"}:
            changes.update(bandwidth_mode="q", bandwidth_octaves=None)
        candidate = replace(band, **changes)
        self.set_band(index, candidate, edited=edited)
        # Discrete edits previously applied inline, numeric drags were throttled.
        if name in {"filter_type", "slope_db_per_octave", "enabled"}:
            self.flush()

    def set_enabled(self, enabled: bool) -> None:
        candidate = copy(self._settings)
        candidate.enabled = EQSettings(enabled=enabled).enabled
        self._request(candidate, label="EQ enable edit")
        self.flush()

    def _request(
        self,
        candidate: EQSettings,
        *,
        label: str | None,
        notify: bool = True,
    ) -> None:
        if candidate == self._settings:
            return
        self.set_auto_eq_diagnostics(None)
        self._settings = candidate
        if notify:
            self.changed.emit()
        self._rate_limiter.call(
            lambda: self._apply(candidate, label=label, notify=not notify)
        )

    def _write(self, settings: EQSettings) -> None:
        self.processor.set_eq_enabled(settings.enabled)
        correction = settings.correction_bands or EQSettings().bands
        self.processor.apply_eq_layers(
            [band.to_native() for band in correction],
            [band.to_native() for band in settings.bands],
        )

    def _apply(
        self,
        candidate: EQSettings,
        *,
        label: str | None = None,
        notify: bool = False,
    ) -> None:
        previous = self._applied_settings
        try:
            self._write(candidate)
        except Exception as error:
            self.cancel()
            try:
                self._write(previous)
            except Exception as rollback_error:
                message = f"EQ update failed: {error}; restoring its previous settings also failed: {rollback_error}"
                self.recoveryFailed.emit(message)
                try:
                    self.processor.stop()
                except Exception as stop_error:
                    message += f"; stopping audio also failed: {stop_error}"
                self.writeFailed.emit(message)
                raise RuntimeError(message) from error
            self.writeFailed.emit(str(error))
            raise RuntimeError(str(error)) from error
        self._applied_settings = copy(candidate)
        if self._settings != candidate:
            self._settings = candidate
            notify = True
        if notify:
            self.changed.emit()
        if label and candidate != previous:
            self.configurationEdited.emit(label)

    def flush(self) -> None:
        self._rate_limiter.flush()

    def cancel(self) -> None:
        self._rate_limiter.cancel()
        if self._settings != self._applied_settings:
            self._settings = copy(self._applied_settings)
            self.changed.emit()

    def _apply_inline(self, candidate: EQSettings, *, label: str | None = None) -> None:
        self._rate_limiter.call_now(lambda: self._apply(candidate, label=label))

    def set_settings(self, settings: dict) -> None:
        if "bands" in settings:
            payload = {
                "schema_version": settings.get("schema_version", EQ_SCHEMA_VERSION),
                "enabled": settings.get("enabled", self._settings.enabled),
                "bands": settings["bands"],
            }
            if "layers" in settings:
                payload["layers"] = settings["layers"]
            candidate = EQSettings.from_dict(payload)
            self._apply_inline(candidate)
        elif "band_gains" in settings:
            candidate = self._with_bands(
                _bands_from_arrays(
                    settings["band_gains"],
                    settings.get("band_qs"),
                    settings.get("band_freqs"),
                )
            )
            candidate.enabled = EQSettings(
                enabled=settings.get("enabled", candidate.enabled)
            ).enabled
            self._apply_inline(candidate)
            self.show_markers = any(
                abs(band.frequency_hz - default) > 1e-6
                for band, default in zip(candidate.bands, EQ_FREQUENCIES, strict=True)
            )
        elif "enabled" in settings:
            candidate = copy(self._settings)
            candidate.enabled = EQSettings(enabled=settings["enabled"]).enabled
            self._apply_inline(candidate)
        self.set_auto_eq_diagnostics(None)
        self.presentationChanged.emit()

    def apply_typed_bands(
        self, bands: tuple[EQBandSettings, ...], *, layer: str = "tone"
    ) -> None:
        if layer not in {"tone", "correction"}:
            raise ValueError(f"Unsupported EQ layer: {layer}")
        candidate = self._with_bands(
            self._settings.bands if layer == "correction" else bands,
            correction=bands
            if layer == "correction"
            else self._settings.correction_bands,
        )
        self._apply_inline(candidate)
        self.set_auto_eq_diagnostics(None)

    def apply_preset(
        self,
        gains: list,
        qs: list | None = None,
        freqs: list | None = None,
        *,
        show_markers: bool = False,
        label: str = "EQ preset edit",
    ) -> None:
        bands = _bands_from_arrays(gains, qs, freqs)
        self._apply_inline(self._with_bands(bands), label=label)
        self.show_markers = show_markers
        self.set_auto_eq_diagnostics(None)
        self.presentationChanged.emit()

    def apply_catalog_preset(self, key: str) -> None:
        bands = BUILTIN_PRESETS[key].eq.bands
        self.apply_preset([band.gain_db for band in bands], [band.q for band in bands])

    def apply_warm_clear_preset(self) -> None:
        self.apply_preset(
            [-12.0, 4.0, 4.0, 3.0, -3.0, -10.0, 0.0, 0.0, 0.0, 0.0],
            [0.707] * 10,
        )

    def reset_tone(self) -> None:
        self.apply_preset([0.0] * 10, label="Reset tone EQ")

    def clear_correction(self) -> None:
        if self._settings.correction_bands is None:
            return
        self._apply_inline(
            EQSettings(enabled=self._settings.enabled, bands=self._settings.bands),
            label="Clear Auto-EQ layer",
        )
        self.set_auto_eq_diagnostics(None)

    def apply_auto_eq_results(
        self, bands: list, diagnostics: dict | None = None, *, layer: str = "correction"
    ) -> None:
        if len(bands) != 10:
            raise ValueError(f"Expected 10 bands, got {len(bands)}")
        typed = tuple(
            EQBandSettings(
                _default_filter_type(index), float(freq), float(gain), float(q)
            )
            for index, (freq, gain, q) in enumerate(bands)
        )
        if layer == "correction":
            candidate = self._with_bands(
                tuple(replace(band, gain_db=0.0) for band in typed), correction=typed
            )
        elif layer == "tone":
            candidate = self._with_bands(typed)
        else:
            raise ValueError(f"Unsupported EQ layer: {layer}")
        self._apply_inline(candidate)
        self.show_markers = True
        self.set_auto_eq_diagnostics(diagnostics)
        self.presentationChanged.emit()

    def set_auto_eq_diagnostics(self, diagnostics: dict | None) -> None:
        candidate = deepcopy(diagnostics) if diagnostics else None
        if candidate != self._diagnostics:
            self._diagnostics = candidate
            self.presentationChanged.emit()

    @property
    def auto_eq_diagnostics(self) -> dict | None:
        return deepcopy(self._diagnostics)

    def select_band(self, index: int) -> None:
        self.get_band(index)
        if index != self.selected_band:
            self.selected_band = index
            self.presentationChanged.emit()

    def step_band(self, direction: int) -> None:
        self.select_band(((self.selected_band or 0) + direction) % 10)

    def begin_curve_edit(self, _index: int) -> None:
        self.configurationEditStarted.emit()

    def edit_curve_band(self, index: int, frequency_hz: float, gain_db: float) -> None:
        self.get_band(index)
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in (frequency_hz, gain_db)
        ):
            raise ValueError("EQ graph coordinates must be finite numbers")
        band = self.get_band(index)
        changes = {
            "frequency_hz": min(20000.0, max(20.0, frequency_hz)),
            "bandwidth_mode": "q",
            "bandwidth_octaves": None,
        }
        if band.filter_type in GAIN_FILTER_TYPES:
            changes["gain_db"] = min(12.0, max(-12.0, gain_db))
        bands = list(self._settings.bands)
        bands[index] = replace(band, **changes)
        # One queue composes graph and numeric edits in arrival order. The graph
        # already follows the pointer itself, so repaint shared views on commit.
        self._request(self._with_bands(tuple(bands)), label=None, notify=False)

    def finish_curve_edit(
        self,
        index: int,
        frequency_hz: float,
        gain_db: float,
        *,
        cancelled: bool = False,
    ) -> None:
        try:
            self.edit_curve_band(index, frequency_hz, gain_db)
            self._curve_rate_limiter.flush()
        finally:
            self.configurationEditFinished.emit(
                "Cancelled EQ graph edit" if cancelled else "EQ graph edit"
            )

    def layer_status(self) -> str:
        def count(bands):
            active = (
                sum(
                    band.enabled
                    and (
                        band.filter_type in PASS_FILTER_TYPES
                        or band.filter_type == "notch"
                        or abs(band.gain_db) >= 0.25
                    )
                    for band in bands
                )
                if self._settings.enabled
                else 0
            )
            return f"{active} active band" if active == 1 else f"{active} active bands"

        correction = self._settings.correction_bands
        return f"Auto-EQ: {count(correction) if correction is not None else 'none'} | Tone: {count(self._settings.bands)}"
