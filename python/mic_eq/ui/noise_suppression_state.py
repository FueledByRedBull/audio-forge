"""Noise suppression settings shared by the widget and Quick views."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import QObject, Signal

from ..config import PresetValidationError, RNNoiseSettings
from ..config_parts.validation import (
    VALIDATION_RANGES,
    _validate_bool,
    _validate_range,
)


class NoiseSuppressionState(QObject):
    """Own exact settings and report rejected backend switches to both views."""

    changed = Signal()
    configurationEdited = Signal(str)
    writeFailed = Signal(str)
    recoveryFailed = Signal(str)

    def __init__(self, processor, parent: QObject | None = None):
        super().__init__(parent)
        self.processor = processor
        self._settings = asdict(RNNoiseSettings())
        self.models = tuple(processor.list_noise_models())

    def get_settings(self) -> dict:
        """Return accepted settings; backend changes apply synchronously."""
        return self._settings.copy()

    @property
    def latency_text(self) -> str:
        if self._settings["model"] == "deepfilter":
            return "Latency: ~30ms (DeepFilterNet)"
        if self._settings["model"] == "deepfilter-ll":
            return "Latency: ~10ms (DeepFilterNet LL)"
        return "Latency: ~10ms (RNNoise)"

    def _validated(self, settings: dict) -> dict:
        candidate = self._settings.copy()
        for name, value in settings.items():
            try:
                if name == "enabled":
                    candidate[name] = _validate_bool(value, name, "rnnoise")
                elif name == "strength":
                    candidate[name] = _validate_range(
                        value, *VALIDATION_RANGES["rnnoise"][name], name, "rnnoise",
                        strict=True,
                    )
                elif name == "model":
                    if not isinstance(value, str) or value not in VALIDATION_RANGES["rnnoise"][name]:
                        raise ValueError(f"Unknown noise model: {value!r}")
                    candidate[name] = value
                else:
                    raise ValueError(f"Unknown noise suppression setting: {name}")
            except PresetValidationError as error:
                raise ValueError(str(error)) from error
        return candidate

    def set_value(self, name: str, value: object) -> None:
        candidate = self._validated({name: value})
        if candidate != self._settings:
            self._apply(candidate, (name,), edited=True)

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline, checking its model before other native writes."""
        candidate = self._validated(settings)
        self._apply(candidate, ("model", "enabled", "strength"), edited=False)

    def _write(self, settings: dict, names: tuple[str, ...]) -> None:
        for name in names:
            if name == "model":
                model = settings[name]
                if not any(model == model_id for model_id, _ in self.models) or not self.processor.set_noise_model(model):
                    raise RuntimeError(
                        f"Noise model {model!r} is unavailable; previous model retained"
                    )
            elif name == "enabled":
                self.processor.set_rnnoise_enabled(settings[name])
            else:
                self.processor.set_rnnoise_strength(settings[name])

    def _apply(self, candidate: dict, names: tuple[str, ...], *, edited: bool) -> None:
        previous = self._settings
        try:
            self._write(candidate, names)
        except Exception as error:
            self.changed.emit()
            try:
                self._write(previous, names)
            except Exception as rollback_error:
                message = (
                    f"Noise suppression update failed: {error}; "
                    f"restoring its previous settings also failed: {rollback_error}"
                )
                self.recoveryFailed.emit(message)
                try:
                    self.processor.stop()
                except Exception as stop_error:
                    message += f"; stopping audio also failed: {stop_error}"
                self.writeFailed.emit(message)
                raise RuntimeError(message) from error
            self.writeFailed.emit(str(error))
            raise RuntimeError(str(error)) from error
        self._settings = candidate
        if candidate != previous:
            self.changed.emit()
            if edited:
                self.configurationEdited.emit("Noise suppression edit")
