"""Noise suppression settings, independent of the view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from PySide6.QtCore import QObject

from ..config import RNNoiseSettings
from ..config_parts.validation import VALIDATION_RANGES
from .stage_state import StageState, validated_settings


def _noise_model(value: object) -> str:
    if not isinstance(value, str) or value not in VALIDATION_RANGES["rnnoise"]["model"]:
        raise ValueError(f"Unknown noise model: {value!r}")
    return value


class NoiseSuppressionState(StageState[dict]):
    """Own exact settings and report rejected backend switches to the view."""

    _stage_name = "Noise suppression"

    def __init__(
        self,
        processor,
        parent: QObject | None = None,
        *,
        stop_audio: Callable[[], object] | None = None,
    ):
        super().__init__(processor, asdict(RNNoiseSettings()), parent, stop_audio=stop_audio)
        self.models = tuple(processor.list_noise_models())

    def get_settings(self) -> dict:
        """Return accepted settings; backend changes apply synchronously."""
        return self._settings.copy()

    @property
    def latency_text(self) -> str:
        if self._settings["model"] == "deepfilter":
            return "Latency: ~35ms (DeepFilterNet)"
        if self._settings["model"] == "deepfilter-ll":
            return "Latency: ~15ms (DeepFilterNet LL)"
        return "Latency: ~10ms (RNNoise)"

    def _validated(self, settings: dict) -> dict:
        return validated_settings(
            self._settings, settings, section="rnnoise", noun="noise suppression",
            booleans={"enabled"},
            ranges={"strength": VALIDATION_RANGES["rnnoise"]["strength"]},
            special={"model": _noise_model},
        )

    def set_value(self, name: str, value: object) -> None:
        candidate = self._validated({name: value})
        if candidate != self._settings:
            self._apply(
                candidate, label="Noise suppression edit",
                write=lambda settings: self._write(settings, (name,)),
            )

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline, checking its model before other native writes."""
        candidate = self._validated(settings)
        self._apply(candidate)

    def _write(
        self, settings: dict, names: tuple[str, ...] = ("model", "enabled", "strength"),
    ) -> None:
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
