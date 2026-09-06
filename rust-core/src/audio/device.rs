//! Audio device enumeration and selection.
#![allow(clippy::useless_conversion)] // PyO3 proc-macro wrappers trigger false positives.

use std::collections::HashMap;
use std::str::FromStr;

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, DeviceId};
use pyo3::prelude::*;

/// Information about an audio device.
#[derive(Clone, Debug)]
#[pyclass(skip_from_py_object)]
pub struct DeviceInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub is_default: bool,
    /// Stable device identifier when available. Windows keeps the raw WASAPI
    /// shape used by existing persisted routes; CPAL's host prefix is restored
    /// only while resolving the endpoint.
    #[pyo3(get)]
    pub endpoint_id: Option<String>,
    /// Audio host used to enumerate and open this endpoint.
    #[pyo3(get)]
    pub host_api: String,
    /// Stream direction ("input" or "output").
    #[pyo3(get)]
    pub direction: String,
    /// Device default sample rate used only as fallback identity evidence.
    #[pyo3(get)]
    pub sample_rate: Option<u32>,
    /// Device default channel count used only as fallback identity evidence.
    #[pyo3(get)]
    pub channels: Option<u16>,
    /// Zero-based occurrence among devices with the same friendly name.
    #[pyo3(get)]
    pub name_ordinal: u32,
}

#[pymethods]
impl DeviceInfo {
    fn __repr__(&self) -> String {
        format!(
            "DeviceInfo(name='{}', is_default={}, host_api='{}', direction='{}', name_ordinal={})",
            self.name, self.is_default, self.host_api, self.direction, self.name_ordinal
        )
    }
}

/// Parse a persisted CPAL device ID.
///
/// v2 resolves through CPAL's serialized `DeviceId` but preserves the raw
/// Windows Core Audio shape in persisted endpoint IDs. v1 used that same raw
/// shape, so accept it by adding the known WASAPI host prefix. No host is
/// guessed on other platforms.
fn parse_endpoint_id(value: &str) -> Result<DeviceId, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("audio endpoint ID must not be empty".to_string());
    }

    if let Ok(id) = DeviceId::from_str(value) {
        return Ok(id);
    }

    #[cfg(target_os = "windows")]
    if !value.contains(':') {
        let legacy_id = format!("wasapi:{value}");
        return DeviceId::from_str(&legacy_id)
            .map_err(|error| format!("invalid legacy WASAPI endpoint ID {value:?}: {error}"));
    }

    Err(format!("invalid CPAL audio endpoint ID {value:?}"))
}

/// Resolve a persisted device ID through CPAL's own host snapshot.
///
/// `device_by_id` performs the lookup and returns the exact backend device,
/// so opening a stream never depends on a second enumeration's ordering.
pub(crate) fn device_for_endpoint_id(endpoint_id: &str, input: bool) -> Result<Device, String> {
    let device_id = parse_endpoint_id(endpoint_id)?;
    let host = cpal::default_host();
    let device = host
        .device_by_id(&device_id)
        .ok_or_else(|| format!("audio endpoint ID {endpoint_id:?} is not available"))?;

    let supports_direction = if input {
        device
            .supported_input_configs()
            .map(|mut configs| configs.next().is_some())
    } else {
        device
            .supported_output_configs()
            .map(|mut configs| configs.next().is_some())
    };
    match supports_direction {
        Ok(true) => Ok(device),
        Ok(false) => Err(format!(
            "audio endpoint ID {endpoint_id:?} does not support {}",
            if input { "input" } else { "output" }
        )),
        Err(error) => Err(format!(
            "failed to inspect audio endpoint ID {endpoint_id:?}: {error}"
        )),
    }
}

/// Serialize a CPAL ID without changing the public Windows endpoint shape.
///
/// Before CPAL exposed stable IDs, AudioForge persisted the raw WASAPI
/// `IMMDevice::GetId` value. Keep those values stable for route/profile keys;
/// `parse_endpoint_id` adds the `wasapi:` host prefix when opening them.
fn persisted_endpoint_id(device_id: &DeviceId) -> String {
    let serialized = device_id.to_string();
    #[cfg(target_os = "windows")]
    if let Some(raw) = serialized.strip_prefix("wasapi:") {
        return raw.to_string();
    }
    serialized
}

pub(crate) fn device_name(device: &Device) -> Result<String, String> {
    device
        .description()
        .map(|description| display_name(&description))
        .map_err(|error| error.to_string())
}

fn display_name(description: &cpal::DeviceDescription) -> String {
    #[cfg(target_os = "windows")]
    if let Some(name) = description
        .extended()
        .first()
        .filter(|name| !name.trim().is_empty())
    {
        return name.clone();
    }

    description.name().to_string()
}

fn collect_devices(input: bool) -> PyResult<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default_id = if input {
        host.default_input_device()
            .and_then(|device| device.id().ok())
    } else {
        host.default_output_device()
            .and_then(|device| device.id().ok())
    };

    let devices: Vec<_> = if input {
        host.input_devices()
            .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?
            .collect()
    } else {
        host.output_devices()
            .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?
            .collect()
    };

    let mut ordinals: HashMap<String, u32> = HashMap::new();
    let mut result = Vec::with_capacity(devices.len());
    for device in devices {
        let Ok(name) = device_name(&device) else {
            continue;
        };
        let ordinal = ordinals.entry(name.clone()).or_default();
        let name_ordinal = *ordinal;
        *ordinal = ordinal.saturating_add(1);
        let device_id = device.id().ok();
        let config = if input {
            device.default_input_config().ok()
        } else {
            device.default_output_config().ok()
        };
        result.push(DeviceInfo {
            name,
            is_default: matches!((&device_id, &default_id), (Some(id), Some(default_id)) if id == default_id),
            endpoint_id: device_id.as_ref().map(persisted_endpoint_id),
            host_api: if cfg!(target_os = "windows") {
                "WASAPI".to_string()
            } else {
                "default".to_string()
            },
            direction: if input { "input" } else { "output" }.to_string(),
            sample_rate: config.as_ref().map(|value| value.sample_rate()),
            channels: config.as_ref().map(|value| value.channels()),
            name_ordinal,
        });
    }
    Ok(result)
}

/// List all available input (microphone) devices.
#[pyfunction]
pub fn list_input_devices() -> PyResult<Vec<DeviceInfo>> {
    collect_devices(true)
}

/// List all available output devices.
#[pyfunction]
pub fn list_output_devices() -> PyResult<Vec<DeviceInfo>> {
    collect_devices(false)
}

#[cfg(test)]
mod tests {
    use super::{display_name, parse_endpoint_id};
    use cpal::DeviceDescriptionBuilder;

    #[test]
    fn endpoint_id_rejects_empty_values() {
        assert!(parse_endpoint_id("").is_err());
        assert!(parse_endpoint_id("   ").is_err());
    }

    #[test]
    fn endpoint_id_accepts_serialized_cpal_ids() {
        let id = parse_endpoint_id("wasapi:stable-device").expect("valid CPAL ID");
        assert_eq!(id.to_string(), "wasapi:stable-device");
    }

    #[test]
    fn endpoint_id_persistence_keeps_the_existing_windows_shape() {
        let id = parse_endpoint_id("wasapi:stable-device").expect("valid CPAL ID");
        let persisted = super::persisted_endpoint_id(&id);
        if cfg!(target_os = "windows") {
            assert_eq!(persisted, "stable-device");
        } else {
            assert_eq!(persisted, "wasapi:stable-device");
        }
    }

    #[test]
    fn display_name_uses_windows_friendly_name_without_appending_driver() {
        let description = DeviceDescriptionBuilder::new("Microphone")
            .driver("Razer Seiren Mini")
            .add_extended_line("Microphone (Razer Seiren Mini)")
            .build();

        let name = display_name(&description);

        if cfg!(target_os = "windows") {
            assert_eq!(name, "Microphone (Razer Seiren Mini)");
            assert_eq!(name.matches("Razer Seiren Mini").count(), 1);
        } else {
            assert_eq!(name, "Microphone");
        }
    }

    #[test]
    fn display_name_falls_back_when_windows_friendly_name_is_missing() {
        let description = DeviceDescriptionBuilder::new("Microphone")
            .driver("Razer Seiren Mini")
            .build();

        assert_eq!(display_name(&description), "Microphone");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn legacy_raw_wasapi_id_gets_explicit_host_prefix() {
        let id = parse_endpoint_id("{legacy-device}").expect("valid legacy WASAPI ID");
        assert_eq!(id.to_string(), "wasapi:{legacy-device}");
    }
}
