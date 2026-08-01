//! Audio device enumeration and selection
#![allow(clippy::useless_conversion)] // PyO3 proc-macro wrappers trigger false positives.

use std::collections::HashMap;

use cpal::traits::{DeviceTrait, HostTrait};
use pyo3::prelude::*;

#[cfg(target_os = "windows")]
use windows::core::Interface;
#[cfg(target_os = "windows")]
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::RPC_E_CHANGED_MODE;
#[cfg(target_os = "windows")]
use windows::Win32::Media::Audio::{
    eAll, eCapture, eConsole, eRender, EDataFlow, IMMDeviceEnumerator, IMMEndpoint,
    MMDeviceEnumerator, DEVICE_STATE_ACTIVE,
};
#[cfg(target_os = "windows")]
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINIT_APARTMENTTHREADED, STGM_READ,
};

/// Information about an audio device
#[derive(Clone, Debug)]
#[pyclass(skip_from_py_object)]
pub struct DeviceInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub is_default: bool,
    /// Stable Windows Core Audio endpoint identifier when available.
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

#[derive(Clone, Debug, Default)]
struct PlatformEndpoint {
    name: String,
    id: Option<String>,
    is_default: bool,
}

fn paired_platform_endpoint(
    endpoints: &[PlatformEndpoint],
    device_count: usize,
    name: &str,
    name_ordinal: u32,
) -> PlatformEndpoint {
    if endpoints.len() != device_count {
        return PlatformEndpoint::default();
    }
    endpoints
        .iter()
        .filter(|endpoint| endpoint.name == name)
        .nth(name_ordinal as usize)
        .cloned()
        .unwrap_or_default()
}

#[cfg(target_os = "windows")]
struct ComGuard(bool);

#[cfg(target_os = "windows")]
impl ComGuard {
    fn initialize() -> Result<Self, String> {
        let result = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
        if result.is_ok() {
            Ok(Self(true))
        } else if result == RPC_E_CHANGED_MODE {
            Ok(Self(false))
        } else {
            Err(format!("Core Audio COM initialization failed: {result:?}"))
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for ComGuard {
    fn drop(&mut self) {
        if self.0 {
            unsafe { CoUninitialize() };
        }
    }
}

#[cfg(target_os = "windows")]
unsafe fn take_endpoint_id(
    device: &windows::Win32::Media::Audio::IMMDevice,
) -> Result<String, String> {
    let raw = unsafe { device.GetId() }.map_err(|error| error.to_string())?;
    let value = unsafe { raw.to_string() }.map_err(|error| error.to_string());
    unsafe { CoTaskMemFree(Some(raw.0.cast())) };
    value
}

#[cfg(target_os = "windows")]
unsafe fn endpoint_friendly_name(
    device: &windows::Win32::Media::Audio::IMMDevice,
) -> Result<String, String> {
    let store =
        unsafe { device.OpenPropertyStore(STGM_READ) }.map_err(|error| error.to_string())?;
    let value =
        unsafe { store.GetValue(&PKEY_Device_FriendlyName) }.map_err(|error| error.to_string())?;
    let name = value.to_string();
    if name.is_empty() {
        Err("Core Audio endpoint has an empty friendly name".to_string())
    } else {
        Ok(name)
    }
}

#[cfg(target_os = "windows")]
fn platform_endpoints(flow: EDataFlow) -> Result<Vec<PlatformEndpoint>, String> {
    let _com = ComGuard::initialize()?;
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|error| error.to_string())?;
        let collection = enumerator
            .EnumAudioEndpoints(eAll, DEVICE_STATE_ACTIVE)
            .map_err(|error| error.to_string())?;
        let default_id = enumerator
            .GetDefaultAudioEndpoint(flow, eConsole)
            .ok()
            .and_then(|device| take_endpoint_id(&device).ok());
        let count = collection.GetCount().map_err(|error| error.to_string())?;
        let mut endpoints = Vec::new();
        for index in 0..count {
            let device = collection.Item(index).map_err(|error| error.to_string())?;
            let endpoint: IMMEndpoint = device.cast().map_err(|error| error.to_string())?;
            if endpoint.GetDataFlow().map_err(|error| error.to_string())? != flow {
                continue;
            }
            // CPAL's WASAPI backend uses this same property for Device::name().
            // Requiring it for every endpoint lets the later join fail closed
            // instead of relying on undocumented global enumeration order.
            let name = endpoint_friendly_name(&device)?;
            let id = take_endpoint_id(&device).ok();
            endpoints.push(PlatformEndpoint {
                name,
                is_default: id
                    .as_ref()
                    .is_some_and(|value| Some(value) == default_id.as_ref()),
                id,
            });
        }
        Ok(endpoints)
    }
}

#[cfg(not(target_os = "windows"))]
fn platform_endpoints(_input: bool) -> Result<Vec<PlatformEndpoint>, String> {
    Ok(Vec::new())
}

fn collect_devices(input: bool) -> PyResult<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    #[cfg(target_os = "windows")]
    let platform = platform_endpoints(if input { eCapture } else { eRender }).unwrap_or_default();
    #[cfg(not(target_os = "windows"))]
    let platform = platform_endpoints(input).unwrap_or_default();

    let devices: Vec<_> = if input {
        host.input_devices()
            .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?
            .collect()
    } else {
        host.output_devices()
            .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?
            .collect()
    };
    let default_name = if input {
        host.default_input_device()
            .and_then(|device| device.name().ok())
    } else {
        host.default_output_device()
            .and_then(|device| device.name().ok())
    };
    let device_count = devices.len();
    let platform_complete = platform.len() == device_count;
    let mut ordinals: HashMap<String, u32> = HashMap::new();
    let mut result = Vec::with_capacity(devices.len());
    for device in devices {
        let Ok(name) = device.name() else {
            continue;
        };
        let ordinal = ordinals.entry(name.clone()).or_default();
        let name_ordinal = *ordinal;
        *ordinal = ordinal.saturating_add(1);
        let config = if input {
            device.default_input_config().ok()
        } else {
            device.default_output_config().ok()
        };
        // Join the two snapshots by the exact property CPAL itself exposes,
        // plus the occurrence among duplicate friendly names. A count mismatch
        // means a hot-plug race may have occurred, so all endpoint-ID pairing
        // fails closed for this snapshot.
        let endpoint = paired_platform_endpoint(&platform, device_count, &name, name_ordinal);
        result.push(DeviceInfo {
            name: name.clone(),
            is_default: if platform_complete && endpoint.id.is_some() {
                endpoint.is_default
            } else {
                name_ordinal == 0 && default_name.as_ref() == Some(&name)
            },
            endpoint_id: endpoint.id,
            host_api: if cfg!(target_os = "windows") {
                "WASAPI".to_string()
            } else {
                "default".to_string()
            },
            direction: if input { "input" } else { "output" }.to_string(),
            sample_rate: config.as_ref().map(|value| value.sample_rate().0),
            channels: config.as_ref().map(|value| value.channels()),
            name_ordinal,
        });
    }
    Ok(result)
}

/// List all available input (microphone) devices
#[pyfunction]
pub fn list_input_devices() -> PyResult<Vec<DeviceInfo>> {
    collect_devices(true)
}

/// List all available output devices
#[pyfunction]
pub fn list_output_devices() -> PyResult<Vec<DeviceInfo>> {
    collect_devices(false)
}

#[cfg(test)]
mod tests {
    use super::{paired_platform_endpoint, PlatformEndpoint};

    #[test]
    fn endpoint_pairing_uses_name_and_duplicate_ordinal() {
        let endpoints = vec![
            PlatformEndpoint {
                name: "Speakers".to_string(),
                id: Some("speaker-a".to_string()),
                is_default: true,
            },
            PlatformEndpoint {
                name: "Microphone".to_string(),
                id: Some("microphone".to_string()),
                is_default: false,
            },
            PlatformEndpoint {
                name: "Speakers".to_string(),
                id: Some("speaker-b".to_string()),
                is_default: false,
            },
        ];

        let first = paired_platform_endpoint(&endpoints, 3, "Speakers", 0);
        let second = paired_platform_endpoint(&endpoints, 3, "Speakers", 1);
        assert_eq!(first.id.as_deref(), Some("speaker-a"));
        assert!(first.is_default);
        assert_eq!(second.id.as_deref(), Some("speaker-b"));
        assert!(!second.is_default);
    }

    #[test]
    fn endpoint_pairing_requires_complete_snapshots() {
        let endpoints = vec![PlatformEndpoint {
            name: "Speakers".to_string(),
            id: Some("endpoint-a".to_string()),
            is_default: true,
        }];

        let mismatched = paired_platform_endpoint(&endpoints, 2, "Speakers", 0);
        assert!(mismatched.id.is_none());
        assert!(!mismatched.is_default);
    }
}
