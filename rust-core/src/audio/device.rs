//! Audio device enumeration and selection
#![allow(clippy::useless_conversion)] // PyO3 proc-macro wrappers trigger false positives.

use cpal::traits::{DeviceTrait, HostTrait};
use pyo3::prelude::*;

/// Information about an audio device
#[derive(Clone, Debug)]
#[pyclass]
pub struct DeviceInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub is_default: bool,
}

#[pymethods]
impl DeviceInfo {
    fn __repr__(&self) -> String {
        format!(
            "DeviceInfo(name='{}', is_default={})",
            self.name, self.is_default
        )
    }
}

/// List all available input (microphone) devices
#[pyfunction]
pub fn list_input_devices() -> PyResult<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default_device = host.default_input_device();
    let default_name = default_device.as_ref().and_then(|d| d.name().ok());

    let devices: Vec<DeviceInfo> = host
        .input_devices()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        .filter_map(|device| {
            device.name().ok().map(|name| {
                let is_default = default_name.as_ref() == Some(&name);
                DeviceInfo { name, is_default }
            })
        })
        .collect();

    Ok(devices)
}

/// List all available output devices
#[pyfunction]
pub fn list_output_devices() -> PyResult<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default_device = host.default_output_device();
    let default_name = default_device.as_ref().and_then(|d| d.name().ok());

    let devices: Vec<DeviceInfo> = host
        .output_devices()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        .filter_map(|device| {
            device.name().ok().map(|name| {
                let is_default = default_name.as_ref() == Some(&name);
                DeviceInfo { name, is_default }
            })
        })
        .collect();

    Ok(devices)
}
