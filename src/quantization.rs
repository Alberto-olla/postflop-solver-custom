//! Quantization support for memory-efficient storage of game data.
//!
//! This module provides types and utilities for quantizing floating-point data
//! to lower-precision representations (16-bit, 8-bit, etc.) to reduce memory usage.

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// Configuration for quantization parameters.
///
/// This struct holds the quantization mode and associated parameters like scale factor.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub struct QuantizationConfig {
    /// The quantization mode
    pub mode: QuantizationMode,

    /// Scale factor for quantization.
    ///
    /// For symmetric quantization:
    /// - quantized = round(value / scale).clamp(min, max)
    /// - dequantized = quantized * scale
    ///
    /// For Float32 and Int16, scale is 1.0 (no scaling needed).
    pub scale: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            mode: QuantizationMode::Float32,
            scale: 1.0,
        }
    }
}

impl QuantizationConfig {
    /// Creates a new quantization config with the given mode and default scale.
    pub fn new(mode: QuantizationMode) -> Self {
        Self {
            mode,
            scale: 1.0,
        }
    }

    /// Creates a quantization config by computing optimal scale from data.
    ///
    /// For Float32 and Int16, this always returns scale = 1.0.
    pub fn from_data(mode: QuantizationMode, _data: &[f32]) -> Self {
        Self {
            mode,
            scale: 1.0,
        }
    }
}

/// Quantization mode for game data storage.
///
/// This enum defines the precision level used for storing strategy and regret values.
/// Lower precision modes use less memory but may introduce some quantization error.
///
/// # Memory Usage
/// - `Float32`: 4 bytes per value (no quantization)
/// - `Int16`: 2 bytes per value (50% memory saving)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub enum QuantizationMode {
    /// Full precision: 32-bit floating point (f32).
    ///
    /// This mode stores values directly as 32-bit floats without any quantization.
    /// It provides the highest precision but uses the most memory.
    #[cfg_attr(feature = "serde", serde(rename = "32bit"))]
    Float32,

    /// Compressed precision: 16-bit signed integer (i16).
    ///
    /// This mode stores values as 16-bit integers with a floating-point scale factor.
    /// It reduces memory usage by 50% compared to Float32 with minimal precision loss.
    #[cfg_attr(feature = "serde", serde(rename = "16bit"))]
    Int16,

    /// Logarithmic compressed precision: 16-bit signed integer (i16).
    ///
    /// This mode stores values as `sign * log(1 + |x|)` in 16-bit integers.
    /// Ideally suited for values with high dynamic range (like CFR+ regrets).
    #[cfg_attr(feature = "serde", serde(rename = "16bit-log"))]
    Int16Log,
}

impl Default for QuantizationMode {
    /// Returns the default quantization mode (Float32 for maximum precision).
    fn default() -> Self {
        Self::Float32
    }
}

impl QuantizationMode {
    /// Returns the number of bytes required per element for this quantization mode.
    #[inline]
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Int16 | Self::Int16Log => 2,
        }
    }

    /// Returns the actual storage size in bytes.
    #[inline]
    pub fn storage_size(&self, num_elements: usize) -> usize {
        match self {
            Self::Float32 => num_elements * 4,
            Self::Int16 | Self::Int16Log => num_elements * 2,
        }
    }

    /// Returns whether this mode uses compression (i.e., not Float32).
    #[inline]
    pub fn is_compressed(&self) -> bool {
        !matches!(self, Self::Float32)
    }

    /// Creates a QuantizationMode from a boolean compression flag.
    ///
    /// This is provided for backward compatibility with the old API.
    /// - `false` maps to Float32
    /// - `true` maps to Int16
    #[inline]
    pub fn from_compression_flag(enable_compression: bool) -> Self {
        if enable_compression {
            Self::Int16
        } else {
            Self::Float32
        }
    }

    /// Converts to a boolean compression flag.
    ///
    /// This is provided for backward compatibility with the old API.
    /// - Float32 maps to `false`
    /// - Int16 maps to `true`
    #[inline]
    pub fn to_compression_flag(&self) -> bool {
        self.is_compressed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_per_element() {
        assert_eq!(QuantizationMode::Float32.bytes_per_element(), 4);
        assert_eq!(QuantizationMode::Int16.bytes_per_element(), 2);
    }

    #[test]
    fn test_storage_size() {
        assert_eq!(QuantizationMode::Float32.storage_size(100), 400);
        assert_eq!(QuantizationMode::Int16.storage_size(100), 200);
    }

    #[test]
    fn test_is_compressed() {
        assert!(!QuantizationMode::Float32.is_compressed());
        assert!(QuantizationMode::Int16.is_compressed());
    }

    #[test]
    fn test_from_compression_flag() {
        assert_eq!(
            QuantizationMode::from_compression_flag(false),
            QuantizationMode::Float32
        );
        assert_eq!(
            QuantizationMode::from_compression_flag(true),
            QuantizationMode::Int16
        );
    }

    #[test]
    fn test_to_compression_flag() {
        assert!(!QuantizationMode::Float32.to_compression_flag());
        assert!(QuantizationMode::Int16.to_compression_flag());
    }

    #[test]
    fn test_default() {
        assert_eq!(QuantizationMode::default(), QuantizationMode::Float32);
    }
}
