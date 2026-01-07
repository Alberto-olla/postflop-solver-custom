//! Low-level encoding/decoding functions for quantization
//!
//! Questo modulo contiene tutte le funzioni per:
//! - Encoding di slice f32 a formati compressi (i16, u16, i8, u8, i4, u4)
//! - Decoding da formati compressi a f32
//! - Helper functions per stochastic rounding e slice operations

use super::super::utility::max;

// ============================================================================
// Helper Functions
// ============================================================================

/// Fast XorShift32 PRNG for stochastic rounding.
/// Used to generate random numbers quickly during encoding.
#[inline]
pub(crate) fn fast_xorshift32(seed: &mut u32) -> u32 {
    let mut x = *seed;
    if x == 0 {
        x = 0xACE1u32;
    } // Avoid zero seed
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x;
    x
}

/// Helper for stochastic rounding.
/// Instead of round(x), it returns floor(x) + 1 with probability fract(x).
/// This preserves expected values over many iterations.
#[inline]
pub(crate) fn stochastic_round(val: f32, seed: &mut u32) -> i32 {
    let floor = val.floor();
    let fract = val - floor;
    // Use 24 bits of entropy for a high-quality fast float in [0, 1)
    let r = (fast_xorshift32(seed) & 0xFFFFFF) as f32 / 16777216.0;
    if r < fract {
        (floor as i32) + 1
    } else {
        floor as i32
    }
}

/// Obtains the maximum absolute value of the given slice.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) fn slice_absolute_max(slice: &[f32]) -> f32 {
    if slice.len() < 32 {
        slice.iter().fold(0.0, |a, x| max(a, x.abs()))
    } else {
        use std::arch::wasm32::*;

        unsafe {
            let slice_ptr = slice.as_ptr() as *const v128;
            let mut tmp: [v128; 4] = [
                f32x4_abs(v128_load(slice_ptr)),
                f32x4_abs(v128_load(slice_ptr.add(1))),
                f32x4_abs(v128_load(slice_ptr.add(2))),
                f32x4_abs(v128_load(slice_ptr.add(3))),
            ];

            let mut iter = slice[16..].chunks_exact(16);
            for chunk in iter.by_ref() {
                let chunk_ptr = chunk.as_ptr() as *const v128;
                tmp[0] = f32x4_max(tmp[0], f32x4_abs(v128_load(chunk_ptr)));
                tmp[1] = f32x4_max(tmp[1], f32x4_abs(v128_load(chunk_ptr.add(1))));
                tmp[2] = f32x4_max(tmp[2], f32x4_abs(v128_load(chunk_ptr.add(2))));
                tmp[3] = f32x4_max(tmp[3], f32x4_abs(v128_load(chunk_ptr.add(3))));
            }

            tmp[0] = f32x4_max(tmp[0], tmp[1]);
            tmp[2] = f32x4_max(tmp[2], tmp[3]);
            tmp[0] = f32x4_max(tmp[0], tmp[2]);
            let tmpmax = max(
                max(
                    f32x4_extract_lane::<0>(tmp[0]),
                    f32x4_extract_lane::<1>(tmp[0]),
                ),
                max(
                    f32x4_extract_lane::<2>(tmp[0]),
                    f32x4_extract_lane::<3>(tmp[0]),
                ),
            );

            iter.remainder().iter().fold(tmpmax, |a, x| max(a, x.abs()))
        }
    }
}

/// Obtains the maximum absolute value of the given slice.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub(crate) fn slice_absolute_max(slice: &[f32]) -> f32 {
    if slice.len() < 16 {
        slice.iter().fold(0.0, |a, x| max(a, x.abs()))
    } else {
        let mut tmp: [f32; 8] = slice[..8].try_into().unwrap();
        tmp.iter_mut().for_each(|x| *x = x.abs());
        let mut iter = slice[8..].chunks_exact(8);
        for chunk in iter.by_ref() {
            for i in 0..8 {
                tmp[i] = max(tmp[i], chunk[i].abs());
            }
        }
        let tmpmax = tmp.iter().fold(0.0f32, |a, &x| max(a, x));
        iter.remainder().iter().fold(tmpmax, |a, x| max(a, x.abs()))
    }
}

/// Obtains the maximum value of the given non-negative slice.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) fn slice_nonnegative_max(slice: &[f32]) -> f32 {
    if slice.len() < 32 {
        slice.iter().fold(0.0, |a, &x| max(a, x))
    } else {
        use std::arch::wasm32::*;

        unsafe {
            let slice_ptr = slice.as_ptr() as *const v128;
            let mut tmp: [v128; 4] = [
                v128_load(slice_ptr),
                v128_load(slice_ptr.add(1)),
                v128_load(slice_ptr.add(2)),
                v128_load(slice_ptr.add(3)),
            ];

            let mut iter = slice[16..].chunks_exact(16);
            for chunk in iter.by_ref() {
                let chunk_ptr = chunk.as_ptr() as *const v128;
                tmp[0] = f32x4_max(tmp[0], v128_load(chunk_ptr));
                tmp[1] = f32x4_max(tmp[1], v128_load(chunk_ptr.add(1)));
                tmp[2] = f32x4_max(tmp[2], v128_load(chunk_ptr.add(2)));
                tmp[3] = f32x4_max(tmp[3], v128_load(chunk_ptr.add(3)));
            }

            tmp[0] = f32x4_max(tmp[0], tmp[1]);
            tmp[2] = f32x4_max(tmp[2], tmp[3]);
            tmp[0] = f32x4_max(tmp[0], tmp[2]);
            let tmpmax = max(
                max(
                    f32x4_extract_lane::<0>(tmp[0]),
                    f32x4_extract_lane::<1>(tmp[0]),
                ),
                max(
                    f32x4_extract_lane::<2>(tmp[0]),
                    f32x4_extract_lane::<3>(tmp[0]),
                ),
            );

            iter.remainder().iter().fold(tmpmax, |a, &x| max(a, x))
        }
    }
}

/// Obtains the maximum value of the given non-negative slice.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub(crate) fn slice_nonnegative_max(slice: &[f32]) -> f32 {
    if slice.len() < 16 {
        slice.iter().fold(0.0, |a, &x| max(a, x))
    } else {
        let mut tmp: [f32; 8] = slice[..8].try_into().unwrap();
        let mut iter = slice[8..].chunks_exact(8);
        for chunk in iter.by_ref() {
            for i in 0..8 {
                tmp[i] = max(tmp[i], chunk[i]);
            }
        }
        let tmpmax = tmp.iter().fold(0.0f32, |a, &x| max(a, x));
        iter.remainder().iter().fold(tmpmax, |a, &x| max(a, x))
    }
}

// ============================================================================
// Int16 Encoding Functions
// ============================================================================

/// Encodes the `f32` slice to the `i16` slice, and returns the scale.
/// Used for signed values (e.g., regrets, cfvalues).
#[inline]
pub fn encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32 {
    let scale = slice_absolute_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i16::MAX as f32 / scale_nonzero;
    dst.iter_mut()
        .zip(slice)
        .for_each(|(d, s)| *d = unsafe { (s * encoder).round().to_int_unchecked::<i32>() as i16 });
    scale
}

/// Encodes the `f32` slice to the `u16` slice, and returns the scale.
/// Used for non-negative values (e.g., cumulative strategy).
#[inline]
pub fn encode_unsigned_slice(dst: &mut [u16], slice: &[f32]) -> f32 {
    let scale = slice_nonnegative_max(slice);
    // Handle NaN/Inf gracefully
    let scale = if scale.is_finite() { scale } else { 0.0 };
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u16::MAX as f32 / scale_nonzero;
    // note: 0.49999997 + 0.49999997 = 0.99999994 < 1.0 | 0.5 + 0.49999997 = 1.0
    dst.iter_mut().zip(slice).for_each(|(d, s)| {
        // Clamp to avoid overflow/UB from floating point errors and handle NaN
        let s_safe = if s.is_finite() { *s } else { 0.0 };
        let value = (s_safe * encoder + 0.49999997)
            .min(u16::MAX as f32)
            .max(0.0);
        *d = unsafe { value.to_int_unchecked::<i32>() as u16 }
    });
    scale
}

/// Encodes the `f32` slice to the `i16` slice using logarithmic compression (signed magnitude biasing).
/// This compresses the dynamic range, allowing better precision for both small and large values.
/// Formula: compressed = sign(x) * log1p(abs(x))
/// Returns the scale factor used.
#[inline]
pub fn encode_signed_slice_log(dst: &mut [i16], slice: &[f32]) -> f32 {
    // Apply log1p transform first: compressed = sign(x) * log1p(abs(x))
    let mut log_values = Vec::with_capacity(slice.len());
    log_values.extend(slice.iter().map(|&x| {
        if x >= 0.0 {
            (x.abs() + 1.0).ln()
        } else {
            -((x.abs() + 1.0).ln())
        }
    }));

    // Now quantize the log-compressed values
    let scale = slice_absolute_max(&log_values);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i16::MAX as f32 / scale_nonzero;

    dst.iter_mut()
        .zip(&log_values)
        .for_each(|(d, s)| *d = unsafe { (s * encoder).round().to_int_unchecked::<i32>() as i16 });

    scale
}

// ============================================================================
// Int8 Encoding Functions
// ============================================================================

/// Encodes the `f32` slice to the `u8` slice for cumulative strategy, and returns the scale.
/// Uses stochastic rounding to preserve expected values over iterations.
#[inline]
pub fn encode_unsigned_strategy_u8(dst: &mut [u8], slice: &[f32], base_seed: u32) -> f32 {
    let scale = slice_nonnegative_max(slice);
    // Handle NaN/Inf gracefully
    let scale = if scale.is_finite() { scale } else { 0.0 };
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u8::MAX as f32 / scale_nonzero;
    dst.iter_mut()
        .enumerate()
        .zip(slice)
        .for_each(|((i, d), s)| {
            let s_safe = if s.is_finite() { *s } else { 0.0 };
            let scaled = (s_safe * encoder).min(u8::MAX as f32).max(0.0);
            let mut seed = base_seed ^ (i as u32);
            *d = stochastic_round(scaled, &mut seed) as u8;
        });
    scale
}

/// Encodes the `f32` slice to the `u8` slice (unsigned) for regrets, and returns the scale.
/// Used for CFR+ where regrets are non-negative.
#[inline]
pub fn encode_unsigned_regrets_u8(dst: &mut [u8], slice: &[f32], base_seed: u32) -> f32 {
    let scale = slice_nonnegative_max(slice);
    let scale = if scale.is_finite() { scale } else { 0.0 };
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u8::MAX as f32 / scale_nonzero;
    dst.iter_mut()
        .enumerate()
        .zip(slice)
        .for_each(|((i, d), s)| {
            let s_safe = if s.is_finite() { *s } else { 0.0 };
            // Clamp negative values to 0 (CFR+ Requirement)
            let scaled = (s_safe * encoder).min(u8::MAX as f32).max(0.0);
            let mut seed = base_seed ^ (i as u32);
            *d = stochastic_round(scaled, &mut seed) as u8;
        });
    scale
}

/// Encodes the `f32` slice to the `i8` slice for signed values (e.g., cfvalues_chance), and returns the scale.
/// Uses signed quantization: maps [-max_abs, max_abs] to [-127, 127].
/// Note: We use i8::MAX (127) instead of full range to avoid overflow issues.
#[inline]
pub fn encode_signed_i8(dst: &mut [i8], slice: &[f32], base_seed: u32) -> f32 {
    let scale = slice_absolute_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i8::MAX as f32 / scale_nonzero; // encoder = 127 / max_abs

    dst.iter_mut()
        .enumerate()
        .zip(slice)
        .for_each(|((i, d), s)| {
            let scaled = s * encoder;
            // Clamp to i8 range before stochastic rounding
            let scaled_clamped = scaled.min(i8::MAX as f32).max(i8::MIN as f32);
            let mut seed = base_seed ^ (i as u32);
            *d = stochastic_round(scaled_clamped, &mut seed) as i8;
        });

    scale
}

/// Decodes the `i8` slice to `f32` for signed values.
#[inline]
pub fn decode_signed_i8(src: &[i8], scale: f32) -> Vec<f32> {
    let decoder = scale / i8::MAX as f32;
    src.iter().map(|&x| x as f32 * decoder).collect()
}

// ============================================================================
// Int4 Packed Encoding Functions (2 values per byte)
// ============================================================================

/// Encodes the `f32` slice to the `u8` slice for signed values (packed 4-bit), and returns the scale.
/// Uses signed quantization: maps [-max_abs, max_abs] to [-7, 7].
/// Two 4-bit values are packed into one `u8`.
#[inline]
pub fn encode_signed_i4_packed(dst: &mut [u8], slice: &[f32], base_seed: u32) -> f32 {
    let scale = slice_absolute_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = 7.0 / scale_nonzero;

    for i in 0..dst.len() {
        let s1 = slice[i * 2];
        let scaled1 = (s1 * encoder).min(7.0).max(-7.0);
        let mut seed1 = base_seed ^ (i as u32 * 2);
        let val1 = stochastic_round(scaled1, &mut seed1) as i8;

        let val2 = if i * 2 + 1 < slice.len() {
            let s2 = slice[i * 2 + 1];
            let scaled2 = (s2 * encoder).min(7.0).max(-7.0);
            let mut seed2 = base_seed ^ (i as u32 * 2 + 1);
            stochastic_round(scaled2, &mut seed2) as i8
        } else {
            0
        };

        dst[i] = (val1 as u8 & 0x0F) | ((val2 as u8 & 0x0F) << 4);
    }

    scale
}

/// Decodes the `u8` slice (packed 4-bit) to `f32` for signed values.
#[inline]
pub fn decode_signed_i4_packed(src: &[u8], scale: f32, len: usize) -> Vec<f32> {
    let decoder = scale / 7.0;
    let mut dst = Vec::with_capacity(len);
    for i in 0..len {
        let byte = src[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        // Sign extension
        let val = ((nibble << 4) as i8) >> 4;
        dst.push(val as f32 * decoder);
    }
    dst
}

/// Encodes the `f32` slice to the `u8` slice (unsigned packed 4-bit), and returns the scale.
/// Used for non-negative regrets (DCFR+). Maps [0, max] to [0, 15].
#[inline]
pub fn encode_unsigned_u4_packed(dst: &mut [u8], slice: &[f32], base_seed: u32) -> f32 {
    let scale = slice_nonnegative_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = 15.0 / scale_nonzero;

    for i in 0..dst.len() {
        let s1 = slice[i * 2];
        let scaled1 = (s1 * encoder).min(15.0).max(0.0);
        let mut seed1 = base_seed ^ (i as u32 * 2);
        let val1 = stochastic_round(scaled1, &mut seed1) as u8;

        let val2 = if i * 2 + 1 < slice.len() {
            let s2 = slice[i * 2 + 1];
            let scaled2 = (s2 * encoder).min(15.0).max(0.0);
            let mut seed2 = base_seed ^ (i as u32 * 2 + 1);
            stochastic_round(scaled2, &mut seed2) as u8
        } else {
            0
        };

        dst[i] = (val1 & 0x0F) | ((val2 & 0x0F) << 4);
    }

    scale
}

/// Decodes the `u8` slice (unsigned packed 4-bit) to `f32`.
#[inline]
#[allow(dead_code)]
pub fn decode_unsigned_u4_packed(src: &[u8], scale: f32, len: usize) -> Vec<f32> {
    let decoder = scale / 15.0;
    let mut dst = Vec::with_capacity(len);
    for i in 0..len {
        let byte = src[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        dst.push(nibble as f32 * decoder);
    }
    dst
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_rounding_preserves_expected_value() {
        // Test that stochastic rounding preserves expected value over many iterations
        const NUM_TRIALS: usize = 10000;
        const NUM_ACTIONS: usize = 3;

        let initial_strategy = vec![0.5, 0.3, 0.2];
        let small_update = vec![0.001, 0.0005, 0.0005];

        let mut accumulated = initial_strategy.clone();
        let mut encoded = vec![0u8; NUM_ACTIONS];

        for _ in 0..NUM_TRIALS {
            for i in 0..NUM_ACTIONS {
                accumulated[i] += small_update[i];
            }
            let _scale = encode_unsigned_strategy_u8(&mut encoded, &accumulated, 0);
        }

        let expected: Vec<f32> = initial_strategy
            .iter()
            .zip(&small_update)
            .map(|(init, update)| init + update * NUM_TRIALS as f32)
            .collect();

        for i in 0..NUM_ACTIONS {
            let error = (accumulated[i] - expected[i]).abs() / expected[i];
            assert!(
                error < 0.05,
                "Action {}: accumulated={}, expected={}, error={}%",
                i,
                accumulated[i],
                expected[i],
                error * 100.0
            );
        }
    }

    #[test]
    fn test_encode_decode_roundtrip_u8() {
        let original = vec![0.5, 0.3, 0.15, 0.05];
        let mut encoded = vec![0u8; 4];

        let scale = encode_unsigned_strategy_u8(&mut encoded, &original, 0);

        let decoded: Vec<f32> = encoded
            .iter()
            .map(|&x| (x as f32) * scale / 255.0)
            .collect();

        for i in 0..4 {
            let error = (decoded[i] - original[i]).abs();
            assert!(
                error < 0.01,
                "Index {}: decoded={}, original={}, error={}",
                i,
                decoded[i],
                original[i],
                error
            );
        }
    }

    #[test]
    fn test_encode_decode_i16() {
        let original = vec![1.5, -2.3, 0.8, 0.0];
        let mut encoded = vec![0i16; 4];

        let scale = encode_signed_slice(&mut encoded, &original);
        assert!(scale > 0.0);

        let decoder = scale / i16::MAX as f32;
        let decoded: Vec<f32> = encoded.iter().map(|&x| x as f32 * decoder).collect();

        for i in 0..4 {
            let error = (decoded[i] - original[i]).abs();
            assert!(
                error < 0.001,
                "Index {}: decoded={}, original={}, error={}",
                i,
                decoded[i],
                original[i],
                error
            );
        }
    }

    #[test]
    fn test_encode_decode_i8() {
        let original = vec![1.5, -2.3, 0.8, 0.0];
        let mut encoded = vec![0i8; 4];

        let scale = encode_signed_i8(&mut encoded, &original, 12345);
        let decoded = decode_signed_i8(&encoded, scale);

        for i in 0..4 {
            let error = (decoded[i] - original[i]).abs();
            assert!(
                error < 0.1,
                "Index {}: decoded={}, original={}, error={}",
                i,
                decoded[i],
                original[i],
                error
            );
        }
    }

    #[test]
    fn test_encode_decode_i4_packed() {
        let original = vec![1.5, -2.3, 0.8, 0.0, -1.2, 2.0];
        let packed_len = (original.len() + 1) / 2;
        let mut encoded = vec![0u8; packed_len];

        let scale = encode_signed_i4_packed(&mut encoded, &original, 12345);
        let decoded = decode_signed_i4_packed(&encoded, scale, original.len());

        for i in 0..original.len() {
            let error = (decoded[i] - original[i]).abs();
            // 4-bit has lower precision
            assert!(
                error < 0.5,
                "Index {}: decoded={}, original={}, error={}",
                i,
                decoded[i],
                original[i],
                error
            );
        }
    }

    #[test]
    fn test_encode_decode_u4_packed() {
        let original = vec![1.5, 2.3, 0.8, 0.0, 1.2, 2.0];
        let packed_len = (original.len() + 1) / 2;
        let mut encoded = vec![0u8; packed_len];

        let scale = encode_unsigned_u4_packed(&mut encoded, &original, 12345);
        let decoded = decode_unsigned_u4_packed(&encoded, scale, original.len());

        for i in 0..original.len() {
            let error = (decoded[i] - original[i]).abs();
            assert!(
                error < 0.5,
                "Index {}: decoded={}, original={}, error={}",
                i,
                decoded[i],
                original[i],
                error
            );
        }
    }

    #[test]
    fn test_slice_absolute_max() {
        assert_eq!(slice_absolute_max(&[1.0, -2.0, 0.5]), 2.0);
        assert_eq!(slice_absolute_max(&[-5.0, 3.0, -1.0]), 5.0);
        assert_eq!(slice_absolute_max(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_slice_nonnegative_max() {
        assert_eq!(slice_nonnegative_max(&[1.0, 2.0, 0.5]), 2.0);
        assert_eq!(slice_nonnegative_max(&[5.0, 3.0, 1.0]), 5.0);
        assert_eq!(slice_nonnegative_max(&[0.0, 0.0, 0.0]), 0.0);
    }
}
