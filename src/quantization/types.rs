use super::traits::QuantizationType;
use crate::utility::*;

/// Float32 quantization (no compression) - 32 bit
pub struct Float32Quant;

impl QuantizationType for Float32Quant {
    type Storage = f32;
    const BITS: u8 = 32;
    const MAX_VALUE: f32 = f32::MAX;
    const IS_SIGNED: bool = true;

    fn encode(dst: &mut [f32], src: &[f32], _seed: u32) -> f32 {
        dst.copy_from_slice(src);
        1.0 // No scaling needed
    }

    fn decode_element(value: f32, _scale: f32) -> f32 {
        value
    }

    fn decode_slice(src: &[f32], _scale: f32, dst: &mut [f32]) {
        dst.copy_from_slice(src);
    }

    fn regret_matching(data: &[f32], _scale: f32, num_actions: usize) -> Vec<f32> {
        use crate::sliceop::*;
        use crate::utility::max;

        // Implementazione identica alla vecchia regret_matching per f32
        let mut strategy = Vec::with_capacity(data.len());
        let uninit = strategy.spare_capacity_mut();
        uninit.iter_mut().zip(data).for_each(|(s, r)| {
            s.write(max(*r, 0.0));
        });
        unsafe { strategy.set_len(data.len()) };

        let row_size = data.len() / num_actions;
        let mut denom = Vec::with_capacity(row_size);
        sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
        unsafe { denom.set_len(row_size) };

        let default = 1.0 / num_actions as f32;
        strategy.chunks_exact_mut(row_size).for_each(|row| {
            div_slice(row, &denom, default);
        });

        strategy
    }
}

/// Int16 quantization (signed 16-bit)
pub struct Int16Quant;

impl QuantizationType for Int16Quant {
    type Storage = i16;
    const BITS: u8 = 16;
    const MAX_VALUE: f32 = i16::MAX as f32;
    const IS_SIGNED: bool = true;

    fn encode(dst: &mut [i16], src: &[f32], _seed: u32) -> f32 {
        encode_signed_slice(dst, src)
    }

    fn decode_element(value: i16, scale: f32) -> f32 {
        value as f32 * scale / i16::MAX as f32
    }

    fn decode_slice(src: &[i16], scale: f32, dst: &mut [f32]) {
        let decode_factor = scale / i16::MAX as f32;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s as f32 * decode_factor;
        }
    }

    fn regret_matching(data: &[i16], scale: f32, num_actions: usize) -> Vec<f32> {
        // Decode e usa l'implementazione Float32
        let decoded = {
            let mut result = vec![0.0; data.len()];
            Self::decode_slice(data, scale, &mut result);
            result
        };
        Float32Quant::regret_matching(&decoded, 1.0, num_actions)
    }
}

/// Int16Log quantization (16-bit with logarithmic encoding)
pub struct Int16LogQuant;

impl QuantizationType for Int16LogQuant {
    type Storage = i16;
    const BITS: u8 = 16;
    const MAX_VALUE: f32 = i16::MAX as f32;
    const IS_SIGNED: bool = true;

    fn encode(dst: &mut [i16], src: &[f32], _seed: u32) -> f32 {
        encode_signed_slice_log(dst, src)
    }

    fn decode_element(value: i16, scale: f32) -> f32 {
        // Decode logarithmic encoding
        let compressed = value as f32 * scale / i16::MAX as f32;
        if compressed >= 0.0 {
            (compressed.exp() - 1.0).max(0.0)
        } else {
            -(-compressed.exp() - 1.0).max(0.0)
        }
    }

    fn decode_slice(src: &[i16], scale: f32, dst: &mut [f32]) {
        let decode_factor = scale / i16::MAX as f32;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            let compressed = s as f32 * decode_factor;
            *d = if compressed >= 0.0 {
                (compressed.exp() - 1.0).max(0.0)
            } else {
                -(-compressed.exp() - 1.0).max(0.0)
            };
        }
    }

    fn regret_matching(data: &[i16], scale: f32, num_actions: usize) -> Vec<f32> {
        // Decode e usa l'implementazione Float32
        let decoded = {
            let mut result = vec![0.0; data.len()];
            Self::decode_slice(data, scale, &mut result);
            result
        };
        Float32Quant::regret_matching(&decoded, 1.0, num_actions)
    }
}

/// Int8 quantization (signed 8-bit)
pub struct Int8Quant;

impl QuantizationType for Int8Quant {
    type Storage = i8;
    const BITS: u8 = 8;
    const MAX_VALUE: f32 = i8::MAX as f32;
    const IS_SIGNED: bool = true;

    fn encode(dst: &mut [i8], src: &[f32], seed: u32) -> f32 {
        encode_signed_i8(dst, src, seed)
    }

    fn decode_element(value: i8, scale: f32) -> f32 {
        value as f32 * scale / i8::MAX as f32
    }

    fn decode_slice(src: &[i8], scale: f32, dst: &mut [f32]) {
        let decode_factor = scale / i8::MAX as f32;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s as f32 * decode_factor;
        }
    }

    fn regret_matching(data: &[i8], scale: f32, num_actions: usize) -> Vec<f32> {
        // Decode e usa l'implementazione Float32
        let decoded = {
            let mut result = vec![0.0; data.len()];
            Self::decode_slice(data, scale, &mut result);
            result
        };
        Float32Quant::regret_matching(&decoded, 1.0, num_actions)
    }
}

/// Uint8 quantization (unsigned 8-bit) - Used by CFR+ for non-negative regrets
pub struct Uint8Quant;

impl QuantizationType for Uint8Quant {
    type Storage = u8;
    const BITS: u8 = 8;
    const MAX_VALUE: f32 = u8::MAX as f32;
    const IS_SIGNED: bool = false;

    fn encode(dst: &mut [u8], src: &[f32], seed: u32) -> f32 {
        encode_unsigned_regrets_u8(dst, src, seed)
    }

    fn decode_element(value: u8, scale: f32) -> f32 {
        value as f32 * scale / u8::MAX as f32
    }

    fn decode_slice(src: &[u8], scale: f32, dst: &mut [f32]) {
        let decode_factor = scale / u8::MAX as f32;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s as f32 * decode_factor;
        }
    }

    fn regret_matching(data: &[u8], _scale: f32, num_actions: usize) -> Vec<f32> {
        // For unsigned, values are already non-negative
        // No need to decode, just convert directly
        let num_hands = data.len() / num_actions;
        let mut strategy = Vec::with_capacity(data.len());

        for hand in 0..num_hands {
            let offset = hand * num_actions;
            let regrets = &data[offset..offset + num_actions];

            // Trova sum dei regrets (tutti già non-negativi)
            let mut sum = 0.0f32;
            for &r in regrets {
                sum += r as f32;
            }

            // Calcola strategy
            if sum > 0.0 {
                for &r in regrets {
                    strategy.push(r as f32 / sum);
                }
            } else {
                // Uniform strategy se nessun regret positivo
                let uniform = 1.0 / num_actions as f32;
                for _ in 0..num_actions {
                    strategy.push(uniform);
                }
            }
        }

        strategy
    }
}

/// Int4Packed quantization (4-bit packed, 2 values per byte)
pub struct Int4PackedQuant;

impl QuantizationType for Int4PackedQuant {
    type Storage = u8; // Stores 2 packed 4-bit values
    const BITS: u8 = 4;
    const MAX_VALUE: f32 = 7.0; // 4-bit signed: -8 to 7
    const IS_SIGNED: bool = true;

    fn encode(dst: &mut [u8], src: &[f32], seed: u32) -> f32 {
        encode_signed_i4_packed(dst, src, seed)
    }

    fn decode_element(value: u8, scale: f32) -> f32 {
        // Decode one 4-bit value from packed format
        // This method is less useful for packed format
        // (usually decode_slice is used instead)
        let signed_val = ((value & 0x0F) as i8) << 4 >> 4; // Sign extend
        signed_val as f32 * scale / 7.0
    }

    fn decode_slice(src: &[u8], scale: f32, dst: &mut [f32]) {
        let decode_factor = scale / 7.0;
        for (i, d) in dst.iter_mut().enumerate() {
            let byte_idx = i / 2;
            let packed = src[byte_idx];
            let nibble = if i % 2 == 0 {
                packed & 0x0F // Low nibble
            } else {
                packed >> 4 // High nibble
            };
            // Sign extend 4-bit to i8
            let signed_val = ((nibble as i8) << 4) >> 4;
            *d = signed_val as f32 * decode_factor;
        }
    }

    fn regret_matching(data: &[u8], _scale: f32, num_actions: usize) -> Vec<f32> {
        // Direct implementation without full decode for better performance
        let num_elements = (data.len() * 2 / num_actions) * num_actions;
        let num_hands = num_elements / num_actions;
        let mut strategy = Vec::with_capacity(num_elements);

        for hand in 0..num_hands {
            let offset = hand * num_actions;

            // Decode and compute sum in one pass
            let mut sum = 0.0f32;
            let mut hand_regrets = Vec::with_capacity(num_actions);

            for i in 0..num_actions {
                let elem_idx = offset + i;
                let byte_idx = elem_idx / 2;
                let packed = data[byte_idx];
                let nibble = if elem_idx % 2 == 0 {
                    packed & 0x0F
                } else {
                    packed >> 4
                };
                // Sign extend 4-bit to i8
                let signed_val = ((nibble as i8) << 4) >> 4;
                let regret = (signed_val as f32).max(0.0);
                hand_regrets.push(regret);
                sum += regret;
            }

            // Compute strategy
            if sum > 0.0 {
                for r in hand_regrets {
                    strategy.push(r / sum);
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for _ in 0..num_actions {
                    strategy.push(uniform);
                }
            }
        }

        strategy
    }
}

/// Uint4Packed quantization (unsigned 4-bit packed) - Used by CFR+ for non-negative regrets
pub struct Uint4PackedQuant;

impl QuantizationType for Uint4PackedQuant {
    type Storage = u8; // Stores 2 packed 4-bit values
    const BITS: u8 = 4;
    const MAX_VALUE: f32 = 15.0; // 4-bit unsigned: 0 to 15
    const IS_SIGNED: bool = false;

    fn encode(dst: &mut [u8], src: &[f32], seed: u32) -> f32 {
        encode_unsigned_u4_packed(dst, src, seed)
    }

    fn decode_element(value: u8, scale: f32) -> f32 {
        let nibble = value & 0x0F;
        nibble as f32 * scale / 15.0
    }

    fn decode_slice(src: &[u8], scale: f32, dst: &mut [f32]) {
        let decode_factor = scale / 15.0;
        for (i, d) in dst.iter_mut().enumerate() {
            let byte_idx = i / 2;
            let packed = src[byte_idx];
            let nibble = if i % 2 == 0 {
                packed & 0x0F
            } else {
                packed >> 4
            };
            *d = nibble as f32 * decode_factor;
        }
    }

    fn regret_matching(data: &[u8], _scale: f32, num_actions: usize) -> Vec<f32> {
        // Direct implementation for unsigned 4-bit
        let num_elements = (data.len() * 2 / num_actions) * num_actions;
        let num_hands = num_elements / num_actions;
        let mut strategy = Vec::with_capacity(num_elements);

        for hand in 0..num_hands {
            let offset = hand * num_actions;

            // Compute sum directly from packed data
            let mut sum = 0.0f32;
            let mut hand_regrets = Vec::with_capacity(num_actions);

            for i in 0..num_actions {
                let elem_idx = offset + i;
                let byte_idx = elem_idx / 2;
                let packed = data[byte_idx];
                let nibble = if elem_idx % 2 == 0 {
                    packed & 0x0F
                } else {
                    packed >> 4
                };
                let regret = nibble as f32;
                hand_regrets.push(regret);
                sum += regret;
            }

            // Compute strategy
            if sum > 0.0 {
                for r in hand_regrets {
                    strategy.push(r / sum);
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for _ in 0..num_actions {
                    strategy.push(uniform);
                }
            }
        }

        strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float32_encode_decode() {
        let src = vec![1.5, -2.3, 0.8, 0.0];
        let mut storage = vec![0.0; src.len()];
        let scale = Float32Quant::encode(&mut storage, &src, 0);

        assert_eq!(scale, 1.0);
        assert_eq!(storage, src);

        let mut decoded = vec![0.0; src.len()];
        Float32Quant::decode_slice(&storage, scale, &mut decoded);
        assert_eq!(decoded, src);
    }

    #[test]
    fn test_int16_encode_decode() {
        let src = vec![1.5, -2.3, 0.8, 0.0];
        let mut storage = vec![0i16; src.len()];
        let scale = Int16Quant::encode(&mut storage, &src, 0);

        assert!(scale > 0.0);

        let mut decoded = vec![0.0; src.len()];
        Int16Quant::decode_slice(&storage, scale, &mut decoded);

        // Verifica approssimazione (tolerance per quantization error)
        for (original, dec) in src.iter().zip(decoded.iter()) {
            assert!((original - dec).abs() < scale / 10000.0);
        }
    }

    #[test]
    fn test_float32_regret_matching() {
        // Test basic regret matching behavior
        // Data layout: actions are interleaved for multiple hands
        // [a0h0, a0h1, a1h0, a1h1, a2h0, a2h1] for 3 actions, 2 hands
        let regrets = vec![
            1.0, -1.0,  // Action 0: hand0=1.0, hand1=-1.0 -> max(0)=[1.0, 0.0]
            2.0, 0.0,   // Action 1: hand0=2.0, hand1=0.0 -> max(0)=[2.0, 0.0]
            1.0, 0.0,   // Action 2: hand0=1.0, hand1=0.0 -> max(0)=[1.0, 0.0]
        ];
        // Sum for hand0: 1.0+2.0+1.0=4.0
        // Sum for hand1: 0.0+0.0+0.0=0.0 (uniform)
        // Strategy hand0: [1/4, 2/4, 1/4] = [0.25, 0.5, 0.25]
        // Strategy hand1: [1/3, 1/3, 1/3] (uniform)

        let strategy = Float32Quant::regret_matching(&regrets, 1.0, 3);

        assert_eq!(strategy.len(), 6);
        // Hand 0 values
        assert!((strategy[0] - 0.25).abs() < 0.001, "strategy[0]={}", strategy[0]);
        assert!((strategy[2] - 0.5).abs() < 0.001, "strategy[2]={}", strategy[2]);
        assert!((strategy[4] - 0.25).abs() < 0.001, "strategy[4]={}", strategy[4]);
        // Hand 1 values (uniform)
        assert!((strategy[1] - 0.333333).abs() < 0.001, "strategy[1]={}", strategy[1]);
        assert!((strategy[3] - 0.333333).abs() < 0.001, "strategy[3]={}", strategy[3]);
        assert!((strategy[5] - 0.333333).abs() < 0.001, "strategy[5]={}", strategy[5]);
    }
}
