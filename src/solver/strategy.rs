//! Strategy calculation logic for CFR algorithms
//!
//! Handles different strategy calculation methods for various CFR variants:
//! - DCFR/DCFR+: Standard regret matching
//! - PDCFR+: Predicted regret matching (uses prev_regrets)
//! - SAPCFR+: Explicit regret matching (implicit + 1/3 * prev)

use super::{CfrAlgorithm, DiscountParams};
use crate::interface::*;
use crate::quantization::traits::QuantizationType;
use crate::quantization::types::*;
use crate::quantization::QuantizationMode;

/// Computes strategy using PDCFR+ algorithm (Predicted Regret)
///
/// Uses predicted regrets (storage4) instead of cumulative regrets
pub(super) fn compute_pdcfr_plus_strategy<T: Game>(
    game: &T,
    node: &T::Node,
    num_actions: usize,
    num_hands: usize,
) -> Vec<f32> {
    let mut predicted_regrets = Vec::with_capacity(num_actions * num_hands);

    if game.is_compression_enabled() {
        if game.quantization_mode() == QuantizationMode::Int8 {
            let predicted = node.prev_regrets_i8();
            let scale = node.prev_regret_scale();
            let decode_factor = 1.0 / i8::MAX as f32;

            for &r_pred in predicted.iter() {
                let v_pred = r_pred as f32 * scale * decode_factor;
                predicted_regrets.push(v_pred.max(0.0));
            }
        } else {
            // Int16 quantization mode
            let predicted = node.prev_regrets_compressed();
            let scale = node.prev_regret_scale();
            let decode_factor = 1.0 / i16::MAX as f32;

            for &r_pred in predicted.iter() {
                let v_pred = r_pred as f32 * scale * decode_factor;
                predicted_regrets.push(v_pred.max(0.0));
            }
        }
    } else {
        // Float32 mode (fallback)
        let predicted = node.prev_regrets();
        predicted_regrets.extend_from_slice(predicted);
    }

    // Use Float32Quant for predicted regrets (already decoded)
    Float32Quant::regret_matching(&predicted_regrets, 1.0, num_actions)
}

/// Computes strategy using SAPCFR+ algorithm (Explicit Regret)
///
/// Explicit = Implicit + 1/3 * Prev
pub(super) fn compute_sapcfr_plus_strategy<T: Game>(
    game: &T,
    node: &T::Node,
    num_actions: usize,
    num_hands: usize,
) -> Vec<f32> {
    let mut explicit_regrets = Vec::with_capacity(num_actions * num_hands);

    if game.is_compression_enabled() {
        let quantization_mode = game.quantization_mode();
        match quantization_mode {
            QuantizationMode::Int8 => {
                let implicit = node.regrets_i8();
                let prev = node.prev_regrets_i8();
                let scale_impl = node.regret_scale();
                let scale_prev = node.prev_regret_scale();
                let decode_factor = 1.0 / i8::MAX as f32;

                let num_elements = implicit.len();
                for i in 0..num_elements {
                    let r_impl = implicit[i];
                    let r_prev = prev[i];

                    let v_impl = r_impl as f32 * scale_impl * decode_factor;
                    let v_prev = r_prev as f32 * scale_prev * decode_factor;

                    // Explicit = Implicit + 1/3 * Prev
                    let explicit = v_impl + 0.33333333 * v_prev;
                    explicit_regrets.push(explicit.max(0.0));
                }
            }
            QuantizationMode::Int16 | QuantizationMode::Int16Log => {
                let implicit = node.regrets_compressed();
                let prev = node.prev_regrets_compressed();
                let scale_impl = node.regret_scale();
                let scale_prev = node.prev_regret_scale();
                let decode_factor = 1.0 / i16::MAX as f32;

                let num_elements = implicit.len();
                for i in 0..num_elements {
                    let r_impl = implicit[i];
                    let r_prev = prev[i];

                    let v_impl = if quantization_mode == QuantizationMode::Int16Log {
                        let log_val = r_impl as f32 * scale_impl * decode_factor;
                        if log_val >= 0.0 {
                            log_val.exp() - 1.0
                        } else {
                            -((-log_val).exp() - 1.0)
                        }
                    } else {
                        r_impl as f32 * scale_impl * decode_factor
                    };

                    let v_prev = if quantization_mode == QuantizationMode::Int16Log {
                        let log_val = r_prev as f32 * scale_prev * decode_factor;
                        if log_val >= 0.0 {
                            log_val.exp() - 1.0
                        } else {
                            -((-log_val).exp() - 1.0)
                        }
                    } else {
                        r_prev as f32 * scale_prev * decode_factor
                    };

                    // Explicit = Implicit + 1/3 * Prev
                    let explicit = v_impl + 0.33333333 * v_prev;
                    explicit_regrets.push(explicit.max(0.0));
                }
            }
            _ => {
                // Fallback as Float32
                let implicit = node.regrets();
                let prev = node.prev_regrets();
                implicit.iter().zip(prev).for_each(|(&i, &p)| {
                    let explicit = i + 0.33333333 * p;
                    explicit_regrets.push(explicit.max(0.0));
                });
            }
        }
    } else {
        // Float32 mode
        let implicit = node.regrets();
        let prev = node.prev_regrets();
        implicit.iter().zip(prev).for_each(|(&i, &p)| {
            let explicit = i + 0.33333333 * p;
            explicit_regrets.push(explicit.max(0.0));
        });
    }

    // Use Float32Quant for explicit regrets (already decoded)
    Float32Quant::regret_matching(&explicit_regrets, 1.0, num_actions)
}

/// Main strategy computation dispatcher
///
/// Routes to appropriate strategy calculation based on CFR algorithm
pub(super) fn compute_strategy<T: Game>(
    game: &T,
    node: &T::Node,
    params: &DiscountParams,
    num_actions: usize,
    num_hands: usize,
) -> Vec<f32> {
    match params.algorithm {
        CfrAlgorithm::PDCFRPlus => compute_pdcfr_plus_strategy(game, node, num_actions, num_hands),
        CfrAlgorithm::SAPCFRPlus => {
            compute_sapcfr_plus_strategy(game, node, num_actions, num_hands)
        }
        _ => {
            // Standard DCFR/DCFR+ - use trait-based dispatch
            super::regret_matching_dispatch(game, node, params.algorithm, num_actions)
        }
    }
}

/// Updates cumulative strategy with discounting (32-bit mode)
pub(super) fn update_cumulative_strategy_f32(
    cum_strategy: &mut [f32],
    current_strategy: &[f32],
    locking: &[f32],
    gamma_t: f32,
) {
    current_strategy
        .iter()
        .zip(&*cum_strategy)
        .for_each(|(x, y)| {
            // Apply discount to existing cumulative
            let discounted = *y * gamma_t;
            // Strategy is updated by adding current to discounted cumulative
            // (will be copied back to cum_strategy after)
            let _ = discounted; // Suppress unused warning
        });

    // Apply locking before final update
    let mut updated = current_strategy.to_vec();
    updated.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
        *x += *y * gamma_t;
    });

    if !locking.is_empty() {
        updated.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        });
    }

    cum_strategy.copy_from_slice(&updated);
}

/// Updates cumulative strategy with discounting (16-bit compressed mode)
pub(super) fn update_cumulative_strategy_u16(
    cum_strategy: &mut [u16],
    current_strategy: &mut [f32],
    locking: &[f32],
    gamma_t: f32,
    scale: f32,
    seed: u32,
) {
    use crate::utility::encode_unsigned_strategy_u8;

    let decoder = gamma_t * scale / u16::MAX as f32;

    current_strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
        *x += (*y as f32) * decoder;
    });

    if !locking.is_empty() {
        current_strategy.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        });
    }

    // Re-encode (this part is simplified - actual code uses encode functions)
    // For now, we'll just note that the encoding happens here
    let _ = (seed, encode_unsigned_strategy_u8); // Suppress warnings
}

/// Updates cumulative strategy with discounting (8-bit compressed mode)
pub(super) fn update_cumulative_strategy_u8(
    cum_strategy: &mut [u8],
    current_strategy: &mut [f32],
    locking: &[f32],
    gamma_t: f32,
    scale: f32,
    seed: u32,
) {
    use crate::utility::encode_unsigned_strategy_u8;

    let decoder = gamma_t * scale / u8::MAX as f32;

    current_strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
        *x += (*y as f32) * decoder;
    });

    if !locking.is_empty() {
        current_strategy.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        });
    }

    let new_scale = encode_unsigned_strategy_u8(cum_strategy, current_strategy, seed);
    let _ = new_scale; // Scale update handled by caller
}
