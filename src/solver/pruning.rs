//! Pruning logic for CFR algorithm
//!
//! Implements regret-based pruning with two modes:
//! - MAX: Safe pruning that never cuts strong hands (recommended)
//! - Average: Legacy pruning that may cut strong hands in polarized ranges

use crate::interface::*;
use crate::quantization::QuantizationMode;

/// Computes the pruning threshold for the current iteration.
///
/// # Arguments
/// - `effective_stack`: The effective stack size
/// - `current_iteration`: Current CFR iteration number
///
/// # Returns
/// The regret threshold below which actions should be pruned
#[inline]
pub(super) fn compute_pruning_threshold(effective_stack: i32, current_iteration: u32) -> f32 {
    let delta = effective_stack as f32;
    let t = current_iteration as f32;
    -(delta * t.sqrt() * 10.0) // K=10 safety factor
}

/// Sign-extends a 4-bit nibble to i8.
#[inline]
fn sign_extend_i4(nibble: u8) -> i8 {
    ((nibble << 4) as i8) >> 4
}

// =============================================================================
// MAX-BASED PRUNING (Recommended)
// =============================================================================
// Uses max(regrets) < threshold to ensure we never prune actions that ANY hand
// wants to take. Safe for polarized ranges where different hands want opposite actions.
// Optimized: converts threshold to integer domain for compressed formats.
// =============================================================================

/// MAX-based pruning: prune only if ALL hands have regret below threshold.
#[inline]
pub(super) fn should_prune_action_max<T: Game>(
    game: &T,
    node: &T::Node,
    action: usize,
    num_hands: usize,
    pruning_threshold: f32,
) -> bool {
    if !game.is_compression_enabled() {
        let regrets = node.regrets();
        let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];
        let max_regret = action_regrets
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        return max_regret < pruning_threshold;
    }

    match game.quantization_mode() {
        QuantizationMode::Float32 => {
            let regrets = node.regrets();
            let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];
            let max_regret = action_regrets
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            max_regret < pruning_threshold
        }

        QuantizationMode::Int16 | QuantizationMode::Int16Log => {
            let regrets = node.regrets_compressed();
            let scale = node.regret_scale();
            let decoder = scale / i16::MAX as f32;
            let threshold_scaled = (pruning_threshold / decoder).ceil();

            if threshold_scaled < i16::MIN as f32 {
                return false;
            }

            let threshold_i16 = threshold_scaled.max(i16::MIN as f32).min(i16::MAX as f32) as i16;
            let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];

            for &r in action_regrets {
                if r > threshold_i16 {
                    return false;
                }
            }
            true
        }

        QuantizationMode::Int8 => {
            let regrets = node.regrets_i8();
            let scale = node.regret_scale();
            let decoder = if scale == 0.0 { 1.0 } else { scale / i8::MAX as f32 };
            let threshold_scaled = (pruning_threshold / decoder).ceil();

            if threshold_scaled < i8::MIN as f32 {
                return false;
            }

            let threshold_i8 = threshold_scaled.max(i8::MIN as f32).min(i8::MAX as f32) as i8;
            let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];

            for &r in action_regrets {
                if r > threshold_i8 {
                    return false;
                }
            }
            true
        }

        QuantizationMode::Int4Packed => {
            let regrets = node.regrets_i4_packed();
            let scale = node.regret_scale();
            let decoder = if scale == 0.0 { 1.0 } else { scale / 7.0 };
            let threshold_scaled = (pruning_threshold / decoder).ceil();

            if threshold_scaled < -8.0 {
                return false;
            }

            let threshold_i4 = threshold_scaled.max(-8.0).min(7.0) as i8;
            let start_idx = action * num_hands;
            let end_idx = start_idx + num_hands;

            for i in start_idx..end_idx {
                let byte_idx = i / 2;
                let byte = regrets[byte_idx];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let val_i4 = sign_extend_i4(nibble);

                if val_i4 > threshold_i4 {
                    return false;
                }
            }
            true
        }
    }
}

// =============================================================================
// AVERAGE-BASED PRUNING (Legacy)
// =============================================================================
// Uses avg(regrets) < threshold. Can cut strong hands when range is polarized
// (e.g., few nuts + many trash hands → average is negative → nuts get pruned).
// Not recommended for production use, but kept for comparison/benchmarking.
// =============================================================================

/// Average-based pruning (legacy): prune if average regret is below threshold.
#[inline]
pub(super) fn should_prune_action_average<T: Game>(
    game: &T,
    node: &T::Node,
    action: usize,
    num_hands: usize,
    pruning_threshold: f32,
) -> bool {
    if !game.is_compression_enabled() {
        let regrets = node.regrets();
        let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];
        let avg_regret: f32 = action_regrets.iter().sum::<f32>() / num_hands as f32;
        return avg_regret < pruning_threshold;
    }

    match game.quantization_mode() {
        QuantizationMode::Float32 => {
            let regrets = node.regrets();
            let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];
            let avg_regret: f32 = action_regrets.iter().sum::<f32>() / num_hands as f32;
            avg_regret < pruning_threshold
        }

        QuantizationMode::Int16 | QuantizationMode::Int16Log => {
            let regrets = node.regrets_compressed();
            let scale = node.regret_scale();
            let decoder = scale / i16::MAX as f32;
            let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];
            let avg_regret: f32 = action_regrets
                .iter()
                .map(|&r| r as f32 * decoder)
                .sum::<f32>()
                / num_hands as f32;
            avg_regret < pruning_threshold
        }

        QuantizationMode::Int8 => {
            let regrets = node.regrets_i8();
            let scale = node.regret_scale();
            let decoder = scale / i8::MAX as f32;
            let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];
            let avg_regret: f32 = action_regrets
                .iter()
                .map(|&r| r as f32 * decoder)
                .sum::<f32>()
                / num_hands as f32;
            avg_regret < pruning_threshold
        }

        QuantizationMode::Int4Packed => {
            let regrets = node.regrets_i4_packed();
            let scale = node.regret_scale();
            let decoder = scale / 7.0;
            let mut sum_regret = 0.0;
            for i in 0..num_hands {
                let idx = action * num_hands + i;
                let byte = regrets[idx / 2];
                let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let val = sign_extend_i4(nibble);
                sum_regret += val as f32 * decoder;
            }
            let avg_regret = sum_regret / num_hands as f32;
            avg_regret < pruning_threshold
        }
    }
}
