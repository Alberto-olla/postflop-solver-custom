//! Pruning logic for CFR algorithm
//!
//! Implements branch pruning to skip actions with very negative average regrets,
//! reducing computation without significantly impacting convergence.

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

/// Checks if an action should be pruned based on its average regret.
///
/// # Arguments
/// - `game`: The game instance
/// - `node`: The current node
/// - `action`: The action index to check
/// - `num_hands`: Number of hands at this node
/// - `pruning_threshold`: The regret threshold for pruning
///
/// # Returns
/// `true` if the action should be skipped (pruned), `false` otherwise
#[inline]
pub(super) fn should_prune_action<T: Game>(
    game: &T,
    node: &T::Node,
    action: usize,
    num_hands: usize,
    pruning_threshold: f32,
) -> bool {
    if !game.is_compression_enabled() {
        // Float32 mode
        let regrets = node.regrets();
        let action_regrets = &regrets[action * num_hands..(action + 1) * num_hands];
        let avg_regret: f32 = action_regrets.iter().sum::<f32>() / num_hands as f32;
        avg_regret < pruning_threshold
    } else {
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
                    let val = ((nibble << 4) as i8) >> 4;
                    sum_regret += val as f32 * decoder;
                }
                let avg_regret = sum_regret / num_hands as f32;
                avg_regret < pruning_threshold
            }
        }
    }
}
