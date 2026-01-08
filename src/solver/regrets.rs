//! Regret update logic for CFR algorithms
//!
//! This module contains functions for updating cumulative regrets during CFR iterations.
//! Supports multiple quantization modes (Float32, Int16, Int8) and algorithms (DCFR, DCFR+, PDCFR+, SAPCFR+).

use super::{CfrAlgorithm, DiscountParams};
use crate::interface::*;
use crate::quantization::QuantizationMode;
use crate::sliceop::*;
use crate::utility::*;

// ============================================================================
// Main Dispatch Function
// ============================================================================

/// Updates cumulative regrets based on game configuration and algorithm.
///
/// Dispatches to the appropriate update function based on compression mode and algorithm.
#[inline]
pub(super) fn update_regrets<T: Game>(
    game: &T,
    node: &mut T::Node,
    cfv_actions: &mut [f32],
    result: &[f32],
    locking: &[f32],
    params: &DiscountParams,
    num_hands: usize,
    player: usize,
) {
    if game.is_compression_enabled() {
        let quantization_mode = game.quantization_mode();
        if quantization_mode == QuantizationMode::Int8
            || quantization_mode == QuantizationMode::Int4Packed
        {
            update_regrets_int8(
                game,
                node,
                cfv_actions,
                result,
                locking,
                params,
                num_hands,
                player,
            );
        } else {
            update_regrets_int16(
                game,
                node,
                cfv_actions,
                result,
                locking,
                params,
                num_hands,
                player,
            );
        }
    } else {
        update_regrets_float32(node, cfv_actions, result, params, num_hands);
    }
}

// ============================================================================
// Float32 (Uncompressed) Regret Updates
// ============================================================================

/// Updates regrets in uncompressed (Float32) mode.
///
/// OPTIMIZATION: All operations are fused into single passes per action row
/// to maximize cache locality and minimize memory bandwidth.
#[inline]
fn update_regrets_float32<N: GameNode>(
    node: &mut N,
    cfv_actions: &[f32],
    result: &[f32],
    params: &DiscountParams,
    num_hands: usize,
) {
    let num_actions = cfv_actions.len() / num_hands;

    match params.algorithm {
        CfrAlgorithm::DCFR => {
            // DCFR: FUSED - discount + add cfv + subtract result in single pass
            let (alpha, beta) = (params.alpha_t, params.beta_t);
            let cum_regret = node.regrets_mut();

            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let regret_row = &mut cum_regret[row_start..row_end];
                let cfv_row = &cfv_actions[row_start..row_end];

                regret_row
                    .iter_mut()
                    .zip(cfv_row)
                    .zip(result)
                    .for_each(|((reg, &cfv), &res)| {
                        let coef = if reg.is_sign_positive() { alpha } else { beta };
                        *reg = *reg * coef + cfv - res;
                    });
            }
        }
        CfrAlgorithm::DCFRPlus => {
            // DCFR+: FUSED - discount + add cfv + subtract result + clip in single pass
            let alpha = params.alpha_t;
            let cum_regret = node.regrets_mut();

            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let regret_row = &mut cum_regret[row_start..row_end];
                let cfv_row = &cfv_actions[row_start..row_end];

                regret_row
                    .iter_mut()
                    .zip(cfv_row)
                    .zip(result)
                    .for_each(|((reg, &cfv), &res)| {
                        let new_val = *reg * alpha + cfv - res;
                        *reg = if new_val < 0.0 { 0.0 } else { new_val };
                    });
            }
        }
        CfrAlgorithm::PDCFRPlus => {
            // PDCFR+: FUSED - compute inst_regret + update cum + compute predicted in single pass
            let alpha = params.alpha_t;
            let t = (params.current_iteration + 1) as f64;
            let pow_alpha = t * t.sqrt(); // t^1.5
            let next_discount = (pow_alpha / (pow_alpha + 1.0)) as f32;

            let (cumulative, predicted) = node.regrets_and_prev_mut();

            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let cum_row = &mut cumulative[row_start..row_end];
                let pred_row = &mut predicted[row_start..row_end];
                let cfv_row = &cfv_actions[row_start..row_end];

                cum_row
                    .iter_mut()
                    .zip(pred_row.iter_mut())
                    .zip(cfv_row)
                    .zip(result)
                    .for_each(|(((reg_cum, reg_pred), &cfv), &res)| {
                        let reg_inst = cfv - res;
                        *reg_cum = (*reg_cum * alpha + reg_inst).max(0.0);
                        *reg_pred = (*reg_cum * next_discount + reg_inst).max(0.0);
                    });
            }
        }
        CfrAlgorithm::SAPCFRPlus => {
            // SAPCFR+: FUSED - compute inst_regret + update implicit + store prev in single pass
            let (implicit, prev) = node.regrets_and_prev_mut();

            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let impl_row = &mut implicit[row_start..row_end];
                let prev_row = &mut prev[row_start..row_end];
                let cfv_row = &cfv_actions[row_start..row_end];

                impl_row
                    .iter_mut()
                    .zip(prev_row.iter_mut())
                    .zip(cfv_row)
                    .zip(result)
                    .for_each(|(((reg_impl, reg_prev), &cfv), &res)| {
                        let reg_inst = cfv - res;
                        *reg_impl = (*reg_impl + reg_inst).max(0.0);
                        *reg_prev = reg_inst;
                    });
            }
        }
    }
}

// ============================================================================
// Int16 Compressed Regret Updates
// ============================================================================

/// Updates regrets in Int16 compressed mode.
#[inline]
fn update_regrets_int16<T: Game>(
    game: &T,
    node: &mut T::Node,
    cfv_actions: &mut [f32],
    result: &[f32],
    locking: &[f32],
    params: &DiscountParams,
    num_hands: usize,
    _player: usize,
) {
    let quantization_mode = game.quantization_mode();
    let scale = node.regret_scale();

    match params.algorithm {
        CfrAlgorithm::PDCFRPlus => {
            update_regrets_int16_pdcfr_plus(
                node,
                cfv_actions,
                result,
                locking,
                params,
                num_hands,
                quantization_mode,
                scale,
            );
        }
        CfrAlgorithm::SAPCFRPlus => {
            update_regrets_int16_sapcfr_plus(
                node,
                cfv_actions,
                result,
                locking,
                quantization_mode,
                scale,
                num_hands,
            );
        }
        _ => {
            update_regrets_int16_dcfr(
                node,
                cfv_actions,
                result,
                locking,
                params,
                num_hands,
                quantization_mode,
                scale,
            );
        }
    }
}

/// Int16 PDCFR+ regret update
#[inline]
fn update_regrets_int16_pdcfr_plus<N: GameNode>(
    node: &mut N,
    cfv_actions: &[f32],
    result: &[f32],
    locking: &[f32],
    params: &DiscountParams,
    num_hands: usize,
    quantization_mode: QuantizationMode,
    scale: f32,
) {
    let (cum_regret, predicted_regret) = node.regrets_and_prev_compressed_mut();
    let decoder = scale / i16::MAX as f32;

    // 1. Decode cumulative regrets and compute instantaneous regrets
    let mut inst_regrets = Vec::with_capacity(cfv_actions.len());
    let mut cum_vals = Vec::with_capacity(cfv_actions.len());

    cfv_actions
        .iter()
        .zip(&*cum_regret)
        .for_each(|(&cfv, &r_cum)| {
            let log_cum = r_cum as f32 * decoder;
            let val_cum = if quantization_mode == QuantizationMode::Int16Log {
                if log_cum >= 0.0 {
                    log_cum.exp() - 1.0
                } else {
                    -((-log_cum).exp() - 1.0)
                }
            } else {
                log_cum
            };
            cum_vals.push(val_cum);
            inst_regrets.push(cfv);
        });

    // 2. Subtract EV to get instantaneous regrets
    inst_regrets.chunks_exact_mut(num_hands).for_each(|row| {
        sub_slice(row, result);
    });

    // 3. Apply locking
    if !locking.is_empty() {
        inst_regrets.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        })
    }

    // 4. Update cumulative with uniform discounting + RM+ clipping
    let alpha = params.alpha_t;
    cum_vals
        .iter_mut()
        .zip(&inst_regrets)
        .for_each(|(reg_cum, &reg_inst)| {
            *reg_cum = (*reg_cum * alpha + reg_inst).max(0.0);
        });

    // 5. Compute predicted regrets for next iteration
    let t = (params.current_iteration + 1) as f64;
    let pow_alpha = t * t.sqrt();
    let next_discount = (pow_alpha / (pow_alpha + 1.0)) as f32;

    let mut predicted_vals = Vec::with_capacity(cum_vals.len());
    cum_vals
        .iter()
        .zip(&inst_regrets)
        .for_each(|(&reg_cum, &reg_inst)| {
            predicted_vals.push((reg_cum * next_discount + reg_inst).max(0.0));
        });

    // 6. Encode cumulative and predicted regrets
    let new_scale_cum = if quantization_mode == QuantizationMode::Int16Log {
        encode_signed_slice_log(cum_regret, &cum_vals)
    } else {
        encode_signed_slice(cum_regret, &cum_vals)
    };

    let new_scale_pred = if quantization_mode == QuantizationMode::Int16Log {
        encode_signed_slice_log(predicted_regret, &predicted_vals)
    } else {
        encode_signed_slice(predicted_regret, &predicted_vals)
    };

    node.set_regret_scale(new_scale_cum);
    node.set_prev_regret_scale(new_scale_pred);
}

/// Int16 SAPCFR+ regret update
#[inline]
fn update_regrets_int16_sapcfr_plus<N: GameNode>(
    node: &mut N,
    cfv_actions: &mut [f32],
    result: &[f32],
    locking: &[f32],
    quantization_mode: QuantizationMode,
    scale: f32,
    num_hands: usize,
) {
    let (cum_regret, prev_regret) = node.regrets_and_prev_compressed_mut();
    let decoder = scale / i16::MAX as f32;

    // 1. ADD: Implicit_New_Unclipped = Implicit_Old + Instantaneous
    cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
        let log_impl = *y as f32 * decoder;
        let val_impl = if quantization_mode == QuantizationMode::Int16Log {
            if log_impl >= 0.0 {
                log_impl.exp() - 1.0
            } else {
                -((-log_impl).exp() - 1.0)
            }
        } else {
            log_impl
        };
        *x += val_impl;
    });

    // 2. Subtract EV
    cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
        sub_slice(row, result);
    });

    // 3. Locking
    if !locking.is_empty() {
        cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        })
    }

    // 4. Compute Prev_Regret, Then Clip Implicit
    let mut prev_vals = Vec::with_capacity(cfv_actions.len());
    cfv_actions
        .iter()
        .zip(&*cum_regret)
        .for_each(|(&sum_val, y)| {
            let log_impl = *y as f32 * decoder;
            let val_impl = if quantization_mode == QuantizationMode::Int16Log {
                if log_impl >= 0.0 {
                    log_impl.exp() - 1.0
                } else {
                    -((-log_impl).exp() - 1.0)
                }
            } else {
                log_impl
            };
            prev_vals.push(sum_val - val_impl);
        });

    // Clip Implicit
    cfv_actions.iter_mut().for_each(|x| {
        if *x < 0.0 {
            *x = 0.0;
        }
    });

    // Encode Prev -> storage4
    let new_scale_prev = if quantization_mode == QuantizationMode::Int16Log {
        encode_signed_slice_log(prev_regret, &prev_vals)
    } else {
        encode_signed_slice(prev_regret, &prev_vals)
    };

    // Encode Implicit -> storage2
    let new_scale = if quantization_mode == QuantizationMode::Int16Log {
        encode_signed_slice_log(cum_regret, cfv_actions)
    } else {
        encode_signed_slice(cum_regret, cfv_actions)
    };

    node.set_prev_regret_scale(new_scale_prev);
    node.set_regret_scale(new_scale);
}

/// Int16 DCFR/DCFR+ regret update
///
/// OPTIMIZATION: All operations fused into single pass per action row
/// (decode + discount + add cfv + subtract result + clip) for cache locality.
#[inline]
fn update_regrets_int16_dcfr<N: GameNode>(
    node: &mut N,
    cfv_actions: &mut [f32],
    result: &[f32],
    locking: &[f32],
    params: &DiscountParams,
    num_hands: usize,
    quantization_mode: QuantizationMode,
    scale: f32,
) {
    let cum_regret = node.regrets_compressed_mut();
    let num_actions = cfv_actions.len() / num_hands;
    let decoder = scale / i16::MAX as f32;

    match (params.algorithm, quantization_mode) {
        (CfrAlgorithm::DCFR, QuantizationMode::Int16Log) => {
            // DCFR Int16Log: FUSED decode + discount + add + subtract
            let (alpha, beta) = (params.alpha_t, params.beta_t);
            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let cfv_row = &mut cfv_actions[row_start..row_end];
                let reg_row = &cum_regret[row_start..row_end];

                cfv_row
                    .iter_mut()
                    .zip(reg_row)
                    .zip(result)
                    .for_each(|((cfv, &reg_enc), &res)| {
                        let log_val = reg_enc as f32 * decoder;
                        let real_prev = if log_val >= 0.0 {
                            log_val.exp() - 1.0
                        } else {
                            -((-log_val).exp() - 1.0)
                        };
                        let coef = if real_prev >= 0.0 { alpha } else { beta };
                        *cfv = real_prev * coef + *cfv - res;
                    });
            }
        }
        (CfrAlgorithm::DCFRPlus, QuantizationMode::Int16Log) => {
            // DCFR+ Int16Log: FUSED decode + discount + add + subtract + clip
            let alpha = params.alpha_t;
            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let cfv_row = &mut cfv_actions[row_start..row_end];
                let reg_row = &cum_regret[row_start..row_end];

                cfv_row
                    .iter_mut()
                    .zip(reg_row)
                    .zip(result)
                    .for_each(|((cfv, &reg_enc), &res)| {
                        let log_val = reg_enc as f32 * decoder;
                        let real_prev = if log_val >= 0.0 {
                            log_val.exp() - 1.0
                        } else {
                            -((-log_val).exp() - 1.0)
                        };
                        let new_val = real_prev * alpha + *cfv - res;
                        *cfv = if new_val < 0.0 { 0.0 } else { new_val };
                    });
            }
        }
        (CfrAlgorithm::DCFR, _) => {
            // DCFR Int16 Linear: FUSED decode + discount + add + subtract
            let (alpha, beta) = (params.alpha_t, params.beta_t);
            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let cfv_row = &mut cfv_actions[row_start..row_end];
                let reg_row = &cum_regret[row_start..row_end];

                cfv_row
                    .iter_mut()
                    .zip(reg_row)
                    .zip(result)
                    .for_each(|((cfv, &reg_enc), &res)| {
                        let real_prev = reg_enc as f32 * decoder;
                        let coef = if reg_enc >= 0 { alpha } else { beta };
                        *cfv = real_prev * coef + *cfv - res;
                    });
            }
        }
        (CfrAlgorithm::DCFRPlus, _) => {
            // DCFR+ Int16 Linear: FUSED decode + discount + add + subtract + clip
            let alpha = params.alpha_t;
            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let cfv_row = &mut cfv_actions[row_start..row_end];
                let reg_row = &cum_regret[row_start..row_end];

                cfv_row
                    .iter_mut()
                    .zip(reg_row)
                    .zip(result)
                    .for_each(|((cfv, &reg_enc), &res)| {
                        let real_prev = reg_enc as f32 * decoder;
                        let new_val = real_prev * alpha + *cfv - res;
                        *cfv = if new_val < 0.0 { 0.0 } else { new_val };
                    });
            }
        }
        _ => {}
    }

    // Apply locking (still separate - only runs if locking is enabled)
    if !locking.is_empty() {
        cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        })
    }

    // Encode back to Int16
    let new_scale = if quantization_mode == QuantizationMode::Int16Log {
        encode_signed_slice_log(cum_regret, cfv_actions)
    } else {
        encode_signed_slice(cum_regret, cfv_actions)
    };
    node.set_regret_scale(new_scale);
}

// ============================================================================
// Int8 Compressed Regret Updates
// ============================================================================

/// Updates regrets in Int8 compressed mode.
#[inline]
fn update_regrets_int8<T: Game>(
    game: &T,
    node: &mut T::Node,
    cfv_actions: &mut [f32],
    result: &[f32],
    locking: &[f32],
    params: &DiscountParams,
    num_hands: usize,
    player: usize,
) {
    let scale = node.regret_scale();
    let decoder = scale / i8::MAX as f32;

    match params.algorithm {
        CfrAlgorithm::PDCFRPlus => {
            update_regrets_int8_pdcfr_plus(
                node,
                cfv_actions,
                result,
                locking,
                params,
                num_hands,
                player,
                decoder,
            );
        }
        CfrAlgorithm::SAPCFRPlus => {
            update_regrets_int8_sapcfr_plus(
                node,
                cfv_actions,
                result,
                locking,
                params,
                num_hands,
                player,
                decoder,
            );
        }
        _ => {
            update_regrets_int8_dcfr(
                game,
                node,
                cfv_actions,
                result,
                locking,
                params,
                num_hands,
                player,
                scale,
                decoder,
            );
        }
    }
}

/// Int8 PDCFR+ regret update
#[inline]
fn update_regrets_int8_pdcfr_plus<N: GameNode>(
    node: &mut N,
    cfv_actions: &[f32],
    result: &[f32],
    locking: &[f32],
    params: &DiscountParams,
    num_hands: usize,
    player: usize,
    decoder: f32,
) {
    let (cum_regret, predicted_regret) = node.regrets_and_prev_i8_mut();

    // 1. Decode cumulative regrets and compute instantaneous regrets
    let mut inst_regrets = Vec::with_capacity(cfv_actions.len());
    let mut cum_vals = Vec::with_capacity(cfv_actions.len());

    cfv_actions
        .iter()
        .zip(&*cum_regret)
        .for_each(|(&cfv, &r_cum)| {
            let val_cum = r_cum as f32 * decoder;
            cum_vals.push(val_cum);
            inst_regrets.push(cfv);
        });

    // 2. Subtract EV
    inst_regrets.chunks_exact_mut(num_hands).for_each(|row| {
        sub_slice(row, result);
    });

    // 3. Locking
    if !locking.is_empty() {
        inst_regrets.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        })
    }

    // 4. Update
    let alpha = params.alpha_t;
    cum_vals
        .iter_mut()
        .zip(&inst_regrets)
        .for_each(|(reg_cum, &reg_inst)| {
            *reg_cum = (*reg_cum * alpha + reg_inst).max(0.0);
        });

    // 5. Predicted
    let t = (params.current_iteration + 1) as f64;
    let pow_alpha = t * t.sqrt();
    let next_discount = (pow_alpha / (pow_alpha + 1.0)) as f32;

    let mut predicted_vals = Vec::with_capacity(cum_vals.len());
    cum_vals
        .iter()
        .zip(&inst_regrets)
        .for_each(|(&reg_cum, &reg_inst)| {
            predicted_vals.push((reg_cum * next_discount + reg_inst).max(0.0));
        });

    // 6. Encode
    let seed = params.current_iteration.wrapping_add(player as u32);
    let new_scale_cum = encode_signed_i8(cum_regret, &cum_vals, seed);
    let new_scale_pred = encode_signed_i8(predicted_regret, &predicted_vals, seed ^ 0x55555555);

    node.set_regret_scale(new_scale_cum);
    node.set_prev_regret_scale(new_scale_pred);
}

/// Int8 SAPCFR+ regret update
#[inline]
fn update_regrets_int8_sapcfr_plus<N: GameNode>(
    node: &mut N,
    cfv_actions: &mut [f32],
    result: &[f32],
    locking: &[f32],
    params: &DiscountParams,
    num_hands: usize,
    player: usize,
    decoder: f32,
) {
    let (cum_regret, prev_regret) = node.regrets_and_prev_i8_mut();

    // 1. ADD Implicit
    cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
        let val_impl = *y as f32 * decoder;
        *x += val_impl;
    });

    // 2. Subtract EV
    cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
        sub_slice(row, result);
    });

    // 3. Locking
    if !locking.is_empty() {
        cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        })
    }

    // 4. Compute Prev & Clip Implicit
    let mut prev_vals = Vec::with_capacity(cfv_actions.len());
    cfv_actions
        .iter()
        .zip(&*cum_regret)
        .for_each(|(&sum_val, y)| {
            let val_impl = *y as f32 * decoder;
            prev_vals.push(sum_val - val_impl);
        });

    cfv_actions.iter_mut().for_each(|x| {
        if *x < 0.0 {
            *x = 0.0;
        }
    });

    // Encode
    let seed = params.current_iteration.wrapping_add(player as u32);
    let new_scale_prev = encode_signed_i8(prev_regret, &prev_vals, seed ^ 0x55555555);
    let new_scale = encode_signed_i8(cum_regret, cfv_actions, seed);

    node.set_prev_regret_scale(new_scale_prev);
    node.set_regret_scale(new_scale);
}

/// Int8 DCFR/DCFR+ regret update
///
/// OPTIMIZATION: All operations fused into single pass per action row
/// (decode + discount + add cfv + subtract result + clip) for cache locality.
#[inline]
fn update_regrets_int8_dcfr<T: Game>(
    game: &T,
    node: &mut T::Node,
    cfv_actions: &mut [f32],
    result: &[f32],
    locking: &[f32],
    params: &DiscountParams,
    num_hands: usize,
    player: usize,
    scale: f32,
    decoder: f32,
) {
    let (alpha, beta) = (params.alpha_t, params.beta_t);
    let can_use_dcfr_plus = params.algorithm == CfrAlgorithm::DCFRPlus;
    let num_actions = cfv_actions.len() / num_hands;
    let quantization_mode = game.quantization_mode();

    if can_use_dcfr_plus {
        if quantization_mode == QuantizationMode::Int4Packed {
            // DCFR+ Int4Packed: FUSED decode + discount + add + subtract + clip
            let cum_regret_packed = node.regrets_u4_packed();
            let decoder_u4 = scale / 15.0;
            for action in 0..num_actions {
                let row_start = action * num_hands;
                for i in 0..num_hands {
                    let idx = row_start + i;
                    let byte = cum_regret_packed[idx / 2];
                    let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    let real_prev = nibble as f32 * decoder_u4;
                    let new_val = real_prev * alpha + cfv_actions[idx] - result[i];
                    cfv_actions[idx] = if new_val < 0.0 { 0.0 } else { new_val };
                }
            }
        } else {
            // DCFR+ Int8: FUSED decode + discount + add + subtract + clip
            let cum_regret = node.regrets_u8_mut();
            let decoder_u8 = scale / u8::MAX as f32;
            for action in 0..num_actions {
                let row_start = action * num_hands;
                let row_end = row_start + num_hands;
                let cfv_row = &mut cfv_actions[row_start..row_end];
                let reg_row = &cum_regret[row_start..row_end];

                cfv_row
                    .iter_mut()
                    .zip(reg_row)
                    .zip(result)
                    .for_each(|((cfv, &reg_enc), &res)| {
                        let real_prev = reg_enc as f32 * decoder_u8;
                        let new_val = real_prev * alpha + *cfv - res;
                        *cfv = if new_val < 0.0 { 0.0 } else { new_val };
                    });
            }
        }
    } else if quantization_mode == QuantizationMode::Int4Packed {
        // DCFR Int4Packed: FUSED decode + discount + add + subtract
        let cum_regret_packed = node.regrets_i4_packed();
        let decoder_i4 = scale / 7.0;
        for action in 0..num_actions {
            let row_start = action * num_hands;
            for i in 0..num_hands {
                let idx = row_start + i;
                let byte = cum_regret_packed[idx / 2];
                let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let val = ((nibble << 4) as i8) >> 4;
                let real_prev = val as f32 * decoder_i4;
                let coef = if real_prev >= 0.0 { alpha } else { beta };
                cfv_actions[idx] = real_prev * coef + cfv_actions[idx] - result[i];
            }
        }
    } else {
        // DCFR Int8: FUSED decode + discount + add + subtract
        let cum_regret = node.regrets_i8_mut();
        for action in 0..num_actions {
            let row_start = action * num_hands;
            let row_end = row_start + num_hands;
            let cfv_row = &mut cfv_actions[row_start..row_end];
            let reg_row = &cum_regret[row_start..row_end];

            cfv_row
                .iter_mut()
                .zip(reg_row)
                .zip(result)
                .for_each(|((cfv, &reg_enc), &res)| {
                    let real_prev = reg_enc as f32 * decoder;
                    let coef = if reg_enc >= 0 { alpha } else { beta };
                    *cfv = real_prev * coef + *cfv - res;
                });
        }
    }

    // Apply locking (still separate - only runs if locking is enabled)
    if !locking.is_empty() {
        cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = 0.0;
            }
        })
    }

    // Encode back
    let seed = params.current_iteration.wrapping_add(player as u32);

    if can_use_dcfr_plus {
        if quantization_mode == QuantizationMode::Int4Packed {
            let new_scale =
                encode_unsigned_u4_packed(node.regrets_u4_packed_mut(), cfv_actions, seed);
            node.set_regret_scale(new_scale);
        } else {
            let new_scale = encode_unsigned_regrets_u8(node.regrets_u8_mut(), cfv_actions, seed);
            node.set_regret_scale(new_scale);
        }
    } else if quantization_mode == QuantizationMode::Int4Packed {
        let new_scale = encode_signed_i4_packed(node.regrets_i4_packed_mut(), cfv_actions, seed);
        node.set_regret_scale(new_scale);
    } else {
        let new_scale = encode_signed_i8(node.regrets_i8_mut(), cfv_actions, seed);
        node.set_regret_scale(new_scale);
    }
}
