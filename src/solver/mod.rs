mod pruning;
mod strategy;

use crate::interface::*;
use crate::mutex_like::*;
use crate::quantization::QuantizationMode;
use crate::sliceop::*;
use crate::utility::*;
use pruning::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;
use strategy::{
    compute_pdcfr_plus_strategy, compute_sapcfr_plus_strategy, compute_strategy,
    regret_matching_dispatch,
};

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

// Import algoritmi CFR principali
use crate::cfr_algorithms::{CfrAlgorithmTrait, DcfrAlgorithm, DcfrPlusAlgorithm};
// Import algoritmi sperimentali (richiedono import esplicito)
use crate::cfr_algorithms::experimental::{PdcfrPlusAlgorithm, SapcfrPlusAlgorithm};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfrAlgorithm {
    DCFR,
    DCFRPlus,
    PDCFRPlus,
    SAPCFRPlus,
}

impl CfrAlgorithm {
    /// Indica se l'algoritmo richiede storage4 (previous instantaneous regrets)
    ///
    /// Usa i trait da cfr_algorithms per determinare i requisiti di storage.
    #[inline]
    pub fn requires_storage4(&self) -> bool {
        match self {
            Self::DCFR => DcfrAlgorithm.requires_storage4(),
            Self::DCFRPlus => DcfrPlusAlgorithm.requires_storage4(),
            Self::PDCFRPlus => PdcfrPlusAlgorithm.requires_storage4(),
            Self::SAPCFRPlus => SapcfrPlusAlgorithm.requires_storage4(),
        }
    }
}

impl Default for CfrAlgorithm {
    fn default() -> Self {
        Self::DCFR
    }
}

pub(super) struct DiscountParams {
    pub(super) alpha_t: f32,
    pub(super) beta_t: f32,
    pub(super) gamma_t: f32,
    pub(super) algorithm: CfrAlgorithm,
    pub(super) current_iteration: u32,
}

impl DiscountParams {
    pub fn new(current_iteration: u32, algorithm: CfrAlgorithm) -> Self {
        // Usa i nuovi trait per calcolare i discount params
        let params = match algorithm {
            CfrAlgorithm::DCFR => DcfrAlgorithm.compute_discounts(current_iteration),
            CfrAlgorithm::DCFRPlus => DcfrPlusAlgorithm.compute_discounts(current_iteration),
            CfrAlgorithm::PDCFRPlus => PdcfrPlusAlgorithm.compute_discounts(current_iteration),
            CfrAlgorithm::SAPCFRPlus => SapcfrPlusAlgorithm.compute_discounts(current_iteration),
        };

        Self {
            alpha_t: params.alpha_t,
            beta_t: params.beta_t,
            gamma_t: params.gamma_t,
            algorithm,
            current_iteration,
        }
    }
    // Le funzioni new_dcfr, new_dcfr_plus, new_sapcfr_plus sono state rimosse
    // Ora usiamo i trait da cfr_algorithms per calcolare i discount params
}

/// Performs Discounted CFR algorithm until the given number of iterations or exploitability is
/// satisfied.
///
/// This method returns the exploitability of the obtained strategy.
pub fn solve<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let mut exploitability = compute_exploitability(game);

    if print_progress {
        println!(); // Newline per separare dall'output di cargo test
        print!("iteration: 0 / {max_num_iterations} ");
        print!("(exploitability = {exploitability:.4e})");
        io::stdout().flush().unwrap();
    }

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            if print_progress {
                println!("\n[STOP] Reached target exploitability at iteration {}", t);
                println!(
                    "       Current: {:.6}, Target: {:.6}",
                    exploitability, target_exploitability
                );

                let mem = game.memory_usage_detailed();
                println!("       Memory usage: {:.2} MB", mem.total_mb());
                println!(
                    "         Strategy:    {:>8.2} MB",
                    mem.strategy as f64 / 1_048_576.0
                );
                println!(
                    "         Regrets:     {:>8.2} MB",
                    (mem.regrets + mem.storage4) as f64 / 1_048_576.0
                );
                println!(
                    "         IP CFVs:     {:>8.2} MB",
                    mem.ip_cfvalues as f64 / 1_048_576.0
                );
                println!(
                    "         Chance CFVs: {:>8.2} MB",
                    mem.chance_cfvalues as f64 / 1_048_576.0
                );
                println!(
                    "         Misc:        {:>8.2} MB",
                    mem.misc as f64 / 1_048_576.0
                );

                io::stdout().flush().unwrap();
            }
            break;
        }

        let params = DiscountParams::new(t, game.cfr_algorithm());

        // alternating updates
        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        // Calculate exploitability every 10 iterations for performance and cleaner output
        if (t + 1) % 10 == 0 {
            exploitability = compute_exploitability(game);

            if print_progress {
                print!("\riteration: {} / {} ", t + 1, max_num_iterations);
                print!("(exploitability = {exploitability:.4e})");
                io::stdout().flush().unwrap();
            }
        }
    }

    if print_progress {
        println!();
        println!(); // Newline extra per separare dal prossimo test
        io::stdout().flush().unwrap();
    }

    finalize(game);

    exploitability
}

/// Proceeds Discounted CFR algorithm for one iteration.
#[inline]
pub fn solve_step<T: Game>(game: &T, current_iteration: u32) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration, game.cfr_algorithm());

    // alternating updates
    for player in 0..2 {
        let mut result = Vec::with_capacity(game.num_private_hands(player));
        solve_recursive(
            result.spare_capacity_mut(),
            game,
            &mut root,
            player,
            game.initial_weights(player ^ 1),
            &params,
        );
    }
}

/// Recursively solves the counterfactual values.
fn solve_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
) {
    // return the counterfactual values when the `node` is terminal
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive(result, game, child, player, cfreach, params);
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // if the `node` is chance
    if node.is_chance() {
        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_updated = Vec::with_capacity_in(cfreach.len(), StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
            );
        });

        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        unsafe { result_f64.set_len(num_hands) };

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // process isomorphic chances
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

            apply_swap(tmp, swap_list);

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            apply_swap(tmp, swap_list);
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });
    }
    // if the current player is `player`
    else if node.player() == player {
        // compute the counterfactual values of each action
        // Use manual iteration to support pruning (branch skipping)
        if game.enable_pruning() {
            // Pruning enabled: check each action before recursing
            let pruning_threshold = compute_pruning_threshold(
                game.tree_config().effective_stack,
                params.current_iteration,
            );

            for action in node.action_indices() {
                // Check if this action should be skipped (pruned)
                let should_skip =
                    should_prune_action(game, node, action, num_hands, pruning_threshold);

                if should_skip {
                    // Skip this branch - fill with zeros
                    let mut cfv_lock = cfv_actions.lock();
                    let row = row_mut(cfv_lock.spare_capacity_mut(), action, num_hands);
                    for elem in row {
                        elem.write(0.0);
                    }
                    drop(cfv_lock); // Release lock before recursion
                } else {
                    // Normal recursion
                    solve_recursive(
                        row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                        game,
                        &mut node.play(action),
                        player,
                        cfreach,
                        params,
                    );
                }
            }
        } else {
            // Pruning disabled: use original parallel for_each
            for_each_child(node, |action| {
                solve_recursive(
                    row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                    game,
                    &mut node.play(action),
                    player,
                    cfreach,
                    params,
                );
            });
        }

        // compute the strategy by regret-maching algorithm
        let mut strategy = compute_strategy(game, node, params, num_actions, num_hands);

        // node-locking
        let locking = game.locking_strategy(node);
        crate::utility::apply_locking_strategy(&mut strategy, locking);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        // Update cumulative strategy - dispatch based on strategy_bits (independent of regret compression)
        match game.strategy_bits() {
            32 => {
                // 32-bit strategy mode (full precision)
                let cum_strategy = node.strategy_mut();

                strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                    *x += *y * params.gamma_t;
                });

                if !locking.is_empty() {
                    strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                        if s.is_sign_positive() {
                            *d = 0.0;
                        }
                    })
                }

                cum_strategy.copy_from_slice(&strategy);
            }
            16 => {
                // 16-bit strategy mode (compressed)
                let scale = node.strategy_scale();
                let decoder = params.gamma_t * scale / u16::MAX as f32;
                let cum_strategy = node.strategy_compressed_mut();

                strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                    *x += (*y as f32) * decoder;
                });

                if !locking.is_empty() {
                    strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                        if s.is_sign_positive() {
                            *d = 0.0;
                        }
                    })
                }

                let new_scale = encode_unsigned_slice(cum_strategy, &strategy);
                node.set_strategy_scale(new_scale);
            }
            8 => {
                // 8-bit strategy mode (compressed)
                let scale = node.strategy_scale();
                let decoder = params.gamma_t * scale / u8::MAX as f32; // u8::MAX = 255
                let cum_strategy = node.strategy_u8_mut();

                strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                    *x += (*y as f32) * decoder;
                });

                if !locking.is_empty() {
                    strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                        if s.is_sign_positive() {
                            *d = 0.0;
                        }
                    })
                }

                let seed = params.current_iteration.wrapping_add(player as u32);
                let new_scale = encode_unsigned_strategy_u8(cum_strategy, &strategy, seed);
                node.set_strategy_scale(new_scale);
            }
            _ => {
                panic!(
                    "Invalid strategy_bits: {}. Valid values: 8, 16, 32",
                    game.strategy_bits()
                );
            }
        }

        // Update cumulative regrets - dispatch based on compression mode
        if game.is_compression_enabled() {
            let quantization_mode = game.quantization_mode(); // Defined once

            if quantization_mode == QuantizationMode::Int8 {
                // --- INT8 IMPLEMENTATION ---
                let scale = node.regret_scale();
                let decoder = scale / i8::MAX as f32;

                if params.algorithm == CfrAlgorithm::PDCFRPlus {
                    // PDCFR+ 8-bit
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
                    let new_scale_pred =
                        encode_signed_i8(predicted_regret, &predicted_vals, seed ^ 0x55555555);

                    node.set_regret_scale(new_scale_cum);
                    node.set_prev_regret_scale(new_scale_pred);
                } else if params.algorithm == CfrAlgorithm::SAPCFRPlus {
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
                    let new_scale_prev =
                        encode_signed_i8(prev_regret, &prev_vals, seed ^ 0x55555555);
                    let new_scale = encode_signed_i8(cum_regret, &cfv_actions, seed);

                    node.set_prev_regret_scale(new_scale_prev);
                    node.set_regret_scale(new_scale);
                } else {
                    // DCFR / DCFR+
                    let (alpha, beta) = (params.alpha_t, params.beta_t);
                    let can_use_dcfr_plus = params.algorithm == CfrAlgorithm::DCFRPlus;

                    if can_use_dcfr_plus {
                        if game.quantization_mode() == QuantizationMode::Int4Packed {
                            let cum_regret_packed = node.regrets_u4_packed();
                            let decoder_u4 = scale / 15.0;
                            cfv_actions.iter_mut().enumerate().for_each(|(i, x)| {
                                let byte = cum_regret_packed[i / 2];
                                let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                                let real_prev = nibble as f32 * decoder_u4;
                                *x += real_prev * alpha;
                            });
                        } else {
                            let cum_regret = node.regrets_u8_mut();
                            let decoder_u8 = scale / u8::MAX as f32;

                            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                                let real_prev = *y as f32 * decoder_u8;
                                *x += real_prev * alpha;
                            });
                        }
                    } else if game.quantization_mode() == QuantizationMode::Int4Packed {
                        let cum_regret_packed = node.regrets_i4_packed();
                        let decoder_i4 = scale / 7.0;
                        cfv_actions.iter_mut().enumerate().for_each(|(i, x)| {
                            let byte = cum_regret_packed[i / 2];
                            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                            let val = ((nibble << 4) as i8) >> 4;
                            let real_prev = val as f32 * decoder_i4;
                            let coef = if real_prev >= 0.0 { alpha } else { beta };
                            *x += real_prev * coef;
                        });
                    } else {
                        let cum_regret = node.regrets_i8_mut();
                        cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                            let real_prev = *y as f32 * decoder;
                            let coef = if real_prev >= 0.0 { alpha } else { beta };
                            *x += real_prev * coef;
                        });
                    }

                    cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
                        sub_slice(row, result);
                    });

                    if !locking.is_empty() {
                        cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
                            if s.is_sign_positive() {
                                *d = 0.0;
                            }
                        })
                    }

                    if can_use_dcfr_plus {
                        if game.quantization_mode() == QuantizationMode::Int4Packed {
                            let seed = params.current_iteration.wrapping_add(player as u32);
                            let new_scale = encode_unsigned_u4_packed(
                                node.regrets_u4_packed_mut(),
                                &cfv_actions,
                                seed,
                            );
                            node.set_regret_scale(new_scale);
                        } else {
                            let seed = params.current_iteration.wrapping_add(player as u32);
                            let new_scale = encode_unsigned_regrets_u8(
                                node.regrets_u8_mut(),
                                &cfv_actions,
                                seed,
                            );
                            node.set_regret_scale(new_scale);
                        }
                    } else {
                        if game.quantization_mode() == QuantizationMode::Int4Packed {
                            let seed = params.current_iteration.wrapping_add(player as u32);
                            let new_scale = encode_signed_i4_packed(
                                node.regrets_i4_packed_mut(),
                                &cfv_actions,
                                seed,
                            );
                            node.set_regret_scale(new_scale);
                        } else {
                            let seed = params.current_iteration.wrapping_add(player as u32);
                            let new_scale =
                                encode_signed_i8(node.regrets_i8_mut(), &cfv_actions, seed);
                            node.set_regret_scale(new_scale);
                        }
                    }
                }
            } else {
                // --- INT16 IMPLEMENTATION (Original) ---
                let scale = node.regret_scale();

                if params.algorithm == CfrAlgorithm::PDCFRPlus {
                    // PDCFR+ compressed mode
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

                    // 4. Update cumulative with uniform discounting (like DCFR+) + RM+ clipping
                    // PDCFR+ uses uniform discounting (only alpha), NOT conditional (alpha/beta)
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
                } else if params.algorithm == CfrAlgorithm::SAPCFRPlus {
                    let (cum_regret, prev_regret) = node.regrets_and_prev_compressed_mut();
                    let decoder = scale / i16::MAX as f32;

                    // 1. ADD: Implicit_New_Unclipped = Implicit_Old + Instantaneous
                    // Implicit_Old is in cum_regret (compressed). cfv_actions holds CFV.
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

                    // 2. Subtract EV -> cfv_actions holds (Implicit_Old + Instantaneous)
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

                    // 4. Compute Prev_Regret (Instantaneous), Then Clip Implicit
                    // Prev = cfv_actions - Implicit_Old
                    // We reuse the logic loop to compute Prev and store it.

                    let mut prev_vals = Vec::with_capacity(cfv_actions.len());

                    // Use temp logic to recover Prev because we modified cfv_actions with clamping updates?
                    // Wait, we haven't clamped cfv_actions yet in this block.
                    // We just accumulated.
                    // But we need Implicit Old to recover Prev.
                    // We have cum_regret which still holds Implicit_Old.

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

                    // Clip Implicit (cfv_actions)
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
                        encode_signed_slice_log(cum_regret, &cfv_actions)
                    } else {
                        encode_signed_slice(cum_regret, &cfv_actions)
                    };

                    // Set scales AFTER encoding to drop mutable borrows of storage slices
                    node.set_prev_regret_scale(new_scale_prev);
                    node.set_regret_scale(new_scale);
                } else {
                    let cum_regret = node.regrets_compressed_mut();

                    match (params.algorithm, quantization_mode) {
                        (CfrAlgorithm::DCFR, QuantizationMode::Int16Log) => {
                            let decoder = scale / i16::MAX as f32;
                            let (alpha, beta) = (params.alpha_t, params.beta_t);
                            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                                let log_val = *y as f32 * decoder;
                                let real_prev = if log_val >= 0.0 {
                                    log_val.exp() - 1.0
                                } else {
                                    -((-log_val).exp() - 1.0)
                                };
                                let coef = if real_prev >= 0.0 { alpha } else { beta };
                                *x += real_prev * coef;
                            });
                        }
                        (CfrAlgorithm::DCFRPlus, QuantizationMode::Int16Log) => {
                            let decoder = scale / i16::MAX as f32;
                            let alpha = params.alpha_t;
                            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                                let log_val = *y as f32 * decoder;
                                let real_prev = if log_val >= 0.0 {
                                    log_val.exp() - 1.0
                                } else {
                                    -((-log_val).exp() - 1.0)
                                };
                                *x += real_prev * alpha;
                            });
                        }
                        (CfrAlgorithm::DCFR, _) => {
                            let alpha_decoder = params.alpha_t * scale / i16::MAX as f32;
                            let beta_decoder = params.beta_t * scale / i16::MAX as f32;
                            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                                let coef = if *y >= 0 { alpha_decoder } else { beta_decoder };
                                *x += *y as f32 * coef;
                            });
                        }
                        (CfrAlgorithm::DCFRPlus, _) => {
                            let alpha_decoder = params.alpha_t * scale / i16::MAX as f32;
                            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                                *x += *y as f32 * alpha_decoder;
                            });
                        }
                        _ => {}
                    }

                    cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
                        sub_slice(row, result);
                    });

                    if !locking.is_empty() {
                        cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
                            if s.is_sign_positive() {
                                *d = 0.0;
                            }
                        })
                    }

                    if params.algorithm == CfrAlgorithm::DCFRPlus {
                        cfv_actions.iter_mut().for_each(|x| {
                            if *x < 0.0 {
                                *x = 0.0;
                            }
                        });
                    }

                    let new_scale = if quantization_mode == QuantizationMode::Int16Log {
                        encode_signed_slice_log(cum_regret, &cfv_actions)
                    } else {
                        encode_signed_slice(cum_regret, &cfv_actions)
                    };
                    node.set_regret_scale(new_scale);
                }
            }
        } else {
            // update the cumulative regret (uncompressed mode)
            let cum_regret = node.regrets_mut();
            match params.algorithm {
                CfrAlgorithm::DCFR => {
                    // DCFR: conditional discounting based on sign
                    let (alpha, beta) = (params.alpha_t, params.beta_t);
                    cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
                        let coef = if x.is_sign_positive() { alpha } else { beta };
                        *x = *x * coef + *y;
                    });
                    cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
                        sub_slice(row, result);
                    });
                }
                CfrAlgorithm::DCFRPlus => {
                    // DCFR+: discount, then add instantaneous regret, then clip
                    let alpha = params.alpha_t;
                    cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
                        *x = *x * alpha + *y;
                    });
                    cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
                        sub_slice(row, result);
                    });
                    // Clip after computing instantaneous regret
                    cum_regret.iter_mut().for_each(|x| {
                        if *x < 0.0 {
                            *x = 0.0;
                        }
                    });
                }
                CfrAlgorithm::PDCFRPlus => {
                    // PDCFR+: Combines DCFR discounting with predictive mechanism
                    // Storage2: cumulative regrets (Rt)
                    // Storage4: predicted regrets (R̃t+1) for next iteration

                    // 1. Calcola instantaneous regrets
                    let mut inst_regrets = Vec::with_capacity(cfv_actions.len());
                    inst_regrets.extend_from_slice(&cfv_actions);
                    inst_regrets.chunks_exact_mut(num_hands).for_each(|row| {
                        sub_slice(row, result);
                    });

                    // 2. Calcola next_discount per predicted regrets
                    // Formula: t^1.5 / (t^1.5 + 1) con t = iterazione (1-indexed come nel paper)
                    // current_iteration è 0-indexed, quindi aggiungo 1 per convertire
                    let t = (params.current_iteration + 1) as f64;
                    let pow_alpha = t * t.sqrt(); // t^1.5
                    let next_discount = (pow_alpha / (pow_alpha + 1.0)) as f32;

                    // 3. Get both cumulative and predicted regrets
                    let (cumulative, predicted) = node.regrets_and_prev_mut();

                    // 4. Update cumulative regrets + compute predicted regrets
                    // PDCFR+ uses uniform discounting (only alpha), NOT conditional (alpha/beta)
                    let alpha = params.alpha_t;
                    cumulative
                        .iter_mut()
                        .zip(predicted.iter_mut())
                        .zip(&inst_regrets)
                        .for_each(|((reg_cum, reg_pred), &reg_inst)| {
                            // Update cumulative with uniform discounting (like DCFR+) + RM+ clipping
                            *reg_cum = (*reg_cum * alpha + reg_inst).max(0.0);

                            // Compute predicted for next iteration
                            *reg_pred = (*reg_cum * next_discount + reg_inst).max(0.0);
                        });
                }
                CfrAlgorithm::SAPCFRPlus => {
                    // SAPCFR+ (Uncompressed / Float32)
                    let mut inst_regrets = Vec::with_capacity(cfv_actions.len());
                    inst_regrets.extend_from_slice(&cfv_actions);
                    inst_regrets.chunks_exact_mut(num_hands).for_each(|row| {
                        sub_slice(row, result);
                    });

                    // Update Implicit (storage2) and Prev (storage4) simultaneously
                    // Use split-borrow method
                    let (implicit, prev) = node.regrets_and_prev_mut();

                    implicit
                        .iter_mut()
                        .zip(prev.iter_mut())
                        .zip(&inst_regrets)
                        .for_each(|((reg_impl, reg_prev), &reg_inst)| {
                            *reg_impl = (*reg_impl + reg_inst).max(0.0);
                            *reg_prev = reg_inst;
                        });
                }
            }
        }
    }
    // if the current player is not `player`
    else {
        // compute the strategy by regret-matching algorithm
        let node_num_hands = game.num_private_hands(node.player());
        let mut cfreach_actions = match params.algorithm {
            CfrAlgorithm::PDCFRPlus => {
                compute_pdcfr_plus_strategy(game, node, num_actions, node_num_hands)
            }
            CfrAlgorithm::SAPCFRPlus => {
                compute_sapcfr_plus_strategy(game, node, num_actions, node_num_hands)
            }
            _ => regret_matching_dispatch(game, node, params.algorithm, num_actions),
        };

        // node-locking
        let locking = game.locking_strategy(node);
        crate::utility::apply_locking_strategy(&mut cfreach_actions, locking);

        // update the reach probabilities
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}
