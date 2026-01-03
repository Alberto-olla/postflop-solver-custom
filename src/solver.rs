use crate::interface::*;
use crate::mutex_like::*;
use crate::quantization::QuantizationMode;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

// Import nuovi moduli trait-based
use crate::cfr_algorithms::{CfrAlgorithmTrait, DcfrAlgorithm, DcfrPlusAlgorithm, SapcfrPlusAlgorithm};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfrAlgorithm {
    DCFR,
    DCRFPlus,
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
            Self::DCRFPlus => DcfrPlusAlgorithm.requires_storage4(),
            Self::SAPCFRPlus => SapcfrPlusAlgorithm.requires_storage4(),
        }
    }
}

impl Default for CfrAlgorithm {
    fn default() -> Self {
        Self::DCFR
    }
}

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
    algorithm: CfrAlgorithm,
}

impl DiscountParams {
    pub fn new(current_iteration: u32, algorithm: CfrAlgorithm) -> Self {
        // Usa i nuovi trait per calcolare i discount params
        let params = match algorithm {
            CfrAlgorithm::DCFR => DcfrAlgorithm.compute_discounts(current_iteration),
            CfrAlgorithm::DCRFPlus => DcfrPlusAlgorithm.compute_discounts(current_iteration),
            CfrAlgorithm::SAPCFRPlus => SapcfrPlusAlgorithm.compute_discounts(current_iteration),
        };

        Self {
            alpha_t: params.alpha_t,
            beta_t: params.beta_t,
            gamma_t: params.gamma_t,
            algorithm,
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
                println!("       Current: {:.6}, Target: {:.6}", exploitability, target_exploitability);
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

        // compute the strategy by regret-maching algorithm
        let mut strategy = if params.algorithm == CfrAlgorithm::SAPCFRPlus {
             // SAPCFR+ Strategy Calculation (Explicit Regret)
             let mut explicit_regrets = Vec::with_capacity(num_actions * num_hands);
             
             if game.is_compression_enabled() {
                 let quantization_mode = game.quantization_mode();
                 match quantization_mode {
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
                                if log_val >= 0.0 { log_val.exp() - 1.0 } else { -((-log_val).exp() - 1.0) }
                            } else {
                                r_impl as f32 * scale_impl * decode_factor
                            };

                            let v_prev = if quantization_mode == QuantizationMode::Int16Log {
                                let log_val = r_prev as f32 * scale_prev * decode_factor;
                                if log_val >= 0.0 { log_val.exp() - 1.0 } else { -((-log_val).exp() - 1.0) }
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
             
             #[cfg(feature = "custom-alloc")]
             let strategy_vec = regret_matching(&explicit_regrets, num_actions);
             #[cfg(not(feature = "custom-alloc"))]
             let strategy_vec = regret_matching(&explicit_regrets, num_actions);
             
             strategy_vec
        } else if game.is_compression_enabled() {
            if game.quantization_mode() == QuantizationMode::Int16Log {
                regret_matching_compressed_log(node.regrets_compressed(), node.regret_scale(), num_actions)
            } else {
                regret_matching_compressed(node.regrets_compressed(), num_actions)
            }
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        if game.is_compression_enabled() {
            // Update cumulative strategy - dispatch based on precision
            match (game.quantization_mode(), game.strategy_bits()) {
                (QuantizationMode::Int16, 8) => {
                    // 8-bit strategy mode (mixed precision)
                    let scale = node.strategy_scale();
                    let decoder = params.gamma_t * scale / u8::MAX as f32;
                    let cum_strategy_u8 = node.strategy_u8_mut();

                    strategy.iter_mut().zip(&*cum_strategy_u8).for_each(|(x, y)| {
                        *x += (*y as f32) * decoder;
                    });

                    if !locking.is_empty() {
                        strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                            if s.is_sign_positive() {
                                *d = 0.0;
                            }
                        })
                    }

                    let new_scale = encode_unsigned_strategy_u8(cum_strategy_u8, &strategy);
                    node.set_strategy_scale(new_scale);
                }
                (QuantizationMode::Int16, 16) | (QuantizationMode::Int16Log, 16) | (QuantizationMode::Float32, _) => {
                    // Normal 16-bit mode (or 32-bit mode)
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
                (QuantizationMode::Int16, 4) => {
                    // 4-bit strategy mode (nibble packing)
                    let scale = node.strategy_scale();
                    let decoder = params.gamma_t * scale / 15.0;  // 4-bit max value is 15
                    let cum_strategy_u4 = node.strategy_i4_packed_mut();
                    let num_elements = strategy.len();

                    // Decode nibbles and accumulate
                    for i in 0..num_elements {
                        let byte_idx = i / 2;
                        let nibble = if i % 2 == 0 {
                            cum_strategy_u4[byte_idx] & 0x0F
                        } else {
                            (cum_strategy_u4[byte_idx] >> 4) & 0x0F
                        };
                        strategy[i] += (nibble as f32) * decoder;
                    }

                    if !locking.is_empty() {
                        strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                            if s.is_sign_positive() {
                                *d = 0.0;
                            }
                        })
                    }

                    let new_scale = encode_unsigned_strategy_u4(cum_strategy_u4, &strategy);
                    node.set_strategy_scale(new_scale);
                }
                _ => {
                    panic!("Invalid quantization/strategy_bits combination");
                }
            }

            // update the cumulative regret
            let scale = node.regret_scale();
            let quantization_mode = game.quantization_mode();

            if params.algorithm == CfrAlgorithm::SAPCFRPlus {
                let (cum_regret, prev_regret) = node.regrets_and_prev_compressed_mut();
                let decoder = scale / i16::MAX as f32;
                
                // 1. ADD: Implicit_New_Unclipped = Implicit_Old + Instantaneous
                // Implicit_Old is in cum_regret (compressed). cfv_actions holds CFV.
                cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                    let log_impl = *y as f32 * decoder;
                    let val_impl = if quantization_mode == QuantizationMode::Int16Log {
                        if log_impl >= 0.0 { log_impl.exp() - 1.0 } else { -((-log_impl).exp() - 1.0) }
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
                        if s.is_sign_positive() { *d = 0.0; }
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
                
                cfv_actions.iter().zip(&*cum_regret).for_each(|(&sum_val, y)| {
                    let log_impl = *y as f32 * decoder;
                     let val_impl = if quantization_mode == QuantizationMode::Int16Log {
                        if log_impl >= 0.0 { log_impl.exp() - 1.0 } else { -((-log_impl).exp() - 1.0) }
                    } else {
                        log_impl
                    };
                    prev_vals.push(sum_val - val_impl);
                });
                
                // Clip Implicit (cfv_actions)
               cfv_actions.iter_mut().for_each(|x| {
                    if *x < 0.0 { *x = 0.0; }
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
                            let real_prev = if log_val >= 0.0 { log_val.exp() - 1.0 } else { -((-log_val).exp() - 1.0) };
                            let coef = if real_prev >= 0.0 { alpha } else { beta };
                            *x += real_prev * coef;
                        });
                    }
                    (CfrAlgorithm::DCRFPlus, QuantizationMode::Int16Log) => {
                        let decoder = scale / i16::MAX as f32;
                        let alpha = params.alpha_t;
                        cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                             let log_val = *y as f32 * decoder;
                            let real_prev = if log_val >= 0.0 { log_val.exp() - 1.0 } else { -((-log_val).exp() - 1.0) };
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
                    (CfrAlgorithm::DCRFPlus, _) => {
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
                        if s.is_sign_positive() { *d = 0.0; }
                    })
                }

                if params.algorithm == CfrAlgorithm::DCRFPlus {
                    cfv_actions.iter_mut().for_each(|x| {
                        if *x < 0.0 { *x = 0.0; }
                    });
                }

                let new_scale = if quantization_mode == QuantizationMode::Int16Log {
                    encode_signed_slice_log(cum_regret, &cfv_actions)
                } else {
                    encode_signed_slice(cum_regret, &cfv_actions)
                };
                node.set_regret_scale(new_scale);
            }
        } else {
            // update the cumulative strategy
            let gamma = params.gamma_t;
            let cum_strategy = node.strategy_mut();
            cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
                *x = *x * gamma + *y;
            });

            // update the cumulative regret
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
                CfrAlgorithm::DCRFPlus => {
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
                         if *x < 0.0 { *x = 0.0; }
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
                    
                    implicit.iter_mut().zip(prev.iter_mut()).zip(&inst_regrets).for_each(|((reg_impl, reg_prev), &reg_inst)| {
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
        let mut cfreach_actions = if params.algorithm == CfrAlgorithm::SAPCFRPlus {
             // SAPCFR+ Strategy Calculation (Explicit Regret) for opponent
             let node_num_hands = game.num_private_hands(node.player());
             let mut explicit_regrets = Vec::with_capacity(num_actions * node_num_hands);
             
             if game.is_compression_enabled() {
                 let quantization_mode = game.quantization_mode();
                 match quantization_mode {
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
                                if log_val >= 0.0 { log_val.exp() - 1.0 } else { -((-log_val).exp() - 1.0) }
                            } else {
                                r_impl as f32 * scale_impl * decode_factor
                            };

                            let v_prev = if quantization_mode == QuantizationMode::Int16Log {
                                let log_val = r_prev as f32 * scale_prev * decode_factor;
                                if log_val >= 0.0 { log_val.exp() - 1.0 } else { -((-log_val).exp() - 1.0) }
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
             
             regret_matching(&explicit_regrets, num_actions)
        } else if game.is_compression_enabled() {
            if game.quantization_mode() == QuantizationMode::Int16Log {
                regret_matching_compressed_log(node.regrets_compressed(), node.regret_scale(), num_actions)
            } else {
                regret_matching_compressed(node.regrets_compressed(), num_actions)
            }
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

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

/// Computes the strategy by regret-matching algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32, StackAlloc> {
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);
    let uninit = strategy.spare_capacity_mut();
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    unsafe { strategy.set_len(regret.len()) };

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    let uninit = strategy.spare_capacity_mut();
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    unsafe { strategy.set_len(regret.len()) };

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32, StackAlloc> {
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);
    strategy.extend(regret.iter().map(|&r| r.max(0) as f32));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm for log-compressed regrets.
/// Must decode first because log(R) is not scale-invariant.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching_compressed_log(regret: &[i16], scale: f32, num_actions: usize) -> Vec<f32, StackAlloc> {
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);
    let decoder = scale / i16::MAX as f32;
    
    strategy.extend(regret.iter().map(|&r| {
        let log_val = r as f32 * decoder;
        let val = if log_val >= 0.0 {
            log_val.exp() - 1.0
        } else {
            -((-log_val).exp() - 1.0)
        };
        val.max(0.0)
    }));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    strategy.extend(regret.iter().map(|&r| r.max(0) as f32));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm for log-compressed regrets.
/// Must decode first because log(R) is not scale-invariant.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching_compressed_log(regret: &[i16], scale: f32, num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    let decoder = scale / i16::MAX as f32;
    
    strategy.extend(regret.iter().map(|&r| {
        let log_val = r as f32 * decoder;
        let val = if log_val >= 0.0 {
            log_val.exp() - 1.0
        } else {
            -((-log_val).exp() - 1.0)
        };
        val.max(0.0)
    }));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

