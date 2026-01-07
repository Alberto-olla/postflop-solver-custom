mod pruning;
mod regrets;
mod strategy;

use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use pruning::{compute_pruning_threshold, should_prune_action_average, should_prune_action_max};
use regrets::update_regrets;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PruningMode {
    /// No pruning - process all branches
    #[default]
    Disabled,
    /// Legacy: Average-based pruning (can cut off strong hands in polarized ranges)
    Average,
    /// Recommended: MAX-based pruning with integer threshold optimization
    Max,
}

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
        let pruning_mode = game.pruning_mode();
        if pruning_mode != PruningMode::Disabled {
            // Pruning enabled: check each action before recursing
            let pruning_threshold = compute_pruning_threshold(
                game.tree_config().effective_stack,
                params.current_iteration,
            );

            for action in node.action_indices() {
                // Check if this action should be skipped (pruned)
                let should_skip = match pruning_mode {
                    PruningMode::Max => {
                        should_prune_action_max(game, node, action, num_hands, pruning_threshold)
                    }
                    PruningMode::Average => {
                        should_prune_action_average(game, node, action, num_hands, pruning_threshold)
                    }
                    PruningMode::Disabled => unreachable!(),
                };

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

        // Update cumulative regrets - dispatch to regrets module
        update_regrets(
            game,
            node,
            &mut cfv_actions,
            result,
            locking,
            params,
            num_hands,
            player,
        );
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
