use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use std::mem::{self, MaybeUninit};
use std::ptr;

#[cfg(feature = "bincode")]
use crate::game::PostFlopGame;
#[cfg(feature = "bincode")]
use std::fs::File;
#[cfg(feature = "bincode")]
use std::io::{BufReader, BufWriter};

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Re-export encoding functions from quantization module for backwards compatibility
pub(crate) use crate::quantization::encoding::{
    decode_signed_i4_packed, decode_signed_i8, encode_signed_i4_packed, encode_signed_i8,
    encode_signed_slice, encode_signed_slice_log, encode_unsigned_regrets_u8,
    encode_unsigned_slice, encode_unsigned_strategy_u8, encode_unsigned_u4_packed,
};

/// Executes `op` for each child potentially in parallel.
#[cfg(feature = "rayon")]
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    if node.enable_parallelization() {
        node.action_indices().into_par_iter().for_each(op);
    } else {
        node.action_indices().for_each(op);
    }
}

/// Executes `op` for each child.
#[cfg(not(feature = "rayon"))]
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    node.action_indices().for_each(op);
}

#[cfg(feature = "rayon")]
pub(crate) fn into_par_iter(range: std::ops::Range<usize>) -> rayon::range::Iter<usize> {
    range.into_par_iter()
}

#[cfg(not(feature = "rayon"))]
pub(crate) fn into_par_iter(range: std::ops::Range<usize>) -> std::ops::Range<usize> {
    range
}

#[inline]
pub(crate) fn max(x: f32, y: f32) -> f32 {
    if x > y {
        x
    } else {
        y
    }
}

#[inline]
pub(crate) fn is_zero(x: f32) -> bool {
    x.to_bits() == 0
}

#[inline]
pub(crate) fn vec_memory_usage<T>(vec: &Vec<T>) -> u64 {
    vec.capacity() as u64 * mem::size_of::<T>() as u64
}

/// Computes the average with given weights.
#[inline]
pub fn compute_average(slice: &[f32], weights: &[f32]) -> f32 {
    let mut weight_sum = 0.0;
    let mut value_sum = 0.0;
    for (&v, &w) in slice.iter().zip(weights.iter()) {
        weight_sum += w as f64;
        value_sum += v as f64 * w as f64;
    }
    (value_sum / weight_sum) as f32
}

#[inline]
fn weighted_sum(values: &[f32], weights: &[f32]) -> f32 {
    let f = |sum: f64, (&v, &w): (&f32, &f32)| sum + v as f64 * w as f64;
    values.iter().zip(weights).fold(0.0, f) as f32
}

/// Applies the given swap to the given slice.
#[inline]
pub(crate) fn apply_swap<T>(slice: &mut [T], swap_list: &[(u16, u16)]) {
    for &(i, j) in swap_list {
        unsafe {
            ptr::swap(
                slice.get_unchecked_mut(i as usize),
                slice.get_unchecked_mut(j as usize),
            );
        }
    }
}

/// Finalizes the solving process.
#[inline]
pub fn finalize<T: Game>(game: &mut T) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    // compute the expected values and save them
    for player in 0..2 {
        let mut cfvalues = Vec::with_capacity(game.num_private_hands(player));
        compute_cfvalue_recursive(
            cfvalues.spare_capacity_mut(),
            game,
            &mut game.root(),
            player,
            game.initial_weights(player ^ 1),
            true,
        );
    }

    // set the game solved
    game.set_solved();

    // free buffer
    #[cfg(all(feature = "custom-alloc", feature = "rayon"))]
    rayon::broadcast(|_| free_custom_alloc_buffer());
    #[cfg(all(feature = "custom-alloc", not(feature = "rayon")))]
    free_custom_alloc_buffer();
}

/// Computes the exploitability of the current strategy.
#[inline]
pub fn compute_exploitability<T: Game>(game: &T) -> f32 {
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mes_ev = compute_mes_ev(game);
    if !game.is_raked() {
        (mes_ev[0] + mes_ev[1]) * 0.5
    } else {
        let current_ev = compute_current_ev(game);
        ((mes_ev[0] - current_ev[0]) + (mes_ev[1] - current_ev[1])) * 0.5
    }
}

/// Computes the expected values of the current strategy of each player.
///
/// The bias, i.e., (starting pot) / 2, is already subtracted to increase the significant figures.
/// This treatment makes the return value zero-sum when not raked.
#[inline]
pub fn compute_current_ev<T: Game>(game: &T) -> [f32; 2] {
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mut cfvalues = [
        Vec::with_capacity(game.num_private_hands(0)),
        Vec::with_capacity(game.num_private_hands(1)),
    ];

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    for player in 0..2 {
        compute_cfvalue_recursive(
            cfvalues[player].spare_capacity_mut(),
            game,
            &mut game.root(),
            player,
            reach[player ^ 1],
            false,
        );
        unsafe { cfvalues[player].set_len(game.num_private_hands(player)) };
    }

    let get_sum = |player: usize| weighted_sum(&cfvalues[player], reach[player]);
    [get_sum(0), get_sum(1)]
}

/// Computes the expected values of the MES (Maximally Exploitative Strategy) of each player.
///
/// The bias, i.e., (starting pot) / 2, is already subtracted to increase the significant figures.
/// Therefore, the average of the return value corresponds to the exploitability value if not raked.
#[inline]
pub fn compute_mes_ev<T: Game>(game: &T) -> [f32; 2] {
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mut cfvalues = [
        Vec::with_capacity(game.num_private_hands(0)),
        Vec::with_capacity(game.num_private_hands(1)),
    ];

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    for player in 0..2 {
        compute_best_cfv_recursive(
            cfvalues[player].spare_capacity_mut(),
            game,
            &game.root(),
            player,
            reach[player ^ 1],
        );
        unsafe { cfvalues[player].set_len(game.num_private_hands(player)) };
    }

    let get_sum = |player: usize| weighted_sum(&cfvalues[player], reach[player]);
    [get_sum(0), get_sum(1)]
}

/// The recursive helper function for computing the counterfactual values of the given strategy.
fn compute_cfvalue_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    save_cfvalues: bool,
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // chance node
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
            compute_cfvalue_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                save_cfvalues,
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

        // save the counterfactual values
        if save_cfvalues && node.cfvalue_storage_player() == Some(player) {
            let result = unsafe { &*(result as *const _ as *const [f32]) };
            // Dispatch based on chance_bits precision
            match game.chance_bits() {
                32 => {
                    node.cfvalues_chance_mut().copy_from_slice(result);
                }
                16 => {
                    let cfv_scale =
                        encode_signed_slice(node.cfvalues_chance_compressed_mut(), result);
                    node.set_cfvalue_chance_scale(cfv_scale);
                }
                8 => {
                    let seed =
                        (node.action_indices().start as u32).wrapping_shl(16) ^ (player as u32);
                    let cfv_scale = encode_signed_i8(node.cfvalues_chance_i8_mut(), result, seed);
                    node.set_cfvalue_chance_scale(cfv_scale);
                }
                4 => {
                    let seed =
                        (node.action_indices().start as u32).wrapping_shl(16) ^ (player as u32);
                    let cfv_scale =
                        encode_signed_i4_packed(node.cfvalues_chance_i4_packed_mut(), result, seed);
                    node.set_cfvalue_chance_scale(cfv_scale);
                }
                _ => panic!(
                    "Invalid chance_bits: {}. Valid values: 4, 8, 16, 32",
                    game.chance_bits()
                ),
            }
        }
    }
    // player node
    else if node.player() == player {
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                save_cfvalues,
            );
        });

        // obtain the strategy
        #[cfg(feature = "custom-alloc")]
        let mut strategy = match game.strategy_bits() {
            32 => normalized_strategy_custom_alloc(node.strategy(), num_actions),
            16 => {
                normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions)
            }
            8 => normalized_strategy_compressed_u8_custom_alloc(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed_custom_alloc(
                node.strategy_u4_packed(),
                num_actions,
                num_hands,
            ),
            _ => panic!(
                "Invalid strategy_bits: {}. Valid values: 8, 16, 32",
                game.strategy_bits()
            ),
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut strategy = match game.strategy_bits() {
            32 => normalized_strategy(node.strategy(), num_actions),
            16 => normalized_strategy_compressed(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed(
                node.strategy_u4_packed(),
                num_actions,
                num_hands,
            ),
            _ => panic!(
                "Invalid strategy_bits: {}. Valid values: 8, 16, 32",
                game.strategy_bits()
            ),
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        fma_slices_uninit(result, &strategy, &cfv_actions);

        // save the counterfactual values
        if save_cfvalues {
            // Use regret_bits for standard CFValues (which share storage with regrets)
            match game.regret_bits() {
                32 => {
                    node.cfvalues_mut().copy_from_slice(&cfv_actions);
                }
                16 => {
                    let cfv_scale =
                        encode_signed_slice(node.cfvalues_compressed_mut(), &cfv_actions);
                    node.set_cfvalue_scale(cfv_scale);
                }
                8 => {
                    let seed =
                        (node.action_indices().start as u32).wrapping_shl(16) ^ (player as u32);
                    let cfv_scale = encode_signed_i8(node.cfvalues_i8_mut(), &cfv_actions, seed);
                    node.set_cfvalue_scale(cfv_scale);
                }
                _ => panic!(
                    "Invalid regret_bits (for cfvalues): {}. Valid values: 8, 16, 32",
                    game.regret_bits()
                ),
            }
        }
    }
    // opponent node
    else if num_actions == 1 {
        // simply recurse when the number of actions is one
        compute_cfvalue_recursive(
            result,
            game,
            &mut node.play(0),
            player,
            cfreach,
            save_cfvalues,
        );
    } else {
        // obtain the strategy
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_actions = match game.strategy_bits() {
            32 => normalized_strategy_custom_alloc(node.strategy(), num_actions),
            16 => {
                normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions)
            }
            8 => normalized_strategy_compressed_u8_custom_alloc(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed_custom_alloc(
                node.strategy_u4_packed(),
                num_actions,
                num_hands,
            ),
            _ => panic!(
                "Invalid strategy_bits: {}. Valid values: 8, 16, 32",
                game.strategy_bits()
            ),
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_actions = match game.strategy_bits() {
            32 => normalized_strategy(node.strategy(), num_actions),
            16 => normalized_strategy_compressed(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed(
                node.strategy_u4_packed(),
                num_actions,
                num_hands,
            ),
            _ => panic!(
                "Invalid strategy_bits: {}. Valid values: 8, 16, 32",
                game.strategy_bits()
            ),
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
            compute_cfvalue_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                save_cfvalues,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }

    // save the counterfactual values for IP
    if save_cfvalues && node.has_cfvalues_ip() && player == 1 {
        let result = unsafe { &*(result as *const _ as *const [f32]) };
        match game.ip_bits() {
            32 => {
                node.cfvalues_ip_mut().copy_from_slice(result);
            }
            16 => {
                let cfv_scale = encode_signed_slice(node.cfvalues_ip_compressed_mut(), result);
                node.set_cfvalue_ip_scale(cfv_scale);
            }
            8 => {
                let seed = (node.action_indices().start as u32).wrapping_shl(16) ^ (player as u32);
                let cfv_scale = encode_signed_i8(node.cfvalues_ip_i8_mut(), result, seed);
                node.set_cfvalue_ip_scale(cfv_scale);
            }
            4 => {
                let seed = (node.action_indices().start as u32).wrapping_shl(16) ^ (player as u32);
                let cfv_scale =
                    encode_signed_i4_packed(node.cfvalues_ip_i4_packed_mut(), result, seed);
                node.set_cfvalue_ip_scale(cfv_scale);
            }
            _ => panic!(
                "Invalid ip_bits: {}. Valid values: 4, 8, 16, 32",
                game.ip_bits()
            ),
        }
    }
}

/// The recursive helper function for computing the counterfactual values of best response.
fn compute_best_cfv_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &T::Node,
    player: usize,
    cfreach: &[f32],
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = game.num_private_hands(player);

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &node.play(0);
        compute_best_cfv_recursive(result, game, child, player, cfreach);
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // chance node
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
            compute_best_cfv_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                &cfreach_updated,
            )
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
    // player node
    else if node.player() == player {
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                cfreach,
            )
        });

        let locking = game.locking_strategy(node);
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        if locking.is_empty() {
            // compute element-wise maximum (take the best response)
            max_slices_uninit(result, &cfv_actions);
        } else {
            // when the node is locked
            max_fma_slices_uninit(result, &cfv_actions, locking);
        }
    }
    // opponent node
    else {
        // obtain the strategy
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_actions = match game.strategy_bits() {
            32 => normalized_strategy_custom_alloc(node.strategy(), num_actions),
            16 => {
                normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions)
            }
            8 => normalized_strategy_compressed_u8_custom_alloc(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed_custom_alloc(
                node.strategy_u4_packed(),
                num_actions,
                num_hands,
            ),
            _ => panic!(
                "Invalid strategy_bits: {}. Valid values: 8, 16, 32",
                game.strategy_bits()
            ),
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_actions = match game.strategy_bits() {
            32 => normalized_strategy(node.strategy(), num_actions),
            16 => normalized_strategy_compressed(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed(
                node.strategy_u4_packed(),
                num_actions,
                num_hands,
            ),
            _ => panic!(
                "Invalid strategy_bits: {}. Valid values: 8, 16, 32",
                game.strategy_bits()
            ),
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
            compute_best_cfv_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

#[cfg(feature = "custom-alloc")]
#[inline]
pub(crate) fn normalized_strategy_custom_alloc(
    strategy: &[f32],
    num_actions: usize,
) -> Vec<f32, StackAlloc> {
    let mut normalized = Vec::with_capacity_in(strategy.len(), StackAlloc);
    let uninit = normalized.spare_capacity_mut();

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    uninit
        .chunks_exact_mut(row_size)
        .zip(strategy.chunks_exact(row_size))
        .for_each(|(n, s)| {
            div_slice_uninit(n, s, &denom, default);
        });

    unsafe { normalized.set_len(strategy.len()) };
    normalized
}

#[inline]
pub(crate) fn normalized_strategy(strategy: &[f32], num_actions: usize) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(strategy.len());
    let uninit = normalized.spare_capacity_mut();

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    uninit
        .chunks_exact_mut(row_size)
        .zip(strategy.chunks_exact(row_size))
        .for_each(|(n, s)| {
            div_slice_uninit(n, s, &denom, default);
        });

    unsafe { normalized.set_len(strategy.len()) };
    normalized
}

#[cfg(feature = "custom-alloc")]
#[inline]
pub(crate) fn normalized_strategy_compressed_custom_alloc(
    strategy: &[u16],
    num_actions: usize,
) -> Vec<f32, StackAlloc> {
    let mut normalized = Vec::with_capacity_in(strategy.len(), StackAlloc);
    let uninit = normalized.spare_capacity_mut();

    uninit.iter_mut().zip(strategy).for_each(|(n, s)| {
        n.write(*s as f32);
    });
    unsafe { normalized.set_len(strategy.len()) };

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    normalized
}

#[inline]
pub(crate) fn normalized_strategy_compressed(strategy: &[u16], num_actions: usize) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(strategy.len());
    let uninit = normalized.spare_capacity_mut();

    uninit.iter_mut().zip(strategy).for_each(|(n, s)| {
        n.write(*s as f32);
    });
    unsafe { normalized.set_len(strategy.len()) };

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    normalized
}

#[cfg(feature = "custom-alloc")]
#[inline]
pub(crate) fn normalized_strategy_compressed_u8_custom_alloc(
    strategy: &[u8],
    num_actions: usize,
) -> Vec<f32, StackAlloc> {
    let mut normalized = Vec::with_capacity_in(strategy.len(), StackAlloc);
    let uninit = normalized.spare_capacity_mut();

    uninit.iter_mut().zip(strategy).for_each(|(n, s)| {
        n.write(*s as f32);
    });
    unsafe { normalized.set_len(strategy.len()) };

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    normalized
}

#[cfg(feature = "custom-alloc")]
#[inline]
pub(crate) fn normalized_strategy_compressed_u4_packed_custom_alloc(
    strategy: &[u8],
    num_actions: usize,
    num_hands: usize,
) -> Vec<f32, StackAlloc> {
    let num_elements = num_actions * num_hands;
    let mut normalized = Vec::with_capacity_in(num_elements, StackAlloc);
    for i in 0..num_elements {
        let byte = strategy[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        normalized.push(nibble as f32);
    }

    let mut denom = Vec::with_capacity_in(num_hands, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    unsafe { denom.set_len(num_hands) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(num_hands).for_each(|row| {
        div_slice(row, &denom, default);
    });

    normalized
}

#[inline]
pub(crate) fn normalized_strategy_compressed_u8(strategy: &[u8], num_actions: usize) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(strategy.len());
    let uninit = normalized.spare_capacity_mut();

    uninit.iter_mut().zip(strategy).for_each(|(n, s)| {
        n.write(*s as f32);
    });
    unsafe { normalized.set_len(strategy.len()) };

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    normalized
}
#[inline]
pub(crate) fn normalized_strategy_compressed_u4_packed(
    strategy: &[u8],
    num_actions: usize,
    num_hands: usize,
) -> Vec<f32> {
    let num_elements = num_actions * num_hands;
    let mut normalized = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let byte = strategy[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        normalized.push(nibble as f32);
    }

    let mut denom = Vec::with_capacity(num_hands);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    unsafe { denom.set_len(num_hands) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(num_hands).for_each(|row| {
        div_slice(row, &denom, default);
    });

    normalized
}

#[inline]
pub(crate) fn apply_locking_strategy(dst: &mut [f32], locking: &[f32]) {
    if !locking.is_empty() {
        dst.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = *s;
            }
        });
    }
}

/// A checkpoint of the game state during solving, including metadata for resumption.
#[cfg(feature = "bincode")]
pub struct GameCheckpoint {
    /// The complete game state (tree, strategies, regrets, etc.)
    pub game: PostFlopGame,
    /// Number of iterations already completed
    pub current_iteration: u32,
    /// Optional: exploitability at checkpoint time (for monitoring)
    pub exploitability: Option<f32>,
}

#[cfg(feature = "bincode")]
impl bincode::Encode for GameCheckpoint {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        self.game.encode(encoder)?;
        self.current_iteration.encode(encoder)?;
        self.exploitability.encode(encoder)?;
        Ok(())
    }
}

#[cfg(feature = "bincode")]
impl<C> bincode::Decode<C> for GameCheckpoint {
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(GameCheckpoint {
            game: PostFlopGame::decode(decoder)?,
            current_iteration: u32::decode(decoder)?,
            exploitability: Option::<f32>::decode(decoder)?,
        })
    }
}

/// Saves a checkpoint of the game state to a file.
/// This is the correct way to save a game in progress for later resumption.
#[cfg(feature = "bincode")]
pub fn save_checkpoint(
    game: &PostFlopGame,
    current_iteration: u32,
    path: &str,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Manually encode each field using encode_into_std_write
    bincode::encode_into_std_write(game, &mut writer, bincode::config::standard())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    bincode::encode_into_std_write(&current_iteration, &mut writer, bincode::config::standard())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let exploitability: Option<f32> = None;
    bincode::encode_into_std_write(&exploitability, &mut writer, bincode::config::standard())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    Ok(())
}

/// Loads a checkpoint from a file, returning the game state and iteration count.
#[cfg(feature = "bincode")]
pub fn load_checkpoint(path: &str) -> std::io::Result<GameCheckpoint> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let checkpoint: GameCheckpoint =
        bincode::decode_from_std_read(&mut reader, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(checkpoint)
}

/// DEPRECATED: Saves the game tree to a file.
/// This function does not save iteration count and is unsuitable for resuming solving.
/// Use `save_checkpoint()` instead for proper resumption support.
#[cfg(feature = "bincode")]
#[deprecated(
    since = "0.1.0",
    note = "Use save_checkpoint() instead for proper resumption"
)]
pub fn save_gametree(game: &PostFlopGame, path: &str) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    bincode::encode_into_std_write(game, &mut writer, bincode::config::standard())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

/// DEPRECATED: Loads the game tree from a file.
/// This function does not restore iteration count and is unsuitable for resuming solving.
/// Use `load_checkpoint()` instead for proper resumption support.
#[cfg(feature = "bincode")]
#[deprecated(
    since = "0.1.0",
    note = "Use load_checkpoint() instead for proper resumption"
)]
pub fn load_gametree(path: &str) -> std::io::Result<PostFlopGame> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let game: PostFlopGame =
        bincode::decode_from_std_read(&mut reader, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(game)
}
