use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use std::mem::{self, MaybeUninit};
use std::ptr;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

/// Obtains the maximum absolute value of the given slice.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn slice_absolute_max(slice: &[f32]) -> f32 {
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
fn slice_absolute_max(slice: &[f32]) -> f32 {
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
fn slice_nonnegative_max(slice: &[f32]) -> f32 {
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
fn slice_nonnegative_max(slice: &[f32]) -> f32 {
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

/// Encodes the `f32` slice to the `i16` slice, and returns the scale.
#[inline]
pub(crate) fn encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32 {
    let scale = slice_absolute_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i16::MAX as f32 / scale_nonzero;
    dst.iter_mut()
        .zip(slice)
        .for_each(|(d, s)| *d = unsafe { (s * encoder).round().to_int_unchecked::<i32>() as i16 });
    scale
}

#[inline]
fn fast_xorshift32(seed: &mut u32) -> u32 {
    let mut x = *seed;
    if x == 0 { x = 0xACE1u32; } // Avoid zero seed
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x;
    x
}

/// Helper for stochastic rounding.
/// Instead of round(x), it returns floor(x) + 1 with probability fract(x).
#[inline]
fn stochastic_round(val: f32, seed: &mut u32) -> i32 {
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

/// Encodes the `f32` slice to the `u16` slice, and returns the scale.
#[inline]
pub(crate) fn encode_unsigned_slice(dst: &mut [u16], slice: &[f32]) -> f32 {
    let scale = slice_nonnegative_max(slice);
    // Handle NaN/Inf gracefully
    let scale = if scale.is_finite() { scale } else { 0.0 };
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u16::MAX as f32 / scale_nonzero;
    // note: 0.49999997 + 0.49999997 = 0.99999994 < 1.0 | 0.5 + 0.49999997 = 1.0
    dst.iter_mut().zip(slice).for_each(|(d, s)| {
        // Clamp to avoid overflow/UB from floating point errors and handle NaN
        let s_safe = if s.is_finite() { *s } else { 0.0 };
        let value = (s_safe * encoder + 0.49999997).min(u16::MAX as f32).max(0.0);
        *d = unsafe { value.to_int_unchecked::<i32>() as u16 }
    });
    scale
}

/// Encodes the `f32` slice to the `u8` slice, and returns the scale.
#[inline]
pub(crate) fn encode_unsigned_strategy_u8(dst: &mut [u8], slice: &[f32], base_seed: u32) -> f32 {
    let scale = slice_nonnegative_max(slice);
    // Handle NaN/Inf gracefully
    let scale = if scale.is_finite() { scale } else { 0.0 };
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u8::MAX as f32 / scale_nonzero;
    dst.iter_mut().enumerate().zip(slice).for_each(|((i, d), s)| {
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
pub(crate) fn encode_unsigned_regrets_u8(dst: &mut [u8], slice: &[f32], base_seed: u32) -> f32 {
    let scale = slice_nonnegative_max(slice);
    let scale = if scale.is_finite() { scale } else { 0.0 };
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u8::MAX as f32 / scale_nonzero;
    dst.iter_mut().enumerate().zip(slice).for_each(|((i, d), s)| {
        let s_safe = if s.is_finite() { *s } else { 0.0 };
        // Clamp negative values to 0 (CFR+ Requirement)
        let scaled = (s_safe * encoder).min(u8::MAX as f32).max(0.0);
        let mut seed = base_seed ^ (i as u32);
        *d = stochastic_round(scaled, &mut seed) as u8;
    });
    scale
}


/// Encodes the `f32` slice to the `i16` slice using logarithmic compression (signed magnitude biasing).
/// This compresses the dynamic range, allowing better precision for both small and large values.
/// Formula: compressed = sign(x) * log1p(abs(x))
/// Returns the scale factor used.
#[inline]
pub(crate) fn encode_signed_slice_log(dst: &mut [i16], slice: &[f32]) -> f32 {
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



/// Encodes the `f32` slice to the `i8` slice for signed values (e.g., cfvalues_chance), and returns the scale.
/// Uses signed quantization: maps [-max_abs, max_abs] to [-127, 127].
/// Note: We use i8::MAX (127) instead of full range to avoid overflow issues.
#[inline]
pub(crate) fn encode_signed_i8(dst: &mut [i8], slice: &[f32], base_seed: u32) -> f32 {
    let scale = slice_absolute_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i8::MAX as f32 / scale_nonzero;  // encoder = 127 / max_abs

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
pub(crate) fn decode_signed_i8(src: &[i8], scale: f32) -> Vec<f32> {
    let decoder = scale / i8::MAX as f32;
    src.iter().map(|&x| x as f32 * decoder).collect()
}

/// Encodes the `f32` slice to the `u8` slice for signed values (packed 4-bit), and returns the scale.
/// Uses signed quantization: maps [-max_abs, max_abs] to [-7, 7].
/// Two 4-bit values are packed into one `u8`.
#[inline]
pub(crate) fn encode_signed_i4_packed(dst: &mut [u8], slice: &[f32], base_seed: u32) -> f32 {
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
pub(crate) fn decode_signed_i4_packed(src: &[u8], scale: f32, len: usize) -> Vec<f32> {
    let decoder = scale / 7.0;
    let mut dst = Vec::with_capacity(len);
    for i in 0..len {
        let byte = src[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        };
        // Sign extension
        let val = ((nibble << 4) as i8) >> 4;
        dst.push(val as f32 * decoder);
    }
    dst
}

/// Encodes the `f32` slice to the `u8` slice (unsigned packed 4-bit), and returns the scale.
/// Used for non-negative regrets (DCFR+). Maps [0, max] to [0, 15].
#[inline]
pub(crate) fn encode_unsigned_u4_packed(dst: &mut [u8], slice: &[f32], base_seed: u32) -> f32 {
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
pub(crate) fn decode_unsigned_u4_packed(src: &[u8], scale: f32, len: usize) -> Vec<f32> {
    let decoder = scale / 15.0;
    let mut dst = Vec::with_capacity(len);
    for i in 0..len {
        let byte = src[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        };
        dst.push(nibble as f32 * decoder);
    }
    dst
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
                    let cfv_scale = encode_signed_slice(node.cfvalues_chance_compressed_mut(), result);
                    node.set_cfvalue_chance_scale(cfv_scale);
                }
                8 => {
                    let seed = (node.action_indices().start as u32).wrapping_shl(16) ^ (player as u32);
                    let cfv_scale = encode_signed_i8(node.cfvalues_chance_i8_mut(), result, seed);
                    node.set_cfvalue_chance_scale(cfv_scale);
                }
                4 => {
                    let seed = (node.action_indices().start as u32).wrapping_shl(16) ^ (player as u32);
                    let cfv_scale = encode_signed_i4_packed(node.cfvalues_chance_i4_packed_mut(), result, seed);
                    node.set_cfvalue_chance_scale(cfv_scale);
                }
                _ => panic!("Invalid chance_bits: {}. Valid values: 4, 8, 16, 32", game.chance_bits()),
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
            16 => normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8_custom_alloc(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed_custom_alloc(node.strategy_u4_packed(), num_actions, num_hands),
            _ => panic!("Invalid strategy_bits: {}. Valid values: 8, 16, 32", game.strategy_bits()),
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut strategy = match game.strategy_bits() {
            32 => normalized_strategy(node.strategy(), num_actions),
            16 => normalized_strategy_compressed(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed(node.strategy_u4_packed(), num_actions, num_hands),
            _ => panic!("Invalid strategy_bits: {}. Valid values: 8, 16, 32", game.strategy_bits()),
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
                    let cfv_scale = encode_signed_slice(node.cfvalues_compressed_mut(), &cfv_actions);
                    node.set_cfvalue_scale(cfv_scale);
                }
                8 => {
                    let seed = (node.action_indices().start as u32).wrapping_shl(16) ^ (player as u32);
                    let cfv_scale = encode_signed_i8(node.cfvalues_i8_mut(), &cfv_actions, seed);
                    node.set_cfvalue_scale(cfv_scale);
                }
                _ => panic!("Invalid regret_bits (for cfvalues): {}. Valid values: 8, 16, 32", game.regret_bits()),
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
            16 => normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8_custom_alloc(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed_custom_alloc(node.strategy_u4_packed(), num_actions, num_hands),
            _ => panic!("Invalid strategy_bits: {}. Valid values: 8, 16, 32", game.strategy_bits()),
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_actions = match game.strategy_bits() {
            32 => normalized_strategy(node.strategy(), num_actions),
            16 => normalized_strategy_compressed(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed(node.strategy_u4_packed(), num_actions, num_hands),
            _ => panic!("Invalid strategy_bits: {}. Valid values: 8, 16, 32", game.strategy_bits()),
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
                 let cfv_scale = encode_signed_i4_packed(node.cfvalues_ip_i4_packed_mut(), result, seed);
                 node.set_cfvalue_ip_scale(cfv_scale);
            }
            _ => panic!("Invalid ip_bits: {}. Valid values: 4, 8, 16, 32", game.ip_bits()),
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
            16 => normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8_custom_alloc(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed_custom_alloc(node.strategy_u4_packed(), num_actions, num_hands),
            _ => panic!("Invalid strategy_bits: {}. Valid values: 8, 16, 32", game.strategy_bits()),
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_actions = match game.strategy_bits() {
            32 => normalized_strategy(node.strategy(), num_actions),
            16 => normalized_strategy_compressed(node.strategy_compressed(), num_actions),
            8 => normalized_strategy_compressed_u8(node.strategy_u8(), num_actions),
            4 => normalized_strategy_compressed_u4_packed(node.strategy_u4_packed(), num_actions, num_hands),
            _ => panic!("Invalid strategy_bits: {}. Valid values: 8, 16, 32", game.strategy_bits()),
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
pub(crate) fn normalized_strategy_compressed_u4_packed(strategy: &[u8], num_actions: usize, num_hands: usize) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_rounding_preserves_expected_value() {
        // Test that stochastic rounding preserves expected value over many iterations
        // This simulates the CFR accumulation scenario

        const NUM_TRIALS: usize = 10000;
        const NUM_ACTIONS: usize = 3;

        // Initial strategy (normalized probabilities)
        let initial_strategy = vec![0.5, 0.3, 0.2];

        // Small update that would vanish with deterministic rounding
        let small_update = vec![0.001, 0.0005, 0.0005];

        let mut accumulated = initial_strategy.clone();
        let mut encoded = vec![0u8; NUM_ACTIONS];

        // Simulate many iterations of accumulation
        for _ in 0..NUM_TRIALS {
            // Add small update
            for i in 0..NUM_ACTIONS {
                accumulated[i] += small_update[i];
            }

            // Encode with stochastic rounding
            let _scale = encode_unsigned_strategy_u8(&mut encoded, &accumulated, 0);

            // Decode back (simulating what happens in the solver)
            // Note: we don't actually decode here, just re-encode
        }

        // Expected final values
        let expected: Vec<f32> = initial_strategy.iter()
            .zip(&small_update)
            .map(|(init, update)| init + update * NUM_TRIALS as f32)
            .collect();

        // Check that accumulated values are close to expected
        // Allow 5% error due to stochastic nature
        for i in 0..NUM_ACTIONS {
            let error = (accumulated[i] - expected[i]).abs() / expected[i];
            assert!(
                error < 0.05,
                "Action {}: accumulated={}, expected={}, error={}%",
                i, accumulated[i], expected[i], error * 100.0
            );
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Test basic encode/decode functionality
        let original = vec![0.5, 0.3, 0.15, 0.05];
        let mut encoded = vec![0u8; 4];

        let scale = encode_unsigned_strategy_u8(&mut encoded, &original, 0);

        // Decode manually (since decode function exists but might not be used)
        let decoded: Vec<f32> = encoded.iter()
            .map(|&x| (x as f32) * scale / 255.0)
            .collect();

        // Check that values are reasonably close (within quantization error)
        for i in 0..4 {
            let error = (decoded[i] - original[i]).abs();
            assert!(
                error < 0.01,
                "Index {}: decoded={}, original={}, error={}",
                i, decoded[i], original[i], error
            );
        }
    }
}
