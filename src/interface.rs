use crate::mutex_like::*;
use std::mem::MaybeUninit;
use std::ops::Range;

/// The trait representing a game.
pub trait Game: Send + Sync {
    /// The type representing a node in game tree.
    #[doc(hidden)]
    type Node: GameNode;

    /// Returns the root node of game tree.
    #[doc(hidden)]
    fn root(&self) -> MutexGuardLike<'_, Self::Node>;

    /// Returns the number of private hands of given player.
    #[doc(hidden)]
    fn num_private_hands(&self, player: usize) -> usize;

    /// Returns the initial reach probabilities of given player.
    #[doc(hidden)]
    fn initial_weights(&self, player: usize) -> &[f32];

    /// Computes the counterfactual values of given node.
    #[doc(hidden)]
    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    );

    /// Returns the effective number of chances.
    #[doc(hidden)]
    fn chance_factor(&self, node: &Self::Node) -> usize;

    /// Returns whether the instance is solved.
    #[doc(hidden)]
    fn is_solved(&self) -> bool;

    /// Sets the instance to be solved.
    #[doc(hidden)]
    fn set_solved(&mut self);

    /// Returns whether the instance is ready to be solved.
    #[doc(hidden)]
    fn is_ready(&self) -> bool {
        true
    }

    /// Returns whether the game is raked.
    #[doc(hidden)]
    fn is_raked(&self) -> bool {
        false
    }

    /// Returns the list of indices that isomorphic chances refer to.
    #[doc(hidden)]
    fn isomorphic_chances(&self, _node: &Self::Node) -> &[u8] {
        &[]
    }

    /// Returns the swap list of the given isomorphic chance.
    #[doc(hidden)]
    fn isomorphic_swap(&self, _node: &Self::Node, _index: usize) -> &[Vec<(u16, u16)>; 2] {
        unreachable!()
    }

    /// Returns the locking strategy.
    #[doc(hidden)]
    fn locking_strategy(&self, _node: &Self::Node) -> &[f32] {
        &[]
    }

    /// Returns whether the compression is enabled.
    #[doc(hidden)]
    fn is_compression_enabled(&self) -> bool {
        false
    }

    /// Returns the quantization mode.
    #[doc(hidden)]
    fn quantization_mode(&self) -> crate::quantization::QuantizationMode {
        crate::quantization::QuantizationMode::Float32
    }

    /// Returns whether lazy normalization is enabled.
    #[doc(hidden)]
    fn is_lazy_normalization_enabled(&self) -> bool {
        false
    }

    /// Returns the lazy normalization frequency (0 = normalize only at finalization).
    #[doc(hidden)]
    fn lazy_normalization_freq(&self) -> u32 {
        0
    }

    /// Returns whether logarithmic encoding (signed magnitude biasing) is enabled for regrets.
    #[doc(hidden)]
    fn is_log_encoding_enabled(&self) -> bool {
        false
    }

    /// Returns the strategy precision in bits (16, 8, or 4).
    #[doc(hidden)]
    fn strategy_bits(&self) -> u8 {
        16  // Default: same as quantization mode
    }

    /// Returns the chance cfvalues precision in bits (16 or 8).
    #[doc(hidden)]
    fn chance_bits(&self) -> u8 {
        16  // Default: same as quantization mode
    }

    /// Returns the regret precision in bits (32 or 16).
    #[doc(hidden)]
    fn regret_bits(&self) -> u8 {
        match self.quantization_mode() {
            crate::quantization::QuantizationMode::Float32 => 32,
            _ => 16,
        }
    }

    /// Returns the IP cfvalues precision in bits (32, 16, or 8).
    #[doc(hidden)]
    fn ip_bits(&self) -> u8 {
        32
    }

    /// Returns the CFR algorithm variant.
    #[doc(hidden)]
    fn cfr_algorithm(&self) -> crate::solver::CfrAlgorithm {
        crate::solver::CfrAlgorithm::DCFR
    }

    /// Returns whether regret-based pruning is enabled.
    #[doc(hidden)]
    fn enable_pruning(&self) -> bool {
        false
    }

    /// Returns the tree configuration (for pruning Delta calculation).
    #[doc(hidden)]
    fn tree_config(&self) -> &crate::action_tree::TreeConfig {
        unreachable!()
    }

    /// Returns the current memory usage in megabytes.
    ///
    /// This includes all storage arrays (strategy, regrets, cfvalues).
    /// Default implementation returns 0.0 for compatibility.
    #[doc(hidden)]
    fn memory_usage_mb(&self) -> f64 {
        0.0
    }

    /// Returns the detailed memory usage for each component in bytes.
    #[doc(hidden)]
    fn memory_usage_detailed(&self) -> MemoryUsage {
        MemoryUsage::default()
    }
}

/// A struct representing the memory usage of a game tree.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoryUsage {
    /// Memory used for strategy storage (storage1) in bytes.
    pub strategy: u64,
    /// Memory used for regret storage (storage2) in bytes.
    pub regrets: u64,
    /// Memory used for IP cfvalues storage (storage_ip) in bytes.
    pub ip_cfvalues: u64,
    /// Memory used for chance cfvalues storage (storage_chance) in bytes.
    pub chance_cfvalues: u64,
    /// Memory used for extra predictive storage (storage4) in bytes.
    pub storage4: u64,
    /// Miscellaneous memory (node arena, isomorphism info, etc.) in bytes.
    pub misc: u64,
}

impl MemoryUsage {
    /// Returns the total memory usage in bytes.
    pub fn total(&self) -> u64 {
        self.strategy + self.regrets + self.ip_cfvalues + self.chance_cfvalues + self.storage4 + self.misc
    }

    /// Returns the total memory usage in megabytes.
    pub fn total_mb(&self) -> f64 {
        self.total() as f64 / 1_048_576.0
    }
}

/// The trait representing a node in game tree.
pub trait GameNode: Send + Sync {
    /// Returns whether the node is terminal.
    #[doc(hidden)]
    fn is_terminal(&self) -> bool;

    /// Returns whether the node is chance.
    #[doc(hidden)]
    fn is_chance(&self) -> bool;

    /// Returns the current player.
    #[doc(hidden)]
    fn player(&self) -> usize;

    /// Returns the number of actions.
    #[doc(hidden)]
    fn num_actions(&self) -> usize;

    /// Returns the node after taking the given action.
    #[doc(hidden)]
    fn play(&self, action: usize) -> MutexGuardLike<'_, Self>;

    /// Returns the strategy.
    #[doc(hidden)]
    fn strategy(&self) -> &[f32];

    /// Returns the mutable reference to the strategy.
    #[doc(hidden)]
    fn strategy_mut(&mut self) -> &mut [f32];

    /// Returns the cumulative regrets.
    #[doc(hidden)]
    fn regrets(&self) -> &[f32];

    /// Returns the mutable reference to the cumulative regrets.
    #[doc(hidden)]
    fn regrets_mut(&mut self) -> &mut [f32];

    /// Returns the counterfactual values.
    #[doc(hidden)]
    fn cfvalues(&self) -> &[f32];

    /// Returns the mutable reference to the counterfactual values.
    #[doc(hidden)]
    fn cfvalues_mut(&mut self) -> &mut [f32];

    /// Returns whether IP's counterfactual values are stored.
    #[doc(hidden)]
    fn has_cfvalues_ip(&self) -> bool {
        false
    }

    /// Returns IP's counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip(&self) -> &[f32] {
        unreachable!()
    }

    /// Returns the mutable reference to IP's counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_mut(&mut self) -> &mut [f32] {
        unreachable!()
    }

    /// Returns the player whose counterfactual values are stored (for chance node).
    #[doc(hidden)]
    fn cfvalue_storage_player(&self) -> Option<usize> {
        None
    }

    /// Returns the buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance(&self) -> &[f32] {
        unreachable!()
    }

    /// Returns the mutable reference to the buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_mut(&mut self) -> &mut [f32] {
        unreachable!()
    }

    /// Returns the [`Range`] struct of actions.
    #[doc(hidden)]
    fn action_indices(&self) -> Range<usize> {
        0..self.num_actions()
    }

    /// Returns the compressed strategy.
    #[doc(hidden)]
    fn strategy_compressed(&self) -> &[u16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed strategy.
    #[doc(hidden)]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unreachable!()
    }

    /// Returns the compressed cumulative regrets.
    #[doc(hidden)]
    fn regrets_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed cumulative regrets.
    #[doc(hidden)]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns IP's compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to IP's compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns IP's 4-bit packed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_i4_packed(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to IP's 4-bit packed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_i4_packed_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the 4-bit packed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_i4_packed(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 4-bit packed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_i4_packed_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    // 8-bit quantization methods
    /// Returns the 8-bit quantized strategy.
    #[doc(hidden)]
    fn strategy_u8(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 8-bit quantized strategy.
    #[doc(hidden)]
    fn strategy_u8_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns the 4-bit packed strategy.
    #[doc(hidden)]
    fn strategy_u4_packed(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 4-bit packed strategy.
    #[doc(hidden)]
    fn strategy_u4_packed_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns the 8-bit quantized cumulative regrets.
    #[doc(hidden)]
    fn regrets_i8(&self) -> &[i8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 8-bit quantized cumulative regrets.
    #[doc(hidden)]
    fn regrets_i8_mut(&mut self) -> &mut [i8] {
        unreachable!()
    }

    /// Returns the 8-bit quantized cumulative regrets as unsigned (for CFR+ w/ non-negative regrets).
    #[doc(hidden)]
    fn regrets_u8(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 8-bit quantized cumulative regrets as unsigned.
    #[doc(hidden)]
    fn regrets_u8_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns the 8-bit quantized counterfactual values.
    #[doc(hidden)]
    fn cfvalues_i8(&self) -> &[i8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 8-bit quantized counterfactual values.
    #[doc(hidden)]
    fn cfvalues_i8_mut(&mut self) -> &mut [i8] {
        unreachable!()
    }

    /// Returns IP's 8-bit quantized counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_i8(&self) -> &[i8] {
        unreachable!()
    }

    /// Returns the mutable reference to IP's 8-bit quantized counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_i8_mut(&mut self) -> &mut [i8] {
        unreachable!()
    }

    /// Returns the 8-bit quantized buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_i8(&self) -> &[i8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 8-bit quantized buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_i8_mut(&mut self) -> &mut [i8] {
        unreachable!()
    }

    /// Returns the scale of the compressed strategy.
    #[doc(hidden)]
    fn strategy_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed strategy.
    #[doc(hidden)]
    fn set_strategy_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed cumulative regrets.
    #[doc(hidden)]
    fn regret_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed cumulative regrets.
    #[doc(hidden)]
    fn set_regret_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalue_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed counterfactual values.
    #[doc(hidden)]
    fn set_cfvalue_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed counterfactual values for IP.
    #[doc(hidden)]
    fn cfvalue_ip_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed counterfactual values for IP.
    #[doc(hidden)]
    fn set_cfvalue_ip_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalue_chance_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn set_cfvalue_chance_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Hint for parallelization. By default, it is set to `false`.
    #[doc(hidden)]
    fn enable_parallelization(&self) -> bool {
        false
    }

    /// Returns the number of elements stored in this node.
    #[doc(hidden)]
    fn num_elements(&self) -> usize {
        self.strategy().len()
    }

    /// Returns the scale of the compressed previous instantaneous regrets.
    #[doc(hidden)]
    fn prev_regret_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed previous instantaneous regrets.
    #[doc(hidden)]
    fn set_prev_regret_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the previous instantaneous regrets (uncompressed).
    #[doc(hidden)]
    fn prev_regrets(&self) -> &[f32] {
        unreachable!()
    }

    /// Returns the mutable reference to the previous instantaneous regrets (uncompressed).
    #[doc(hidden)]
    fn prev_regrets_mut(&mut self) -> &mut [f32] {
        unreachable!()
    }
    
    /// Returns mutable references to both regrets and previous regrets (uncompressed).
    /// Used to avoid borrow checker issues when accessing both simultaneously.
    #[doc(hidden)]
    fn regrets_and_prev_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        unreachable!()
    }

    /// Returns the compressed previous instantaneous regrets.
    #[doc(hidden)]
    fn prev_regrets_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed previous instantaneous regrets.
    #[doc(hidden)]
    fn prev_regrets_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }
    
    /// Returns mutable references to both regrets and previous regrets (compressed).
    /// Used to avoid borrow checker issues when accessing both simultaneously.
    #[doc(hidden)]
    fn regrets_and_prev_compressed_mut(&mut self) -> (&mut [i16], &mut [i16]) {
        unreachable!()
    }
    
    /// Returns the 8-bit quantized previous instantaneous regrets.
    #[doc(hidden)]
    fn prev_regrets_i8(&self) -> &[i8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 8-bit quantized previous instantaneous regrets.
    #[doc(hidden)]
    fn prev_regrets_i8_mut(&mut self) -> &mut [i8] {
        unreachable!()
    }

    /// Returns the 8-bit quantized previous instantaneous regrets as unsigned.
    #[doc(hidden)]
    fn prev_regrets_u8(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 8-bit quantized previous instantaneous regrets as unsigned.
    #[doc(hidden)]
    fn prev_regrets_u8_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns mutable references to both regrets and previous regrets (8-bit quantized).
    /// Used to avoid borrow checker issues when accessing both simultaneously.
    #[doc(hidden)]
    fn regrets_and_prev_i8_mut(&mut self) -> (&mut [i8], &mut [i8]) {
        unreachable!()
    }

    /// Returns mutable references to both regrets and previous regrets (8-bit quantized unsigned).
    #[doc(hidden)]
    fn regrets_and_prev_u8_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        unreachable!()
    }

    // 4-bit quantization methods
    /// Returns the 4-bit packed cumulative regrets (signed).
    #[doc(hidden)]
    fn regrets_i4_packed(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 4-bit packed cumulative regrets (signed).
    #[doc(hidden)]
    fn regrets_i4_packed_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns the 4-bit packed cumulative regrets (unsigned).
    #[doc(hidden)]
    fn regrets_u4_packed(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 4-bit packed cumulative regrets (unsigned).
    #[doc(hidden)]
    fn regrets_u4_packed_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns the 4-bit packed previous instantaneous regrets (signed).
    #[doc(hidden)]
    fn prev_regrets_i4_packed(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 4-bit packed previous instantaneous regrets (signed).
    #[doc(hidden)]
    fn prev_regrets_i4_packed_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns the 4-bit packed previous instantaneous regrets (unsigned).
    #[doc(hidden)]
    fn prev_regrets_u4_packed(&self) -> &[u8] {
        unreachable!()
    }

    /// Returns the mutable reference to the 4-bit packed previous instantaneous regrets (unsigned).
    #[doc(hidden)]
    fn prev_regrets_u4_packed_mut(&mut self) -> &mut [u8] {
        unreachable!()
    }

    /// Returns mutable references to both regrets and previous regrets (4-bit packed signed).
    #[doc(hidden)]
    fn regrets_and_prev_i4_packed_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        unreachable!()
    }

    /// Returns mutable references to both regrets and previous regrets (4-bit packed unsigned).
    #[doc(hidden)]
    fn regrets_and_prev_u4_packed_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        unreachable!()
    }
}
