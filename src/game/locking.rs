use super::*;

/// Strategy locking logic for PostFlopGame
impl PostFlopGame {
    /// Returns the locking strategy if the node is locked, otherwise returns an empty slice.
    #[inline]
    pub(crate) fn locking_strategy_for_node(&self, node: &PostFlopNode) -> &[f32] {
        if !node.get_is_locked() {
            &[]
        } else {
            let index = self.node_index(node);
            self.locking_strategy.get(&index).unwrap()
        }
    }
}
