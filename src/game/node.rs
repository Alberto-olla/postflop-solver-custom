use super::*;
use crate::interface::*;
use std::ptr;
use std::slice;

impl GameNode for PostFlopNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        self.player & PLAYER_CHANCE_FLAG != 0
    }

    #[inline]
    fn cfvalue_storage_player(&self) -> Option<usize> {
        let prev_player = self.player & PLAYER_MASK;
        match prev_player {
            0 => Some(1),
            1 => Some(0),
            _ => None,
        }
    }

    #[inline]
    fn player(&self) -> usize {
        self.player as usize
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.num_children as usize
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<'_, Self> {
        self.children()[action].lock()
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage1 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn has_cfvalues_ip(&self) -> bool {
        self.num_elements_ip != 0
    }

    #[inline]
    fn cfvalues_ip(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage3 as *const f32, self.num_elements_ip as usize) }
    }

    #[inline]
    fn cfvalues_ip_mut(&mut self) -> &mut [f32] {
        unsafe {
            slice::from_raw_parts_mut(self.storage3 as *mut f32, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_chance(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage1 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_compressed(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.storage1 as *const u16, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_ip_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage3 as *const i16, self.num_elements_ip as usize) }
    }

    #[inline]
    fn cfvalues_ip_compressed_mut(&mut self) -> &mut [i16] {
        unsafe {
            slice::from_raw_parts_mut(self.storage3 as *mut i16, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_chance_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage1 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut i16, self.num_elements as usize) }
    }

    // 8-bit quantization methods
    #[inline]
    fn strategy_u8(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.storage1 as *const u8, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_u8_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u8, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_i8(&self) -> &[i8] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i8, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_i8_mut(&mut self) -> &mut [i8] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i8, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_u8(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.storage2 as *const u8, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_u8_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut u8, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_i8(&self) -> &[i8] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i8, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_i8_mut(&mut self) -> &mut [i8] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i8, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_ip_i8(&self) -> &[i8] {
        unsafe { slice::from_raw_parts(self.storage3 as *const i8, self.num_elements_ip as usize) }
    }

    #[inline]
    fn cfvalues_ip_i8_mut(&mut self) -> &mut [i8] {
        unsafe {
            slice::from_raw_parts_mut(self.storage3 as *mut i8, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_ip_i4_packed(&self) -> &[u8] {
        let len = (self.num_elements_ip as usize + 1) / 2;
        unsafe { slice::from_raw_parts(self.storage3 as *const u8, len) }
    }

    #[inline]
    fn cfvalues_ip_i4_packed_mut(&mut self) -> &mut [u8] {
        let len = (self.num_elements_ip as usize + 1) / 2;
        unsafe { slice::from_raw_parts_mut(self.storage3 as *mut u8, len) }
    }

    #[inline]
    fn cfvalues_chance_i8(&self) -> &[i8] {
        unsafe { slice::from_raw_parts(self.storage1 as *const i8, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_i8_mut(&mut self) -> &mut [i8] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut i8, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_i4_packed(&self) -> &[u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts(self.storage1 as *const u8, len) }
    }

    #[inline]
    fn cfvalues_chance_i4_packed_mut(&mut self) -> &mut [u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u8, len) }
    }

    #[inline]
    fn strategy_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_strategy_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn regret_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_regret_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_cfvalue_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_ip_scale(&self) -> f32 {
        self.scale3
    }

    #[inline]
    fn set_cfvalue_ip_scale(&mut self, scale: f32) {
        self.scale3 = scale;
    }

    #[inline]
    fn cfvalue_chance_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_cfvalue_chance_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn prev_regret_scale(&self) -> f32 {
        self.scale4
    }

    #[inline]
    fn set_prev_regret_scale(&mut self, scale: f32) {
        self.scale4 = scale;
    }

    #[inline]
    fn prev_regrets(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage4 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn prev_regrets_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage4 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_and_prev_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        unsafe {
            (
                slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize),
                slice::from_raw_parts_mut(self.storage4 as *mut f32, self.num_elements as usize),
            )
        }
    }

    #[inline]
    fn prev_regrets_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage4 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn prev_regrets_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage4 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_and_prev_compressed_mut(&mut self) -> (&mut [i16], &mut [i16]) {
        unsafe {
            (
                slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize),
                slice::from_raw_parts_mut(self.storage4 as *mut i16, self.num_elements as usize),
            )
        }
    }

    #[inline]
    fn prev_regrets_i8(&self) -> &[i8] {
        unsafe { slice::from_raw_parts(self.storage4 as *const i8, self.num_elements as usize) }
    }

    #[inline]
    fn prev_regrets_i8_mut(&mut self) -> &mut [i8] {
        unsafe { slice::from_raw_parts_mut(self.storage4 as *mut i8, self.num_elements as usize) }
    }

    #[inline]
    fn prev_regrets_u8(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.storage4 as *const u8, self.num_elements as usize) }
    }

    #[inline]
    fn prev_regrets_u8_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.storage4 as *mut u8, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_and_prev_i8_mut(&mut self) -> (&mut [i8], &mut [i8]) {
        unsafe {
            (
                slice::from_raw_parts_mut(self.storage2 as *mut i8, self.num_elements as usize),
                slice::from_raw_parts_mut(self.storage4 as *mut i8, self.num_elements as usize),
            )
        }
    }

    #[inline]
    fn regrets_and_prev_u8_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        unsafe {
            (
                slice::from_raw_parts_mut(self.storage2 as *mut u8, self.num_elements as usize),
                slice::from_raw_parts_mut(self.storage4 as *mut u8, self.num_elements as usize),
            )
        }
    }

    #[inline]
    fn enable_parallelization(&self) -> bool {
        self.river == NOT_DEALT
    }

    // 4-bit quantization methods
    #[inline]
    fn regrets_i4_packed(&self) -> &[u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts(self.storage2 as *const u8, len) }
    }

    #[inline]
    fn regrets_i4_packed_mut(&mut self) -> &mut [u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut u8, len) }
    }

    #[inline]
    fn regrets_u4_packed(&self) -> &[u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts(self.storage2 as *const u8, len) }
    }

    #[inline]
    fn regrets_u4_packed_mut(&mut self) -> &mut [u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut u8, len) }
    }

    #[inline]
    fn prev_regrets_i4_packed(&self) -> &[u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts(self.storage4 as *const u8, len) }
    }

    #[inline]
    fn prev_regrets_i4_packed_mut(&mut self) -> &mut [u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts_mut(self.storage4 as *mut u8, len) }
    }

    #[inline]
    fn prev_regrets_u4_packed(&self) -> &[u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts(self.storage4 as *const u8, len) }
    }

    #[inline]
    fn prev_regrets_u4_packed_mut(&mut self) -> &mut [u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts_mut(self.storage4 as *mut u8, len) }
    }

    #[inline]
    fn regrets_and_prev_i4_packed_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe {
            (
                slice::from_raw_parts_mut(self.storage2 as *mut u8, len),
                slice::from_raw_parts_mut(self.storage4 as *mut u8, len),
            )
        }
    }

    #[inline]
    fn regrets_and_prev_u4_packed_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe {
            (
                slice::from_raw_parts_mut(self.storage2 as *mut u8, len),
                slice::from_raw_parts_mut(self.storage4 as *mut u8, len),
            )
        }
    }

    #[inline]
    fn strategy_u4_packed(&self) -> &[u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts(self.storage1 as *const u8, len) }
    }

    #[inline]
    fn strategy_u4_packed_mut(&mut self) -> &mut [u8] {
        let len = (self.num_elements as usize + 1) / 2;
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u8, len) }
    }
}

impl Default for PostFlopNode {
    #[inline]
    fn default() -> Self {
        Self {
            prev_action: Action::None,
            player: PLAYER_OOP,
            turn: NOT_DEALT,
            river: NOT_DEALT,
            is_locked: false,
            amount: 0,
            children_offset: 0,
            num_children: 0,
            num_elements_ip: 0,
            num_elements: 0,
            scale1: 0.0,
            scale2: 0.0,
            scale3: 0.0,
            scale4: 0.0,
            storage1: ptr::null_mut(),
            storage2: ptr::null_mut(),
            storage3: ptr::null_mut(),
            storage4: ptr::null_mut(),
        }
    }
}

impl PostFlopNode {
    #[inline]
    pub(super) fn children(&self) -> &[MutexLike<Self>] {
        // This is safe because `MutexLike<T>` is a `repr(transparent)` wrapper around `T`.
        let self_ptr = self as *const _ as *const MutexLike<PostFlopNode>;
        unsafe {
            slice::from_raw_parts(
                self_ptr.add(self.children_offset as usize),
                self.num_children as usize,
            )
        }
    }
}
