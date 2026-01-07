#[cfg(feature = "bincode")]
use super::*;
use crate::interface::*;
use crate::quantization::*;
use crate::utility::*;
use std::cell::Cell;
use std::ptr;

use bincode::{
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
};

static VERSION_STR: &str = "2026-01-04-mixed-precision-fixed";

impl PostFlopGame {
    /// Returns the storage mode of this instance.
    #[inline]
    pub fn storage_mode(&self) -> BoardState {
        self.storage_mode
    }

    /// Returns the target storage mode, which is used for serialization.
    #[inline]
    pub fn target_storage_mode(&self) -> BoardState {
        self.target_storage_mode
    }

    /// Sets the target storage mode.
    #[inline]
    pub fn set_target_storage_mode(&mut self, mode: BoardState) -> Result<(), String> {
        if mode > self.storage_mode {
            return Err("Cannot set target to a higher value than the current storage".to_string());
        }
        if mode < self.tree_config.initial_state {
            return Err("Cannot set target to a lower value than the initial state".to_string());
        }
        self.target_storage_mode = mode;
        Ok(())
    }

    /// Returns the memory usage when the target storage mode is used for serialization.
    #[inline]
    pub fn target_memory_usage(&self) -> u64 {
        match self.target_storage_mode {
            BoardState::River => match self.quantization_mode.is_compressed() {
                false => self.memory_usage().0,
                true => self.memory_usage().1,
            },
            _ => {
                let num_target_storage = self.num_target_storage();
                num_target_storage.iter().map(|&x| x as u64).sum::<u64>() + self.misc_memory_usage
            }
        }
    }

    /// Returns the number of storage elements required for the target storage mode.
    /// Returns [storage1, storage2, storage_ip, storage_chance, storage4]
    fn num_target_storage(&self) -> [usize; 5] {
        if self.state <= State::TreeBuilt {
            return [0; 5];
        }

        // Determine bytes per element for each storage based on *_bits settings
        let strategy_bytes = match self.strategy_bits {
            32 => 4,
            16 => 2,
            8 => 1,
            4 => 1,
            _ => 2,
        };

        let regrets_bytes = match self.regret_bits {
            32 => 4,
            16 => 2,
            8 => 1,
            4 => 0,
            _ => 2,
        };

        let ip_bytes = match self.ip_bits {
            32 => 4,
            16 => 2,
            8 => 1,
            4 => 0,
            _ => 2,
        };

        let chance_bytes = match self.chance_bits {
            32 => 4,
            16 => 2,
            8 => 1,
            4 => 0,
            _ => 2,
        };

        let is_nibble_mode = self.strategy_bits == 4;

        if self.target_storage_mode == BoardState::River {
            // NOTE: Previous logic returned [size, 0, 0, 0] here, stripping regrets.
            // This prevented resumption. We now fall through to calculate full sizes.
        }

        let mut node_index = match self.target_storage_mode {
            BoardState::Flop => self.num_nodes[0],
            BoardState::Turn => self.num_nodes[0] + self.num_nodes[1],
            BoardState::River => self.node_arena.len() as u64, // Use full arena for River
        } as usize;

        let mut num_storage = [0; 5];

        // Flags to check if we should wait for a storage type
        let check_s1 = true; // Always have strategy
        let check_s2 = self.regret_bits > 0;
        let check_s3 = self.ip_bits > 0;
        let check_chance = self.chance_bits > 0;
        let check_s4 = !self.storage4.is_empty();

        while (check_s1 && num_storage[0] == 0)
            || (check_s2 && num_storage[1] == 0)
            || (check_s3 && num_storage[2] == 0)
            || (check_chance && num_storage[3] == 0)
            || (check_s4 && num_storage[4] == 0)
        {
            if node_index == 0 {
                break;
            }
            node_index -= 1;

            let node = self.node_arena[node_index].lock();
            if num_storage[0] == 0 && !node.is_terminal() && !node.is_chance() {
                let offset_strategy = unsafe { node.storage1.offset_from(self.storage1.as_ptr()) };
                let offset_regrets = unsafe { node.storage2.offset_from(self.storage2.as_ptr()) };
                let offset_ip = unsafe { node.storage3.offset_from(self.storage_ip.as_ptr()) };

                // Storage4: SAPCFR+ regrets (might be null if unused)
                let offset_storage4 = if !node.storage4.is_null() && !self.storage4.is_empty() {
                    unsafe { node.storage4.offset_from(self.storage4.as_ptr()) }
                } else {
                    0
                };

                // For 4-bit mode, calculate packed length
                let len_strategy = if is_nibble_mode {
                    (node.num_elements as usize + 1) / 2
                } else {
                    strategy_bytes * node.num_elements as usize
                };

                let len_regrets = if self.regret_bits == 4 {
                    (node.num_elements as usize + 1) / 2
                } else {
                    regrets_bytes * node.num_elements as usize
                };
                let len_ip = if self.ip_bits == 4 {
                    (node.num_elements_ip as usize + 1) / 2
                } else {
                    ip_bytes * node.num_elements_ip as usize
                };

                if num_storage[0] == 0 {
                    num_storage[0] = offset_strategy as usize + len_strategy;
                }
                if num_storage[1] == 0 {
                    num_storage[1] = offset_regrets as usize + len_regrets;
                }
                if num_storage[2] == 0 {
                    num_storage[2] = offset_ip as usize + len_ip;
                }

                if offset_storage4 > 0 && num_storage[4] == 0 {
                    num_storage[4] = offset_storage4 as usize + len_regrets;
                }
            }
            if num_storage[3] == 0 && node.is_chance() {
                let offset = unsafe { node.storage1.offset_from(self.storage_chance.as_ptr()) };
                let len = if self.chance_bits == 4 {
                    (node.num_elements as usize + 1) / 2
                } else {
                    chance_bytes * node.num_elements as usize
                };
                num_storage[3] = offset as usize + len;
            }
        }

        num_storage
    }
}

thread_local! {
    // Increase size to hold storage1, storage2, storage_ip, storage4
    static PTR_BASE: Cell<[*const u8; 4]> = Cell::new([ptr::null(); 4]);
    static CHANCE_BASE: Cell<*const u8> = Cell::new(ptr::null());
    static PTR_BASE_MUT: Cell<[*mut u8; 4]> = Cell::new([ptr::null_mut(); 4]);
    static CHANCE_BASE_MUT: Cell<*mut u8> = Cell::new(ptr::null_mut());
}

impl Encode for PostFlopGame {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        if self.state <= State::Uninitialized {
            return Err(EncodeError::Other("Game is not successfully initialized"));
        }

        let num_storage = self.num_target_storage();

        // version
        VERSION_STR.to_string().encode(encoder)?;

        // contents
        self.state.encode(encoder)?;
        self.card_config.encode(encoder)?;
        self.tree_config.encode(encoder)?;
        self.added_lines.encode(encoder)?;
        self.removed_lines.encode(encoder)?;
        self.action_root.encode(encoder)?;
        self.target_storage_mode.encode(encoder)?;
        self.num_nodes.encode(encoder)?;
        self.quantization_mode
            .to_compression_flag()
            .encode(encoder)?;
        self.strategy_bits.encode(encoder)?;
        self.chance_bits.encode(encoder)?;
        self.num_storage.encode(encoder)?;
        self.num_storage_ip.encode(encoder)?;
        self.num_storage_chance.encode(encoder)?;
        self.misc_memory_usage.encode(encoder)?;
        self.storage1[0..num_storage[0]].encode(encoder)?;
        self.storage2[0..num_storage[1]].encode(encoder)?;
        self.storage_ip[0..num_storage[2]].encode(encoder)?;
        self.storage_chance[0..num_storage[3]].encode(encoder)?;
        // Encode storage4
        if num_storage[4] > 0 {
            self.storage4[0..num_storage[4]].encode(encoder)?;
        } else {
            // Encode empty if unused
            Vec::<u8>::new().encode(encoder)?;
        }

        let num_nodes = match self.target_storage_mode {
            BoardState::Flop => self.num_nodes[0] as usize,
            BoardState::Turn => (self.num_nodes[0] + self.num_nodes[1]) as usize,
            BoardState::River => self.node_arena.len(),
        };

        // locking strategy
        let mut locking_strategy = self.locking_strategy.clone();
        locking_strategy.retain(|&i, _| i < num_nodes);
        locking_strategy.encode(encoder)?;

        // store base pointers
        PTR_BASE.with(|c| {
            if self.state >= State::MemoryAllocated {
                c.set([
                    self.storage1.as_ptr(),
                    self.storage2.as_ptr(),
                    self.storage_ip.as_ptr(),
                    self.storage4.as_ptr(),
                ]);
            } else {
                c.set([ptr::null(); 4]);
            }
        });

        CHANCE_BASE.with(|c| {
            if self.state >= State::MemoryAllocated {
                c.set(self.storage_chance.as_ptr());
            } else {
                c.set(ptr::null());
            }
        });

        // game tree
        self.node_arena[0..num_nodes].encode(encoder)?;

        Ok(())
    }
}

impl<C> Decode<C> for PostFlopGame {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        // version check
        let version = String::decode(decoder)?;
        if version != VERSION_STR {
            return Err(DecodeError::OtherString(format!(
                "Version mismatch: expected '{VERSION_STR}', but got '{version}'"
            )));
        }

        // game instance
        let mut game = Self {
            state: Decode::decode(decoder)?,
            card_config: Decode::decode(decoder)?,
            tree_config: Decode::decode(decoder)?,
            added_lines: Decode::decode(decoder)?,
            removed_lines: Decode::decode(decoder)?,
            action_root: Decode::decode(decoder)?,
            storage_mode: Decode::decode(decoder)?,
            num_nodes: Decode::decode(decoder)?,
            // Decode compression flag
            quantization_mode: {
                let is_compression_enabled: bool = Decode::decode(decoder)?;
                QuantizationMode::from_compression_flag(is_compression_enabled)
            },
            strategy_bits: Decode::decode(decoder)?,
            chance_bits: Decode::decode(decoder)?,
            num_storage: Decode::decode(decoder)?,
            num_storage_ip: Decode::decode(decoder)?,
            num_storage_chance: Decode::decode(decoder)?,
            misc_memory_usage: Decode::decode(decoder)?,
            storage1: Decode::decode(decoder)?,
            storage2: Decode::decode(decoder)?,
            storage_ip: Decode::decode(decoder)?,
            storage_chance: Decode::decode(decoder)?,
            storage4: Decode::decode(decoder)?, // Decode storage4
            locking_strategy: Decode::decode(decoder)?,
            ..Default::default()
        };

        game.target_storage_mode = game.storage_mode;

        // Handle "River" mode re-allocation logic if needed (skipping for now as we focus on Resume)

        // store base pointers
        PTR_BASE_MUT.with(|c| {
            if game.state >= State::MemoryAllocated {
                c.set([
                    game.storage1.as_mut_ptr(),
                    game.storage2.as_mut_ptr(),
                    game.storage_ip.as_mut_ptr(),
                    game.storage4.as_mut_ptr(),
                ]);
            } else {
                c.set([ptr::null_mut(); 4]);
            }
        });

        CHANCE_BASE_MUT.with(|c| {
            if game.state >= State::MemoryAllocated {
                c.set(game.storage_chance.as_mut_ptr());
            } else {
                c.set(ptr::null_mut());
            }
        });

        // game tree
        game.node_arena = Decode::decode(decoder)?;

        // initialization
        game.check_card_config().map_err(DecodeError::OtherString)?;
        game.init_card_fields();
        game.init_interpreter();
        game.back_to_root();

        // restore the counterfactual values
        if game.storage_mode == BoardState::River && game.state == State::Solved {
            game.state = State::MemoryAllocated;
            finalize(&mut game);
        }

        Ok(game)
    }
}

impl Encode for PostFlopNode {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // contents
        self.prev_action.encode(encoder)?;
        self.packed_state.encode(encoder)?;
        self.amount.encode(encoder)?;
        self.children_offset.encode(encoder)?;
        self.num_children.encode(encoder)?;
        self.num_elements_ip.encode(encoder)?;
        self.num_elements.encode(encoder)?;
        self.scale1.encode(encoder)?;
        self.scale2.encode(encoder)?;
        self.scale3.encode(encoder)?;

        // pointer offset
        if !self.storage1.is_null() {
            if self.is_terminal() {
                // do nothing
            } else if self.is_chance() {
                let base = CHANCE_BASE.with(|c| c.get());
                unsafe { self.storage1.offset_from(base).encode(encoder)? };
            } else {
                let bases = PTR_BASE.with(|c| c.get());
                unsafe {
                    self.storage1.offset_from(bases[0]).encode(encoder)?; // Strategy offset (storage1)

                    // Explicitly encode storage2 offset (Regrets)
                    // If storage2 is not null
                    if !self.storage2.is_null() && !bases[1].is_null() {
                        self.storage2.offset_from(bases[1]).encode(encoder)?;
                    } else {
                        // Should probably panic or encode 0 marker?
                        // But for now assume structure holds.
                        // But to keep Decode aligned, we MUST encode something if we act like Play Node.
                        // Assume storage2 is always present for Play Node.
                        self.storage2.offset_from(bases[1]).encode(encoder)?;
                    }

                    self.storage3.offset_from(bases[2]).encode(encoder)?; // IP CFV offset (storage_ip) -- bases[2] now

                    // Storage4 (SAPCFR+)
                    // We need a flag? Or just always encode it?
                    // bincode relies on fixed structure or length prefix.
                    // We can't really add optional field in middle without breaking struct.
                    // But we can encode it conditionally if we assume Decode knows?
                    // Decode knows nothing locally.
                    // So we must always encode it for Play Node, or use sentinel.
                    // If we always encode, we waste space for DCFR.
                    // But for robustness, let's encode it if bases[3] is not null?
                    // No, Decode needs to know IF it should read it.
                    // Let's decide: Always encode storage4 offset. If null, encode 0.
                    if !bases[3].is_null() && !self.storage4.is_null() {
                        self.storage4.offset_from(bases[3]).encode(encoder)?;
                    } else {
                        0isize.encode(encoder)?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl<C> Decode<C> for PostFlopNode {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        // node instance
        let mut node = Self {
            prev_action: Decode::decode(decoder)?,
            packed_state: Decode::decode(decoder)?,
            amount: Decode::decode(decoder)?,
            children_offset: Decode::decode(decoder)?,
            num_children: Decode::decode(decoder)?,
            num_elements_ip: Decode::decode(decoder)?,
            num_elements: Decode::decode(decoder)?,
            scale1: Decode::decode(decoder)?,
            scale2: Decode::decode(decoder)?,
            scale3: Decode::decode(decoder)?,
            ..Default::default()
        };

        // pointers
        if node.is_terminal() {
            // do nothing
        } else if node.is_chance() {
            let base = CHANCE_BASE_MUT.with(|c| c.get());
            if !base.is_null() {
                node.storage1 = unsafe { base.offset(isize::decode(decoder)?) };
            }
        } else {
            let bases = PTR_BASE_MUT.with(|c| c.get());
            if !bases[0].is_null() {
                let offset = isize::decode(decoder)?; // Strategy
                let offset_regret = isize::decode(decoder)?; // Regrets (New!)
                let offset_ip = isize::decode(decoder)?; // IP CFV
                let offset_s4 = isize::decode(decoder)?; // Storage4 (New!)

                node.storage1 = unsafe { bases[0].offset(offset) };

                // Use explicit offset for storage2
                if !bases[1].is_null() {
                    node.storage2 = unsafe { bases[1].offset(offset_regret) };
                }

                if !bases[2].is_null() {
                    node.storage3 = unsafe { bases[2].offset(offset_ip) };
                }

                // Storage4
                if !bases[3].is_null() && offset_s4 != 0 {
                    node.storage4 = unsafe { bases[3].offset(offset_s4) };
                }
            }
        }

        Ok(node)
    }
}
