use super::*;
use crate::interface::GameNode;
use crate::quantization::*;

/// Memory allocation logic for PostFlopGame
impl PostFlopGame {
    /// Allocates memory for all storage arrays using the configured precision settings.
    ///
    /// Uses the granular *_bits parameters to determine bytes per element for each storage:
    /// - storage1 (strategy): `strategy_bits` (16, or 32)
    /// - storage2/4 (regrets): `regret_bits` (16 or 32)
    /// - storage_ip (IP cfvalues): `ip_bits` (16 or 32)
    /// - storage_chance (chance cfvalues): `chance_bits` (8, 16, or 32)
    ///
    /// # Panics
    /// Panics if the game is not successfully initialized or if memory usage exceeds maximum size.
    pub fn allocate_memory(&mut self) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.state == State::MemoryAllocated && self.storage_mode == BoardState::River {
            return;
        }

        // Determine bytes per element for each storage based on *_bits settings
        let strategy_bytes_per_elem = match self.strategy_bits {
            32 => 4, // f32
            16 => 2, // u16
            8 => 1,  // u8
            _ => panic!(
                "Invalid strategy_bits: {}. Valid values: 8, 16, 32",
                self.strategy_bits
            ),
        };

        let regret_bytes_per_elem = match self.regret_bits {
            32 => 4, // f32
            16 => 2, // i16
            8 => 1,  // i8
            4 => 0,  // Special case: packed
            _ => panic!(
                "Invalid regret_bits: {}. Valid values: 4, 8, 16, 32",
                self.regret_bits
            ),
        };

        let ip_bytes_per_elem = match self.ip_bits {
            32 => 4, // f32
            16 => 2, // i16
            8 => 1,  // i8
            4 => 0,  // Special case: packed
            _ => panic!(
                "Invalid ip_bits: {}. Valid values: 4, 8, 16, 32",
                self.ip_bits
            ),
        };

        let chance_bytes_per_elem = match self.chance_bits {
            32 => 4, // f32
            16 => 2, // i16
            8 => 1,  // i8
            4 => 0,  // Special case: packed
            _ => panic!(
                "Invalid chance_bits: {}. Valid values: 4, 8, 16, 32",
                self.chance_bits
            ),
        };

        // Calculate storage size for strategy
        // Calculate packed sizes if needed
        let (packed_regrets, packed_ip, packed_chance) =
            if self.regret_bits == 4 || self.ip_bits == 4 || self.chance_bits == 4 {
                self.calculate_packed_sizes()
            } else {
                (0, 0, 0)
            };

        // Calculate storage size for strategy
        let storage1_bytes = (strategy_bytes_per_elem * self.num_storage) as usize;
        let storage_chance_bytes = match self.chance_bits {
            4 => packed_chance,
            _ => (chance_bytes_per_elem * self.num_storage_chance) as usize,
        };

        if storage1_bytes > isize::MAX as usize
            || regret_bytes_per_elem * self.num_storage > isize::MAX as u64
            || ip_bytes_per_elem * self.num_storage_ip > isize::MAX as u64
            || storage_chance_bytes as u64 > isize::MAX as u64
        {
            panic!("Memory usage exceeds maximum size");
        }

        self.state = State::MemoryAllocated;

        // Set quantization_mode for backward compatibility with trait methods
        // Use regret_bits as the representative quantization mode
        self.quantization_mode = match self.regret_bits {
            32 => QuantizationMode::Float32,
            16 => {
                if self.log_encoding_enabled {
                    QuantizationMode::Int16Log
                } else {
                    QuantizationMode::Int16
                }
            }
            8 => QuantizationMode::Int8,
            4 => QuantizationMode::Int4Packed,
            _ => QuantizationMode::Int16,
        };

        self.clear_storage();
        let storage2_bytes = match self.regret_bits {
            4 => packed_regrets,
            _ => (regret_bytes_per_elem * self.num_storage) as usize,
        };
        let storage_ip_bytes = match self.ip_bits {
            4 => packed_ip,
            _ => (ip_bytes_per_elem * self.num_storage_ip) as usize,
        };

        self.storage1 = vec![0; storage1_bytes];
        self.storage2 = vec![0; storage2_bytes];
        self.storage_ip = vec![0; storage_ip_bytes];
        self.storage_chance = vec![0; storage_chance_bytes];

        if self.cfr_algorithm.requires_storage4() {
            let storage4_bytes = match self.regret_bits {
                4 => packed_regrets,
                _ => (regret_bytes_per_elem * self.num_storage) as usize,
            };
            self.storage4 = vec![0; storage4_bytes];
        }

        self.allocate_memory_nodes();

        self.storage_mode = BoardState::River;
        self.target_storage_mode = BoardState::River;
    }

    /// DEPRECATED: Use `allocate_memory()` instead with `set_*_bits()` setters.
    ///
    /// This method is provided for backward compatibility only.
    /// It sets all *_bits parameters based on the quantization mode and then allocates memory.
    #[deprecated(
        since = "0.1.0",
        note = "Use allocate_memory() with set_*_bits() instead"
    )]
    pub fn allocate_memory_with_mode(&mut self, mode: QuantizationMode) {
        // Convert mode to bits
        let bits = mode.bytes_per_element() as u8 * 8;
        self.strategy_bits = bits.min(16); // Strategy max 16-bit in old mode
        self.regret_bits = bits;
        self.ip_bits = bits;
        self.chance_bits = bits.min(16); // Chance max 16-bit in old mode
        self.quantization_mode = mode;
        self.allocate_memory();
    }

    /// Sets the strategy precision in bits (mixed precision mode).
    ///
    /// Must be called BEFORE allocate_memory_with_mode().
    /// Only works when quantization_mode is Int16.
    ///
    /// Valid values:
    /// - `16`: Default, uses u16 (2 bytes per element)
    /// - `32`: Full precision, uses f32 (4 bytes per element)
    ///
    /// # Panics
    /// Panics if memory has already been allocated or if an invalid value is provided.
    #[inline]
    pub fn set_strategy_bits(&mut self, bits: u8) {
        if self.state >= State::MemoryAllocated {
            panic!("Cannot change strategy precision after memory allocation");
        }

        match bits {
            16 | 32 | 8 => {
                self.strategy_bits = bits;
            }
            _ => {
                panic!("Invalid strategy_bits: {}. Valid values: 8, 16, 32", bits);
            }
        }
    }

    /// Sets the chance cfvalues precision in bits (mixed precision mode).
    ///
    /// Must be called BEFORE allocate_memory_with_mode().
    /// Only works when quantization_mode is Int16.
    ///
    /// Valid values:
    /// - `16`: Default, uses i16 (same as quantization mode)
    /// - `8`: Uses i8 (50% less memory for chance cfvalues)
    ///
    /// Benefits of 8-bit chance:
    /// - 50% less memory for chance cfvalues storage
    /// - Regrets stay at 16-bit (preserves convergence)
    /// - Useful for large game trees with many chance nodes
    ///
    /// # Panics
    /// Panics if memory has already been allocated or if an invalid value is provided.
    #[inline]
    pub fn set_chance_bits(&mut self, bits: u8) {
        if self.state >= State::MemoryAllocated {
            panic!("Cannot change chance precision after memory allocation");
        }

        match bits {
            32 | 16 | 8 | 4 => {
                self.chance_bits = bits;
            }
            _ => {
                panic!("Invalid chance_bits: {}. Valid values: 32, 16, 8, 4", bits);
            }
        }
    }

    /// Sets the regret precision in bits (for storage2 and storage4).
    ///
    /// Must be called BEFORE allocate_memory_with_mode().
    ///
    /// Valid values:
    /// - `16`: Default, uses i16 (2 bytes per element)
    /// - `32`: Full precision, uses f32 (4 bytes per element)
    ///
    /// # Panics
    /// Panics if memory has already been allocated or if an invalid value is provided.
    #[inline]
    pub fn set_regret_bits(&mut self, bits: u8) {
        if self.state >= State::MemoryAllocated {
            panic!("Cannot change regret precision after memory allocation");
        }

        match bits {
            32 | 16 | 8 | 4 => {
                self.regret_bits = bits;
            }
            _ => {
                panic!("Invalid regret_bits: {}. Valid values: 4, 8, 16, 32", bits);
            }
        }
    }

    /// Sets the IP counterfactual values precision in bits (for storage_ip).
    ///
    /// Must be called BEFORE allocate_memory_with_mode().
    ///
    /// Valid values:
    /// - `16`: Default, uses i16 (2 bytes per element)
    /// - `8`: Uses i8 (50% less memory for IP cfvalues)
    /// - `32`: Full precision, uses f32 (4 bytes per element)
    ///
    /// # Panics
    /// Panics if memory has already been allocated or if an invalid value is provided.
    #[inline]
    pub fn set_ip_bits(&mut self, bits: u8) {
        if self.state >= State::MemoryAllocated {
            panic!("Cannot change IP cfvalues precision after memory allocation");
        }

        match bits {
            32 | 16 | 8 | 4 => {
                self.ip_bits = bits;
            }
            _ => {
                panic!("Invalid ip_bits: {}. Valid values: 4, 8, 16, 32", bits);
            }
        }
    }

    /// Calculates the exact required bytes for packed 4-bit storage by iterating the arena.
    /// Returns (regrets_bytes, ip_bytes, chance_bytes).
    pub(crate) fn calculate_packed_sizes(&self) -> (usize, usize, usize) {
        let mut packed_regrets = 0usize;
        let mut packed_ip = 0usize;
        let mut packed_chance = 0usize;

        for node in &self.node_arena {
            let node = node.lock();
            if node.is_terminal() {
                continue;
            }

            if node.is_chance() {
                if self.chance_bits == 4 {
                    packed_chance += (node.num_elements as usize + 1) / 2;
                }
            } else {
                // Play node
                if self.regret_bits == 4 {
                    packed_regrets += (node.num_elements as usize + 1) / 2;
                }
                if self.ip_bits == 4 {
                    packed_ip += (node.num_elements_ip as usize + 1) / 2;
                }
            }
        }
        (packed_regrets, packed_ip, packed_chance)
    }

    /// Allocates memory recursively.
    pub(crate) fn allocate_memory_nodes(&mut self) {
        // Determine bytes per element for each storage based on *_bits settings
        let strategy_bytes = match self.strategy_bits {
            32 => 4,
            16 => 2,
            8 => 1,
            _ => 2,
        };

        let regrets_bytes = match self.regret_bits {
            32 => 4,
            16 => 2,
            8 => 1,
            _ => 2,
        };

        let ip_bytes = match self.ip_bits {
            32 => 4,
            16 => 2,
            8 => 1,
            4 => 0, // Special case
            _ => 2,
        };

        let chance_bytes = match self.chance_bits {
            32 => 4,
            16 => 2,
            8 => 1,
            4 => 0, // Special case
            _ => 2,
        };

        let mut strategy_counter = 0;
        let mut regrets_counter = 0;
        let mut ip_counter = 0;
        let mut chance_counter = 0;
        let mut storage4_counter = 0;
        let use_storage4 = !self.storage4.is_empty();

        for node in &self.node_arena {
            let mut node = node.lock();
            if node.is_terminal() {
                // do nothing
            } else if node.is_chance() {
                unsafe {
                    let ptr = self.storage_chance.as_mut_ptr();
                    node.storage1 = ptr.add(chance_counter);
                }
                if self.chance_bits == 4 {
                    chance_counter += (node.num_elements as usize + 1) / 2;
                } else {
                    chance_counter += chance_bytes * node.num_elements as usize;
                }
                // Initialize scale factors to 1.0 (will be updated on first write)
                node.scale1 = 1.0;
                node.scale2 = 1.0;
                node.scale3 = 1.0;
            } else {
                unsafe {
                    let ptr1 = self.storage1.as_mut_ptr();
                    let ptr2 = self.storage2.as_mut_ptr();
                    let ptr3 = self.storage_ip.as_mut_ptr();
                    node.storage1 = ptr1.add(strategy_counter);
                    node.storage2 = ptr2.add(regrets_counter);
                    node.storage3 = ptr3.add(ip_counter);
                    if use_storage4 {
                        let ptr4 = self.storage4.as_mut_ptr();
                        node.storage4 = ptr4.add(storage4_counter);
                    }
                }
                strategy_counter += strategy_bytes * node.num_elements as usize;
                if self.regret_bits == 4 {
                    regrets_counter += (node.num_elements as usize + 1) / 2;
                } else {
                    regrets_counter += regrets_bytes * node.num_elements as usize;
                }
                if self.ip_bits == 4 {
                    ip_counter += (node.num_elements_ip as usize + 1) / 2;
                } else {
                    ip_counter += ip_bytes * node.num_elements_ip as usize;
                }
                if use_storage4 {
                    if self.regret_bits == 4 {
                        storage4_counter += (node.num_elements as usize + 1) / 2;
                    } else {
                        storage4_counter += regrets_bytes * node.num_elements as usize;
                    }
                }
                // Initialize scale factors to 1.0 (will be updated on first write)
                node.scale1 = 1.0;
                node.scale2 = 1.0;
                node.scale3 = 1.0;
            }
        }
    }

    /// Clears the storage.
    #[inline]
    pub(crate) fn clear_storage(&mut self) {
        self.storage1 = Vec::new();
        self.storage2 = Vec::new();
        self.storage_ip = Vec::new();
        self.storage_chance = Vec::new();
        self.storage4 = Vec::new();
    }
}
