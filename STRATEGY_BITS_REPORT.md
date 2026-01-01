# Report: Feature `strategy_bits = 8`

## Sommario Esecutivo

La feature `strategy_bits` implementa la **mixed precision** nel solver, permettendo di ridurre l'uso di memoria per le strategie mantenendo i regrets a 16-bit. Con `strategy_bits = 8`, si ottiene un risparmio del **~25% di memoria totale** (50% per le strategie) senza compromettere significativamente la convergenza.

---

## 1. Configurazione

### 1.1 Parametro TOML

```toml
[solver]
quantization = "16bit"  # OBBLIGATORIO
strategy_bits = 8       # Opzionale: 16 (default), 8, o 4 (futuro)
```

**Valori supportati:**
- `16`: Default - stessa precisione del quantization mode (u16)
- `8`: Mixed precision - usa u8 per le strategie (50% meno memoria)
- `4`: Futuro - nibble packing (75% meno memoria) - NON ANCORA IMPLEMENTATO

**Vincoli:**
- Funziona **SOLO** con `quantization = "16bit"`
- Con `quantization = "32bit"` viene ignorato con warning

### 1.2 Parsing della Configurazione

**File:** `examples/solve_from_config.rs`

```rust
#[derive(Debug, Deserialize)]
struct SolverSettings {
    // ...
    /// Mixed precision: strategy precision in bits (16, 8, or 4)
    /// Only works when quantization = "16bit"
    /// - 16: Default (same precision as quantization mode)
    /// - 8: Mixed precision (50% less memory for strategy, ~25% overall)
    /// - 4: Future (75% less memory for strategy)
    #[serde(default = "default_strategy_bits")]
    strategy_bits: u8,
}

fn default_strategy_bits() -> u8 { 16 }
```

**Validazione (linee 408-416):**
```rust
if config.solver.strategy_bits != 16 {
    if quantization_mode != QuantizationMode::Int16 {
        eprintln!("Warning: strategy_bits only works with quantization='16bit', ignoring");
    } else {
        game.set_strategy_bits(config.solver.strategy_bits);
        println!("Mixed precision: {}-bit strategy (regrets stay 16-bit)",
                 config.solver.strategy_bits);
    }
}
```

---

## 2. Implementazione Core

### 2.1 Struttura Dati

**File:** `src/game/mod.rs`

```rust
pub struct PostFlopGame {
    // ...
    quantization_mode: QuantizationMode,
    strategy_bits: u8,  // Mixed precision: strategy precision (16, 8, or 4 bits)
    // ...
    storage1: Vec<u8>,  // Strategy storage (byte array)
    storage2: Vec<u8>,  // Regrets storage (byte array)
    // ...
}
```

**Nota:** Tutti gli storage sono `Vec<u8>` - l'interpretazione come u16/u8/f32 avviene tramite cast di puntatori.

### 2.2 Setter del Parametro

**File:** `src/game/base.rs` (linee 533-549)

```rust
pub fn set_strategy_bits(&mut self, bits: u8) {
    if self.state >= State::MemoryAllocated {
        panic!("Cannot change strategy precision after memory allocation");
    }

    match bits {
        16 | 8 => {
            self.strategy_bits = bits;
        }
        4 => {
            panic!("4-bit strategy not yet implemented (future feature)");
        }
        _ => {
            panic!("Invalid strategy_bits: {}. Valid values: 16, 8 (4 in future)", bits);
        }
    }
}
```

**Vincolo importante:** Deve essere chiamato **PRIMA** di `allocate_memory_with_mode()`.

### 2.3 Allocazione Memoria

**File:** `src/game/base.rs` (linee 424-435)

```rust
let strategy_bytes_per_elem = if mode == QuantizationMode::Int16 {
    // For 16-bit mode, allow mixed precision
    match self.strategy_bits {
        16 => 2,  // u16
        8 => 1,   // u8
        4 => 1,   // Future: nibbles (2 per byte, but allocate per byte)
        _ => panic!("Invalid strategy_bits value"),
    }
} else {
    // For Float32 mode, ignore strategy_bits and use full precision
    mode.bytes_per_element() as u64
};
```

**Allocazione separata (linee 450-458):**
```rust
let storage1_bytes = (strategy_bytes_per_elem * self.num_storage) as usize;  // Strategy
let storage2_bytes = (regrets_bytes_per_elem * self.num_storage) as usize;   // Regrets

self.storage1 = vec![0; storage1_bytes];  // Strategy: 1 byte/elem se strategy_bits=8
self.storage2 = vec![0; storage2_bytes];  // Regrets: sempre 2 bytes/elem
```

**Assegnazione puntatori ai nodi (linee 1678-1728):**
```rust
fn allocate_memory_nodes(&mut self) {
    let regrets_bytes = self.quantization_mode.bytes_per_element();
    
    let strategy_bytes = if self.quantization_mode == QuantizationMode::Int16 {
        match self.strategy_bits {
            16 => 2,  // u16
            8 => 1,   // u8
            4 => 1,   // Future: nibbles
            _ => 2,
        }
    } else {
        regrets_bytes  // Float32 mode: same as regrets
    };

    for node in &self.node_arena {
        let mut node = node.lock();
        if !node.is_terminal() && !node.is_chance() {
            unsafe {
                node.storage1 = ptr1.add(strategy_counter);  // Strategy
                node.storage2 = ptr2.add(regrets_counter);   // Regrets
            }
            strategy_counter += strategy_bytes * node.num_elements as usize;
            regrets_counter += regrets_bytes * node.num_elements as usize;
        }
    }
}
```

---

## 3. Accesso ai Dati

### 3.1 Metodi del Nodo

**File:** `src/game/node.rs`

```rust
impl GameNode for PostFlopNode {
    // 16-bit strategy (u16)
    fn strategy_compressed(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.storage1 as *const u16, self.num_elements as usize) }
    }

    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u16, self.num_elements as usize) }
    }

    // 8-bit strategy (u8) - MIXED PRECISION
    fn strategy_u8(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.storage1 as *const u8, self.num_elements as usize) }
    }

    fn strategy_u8_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u8, self.num_elements as usize) }
    }
}
```

**Nota:** Lo stesso puntatore `storage1` viene interpretato come `u16` o `u8` a seconda di `strategy_bits`.

---

## 4. Encoding/Decoding

### 4.1 Encoding u8 (Quantizzazione)

**File:** `src/utility.rs` (linee 278-288)

```rust
pub(crate) fn encode_unsigned_strategy_u8(dst: &mut [u8], slice: &[f32]) -> f32 {
    let scale = slice_nonnegative_max(slice);  // Trova il massimo
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u8::MAX as f32 / scale_nonzero;  // encoder = 255 / max

    dst.iter_mut().zip(slice).for_each(|(d, s)| {
        *d = unsafe { 
            (s * encoder + 0.49999997).to_int_unchecked::<i32>().clamp(0, 255) as u8 
        }
    });

    scale  // Ritorna lo scale factor per il decoding
}
```

**Formula:** `quantized = round(value * 255 / max).clamp(0, 255)`

### 4.2 Decoding u8

**File:** `src/utility.rs` (linee 292-295)

```rust
pub(crate) fn decode_unsigned_strategy_u8(src: &[u8], scale: f32) -> Vec<f32> {
    let decoder = scale / u8::MAX as f32;  // decoder = max / 255
    src.iter().map(|&x| x as f32 * decoder).collect()
}
```

**Formula:** `value = quantized * max / 255`

### 4.3 Normalizzazione

**File:** `src/utility.rs` (linee 907-927)

```rust
pub(crate) fn normalized_strategy_u8(strategy: &[u8], num_actions: usize) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(strategy.len());
    
    // Converti u8 -> f32
    normalized.extend(strategy.iter().map(|&s| s as f32));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);

    // Normalizza: strategy[i] / sum(strategy)
    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    normalized
}
```

**Processo:**
1. Converti u8 → f32
2. Somma per ogni mano
3. Normalizza dividendo per la somma (default uniforme se somma = 0)

---

## 5. Uso nel Solver

### 5.1 Update della Strategia Cumulativa

**File:** `src/solver.rs` (linee 256-307)

```rust
if game.is_compression_enabled() {
    match (game.quantization_mode(), game.strategy_bits()) {
        (QuantizationMode::Int16, 8) => {
            // 8-bit strategy mode (mixed precision)
            let scale = node.strategy_scale();
            let decoder = params.gamma_t * scale / u8::MAX as f32;
            let cum_strategy_u8 = node.strategy_u8_mut();

            // Decode e accumula
            strategy.iter_mut().zip(&*cum_strategy_u8).for_each(|(x, y)| {
                *x += (*y as f32) * decoder;
            });

            // Applica locking se necessario
            if !locking.is_empty() {
                strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                    if s.is_sign_positive() { *d = 0.0; }
                })
            }

            // Encode e salva
            let new_scale = encode_unsigned_strategy_u8(cum_strategy_u8, &strategy);
            node.set_strategy_scale(new_scale);
        }
        (QuantizationMode::Int16, 16) | (QuantizationMode::Float32, _) => {
            // Normal 16-bit mode (o 32-bit)
            // ... usa encode_unsigned_slice (u16) ...
        }
        // ...
    }
}
```

**Flusso:**
1. **Decode:** `cum_strategy_u8` → f32 usando `scale / 255`
2. **Accumula:** Aggiungi alla strategia corrente con peso `gamma_t`
3. **Encode:** f32 → `cum_strategy_u8` usando nuovo scale factor

### 5.2 Lettura della Strategia

**File:** `src/game/interpreter.rs` (linee 833-840)

```rust
let mut ret = if self.is_compression_enabled() {
    match self.strategy_bits() {
        8 => normalized_strategy_u8(node.strategy_u8(), num_actions),
        _ => normalized_strategy_compressed(node.strategy_compressed(), num_actions),
    }
} else {
    normalized_strategy(node.strategy(), num_actions)
};
```

**File:** `src/utility.rs` - Compute CFValue (linee 535-551)

```rust
let mut strategy = if game.is_compression_enabled() {
    match game.strategy_bits() {
        8 => normalized_strategy_u8(node.strategy_u8(), num_actions),
        _ => normalized_strategy_compressed(node.strategy_compressed(), num_actions),
    }
} else {
    normalized_strategy(node.strategy(), num_actions)
};
```

---

## 6. Risparmio di Memoria

### 6.1 Calcolo Teorico

**Assunzioni:**
- Quantization mode: 16-bit
- Num elementi totali: N

**Senza mixed precision (strategy_bits = 16):**
- Strategy: N × 2 bytes (u16)
- Regrets: N × 2 bytes (i16)
- **Totale:** N × 4 bytes

**Con mixed precision (strategy_bits = 8):**
- Strategy: N × 1 byte (u8)
- Regrets: N × 2 bytes (i16)
- **Totale:** N × 3 bytes

**Risparmio:**
- Strategy: 50% (da 2N a N bytes)
- Totale: 25% (da 4N a 3N bytes)

### 6.2 Esempio Pratico

Per un tree con 1 milione di elementi:

| Componente | 16-bit | 8-bit | Risparmio |
|------------|--------|-------|-----------|
| Strategy   | 2 MB   | 1 MB  | 50%       |
| Regrets    | 2 MB   | 2 MB  | 0%        |
| **Totale** | **4 MB** | **3 MB** | **25%** |

---

## 7. Trade-off Precisione vs Memoria

### 7.1 Precisione

**16-bit (u16):**
- Range: 0 - 65535
- Risoluzione: ~0.0015% (1/65535)

**8-bit (u8):**
- Range: 0 - 255
- Risoluzione: ~0.39% (1/255)

**Impatto:**
- Le strategie sono **probabilità normalizzate** (somma = 1)
- Errore di quantizzazione: ~0.4% per azione
- I **regrets rimangono a 16-bit** → convergenza preservata

### 7.2 Quando Usare strategy_bits = 8

**✅ Consigliato:**
- Tree molto grandi (memoria limitata)
- Solver in produzione (dopo tuning)
- Quando la precisione dello 0.4% è accettabile

**❌ Sconsigliato:**
- Ricerca scientifica (massima precisione)
- Debugging/analisi dettagliata
- Tree piccoli (risparmio trascurabile)

---

## 8. File di Configurazione Analizzati

### 8.1 hand_0000007438_node_03_turn_DeepStack.toml

```toml
[solver]
max_iterations = 2500
target_exploitability_pct = 0.1
quantization = "16bit"
strategy_bits = 8              # ← MIXED PRECISION ATTIVA
zstd_compression_level = 0
```

### 8.2 hand_0000007438_node_05_river_DeepStack.toml

```toml
[solver]
max_iterations = 1000
target_exploitability_pct = 0.1
quantization = "16bit"
zstd_compression_level = 3
strategy_bits = 8              # ← MIXED PRECISION ATTIVA
```

**Osservazione:** Entrambi i file usano `strategy_bits = 8` con `quantization = "16bit"`.

---

## 9. Flusso Completo

```
1. CONFIGURAZIONE
   ├─ Parsing TOML: strategy_bits = 8
   ├─ Validazione: quantization = "16bit" ✓
   └─ game.set_strategy_bits(8)

2. ALLOCAZIONE MEMORIA
   ├─ Calcolo bytes: strategy_bytes = 1 (u8)
   ├─ Allocazione: storage1 = vec![0; N × 1]
   └─ Assegnazione puntatori ai nodi

3. SOLVING (ogni iterazione)
   ├─ Regret Matching: regrets (i16) → strategy (f32)
   ├─ Decode: cum_strategy_u8 → f32 (scale / 255)
   ├─ Accumula: strategy += cum_strategy × gamma_t
   ├─ Encode: strategy → cum_strategy_u8 (nuovo scale)
   └─ Update regrets (sempre i16)

4. LETTURA RISULTATI
   ├─ normalized_strategy_u8(node.strategy_u8())
   ├─ Converti u8 → f32
   └─ Normalizza per somma
```

---

## 10. Conclusioni

### 10.1 Punti Chiave

1. **Mixed precision** separa la precisione di strategy e regrets
2. **Risparmio memoria:** ~25% totale, 50% per le strategie
3. **Convergenza preservata:** I regrets rimangono a 16-bit
4. **Implementazione trasparente:** Dispatch automatico basato su `strategy_bits`
5. **Vincolo:** Richiede `quantization = "16bit"`

### 10.2 Limitazioni Attuali

- `strategy_bits = 4` non ancora implementato
- Nessun supporto per `quantization = "32bit"` (viene ignorato)
- Deve essere impostato **prima** dell'allocazione memoria

### 10.3 Possibili Estensioni Future

- Implementare nibble packing (4-bit) per 75% di risparmio
- Adaptive precision basata sulla profondità del tree
- Profiling automatico per scegliere il miglior trade-off

---

## Riferimenti Codice

| Componente | File | Linee |
|------------|------|-------|
| Config parsing | `examples/solve_from_config.rs` | 98-104, 408-416 |
| Setter | `src/game/base.rs` | 533-549 |
| Allocazione | `src/game/base.rs` | 424-435, 1678-1728 |
| Node accessors | `src/game/node.rs` | 100-119 |
| Encoding | `src/utility.rs` | 278-295 |
| Normalizzazione | `src/utility.rs` | 907-927 |
| Solver update | `src/solver.rs` | 256-307 |
| Interpreter | `src/game/interpreter.rs` | 833-840 |

---

**Report generato il:** 2026-01-01  
**Versione codebase:** postflop-solver-custom

