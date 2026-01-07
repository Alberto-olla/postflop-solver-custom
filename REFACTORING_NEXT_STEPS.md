# Piano Refactoring - Prossimi Step

## Stato Attuale (Completato)

### ✅ Refactoring Fase 1 - Completato

1. **Split game/base.rs** - Da 2,182 linee a ~1,800
   - `game/allocation.rs` - Logica di allocazione memoria
   - `game/locking.rs` - Strategy locking
   - `game/base.rs` - Core game logic

2. **Separazione Algoritmi Experimental**
   - `cfr_algorithms/algorithms/` - DCFR, DCFR+ (raccomandati)
   - `cfr_algorithms/experimental/` - PDCFR+, SAPCFR+ (sperimentali)

3. **Quantization Module Trait-based**
   - `quantization/traits.rs` - Trait `QuantizationType`
   - `quantization/types.rs` - Implementazioni (Float32, Int16, Int8, Int4)
   - `quantization/mode.rs` - QuantizationMode enum
   - Foundation pronta per eliminare duplicazione

4. **Utility e Solver Modules**
   - `utility/` - Modulo creato
   - `solver/` - Modulo creato

---

## Prossimi Step Prioritari

### Step 6: Integrare Quantization Traits nel Solver ⚠️ PRIORITÀ ALTA

**Obiettivo:** Eliminare duplicazione delle funzioni `regret_matching_*` usando i trait.

**File da modificare:**
- `src/solver/mod.rs` (o creare `solver/matching.rs`)

**Problema attuale:**
```rust
// Duplicazione in solver/mod.rs (6+ funzioni simili):
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32>
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32>
fn regret_matching_compressed_i8(regret: &[i8], num_actions: usize) -> Vec<f32>
fn regret_matching_compressed_u8(regret: &[u8], num_actions: usize) -> Vec<f32>
fn regret_matching_compressed_i4_packed(regret: &[u8], ...) -> Vec<f32>
fn regret_matching_compressed_u4_packed(regret: &[u8], ...) -> Vec<f32>
```

**Soluzione proposta:**
```rust
// Usare i trait già creati in quantization/types.rs
use crate::quantization::types::*;

fn regret_matching_generic<Q: QuantizationType>(
    data: &[Q::Storage],
    scale: f32,
    num_actions: usize
) -> Vec<f32> {
    Q::regret_matching(data, scale, num_actions)
}

// Nel solver, dispatch basato su game.quantization_mode():
match game.quantization_mode() {
    QuantizationMode::Float32 => {
        regret_matching_generic::<Float32Quant>(node.regrets(), scale, num_actions)
    }
    QuantizationMode::Int16 => {
        regret_matching_generic::<Int16Quant>(node.regrets_compressed(), scale, num_actions)
    }
    // ... altri casi
}
```

**Benefici:**
- Elimina ~300+ linee di codice duplicato
- Più facile aggiungere nuovi tipi di quantizzazione
- Testing centralizzato (trait tests in quantization/types.rs)

**Stima complessità:** Media-Alta (richiede refactoring attento del solver loop)

---

### Step 7: Split solver/mod.rs per Responsabilità ⚠️ PRIORITÀ MEDIA

**Obiettivo:** Spezzettare solver/mod.rs (1,517 linee) in moduli più piccoli.

**Struttura proposta:**
```
solver/
├── mod.rs (~200 lines)
│   - PostFlopSolver struct
│   - Public API: solve(), iterate()
│   - Main CFR loop orchestration
│
├── strategy.rs
│   - Strategy calculation logic (lines 373-509 del vecchio solver.rs)
│   - Strategy update e discounting (lines 522-587)
│   - Locked strategy handling
│
├── regrets.rs
│   - Regret update logic (lines 590-1020+)
│   - CFV calculations
│   - Algorithm-specific regret logic (DCFR/DCFR+/PDCFR+/SAPCFR+)
│
├── pruning.rs
│   - Threshold computation (lines 289-336)
│   - Pruning decision logic
│
└── matching.rs (OPZIONALE se Step 6 non completato)
    - Regret matching functions
    - O meglio: eliminare creando wrapper ai trait
```

**Implementazione:**

1. **Creare solver/strategy.rs:**
```rust
use super::*;
use crate::cfr_algorithms::*;

/// Calculate strategy from regrets based on CFR algorithm
pub(super) fn calculate_strategy<T: Game>(
    game: &T,
    node: &T::Node,
    algorithm: CfrAlgorithm,
    params: &DiscountParams,
) -> Vec<f32> {
    // Logica estratta da solver.rs lines 373-509
    match algorithm {
        CfrAlgorithm::DCFR | CfrAlgorithm::DCFRPlus => {
            // Standard regret matching
        }
        CfrAlgorithm::PDCFRPlus => {
            // Predicted regrets logic
        }
        CfrAlgorithm::SAPCFRPlus => {
            // Explicit regrets logic
        }
    }
}

/// Update stored cumulative strategy
pub(super) fn update_strategy<T: Game>(
    game: &T,
    node: &T::Node,
    new_strategy: &[f32],
    gamma_t: f32,
    seed: u32,
) {
    // Logica estratta da solver.rs lines 522-587
}
```

2. **Creare solver/regrets.rs:**
```rust
pub(super) fn update_regrets<T: Game>(
    game: &T,
    node: &T::Node,
    cfvalues: &[f32],
    params: &DiscountParams,
) {
    // Logica complessa per update regrets (lines 590-1020+)
}
```

3. **Aggiornare solver/mod.rs:**
```rust
mod strategy;
mod regrets;
mod pruning;

use strategy::*;
use regrets::*;
use pruning::*;

pub fn solve<T: Game>(game: &T, ...) {
    // Orchestrazione usando i moduli
    let strategy = calculate_strategy(game, node, algorithm, &params);
    update_strategy(game, node, &strategy, gamma_t, seed);
    update_regrets(game, node, &cfvalues, &params);
}
```

**Benefici:**
- solver/mod.rs da 1,517 linee a ~200 linee
- Ogni modulo ha responsabilità chiara
- Più facile testing e debugging
- Migliore comprensione del flusso CFR

**Stima complessità:** Alta (molti refactoring, testing accurato richiesto)

---

### Step 8: Migrare Encoding Functions ⚠️ PRIORITÀ BASSA

**Obiettivo:** Spostare funzioni di encoding da `utility/mod.rs` a `quantization/encoding.rs`.

**Funzioni da spostare (da utility/mod.rs):**
```rust
// Lines ~214-465 in utility/mod.rs
encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32
encode_unsigned_slice(dst: &mut [u16], slice: &[f32]) -> f32
encode_unsigned_strategy_u8(dst: &mut [u8], slice: &[f32], seed: u32) -> f32
encode_unsigned_regrets_u8(dst: &mut [u8], slice: &[f32], seed: u32) -> f32
encode_signed_slice_log(dst: &mut [i16], slice: &[f32]) -> f32
encode_signed_i8(dst: &mut [i8], slice: &[f32], seed: u32) -> f32
encode_signed_i4_packed(dst: &mut [u8], slice: &[f32], seed: u32) -> f32
encode_unsigned_u4_packed(dst: &mut [u8], slice: &[f32], seed: u32) -> f32
decode_signed_i8(src: &[i8], scale: f32) -> Vec<f32>
decode_signed_i4_packed(src: &[u8], scale: f32, len: usize) -> Vec<f32>
decode_unsigned_u4_packed(src: &[u8], scale: f32, len: usize) -> Vec<f32>
```

**Helper functions da spostare:**
```rust
fast_xorshift32(seed: &mut u32) -> u32
stochastic_round(val: f32, seed: &mut u32) -> i32
slice_absolute_max(slice: &[f32]) -> f32
slice_nonnegative_max(slice: &[f32]) -> f32
```

**Struttura proposta:**

`quantization/encoding.rs`:
```rust
//! Low-level encoding/decoding functions

mod helpers {
    pub(super) fn fast_xorshift32(seed: &mut u32) -> u32 { ... }
    pub(super) fn stochastic_round(val: f32, seed: &mut u32) -> i32 { ... }
    pub(super) fn slice_absolute_max(slice: &[f32]) -> f32 { ... }
    pub(super) fn slice_nonnegative_max(slice: &[f32]) -> f32 { ... }
}

// Int16 encoding
pub fn encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32 { ... }
pub fn encode_unsigned_slice(dst: &mut [u16], slice: &[f32]) -> f32 { ... }
pub fn encode_signed_slice_log(dst: &mut [i16], slice: &[f32]) -> f32 { ... }

// Int8 encoding
pub fn encode_signed_i8(dst: &mut [i8], slice: &[f32], seed: u32) -> f32 { ... }
pub fn encode_unsigned_strategy_u8(dst: &mut [u8], slice: &[f32], seed: u32) -> f32 { ... }
pub fn encode_unsigned_regrets_u8(dst: &mut [u8], slice: &[f32], seed: u32) -> f32 { ... }

// Int4 encoding
pub fn encode_signed_i4_packed(dst: &mut [u8], slice: &[f32], seed: u32) -> f32 { ... }
pub fn encode_unsigned_u4_packed(dst: &mut [u8], slice: &[f32], seed: u32) -> f32 { ... }

// Decoding functions
pub fn decode_signed_i8(src: &[i8], scale: f32) -> Vec<f32> { ... }
pub fn decode_signed_i4_packed(src: &[u8], scale: f32, len: usize) -> Vec<f32> { ... }
pub fn decode_unsigned_u4_packed(src: &[u8], scale: f32, len: usize) -> Vec<f32> { ... }
```

**utility/mod.rs cleanup:**
```rust
// Re-export per backwards compatibility (deprecate in future)
pub use crate::quantization::encoding::{
    encode_signed_slice,
    encode_unsigned_slice,
    // ... altri
};

// Mantenere solo:
// - Terminal utility calculations
// - Equity computations
// - Game-specific utilities (apply_swap, etc.)
```

**Benefici:**
- Quantization logic tutta in un modulo
- utility/mod.rs più focused (solo game utilities)
- Coerenza architetturale

**Stima complessità:** Bassa-Media (principalmente moving code + re-exports)

---

## Step Opzionali (Minore Priorità)

### Opzionale A: Split game/base.rs Ulteriore

**Situazione attuale:** game/base.rs è ancora ~1,800 linee dopo lo split.

**Possibili suddivisioni:**
```
game/
├── base.rs (~800 lines) - Core PostFlopGame struct + main methods
├── allocation.rs (✅ già fatto)
├── locking.rs (✅ già fatto)
├── construction.rs - Game tree building logic
├── queries.rs - Query methods (node access, indices, etc.)
└── metadata.rs - Memory usage, node counting, etc.
```

**Quando considerare:** Solo se game/base.rs diventa difficile da navigare.

---

### Opzionale B: Refactor game/interpreter.rs

**Situazione:** game/interpreter.rs è 1,297 linee e abbastanza coeso.

**Possibile split solo se necessario:**
```
game/
└── interpreter/
    ├── mod.rs - Main interpreter struct
    ├── queries.rs - Strategy queries
    ├── weights.rs - Weight calculations
    └── results.rs - Result formatting
```

**Quando considerare:** Solo se si aggiungono molte nuove query APIs.

---

### Opzionale C: Astrarre Storage Access con Trait

**Idea:** Creare trait per accesso storage type-safe.

```rust
// quantization/storage.rs
pub trait QuantizedStorage {
    type Quant: QuantizationType;

    fn data(&self) -> &[<Self::Quant as QuantizationType>::Storage];
    fn data_mut(&mut self) -> &mut [<Self::Quant as QuantizationType>::Storage];
    fn scale(&self) -> f32;
    fn set_scale(&mut self, scale: f32);
}

// Esempio uso in solver
fn process_node<S: QuantizedStorage>(storage: &S) {
    let strategy = S::Quant::regret_matching(
        storage.data(),
        storage.scale(),
        num_actions
    );
}
```

**Benefici:**
- Type safety a compile-time
- Eliminazione completa di pattern matching runtime

**Sfida:** Richiede refactoring significativo di PostFlopNode storage pointers.

**Quando considerare:** Dopo Step 6 completato, se si vuole ulteriore type safety.

---

## Ordine di Esecuzione Raccomandato

### Fase 2 (Prossima)
1. **Step 6** - Integrare quantization traits (elimina duplicazione) ⚠️
2. **Step 8** - Migrare encoding functions (cleanup modulare) ✅

### Fase 3 (Dopo Step 6+8)
3. **Step 7** - Split solver/mod.rs (ora più facile senza duplicazione) ⚠️

### Fase 4 (Opzionale)
4. Opzionali A/B/C - Solo se necessario

---

## Metriche di Successo

### Dopo Step 6 (Quantization Integration):
- ✅ Eliminazione di 6+ funzioni `regret_matching_*`
- ✅ Riduzione di ~300-400 linee di codice duplicato
- ✅ Test passano (93+ test)
- ✅ Nessun peggioramento performance (benchmark)

### Dopo Step 7 (Solver Split):
- ✅ solver/mod.rs < 300 linee
- ✅ Ogni submodule < 500 linee
- ✅ Separazione chiara responsabilità
- ✅ Test passano

### Dopo Step 8 (Encoding Migration):
- ✅ utility/mod.rs < 800 linee (da ~1,373)
- ✅ quantization/encoding.rs contiene tutto l'encoding logic
- ✅ Re-exports per backwards compatibility
- ✅ Test passano

---

## Note Implementative

### Testing Strategy
Per ogni step:
1. **Unit tests** - Test dei singoli moduli
2. **Integration tests** - Verifica che solver converga correttamente
3. **Performance benchmarks** - Nessuna regressione
4. **Backward compatibility** - Verificare esempi esistenti

### Gestione Breaking Changes
- Usare deprecation warnings prima di rimuovere API pubbliche
- Mantenere re-exports per internal API quando possibile
- Documentare migration path in CHANGELOG

### Performance
- Step 6: Usare generics/monomorphization (zero-cost abstraction)
- Step 7: Nessun impatto (solo code organization)
- Step 8: Nessun impatto (solo moving code)

---

## Risorse Utili

### File Chiave da Leggere
- `src/quantization/traits.rs` - Trait già implementato
- `src/quantization/types.rs` - Implementazioni pronte
- `src/solver/mod.rs` - Codice da refactorare (Step 6, 7)
- `src/utility/mod.rs` - Funzioni da spostare (Step 8)

### Pattern da Seguire
- `cfr_algorithms/` - Esempio di buona separazione trait-based
- `game/allocation.rs` - Esempio di split per responsabilità

---

## Conclusione

Il refactoring Fase 1 ha creato una solida foundation. I prossimi step si concentrano su:
1. **Eliminare duplicazione** (Step 6) - massimo impatto
2. **Migliorare modularità** (Step 7) - migliore manutenibilità
3. **Completare separazione logica** (Step 8) - coerenza architetturale

Ogni step è indipendente e può essere fatto separatamente, ma l'ordine raccomandato ottimizza il flusso di lavoro.

**Priorità:** Step 6 > Step 8 > Step 7 > Opzionali
