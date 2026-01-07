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

### ✅ Refactoring Fase 2 - Completato

5. **Step 6: Quantization Traits Integration** ✅
   - `regret_matching_dispatch` ora usa trait-based dispatch
   - Eliminata duplicazione di 6+ funzioni `regret_matching_*`
   - Riduzione ~70 linee di codice

6. **Step 8: Encoding Functions Migration** ✅
   - `quantization/encoding.rs` - 580 linee con tutte le funzioni di encoding
   - `utility/mod.rs` - Ridotto da ~1,466 a 1,012 linee (~31% reduction)
   - Re-exports per backwards compatibility
   - 8 test per encoding functions

7. **Step 7 Parziale: Solver Module Consolidation** ✅
   - `solver/strategy.rs` - 337 linee (strategy calculation + regret_matching_dispatch)
   - `solver/mod.rs` - Ridotto da 1,191 a 1,026 linee
   - Eliminata duplicazione opponent strategy calculation
   - Imports consolidati (compute_pdcfr_plus_strategy, compute_sapcfr_plus_strategy)

---

## Prossimi Step Prioritari

### ✅ Step 6: Integrare Quantization Traits nel Solver - COMPLETATO

**Risultato:** `regret_matching_dispatch` in `solver/strategy.rs` ora usa trait-based dispatch.
- Eliminata duplicazione di 6+ funzioni
- Dispatch basato su `QuantizationMode`
- DCFR+ usa unsigned (Uint8Quant, Uint4PackedQuant)
- Altri algoritmi usano signed (Int8Quant, Int4PackedQuant)

---

### Step 7: Split solver/mod.rs per Responsabilità ⚠️ IN CORSO (Parzialmente Completato)

**Stato attuale:**
- `solver/mod.rs`: 1,026 linee (da 1,191)
- `solver/strategy.rs`: 337 linee (strategy calculation, regret_matching_dispatch)
- `solver/pruning.rs`: 103 linee (pruning logic)

**Obiettivo rimanente:** Estrarre regret update logic in `solver/regrets.rs`.

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

### ✅ Step 8: Migrare Encoding Functions - COMPLETATO

**Risultato:** Tutte le encoding functions migrate in `quantization/encoding.rs`.
- `quantization/encoding.rs`: 580 linee con 8 test
- `utility/mod.rs`: Ridotto da ~1,466 a 1,012 linee (~31%)
- Re-exports per backwards compatibility

**Funzioni migrate:**
- `encode_signed_slice`, `encode_unsigned_slice`, `encode_signed_slice_log` (Int16)
- `encode_signed_i8`, `encode_unsigned_strategy_u8`, `encode_unsigned_regrets_u8` (Int8)
- `encode_signed_i4_packed`, `encode_unsigned_u4_packed` (Int4)
- `decode_signed_i8`, `decode_signed_i4_packed`, `decode_unsigned_u4_packed`
- Helper functions: `fast_xorshift32`, `stochastic_round`, `slice_absolute_max`, `slice_nonnegative_max`

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

### ✅ Fase 2 - COMPLETATA
1. **Step 6** - Integrare quantization traits ✅
2. **Step 8** - Migrare encoding functions ✅

### Fase 3 (In Corso)
3. **Step 7** - Split solver/mod.rs (parzialmente completato, rimane regrets.rs) ⚠️

### Fase 4 (Opzionale)
4. Opzionali A/B/C - Solo se necessario

---

## Metriche di Successo

### ✅ Dopo Step 6 (Quantization Integration) - RAGGIUNTO:
- ✅ Eliminazione di 6+ funzioni `regret_matching_*`
- ✅ `regret_matching_dispatch` usa trait-based dispatch
- ✅ 99/100 test passano (1 failure pre-esistente)

### Dopo Step 7 (Solver Split) - IN CORSO:
- ⚠️ solver/mod.rs: 1,026 linee (obiettivo: < 500)
- ✅ solver/strategy.rs: 337 linee
- ✅ solver/pruning.rs: 103 linee
- ❌ solver/regrets.rs: Da creare

### ✅ Dopo Step 8 (Encoding Migration) - RAGGIUNTO:
- ✅ utility/mod.rs: 1,012 linee (da ~1,466, -31%)
- ✅ quantization/encoding.rs: 580 linee con 8 test
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
