# Piano di Ottimizzazione - Profiling Analysis

## Risultati Profiling (30s sample, 281 iterazioni, ~20.85s solving)

```
Funzione                    | % Tempo | Note
----------------------------|---------|---------------------------
solve_recursive             | ~55%    | Ricorsione + Rayon overhead
evaluate_internal           | ~25%    | Valutazione showdown/fold
regret_matching             | ~10%    | Calcolo strategia
compute_exploitability      | ~5%     | Convergenza check
I/O                         | ~5%     | Serializzazione + ZSTD
```

---

## 1. Ottimizzazione `evaluate_internal`

**File:** `src/game/evaluation.rs:15-253`

### Problema Identificato

Ad ogni chiamata di `evaluate_internal`, vengono allocati sullo stack e inizializzati array di 52 elementi:

```rust
// Linea 32 - sempre allocato
let mut cfreach_minus = [0.0; 52];

// Linee 127 - reset in mezzo al 2-pass
cfreach_minus.fill(0.0);

// Linee 184-185 - caso raked (3-pass), due array aggiuntivi
let mut cfreach_minus_win = [0.0; 52];
let mut cfreach_minus_tie = [0.0; 52];
```

Questi array sono piccoli (52 * 8 = 416 bytes ciascuno), ma:
- Vengono inizializzati/azzerati migliaia di volte per iterazione
- Il `.fill(0.0)` ha un costo non trascurabile in loop tight
- Nel caso raked, si copiano interi array (linee 198, 214)

### Proposta: Thread-Local Buffer Pool

Creare un buffer pool thread-local per riutilizzare questi array:

```rust
// src/solver/buffer_pool.rs (nuovo file)
use std::cell::RefCell;

thread_local! {
    static EVAL_BUFFERS: RefCell<EvalBuffers> = RefCell::new(EvalBuffers::new());
}

pub struct EvalBuffers {
    pub cfreach_minus: [f64; 52],
    pub cfreach_minus_win: [f64; 52],
    pub cfreach_minus_tie: [f64; 52],
}

impl EvalBuffers {
    pub fn new() -> Self {
        Self {
            cfreach_minus: [0.0; 52],
            cfreach_minus_win: [0.0; 52],
            cfreach_minus_tie: [0.0; 52],
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.cfreach_minus.fill(0.0);
        // win/tie resettati solo se usati
    }

    #[inline]
    pub fn reset_all(&mut self) {
        self.cfreach_minus.fill(0.0);
        self.cfreach_minus_win.fill(0.0);
        self.cfreach_minus_tie.fill(0.0);
    }
}

pub fn with_eval_buffers<F, R>(f: F) -> R
where
    F: FnOnce(&mut EvalBuffers) -> R,
{
    EVAL_BUFFERS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();
        buffers.reset();
        f(&mut buffers)
    })
}
```

### Modifica a `evaluate_internal`

```rust
pub(super) fn evaluate_internal(
    &self,
    result: &mut [MaybeUninit<f32>],
    node: &PostFlopNode,
    player: usize,
    cfreach: &[f32],
) {
    crate::solver::buffer_pool::with_eval_buffers(|buffers| {
        self.evaluate_internal_with_buffers(result, node, player, cfreach, buffers)
    })
}

fn evaluate_internal_with_buffers(
    &self,
    result: &mut [MaybeUninit<f32>],
    node: &PostFlopNode,
    player: usize,
    cfreach: &[f32],
    buffers: &mut EvalBuffers,
) {
    // ... usa buffers.cfreach_minus invece di stack allocation
}
```

### Benefici Attesi
- Elimina allocazione stack ripetuta
- Riduce inizializzazioni a zero
- Cache più calda (stesso indirizzo memoria)
- Stima: **5-10% speedup** su evaluate_internal

---

## 2. Ottimizzazione `regret_matching`

**File:** `src/quantization/types.rs`

### Problema Identificato

Per i tipi compressi (`Int16Quant`, `Int8Quant`, `Int4PackedQuant`), il pattern attuale:

```rust
fn regret_matching(data: &[i16], scale: f32, num_actions: usize) -> Vec<f32> {
    // STEP 1: Decodifica completa in Vec temporaneo
    let decoded = {
        let mut result = vec![0.0; data.len()];  // ALLOCAZIONE
        Self::decode_slice(data, scale, &mut result);
        result
    };
    // STEP 2: Chiama Float32Quant::regret_matching
    Float32Quant::regret_matching(&decoded, 1.0, num_actions)  // ALTRA ALLOCAZIONE
}
```

Problemi:
1. **Doppia allocazione**: `decoded` + `strategy` dentro Float32Quant
2. **Doppio loop**: decode loop + regret matching loop
3. **Cache miss**: dati attraversati due volte

### Proposta: Fused Decode + Regret Matching

Implementare regret matching direttamente sui dati compressi:

```rust
impl QuantizationType for Int16Quant {
    fn regret_matching(data: &[i16], scale: f32, num_actions: usize) -> Vec<f32> {
        use crate::sliceop::*;

        let decode_factor = scale / i16::MAX as f32;
        let row_size = data.len() / num_actions;
        let default = 1.0 / num_actions as f32;

        // Allocazione singola per output
        let mut strategy = Vec::with_capacity(data.len());

        // Decode + max(0) fused in un singolo loop
        let uninit = strategy.spare_capacity_mut();
        uninit.iter_mut().zip(data).for_each(|(s, &r)| {
            let decoded = r as f32 * decode_factor;
            s.write(decoded.max(0.0));
        });
        unsafe { strategy.set_len(data.len()) };

        // Calcolo denominatore (somma per riga)
        let mut denom = Vec::with_capacity(row_size);
        sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
        unsafe { denom.set_len(row_size) };

        // Normalizzazione
        strategy.chunks_exact_mut(row_size).for_each(|row| {
            div_slice(row, &denom, default);
        });

        strategy
    }
}
```

### Variante SIMD (opzionale, richiede nightly o portable-simd)

Per Int16, possiamo usare SIMD per decodificare 8 valori alla volta:

```rust
#[cfg(target_arch = "aarch64")]
fn regret_matching_simd(data: &[i16], scale: f32, num_actions: usize) -> Vec<f32> {
    use std::arch::aarch64::*;
    // ... implementazione SIMD per ARM NEON
}
```

### Benefici Attesi
- Elimina allocazione temporanea `decoded`
- Single-pass su dati compressi
- Migliore cache locality
- Stima: **15-25% speedup** su regret_matching per tipi compressi

---

## 3. Piano di Implementazione

### Fase 1: Buffer Pool (basso rischio)
1. [ ] Creare `src/solver/buffer_pool.rs`
2. [ ] Implementare `EvalBuffers` con thread-local storage
3. [ ] Modificare `evaluate_internal` per usare buffer pool
4. [ ] Benchmark prima/dopo

### Fase 2: Fused Regret Matching (medio rischio)
1. [ ] Implementare versione fused per `Int16Quant`
2. [ ] Verificare correttezza con test esistenti
3. [ ] Estendere a `Int8Quant` e tipi packed
4. [ ] Benchmark prima/dopo

### Fase 3: Validazione
1. [ ] Test di correttezza: confrontare output con versione originale
2. [ ] Benchmark su configurazioni diverse (16-bit, 32-bit, miste)
3. [ ] Profilare nuovamente per verificare miglioramenti

---

## 4. Note Tecniche

### Thread Safety
- `EvalBuffers` usa `thread_local!` quindi ogni worker Rayon ha il suo buffer
- Nessun lock necessario, zero contention

### Compatibilita
- Buffer pool richiede solo Rust stable
- SIMD opzionale richiede feature flag

### Rischi
- Buffer pool: minimo (fallback a stack allocation sempre possibile)
- Fused regret: medio (richiede test accurati per precisione numerica)

---

## 5. Metriche di Successo

| Metrica | Attuale | Target |
|---------|---------|--------|
| Tempo solving (hand_7438) | 20.85s | < 18s |
| evaluate_internal % | ~25% | < 20% |
| regret_matching % | ~10% | < 7% |