# Analisi delle Opportunità di Ottimizzazione del Solver

Questo documento descrive le aree del codice dove è possibile migliorare le performance del solving.

## Stato Implementazione

| Ottimizzazione | Stato | File Modificati |
|----------------|-------|-----------------|
| Parallelizzazione durante pruning | ✅ IMPLEMENTATA | `src/solver/mod.rs` |
| Loop FMA ottimizzati | ✅ IMPLEMENTATA | `src/sliceop.rs`, `src/sliceop_simd.rs` |
| Fusione operazioni regret updates | ✅ IMPLEMENTATA | `src/solver/regrets.rs` |
| Lock contention cfv_actions | ✅ IMPLEMENTATA | `src/solver/mod.rs`, `src/buffer_pool.rs` |
| Allocazioni memoria hot path | ✅ IMPLEMENTATA | `src/buffer_pool.rs` (thread-local stack, Rust stable) |

---

## 1. Ottimizzazioni Critiche (Massimo Impatto)

### 1.1 Allocazioni di Memoria nel Hot Path

**Posizione**: `src/solver/mod.rs:254-288`

**Problema**: Vengono create nuove allocazioni `Vec` dentro `solve_recursive()` ad ogni visita di nodo:

```rust
let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));
let mut cfreach_updated = Vec::with_capacity(cfreach.len());
let mut result_f64 = Vec::with_capacity(num_hands);
```

**Impatto**: Questo è nel loop più interno, causando un overhead massivo di allocazione. Con milioni di nodi visitati per iterazione, questo diventa un collo di bottiglia significativo.

**Soluzioni proposte**:
- Implementare object pooling o arena allocation thread-local
- Pre-allocare buffer di workspace all'inizio di ogni iterazione
- Utilizzare più aggressivamente lo `StackAlloc` esistente (feature `custom-alloc`)

---

### 1.2 Loop FMA Inefficienti

**Posizione**: `src/sliceop.rs:120-151` (versione scalare), `src/sliceop_simd.rs:107-152` (SIMD)

**Problema**: La versione scalare usa closure annidate con potenziale branch misprediction:

```rust
let len = dst.len();
dst.iter_mut().zip(src1.iter().zip(src2)).for_each(...);
src1[len..].chunks_exact(len).zip(src2[len..].chunks_exact(len)).for_each(...)
```

**Soluzioni proposte**:
- Inline manuale del loop (eliminare overhead delle closure)
- Usare loop esplicito con contatore invece di `chunks_exact().zip()`
- Unrolling 2-4x per migliorare la località della cache

---

### 1.3 Parallelizzazione Disabilitata Durante il Pruning

**Posizione**: `src/solver/mod.rs:322-373`

**Problema**: Quando il pruning è attivo, la parallelizzazione viene completamente disabilitata:

```rust
if pruning_mode != PruningMode::Disabled {
    for action in node.action_indices() {  // SEQUENZIALE!
        let should_skip = should_prune_action_max(...);
        // ...
    }
} else {
    for_each_child(node, |action| {  // PARALLELO
        solve_recursive(...);
    });
}
```

**Impatto**: Rallentamento significativo su alberi di gioco grandi quando il pruning è attivo.

**Soluzioni proposte**:
- Mantenere la parallelizzazione sulle azioni anche con pruning
- Usare check di pruning thread-safe (atomic se necessario)
- Sincronizzare solo dopo che la ricorsione dell'azione è completata

---

## 2. Ottimizzazioni Alte (Impatto Significativo)

### 2.1 Lock Contention su cfv_actions

**Posizione**: `src/solver/mod.rs:275, 291, 343-352, 365, 383, 491, 501`

**Problema**: Multiple chiamate `.lock()` su `cfv_actions` per ogni nodo:

```rust
row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands)
```

**Soluzioni proposte**:
- Ristrutturare per acquisire il lock una volta sola, usare riferimento per operazioni multiple
- Considerare lane-splitting: cfv_actions separati per thread per evitare sincronizzazione

---

### 2.2 Cache Thrashing nei Regret Updates

**Posizione**: `src/solver/regrets.rs:75-104`

**Problema**: Due passate separate sugli stessi dati (scarso riutilizzo cache):

```rust
cum_regret.iter_mut().zip(cfv_actions).for_each(|(x, y)| {
    let coef = if x.is_sign_positive() { alpha } else { beta };
    *x = *x * coef + *y;
});
cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
    sub_slice(row, result);
});
```

**Impatto**: Cache miss L1 per action space grandi.

**Soluzioni proposte**:
- Fondere le due operazioni: applicare alpha/beta E sottrarre in una singola passata
- Processare action-row-wise invece che element-wise

---

### 2.3 SIMD Sottoutilizzato

**Posizione**: `src/sliceop_simd.rs:86-98`

**Problema**: Uso di `for_each` dentro il loop SIMD che impedisce la vettorizzazione del loop esterno:

```rust
src[len..].chunks_exact(len).for_each(|chunk| {
    let mut i = 0;
    while i + 8 <= len {
        // SIMD 8 elementi
    }
    // Gestione coda scalare
});
```

**Soluzioni proposte**:
- Usare loop esplicito per i chunk delle azioni (rimuovere closure)
- Permettere al compilatore di vettorizzare l'iterazione sui chunk
- Potenziale speedup 2-3x con struttura loop corretta

---

## 3. Ottimizzazioni Medie

### 3.1 Double-Lock nel Pruning - ✅ RISOLTO

**Stato**: Automaticamente risolto con l'introduzione di `ConcurrentCfvBuffer` che elimina completamente il lock.

---

### 3.2 Decode Ripetuto nei Regret Updates - ✅ OTTIMIZZATO

**Posizione**: `src/solver/regrets.rs`

**Ottimizzazioni implementate**:
1. **Branch fuori dal loop**: Il check `quantization_mode` è stato spostato fuori dal loop interno
2. **Funzioni helper inline**: `decode_i16_log()` e `decode_i16_linear()` per specializzazione
3. **Single decode pass**: SAPCFR+ ora decodifica una sola volta invece di due
4. **Fused operations**: Decode + compute inst + update cum + compute predicted in un'unica passata

```rust
// Nuovo pattern ottimizzato
match quantization_mode {
    QuantizationMode::Int16Log => {
        for action in 0..num_actions {
            // Loop interno senza branch
            let val_cum = decode_i16_log(r_cum_encoded, decoder);
            // ...
        }
    }
    _ => { /* Linear path */ }
}
```

---

### 3.3 Overhead Passata di Encoding - ✅ OTTIMIZZATO

Ora parte del refactoring di 3.2 - le operazioni sono fuse dove possibile.

---

### 3.4 Operazioni Int4 Packed Manuali - ⏸️ BASSA PRIORITÀ

**Posizione**: `src/solver/regrets.rs:740-765`

Il branch `if i % 2 == 0` nel loop è subottimale ma l'impatto è basso perché:
1. Int4 è usato raramente (solo per casi specifici di compressione estrema)
2. Il predictor di branch gestisce bene pattern alternati

**Possibile ottimizzazione futura**: Processare 2 nibble per iterazione.

---

### 3.5 Copia Strategia + Encoding - ✅ GIÀ OTTIMIZZATO

Il codice attuale non ha il problema descritto - l'encoding avviene direttamente senza copia ridondante.

---

## 4. Ottimizzazioni Basse

### 4.1 Inner Product Implementation

**Posizione**: `src/sliceop.rs:215-248`

La versione scalare usa array accumulatore a 8 elementi con accesso cache-unfriendly nel loop interno.

**Soluzione**: Ristrutturare a column-major, usare 4 accumulatori indipendenti per nascondere latenza.

---

### 4.2 Frequenza Calcolo Exploitability

**Posizione**: `src/solver/mod.rs:177-184`

```rust
if (t + 1) % 10 == 0 {
    exploitability = compute_exploitability(game);
}
```

**Soluzioni proposte**:
- Rendere configurabile (attualmente hardcoded 10)
- Usare sampling adattivo (calcolare meno frequentemente per convergenza lenta)

---

### 4.3 Ricalcolo Threshold Pruning

**Posizione**: `src/solver/mod.rs:324`, `src/solver/pruning.rs:20-24`

**Problema**: Ricalcolato per ogni update del player, anche se stesso risultato.

**Soluzione**: Calcolare una volta per iterazione, passare come parametro.

---

### 4.4 Check AVX2 Ripetuto

**Posizione**: `src/sliceop.rs:10, 22, 34, 69, etc.`

**Problema**: Chiamate ripetute a `sliceop_simd::has_avx2()`.

**Soluzione**: Cache del risultato a livello di modulo o inizio iterazione.

---

### 4.5 Custom Allocator Non Default

**Posizione**: `Cargo.toml:27`, `src/solver/mod.rs:253-256`

Il custom stack allocator è una feature opzionale, non abilitata di default.

**Soluzione**: Considerare di renderla default (con opzione per disabilitare).

---

## 5. Riepilogo e Priorità

| Priorità | Ottimizzazione | File | Speedup Stimato |
|----------|----------------|------|-----------------|
| CRITICA | Allocazioni hot path | solver/mod.rs:254-288 | 15-25% |
| CRITICA | Loop FMA inefficienti | sliceop.rs, sliceop_simd.rs | 10-15% |
| CRITICA | Parallelizzazione pruning | solver/mod.rs:322-373 | 10-20% |
| ALTA | Lock contention | solver/mod.rs | 5-10% |
| ALTA | Cache thrashing regrets | solver/regrets.rs:75-104 | 5-10% |
| ALTA | SIMD sottoutilizzato | sliceop_simd.rs:86-98 | 5-10% |
| MEDIA | Varie | Multipli | 2-5% ciascuna |
| BASSA | Varie | Multipli | <2% ciascuna |

---

## 6. Stima Speedup Totale

| Livello Implementazione | Speedup Atteso |
|-------------------------|----------------|
| Solo ottimizzazioni critiche | 30-50% |
| Critiche + alte | 50-100% (2x) |
| Tutte le ottimizzazioni | 100-200% (2-3x) |

Le fix più impattanti sono nel core loop del solver (allocazione memoria e pattern di lock) e nelle operazioni SIMD FMA, che vengono eseguite miliardi di volte durante il solving.

---

## 7. Analisi Dettagliata Ottimizzazioni PENDING

### 7.1 Allocazioni Memoria Hot Path - ✅ IMPLEMENTATA

**Soluzione implementata**: Thread-local buffer stack in `src/buffer_pool.rs`

La soluzione usa un buffer stack thread-local che elimina le allocazioni heap nel hot path:

```rust
// BufferStack: pre-allocato 4MB per thread
// ConcurrentCfvBuffer::new() prende spazio dallo stack invece di allocare
// ConcurrentCfvBuffer::drop() restituisce lo spazio (LIFO)
```

**Vantaggi**:
- Zero allocazioni heap dopo il warmup iniziale
- Funziona su Rust stable (non richiede `allocator_api`)
- Compatibile con la ricorsione (stack LIFO)
- Nessun overhead di sincronizzazione (thread-local)

**Stato precedente del codice** (`src/solver/mod.rs:252-288`):

```rust
// Allocazione 1: cfv_actions (CRITICA - ogni nodo)
#[cfg(feature = "custom-alloc")]
let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
#[cfg(not(feature = "custom-alloc"))]
let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

// Allocazione 2: cfreach_updated (solo nodi chance)
#[cfg(feature = "custom-alloc")]
let mut cfreach_updated = Vec::with_capacity_in(cfreach.len(), StackAlloc);
#[cfg(not(feature = "custom-alloc"))]
let mut cfreach_updated = Vec::with_capacity(cfreach.len());

// Allocazione 3: result_f64 (solo nodi chance)
#[cfg(feature = "custom-alloc")]
let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
#[cfg(not(feature = "custom-alloc"))]
let mut result_f64 = Vec::with_capacity(num_hands);
```

**Analisi del problema**:

| Allocazione | Frequenza | Dimensione Tipica | Impatto |
|-------------|-----------|-------------------|---------|
| `cfv_actions` | Ogni nodo | `num_actions * num_hands` (es. 6 * 1326 = 7956 floats = ~32KB) | CRITICO |
| `cfreach_updated` | Solo nodi chance | `num_hands` (es. 1326 floats = ~5KB) | MEDIO |
| `result_f64` | Solo nodi chance | `num_hands` (es. 1326 f64 = ~10KB) | MEDIO |

Con un albero di ~10M nodi e 1000 iterazioni, abbiamo:
- ~10 miliardi di allocazioni di `cfv_actions`
- Overhead stimato: 50-100ns per allocazione = 500-1000 secondi di overhead puro

**Soluzione Proposta: Thread-Local Arena Allocator**

```rust
// Nuova struttura da aggiungere in src/solver/mod.rs

use std::cell::RefCell;

/// Buffer di workspace pre-allocato per thread
struct SolverWorkspace {
    // Buffer per cfv_actions - dimensione max stimata
    cfv_buffer: Vec<f32>,
    // Buffer per cfreach_updated
    cfreach_buffer: Vec<f32>,
    // Buffer per result_f64
    result_f64_buffer: Vec<f64>,
}

impl SolverWorkspace {
    fn new(max_actions: usize, max_hands: usize) -> Self {
        Self {
            cfv_buffer: vec![0.0; max_actions * max_hands],
            cfreach_buffer: vec![0.0; max_hands],
            result_f64_buffer: vec![0.0; max_hands],
        }
    }
}

thread_local! {
    static WORKSPACE: RefCell<Option<SolverWorkspace>> = RefCell::new(None);
}

/// Inizializza workspace thread-local (chiamare prima di solve)
pub fn init_thread_workspace(max_actions: usize, max_hands: usize) {
    WORKSPACE.with(|ws| {
        *ws.borrow_mut() = Some(SolverWorkspace::new(max_actions, max_hands));
    });
}

/// Accede al workspace thread-local
fn with_workspace<F, R>(f: F) -> R
where
    F: FnOnce(&mut SolverWorkspace) -> R,
{
    WORKSPACE.with(|ws| {
        let mut ws = ws.borrow_mut();
        let ws = ws.as_mut().expect("Workspace not initialized");
        f(ws)
    })
}
```

**Modifiche a `solve_recursive`**:

```rust
fn solve_recursive(/* params */) {
    // ... codice esistente fino a num_actions/num_hands ...

    // INVECE DI allocare nuovo Vec:
    // let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // USARE il buffer thread-local:
    with_workspace(|ws| {
        let cfv_slice = &mut ws.cfv_buffer[..num_actions * num_hands];
        // Azzera il buffer (SIMD friendly)
        cfv_slice.iter_mut().for_each(|x| *x = 0.0);

        // Usa cfv_slice invece di cfv_actions per le operazioni
        // ...
    });
}
```

**Vantaggi della soluzione**:
- Elimina allocazioni nel hot path
- Zero overhead di sincronizzazione (thread-local)
- Cache-friendly (riutilizzo stesso buffer)
- Compatibile con la parallelizzazione esistente (ogni thread ha il suo workspace)

**Svantaggi/Considerazioni**:
- Richiede refactoring significativo di `solve_recursive`
- Necessita stima accurata di `max_actions` e `max_hands`
- Il pattern ricorsivo complica l'uso di buffer fissi (servono stack di buffer)

**Alternativa: Stack-based Arena per ricorsione**

```rust
/// Arena che cresce come stack per supportare ricorsione
struct RecursiveArena {
    buffer: Vec<f32>,
    stack_pointer: usize,
}

impl RecursiveArena {
    fn push(&mut self, size: usize) -> &mut [f32] {
        let start = self.stack_pointer;
        self.stack_pointer += size;
        &mut self.buffer[start..self.stack_pointer]
    }

    fn pop(&mut self, size: usize) {
        self.stack_pointer -= size;
    }
}
```

---

### 7.2 Lock Contention cfv_actions - ✅ IMPLEMENTATA

**Soluzione implementata**: `ConcurrentCfvBuffer` in `src/buffer_pool.rs`

La soluzione adottata usa `UnsafeCell` per permettere scritture concorrenti su regioni disgiunte del buffer senza overhead di sincronizzazione. Ogni action scrive su una riga diversa, quindi non c'è data race.

```rust
// Nuovo pattern (senza lock)
let cfv_actions = ConcurrentCfvBuffer::new(num_actions, num_hands);
for_each_child(node, |action| {
    solve_recursive(cfv_actions.row_mut(action), ...);
});
```

**Stato precedente del codice**:

Le chiamate a `.lock()` su `cfv_actions` avvengono:

| Linea | Contesto | Frequenza |
|-------|----------|-----------|
| 275 | `for_each_child` (chance node) | Per ogni azione di chance |
| 346 | Branch pruning skip | Per ogni azione prunata |
| 355 | Branch normale (player node) | Per ogni azione non prunata |
| 373 | Dopo il loop (player node) | 1 volta per nodo |
| 481 | `for_each_child` (opponent node) | Per ogni azione opponent |
| 491 | Dopo il loop (opponent node) | 1 volta per nodo |

**Pattern problematico**:

```rust
for_each_child(node, |action| {
    solve_recursive(
        row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
        // ...
    );
});
```

Il problema: `for_each_child` può essere parallelo (rayon), quindi ogni thread compete per lo stesso lock.

**Soluzione Proposta: Pre-allocazione + Write-Once Pattern**

L'idea è separare la fase di allocazione/inizializzazione (sequenziale) dalla fase di scrittura (parallela senza lock):

```rust
// PRIMA: Lock per ogni azione nel loop parallelo
for_each_child(node, |action| {
    solve_recursive(
        row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
        // ...
    );
});

// DOPO: Pre-alloca tutto, poi accesso lock-free
let mut cfv_actions = vec![0.0f32; num_actions * num_hands];

// Crea slice mutabili disgiunti per ogni azione
let action_slices: Vec<&mut [f32]> = cfv_actions
    .chunks_exact_mut(num_hands)
    .collect();

// Iterazione parallela senza lock (ogni thread scrive su slice diverso)
action_slices.into_par_iter().enumerate().for_each(|(action, slice)| {
    solve_recursive_into_slice(
        slice,  // Slice esclusivo, no lock needed
        game,
        &mut node.play(action),
        player,
        cfreach,
        params,
    );
});
```

**Problema con il pattern attuale**: La funzione `solve_recursive` scrive su `MaybeUninit<f32>` slice, non su slice normali. Questo richiede:

1. Modificare la signature di `solve_recursive` per accettare `&mut [f32]` invece di `&mut [MaybeUninit<f32>]`
2. Oppure usare un wrapper che converte slice inizializzati in MaybeUninit

**Soluzione alternativa: Atomic-free parallel writes**

```rust
use std::sync::atomic::{AtomicPtr, Ordering};

// Struttura per permettere writes parallele senza lock
struct ParallelCfvBuffer {
    data: Vec<f32>,
    num_hands: usize,
}

impl ParallelCfvBuffer {
    fn new(num_actions: usize, num_hands: usize) -> Self {
        Self {
            data: vec![0.0; num_actions * num_hands],
            num_hands,
        }
    }

    /// Ritorna un puntatore raw alla riga dell'azione
    /// SAFETY: chiamante deve garantire che ogni azione venga scritta da un solo thread
    unsafe fn row_ptr(&self, action: usize) -> *mut f32 {
        self.data.as_ptr().add(action * self.num_hands) as *mut f32
    }

    fn into_vec(self) -> Vec<f32> {
        self.data
    }
}
```

**Approccio più semplice: Rimuovere il MutexLike**

Dato che `for_each_child` itera su azioni distinte, possiamo sfruttare il fatto che ogni azione scrive su una porzione diversa del buffer:

```rust
// Alloca buffer completo upfront
let mut cfv_actions = vec![MaybeUninit::<f32>::uninit(); num_actions * num_hands];

// Split in slices disgiunti
let cfv_chunks: Vec<_> = cfv_actions.chunks_exact_mut(num_hands).collect();

// Parallel iteration - ogni thread ha il suo slice esclusivo
cfv_chunks.into_par_iter().enumerate().for_each(|(action, chunk)| {
    solve_recursive(chunk, game, &mut node.play(action), player, cfreach, params);
});

// Dopo il loop, tutti i dati sono scritti - possiamo convertire a inizializzato
let cfv_actions: Vec<f32> = unsafe {
    std::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(cfv_actions)
};
```

---

## 8. Piano di Implementazione

### Fase 1: Rimozione Lock Contention (Impatto alto, rischio medio)

1. **Modifica `solve_recursive` signature** per accettare slice pre-allocato
2. **Rimuovere `MutexLike`** wrapper da `cfv_actions`
3. **Pre-allocare buffer** prima di `for_each_child`
4. **Usare `chunks_exact_mut`** per dividere in slice disgiunti
5. **Test**: Verificare correttezza con test esistenti
6. **Benchmark**: Misurare speedup su casi reali

### Fase 2: Thread-Local Workspace (Impatto molto alto, rischio alto)

1. **Implementare `SolverWorkspace`** struct
2. **Aggiungere `init_thread_workspace`** in `solve()`
3. **Refactor `solve_recursive`** per usare workspace
4. **Gestire ricorsione** con stack di offset
5. **Test**: Verificare correttezza con tutti gli algoritmi (DCFR, DCFR+, PDCFR+, SAPCFR+)
6. **Benchmark**: Confronto memoria e performance

### Fase 3: Ottimizzazioni Minori

1. Cache del check AVX2
2. Frequenza exploitability configurabile
3. Pruning threshold pre-calcolato

---

## 9. Benchmark Suggeriti

### Setup benchmark

```bash
# Creare script benchmark standardizzato
./scripts/benchmark.sh --scenario=standard --iterations=1000

# Scenari consigliati:
# 1. Small tree: ~100K nodi, misura overhead allocazione
# 2. Medium tree: ~1M nodi, misura lock contention
# 3. Large tree: ~10M nodi, misura impatto complessivo
```

### Metriche da tracciare

| Metrica | Tool | Target |
|---------|------|--------|
| Tempo totale solving | `std::time::Instant` | Riduzione 30-50% |
| Allocazioni/secondo | Valgrind massif | Riduzione 90%+ |
| Lock wait time | perf/dtrace | Riduzione 80%+ |
| Cache miss rate | perf stat | Riduzione 20-30% |
| Memoria peak | `/usr/bin/time -v` | Invariata o ridotta |

### Confronto A/B

```rust
// Aggiungere feature flag per confronto
#[cfg(feature = "opt-workspace")]
// Usa thread-local workspace

#[cfg(not(feature = "opt-workspace"))]
// Usa allocazioni standard
```

---

## 10. Rischi e Mitigazioni

| Rischio | Probabilità | Impatto | Mitigazione |
|---------|-------------|---------|-------------|
| Bug sottili per race condition | Media | Alto | Test estensivi, sanitizer TSAN |
| Regressione performance su alberi piccoli | Bassa | Medio | Benchmark su multiple dimensioni |
| Incompatibilità con custom-alloc | Media | Medio | Testare entrambe le configurazioni |
| Overflow buffer workspace | Bassa | Alto | Validazione dimensioni upfront |

---

## 11. Conclusioni e Prossimi Passi

### Stato Attuale (Gennaio 2026)

**TUTTE le ottimizzazioni critiche e medie sono state implementate:**

#### Ottimizzazioni Critiche (✅ Complete)
- ✅ Parallelizzazione durante pruning
- ✅ Loop FMA ottimizzati
- ✅ Fusione operazioni regret updates
- ✅ **Lock contention cfv_actions** - `ConcurrentCfvBuffer` con accesso lock-free
- ✅ **Allocazioni memoria hot path** - Thread-local buffer stack (zero heap alloc dopo warmup)

#### Ottimizzazioni Medie (✅ Complete)
- ✅ **Double-Lock nel Pruning** - Risolto con ConcurrentCfvBuffer
- ✅ **Decode Ripetuto nei Regret Updates** - Branch fuori dal loop + single decode pass
- ✅ **Overhead Passata di Encoding** - Fuso con altre operazioni
- ⏸️ **Int4 Packed** - Bassa priorità, impatto minimo

### Impatto Stimato Complessivo

| Ottimizzazione | Speedup Stimato |
|----------------|-----------------|
| Lock contention eliminata | 5-10% |
| Allocazioni hot path eliminate | 15-25% |
| Decode ottimizzato (Int16) | 3-5% |
| **Totale combinato** | **25-40%** |

L'impatto reale dipende dalla dimensione dell'albero di gioco, numero di iterazioni, e se si usa compressione Int16.

### Prossimi Passi

1. **Benchmark**: Eseguire benchmark comparativi per misurare l'impatto reale
2. **Profiling**: Usare `perf` o Instruments (macOS) per identificare nuovi colli di bottiglia
3. **Allocatori alternativi**: Testare jemalloc/mimalloc per ulteriori miglioramenti:
   ```bash
   cargo build --release --features jemalloc
   ```
4. **SIMD avanzato**: Considerare AVX-512 per processori che lo supportano
