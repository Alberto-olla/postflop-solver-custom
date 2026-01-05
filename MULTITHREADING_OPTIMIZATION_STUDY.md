# Studio delle Ottimizzazioni Multithread per DCFR/DCFR+

Questo documento analizza la fattibilità e i benefici dell'applicazione di tecniche di multithreading avanzate e ottimizzazioni di basso livello per massimizzare l'uso della CPU negli algoritmi **DCFR** e **DCFR+**.

## Stato Attuale della Parallelizzazione

Attualmente, il solver utilizza la libreria `rayon` in `src/utility.rs` tramite la funzione `for_each_child`. 
- **Livello di Parallelizzazione**: A livello di "azione" (branch) nei nodi del gioco.
- **Limitazioni**:
    - **Granularità**: Se un nodo ha poche azioni (es. 2-3), solo 2-3 thread possono lavorare su quel sotto-albero, lasciando gli altri core inattivi se i rami sono sbilanciati.
    - **Overhead**: La creazione di task ricorsivi per ogni nodo può introdurre un overhead significativo se il lavoro per nodo è troppo piccolo.
    - **Efficienza SIMD**: Le operazioni sui vettori (hands/buckets) sono scritte in Rust idiomatico. Sebbene il compilatore possa auto-vettorizzare, non c'è un uso esplicito di AVX2/AVX-512.

## Proposte di Ottimizzazione

Per "sovraccaricare" correttamente la CPU e sfruttare ogni ciclo disponibile, si propongono i seguenti interventi:

### 1. Parallelizzazione Multi-Livello (Hybrid Parallelism)
Invece di parallelizzare solo le azioni, dovremmo parallelizzare anche l'elaborazione delle mani (hands) all'interno di ogni operazione sui vettori.
- **Azione-livello**: Continua a usare work-stealing (Rayon) per i rami grandi.
- **Mano-livello**: Per nodi con molte mani (es. > 1024), parallelizzare le operazioni di `fma_slices`, `regret_matching` e `encode/decode`.

### 2. Soglia di Parallelizzazione Adattiva
Per evitare l'overhead dei thread in nodi vicini alle foglie (terminal nodes), implementare una soglia basata sulla profondità o sul numero di nodi stimati nel sotto-albero. Solo sopra questa soglia si attiva il fork-join di Rayon.

### 3. Vettorizzazione SIMD Esplicita (AVX2 / AVX-512)
Le operazioni in `src/sliceop.rs` sono il cuore pulsante del calcolo:
- Implementare versioni AVX2/AVX-512 per `sub_slice`, `fma_slices`, e `sum_slices`.
- **Ottimizzazione Quantizzazione**: Sfruttare istruzioni come `_mm256_maddubs_epi16` per eseguire calcoli direttamente su dati a 8 bit (Int8), riducendo i trasferimenti di memoria e aumentando il throughput.

### 4. Gestione Avanzata del Carico (CPU Overloading)
Per spingere la CPU al limite senza causare instabilità di sistema:
- **Thread Pinning (Affinity)**: Vincolare i thread ai core fisici per minimizzare i context switch e migliorare la cache locality (L1/L2).
- **NUMA-Awareness**: Su CPU con più socket (es. Threadripper, EPYC), assicurarsi che la memoria sia allocata vicino al core che la processa per evitare colli di bottiglia sul bus Infinity Fabric.
- **Controllo Dinamico del Concurrency**: Permettere all'utente di definire un "Overload Factor" (es. 1.5x rispetto ai core fisici) per sfruttare meglio l'Hyper-Threading in scenari I/O-bound (anche se CFR è tipicamente Compute-bound).

## Esempi Tecnici di Implementazione

### Esempio SIMD AVX2 per `sub_slice`
Attualmente Rust auto-vettorizza, ma un'implementazione esplicita garantisce l'uso delle istruzioni più efficienti:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub(crate) unsafe fn sub_slice_avx2(lhs: &mut [f32], rhs: &[f32]) {
    let mut i = 0;
    while i + 8 <= lhs.len() {
        let l_vec = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let r_vec = _mm256_loadu_ps(rhs.as_ptr().add(i));
        let res = _mm256_sub_ps(l_vec, r_vec);
        _mm256_storeu_ps(lhs.as_mut_ptr().add(i), res);
        i += 8;
    }
    // Gestione rimanente...
    for j in i..lhs.len() {
        *lhs.get_unchecked_mut(j) -= *rhs.get_unchecked(j);
    }
}
```

### Esempio Parallelizzazione Granulare in `solve_recursive`
Modifica a `src/solver.rs` per parallelizzare sotto-alberi solo se "pesanti":

```rust
// In src/solver.rs
fn solve_recursive(...) {
    // ...
    if node.is_complex_subtree() {
        // Usa Rayon per parallelizzare le azioni
        node.action_indices().into_par_iter().for_each(|action| {
            // ...
        });
    } else {
        // Elaborazione sequenziale veloce per ridurre l'overhead
        for action in node.action_indices() {
            // ...
        }
    }
}
```

## Analisi Specifica per DCFR+

L'algoritmo **DCFR+** beneficia maggiormente di queste ottimizzazioni poiché:
1. **Regret Matching+ (RM+)**: La rimozione dei rimpianti negativi è un'operazione di `max(0, x)` che si presta perfettamente alla vettorizzazione (istruzioni `MAXPS` o `PMAXSD`).
2. **Cumulative Calculations**: DCFR+ aggiorna continuamente le strategie medie, operazione che richiede molti accessi in memoria. L'ottimizzazione del bandwidth tramite SIMD e NUMA è critica.

### 5. Controllo del Sovraccarico (Overload Control)
Per gestire il "sovraccarico" desiderato:
- **Rayon ThreadPool**: Configurare dinamicamente il numero di thread in base alla fase dell'algoritmo (es. più thread durante la fase di risoluzione del full tree, meno durante il calcolo dell'exploitability se non è critica).
- **Work Stealing Tweaks**: Regolare `RAYON_NUM_THREADS` e usare `scope` per task a bassa priorità.

## Conclusione

L'applicazione di queste ottimizzazioni può portare a incrementi di velocità stimati tra il **30% e il 150%**, a seconda dell'architettura hardware e della topologia dell'albero di gioco. La combinazione di SIMD esplicito per il calcolo vettoriale e parallelizzazione ibrida (azioni + mani) permetterà di saturare completamente la CPU, riducendo drasticamente i tempi di convergenza per DCFR e DCFR+.

---
*Studio redatto per il progetto postflop-solver-custom.*
