# SIMD Optimization Candidates

Analisi delle opportunità di ottimizzazione SIMD nel codebase del postflop solver.

## Stato Attuale

### SIMD già implementato (`src/sliceop_simd.rs`)

| Funzione | Operazione | Implementazione |
|----------|------------|-----------------|
| `sub_slice_avx2` | `lhs[i] -= rhs[i]` | `_mm256_sub_ps` (8 float/iter) |
| `mul_slice_avx2` | `lhs[i] *= rhs[i]` | `_mm256_mul_ps` (8 float/iter) |
| `sum_slices_uninit_avx2` | Somma riduzione su chunk | `_mm256_add_ps` |
| `fma_slices_uninit_avx2` | `dst += src1[i] * src2[i]` | `_mm256_fmadd_ps` |

Il dispatch avviene tramite `sliceop_simd::has_avx2()` con fallback scalare.

---

## Candidati Priorità 1 (Alto Impatto)

### 1. `div_slice` e `div_slice_uninit`

**File:** `src/sliceop.rs:31-49`

**Codice attuale:**
```rust
pub(crate) fn div_slice(lhs: &mut [f32], rhs: &[f32], default: f32) {
    lhs.iter_mut()
        .zip(rhs)
        .for_each(|(l, r)| *l = if is_zero(*r) { default } else { *l / *r });
}
```

**Perché è critico:**
- Usato in regret matching e normalizzazione strategia
- Chiamato con migliaia di elementi
- 100% scalare nonostante sia vettorizzabile

**Strategia SIMD:**
- `_mm256_div_ps` per la divisione
- `_mm256_cmp_ps` per creare maschera di confronto con zero
- `_mm256_blendv_ps` per selezionare tra risultato divisione e default

**Speedup stimato:** 4-6x

---

### 2. `max_fma_slices_uninit`

**File:** `src/sliceop.rs:148-179`

**Codice attuale:**
```rust
pub(crate) fn max_fma_slices_uninit<'a>(
    dst: &'a mut [MaybeUninit<f32>],
    src1: &[f32],
    src2: &[f32],
) -> &'a mut [f32] {
    // if s2.is_sign_positive() { *s1 * *s2 } else { *s1 }
}
```

**Perché è critico:**
- Operazione FMA condizionale complessa
- Usato negli algoritmi CFR per azioni attive/pruned
- Logica condizionale blocca la vettorizzazione automatica

**Strategia SIMD:**
- `_mm256_cmp_ps(s2_vec, zero, _CMP_GE_OS)` per confronto segno
- `_mm256_fmadd_ps` per casi positivi
- `_mm256_blendv_ps` per blend dei risultati

**Speedup stimato:** 3-5x

---

### 3. `inner_product`

**File:** `src/sliceop.rs:182-208`

**Codice attuale:**
```rust
pub(crate) fn inner_product(src1: &[f32], src2: &[f32]) -> f32 {
    // Chunking manuale in CHUNK_SIZE=8 con accumulatori
    // Usa f64 per stabilità numerica
    // NON usa intrinsics SIMD reali
}
```

**Perché è importante:**
- Già ha chunking manuale SIMD-like
- Non usa istruzioni SIMD effettive
- Usato nella valutazione strategia

**Strategia SIMD:**
- `_mm256_mul_ps` per moltiplicazione vettoriale
- Riduzione orizzontale con cascade di shuffle e add
- Mantenere accumulazione f64 per stabilità

**Speedup stimato:** 2-4x

---

## Candidati Priorità 2 (Impatto Medio)

### 4. `inner_product_cond`

**File:** `src/sliceop.rs:211-268`

**Codice attuale:**
```rust
pub(crate) fn inner_product_cond(
    src1: &[f32],
    src2: &[f32],
    cond: &[u16],
    threshold: u16,
    less: f32,
    greater: f32,
    equal: f32,
) -> f32 {
    // Commento nel codice: "'match' prevents vectorization"
}
```

**Perché è importante:**
- Inner product condizionale con branching a tre vie
- Il commento ammette esplicitamente che il `match` blocca la vettorizzazione
- Usato per logica di compressione mani

**Strategia SIMD:**
- `_mm256_cvtepu16_epi32` per convertire condizioni u16 a i32
- Creare tre maschere per `<threshold`, `>threshold`, `==threshold`
- Moltiplicazioni mascherate e blend

**Speedup stimato:** 2-3x

---

### 5. `mul_slice_scalar_uninit`

**File:** `src/sliceop.rs:52-55`

**Codice attuale:**
```rust
pub(crate) fn mul_slice_scalar_uninit(dst: &mut [MaybeUninit<f32>], src: &[f32], scalar: f32) {
    dst.iter_mut().zip(src).for_each(|(d, s)| {
        d.write(*s * scalar);
    });
}
```

**Perché è importante:**
- Operazione semplice ma frequente
- Basso sforzo di implementazione

**Strategia SIMD:**
- `_mm256_set1_ps(scalar)` per broadcast
- `_mm256_mul_ps` per moltiplicazione

**Speedup stimato:** 3-4x

---

### 6. `max_slices_uninit`

**File:** `src/sliceop.rs:133-145`

**Codice attuale:**
```rust
pub(crate) fn max_slices_uninit<'a>(dst: &'a mut [MaybeUninit<f32>], src: &[f32]) -> &'a mut [f32] {
    // Max element-wise di array multipli
}
```

**Strategia SIMD:**
- `_mm256_max_ps` per-element sui chunk

**Speedup stimato:** 3-4x

---

## Candidati Priorità 3 (Altre Aree)

### 7. Quantization Encoding

**File:** `src/quantization/encoding.rs:92-107, 156-170`

**Funzioni:**
- `slice_absolute_max()` - ha WASM SIMD ma usa loop manuale su x86
- `slice_nonnegative_max()` - stesso problema

**Stato:** Loop unrolling manuale con array di 8 elementi, non usa AVX2.

**Strategia SIMD:**
- Usare `_mm256_max_ps` con riduzione orizzontale

---

### 8. Pruning Threshold

**File:** `src/solver/pruning.rs:41-85`

**Codice attuale:**
```rust
let max_regret = action_regrets.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
```

**Strategia SIMD:**
- `_mm256_max_ps` con riduzione orizzontale per trovare il massimo

---

### 9. Strategy Update Loop

**File:** `src/solver/mod.rs:393-405`

**Codice attuale:**
```rust
strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
    *x += *y * params.gamma_t;
});
```

**Strategia SIMD:**
- Creare funzione `mul_add_slice` con `_mm256_fmadd_ps`

---

## Riepilogo Copertura

| Operazione | Stato | Priorità |
|------------|-------|----------|
| `sub_slice` | ✅ Vettorizzato | - |
| `mul_slice` | ✅ Vettorizzato | - |
| `sum_slices` | ✅ Vettorizzato | - |
| `fma_slices` | ✅ Vettorizzato | - |
| `div_slice` | ❌ Scalare | **P1** |
| `div_slice_uninit` | ❌ Scalare | **P1** |
| `max_fma_slices_uninit` | ❌ Scalare | **P1** |
| `inner_product` | ⚠️ Chunking manuale | **P1** |
| `inner_product_cond` | ❌ Scalare | **P2** |
| `mul_slice_scalar_uninit` | ❌ Scalare | **P2** |
| `max_slices_uninit` | ❌ Scalare | **P2** |
| Quantization max | ⚠️ Loop manuale | **P3** |
| Pruning max | ❌ Scalare | **P3** |
| Strategy update | ❌ Scalare | **P3** |

## Impatto Stimato Complessivo

- **Priorità 1 completata:** 15-20% miglioramento solver
- **Priorità 1+2 completata:** 20-25% miglioramento solver
- **Tutte le priorità:** 25-30% miglioramento solver

## Pattern di Implementazione

Seguire il pattern esistente in `sliceop_simd.rs`:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn operation_name_avx2(/* params */) {
    // Implementazione con _mm256_* intrinsics
}
```

Wrapper in `sliceop.rs`:
```rust
#[inline]
pub(crate) fn operation_name(/* params */) {
    #[cfg(target_arch = "x86_64")]
    {
        if sliceop_simd::has_avx2() && data.len() >= 8 {
            return unsafe { sliceop_simd::operation_name_avx2(/* params */) };
        }
    }
    // Fallback scalare
}
```