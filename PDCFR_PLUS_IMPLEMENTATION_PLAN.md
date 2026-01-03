# Piano di Implementazione: PDCFR+ Algorithm

## Obiettivo
Implementare l'algoritmo PDCFR+ (Predictive Discounted Counterfactual Regret Minimization) come descritto nel paper `PDCFRplus.md`, basandosi sull'implementazione di DCFR esistente.

**NOTA**: Questa implementazione si concentra sulla quantizzazione **Int16** (16-bit). Altri modi di quantizzazione non sono supportati.

## Contesto Tecnico

### Algoritmo PDCFR+
PDCFR+ combina:
- **Discounting di DCFR**: Mitiga l'impatto degli errori nelle prime iterazioni
  - alpha_t = (t-1)^1.5 / ((t-1)^1.5 + 1) per regret positivi
  - beta_t = 0.5 (costante) per regret negativi
- **Meccanismo Predittivo**: Usa predicted regrets per calcolare la strategy (come PCFR+)
  - Richiede storage4 per memorizzare i predicted regrets
  - Strategy derivata da R̃t+1 invece che da Rt

### Formule Chiave (dal paper)
1. **Cumulative Regret Update**:
   ```
   Rt[j] = [Rt-1[j] * ((t-1)^α / ((t-1)^α+1)) + rt[j]]+
   ```
   (Identico a DCFR con conditional discounting per segno)

2. **Predictive Regret Update**:
   ```
   R̃t+1[j] = [Rt[j] * (t^α / (t^α+1)) + vt+1[j]]+
   ```
   Dove vt+1[j] ≈ rt[j] (uso regret istantaneo corrente come predizione)

3. **Strategy**: Calcolata da R̃t (predicted regrets), non da Rt (cumulative regrets)

## Piano di Implementazione

### 1. Creare Implementazione PDCFR+ Algorithm
**File**: `src/cfr_algorithms/algorithms/pdcfr_plus.rs`

Implementare struct PdcfrPlusAlgorithm con:
- Discounting identico a DCFR
- requires_storage4() = true
- Test suite completa

### 2. Registrare PDCFR+ nell'Enum CfrAlgorithmEnum
**File**: `src/cfr_algorithms/algorithm.rs`

Aggiungere variante PDCFRPlus e supporto in from_name() per "pdcfr+", "pdcfrplus"

### 3. Aggiornare Enum CfrAlgorithm nel Solver
**File**: `src/solver.rs`

Aggiungere PDCFRPlus all'enum e ai metodi requires_storage4(), DiscountParams::new()

### 4. Implementare Logica di Regret Update per PDCFR+
**File**: `src/solver.rs` (circa riga 588-634)

Logica:
1. Calcola instantaneous regrets
2. Update cumulative regrets con DCFR conditional discounting + RM+ clipping
3. Calcola predicted regrets = cumulative * next_discount + instantaneous
4. Store predicted regrets in storage4

**Storage mapping**:
- storage2: Cumulative regrets (Rt)
- storage4: Predicted regrets (R̃t+1)

### 5. Implementare Strategy Calculation per PDCFR+
**File**: `src/solver.rs` (riga ~278 e ~639)

Strategy calculation usa predicted regrets da storage4, non cumulative regrets da storage2.

Supporto **solo Int16 quantization mode** (non Float32, non Int16Log).

### 6. Aggiornare Module Exports
**File**: `src/cfr_algorithms/algorithms/mod.rs` e `src/cfr_algorithms/mod.rs`

Aggiungere export di PdcfrPlusAlgorithm.

## File da Modificare (Riepilogo)

1. **NUOVO**: `src/cfr_algorithms/algorithms/pdcfr_plus.rs`
2. **MODIFICA**: `src/cfr_algorithms/algorithms/mod.rs`
3. **MODIFICA**: `src/cfr_algorithms/algorithm.rs`
4. **MODIFICA**: `src/cfr_algorithms/mod.rs`
5. **MODIFICA**: `src/solver.rs`

## Considerazioni Chiave

### Differenze vs DCFR
- **DCFR**: NO storage4, strategy da cumulative regrets
- **PDCFR+**: SI storage4, strategy da predicted regrets

### Differenze vs SAPCFR+
- **SAPCFR+**: alpha=1.0 (no discount), explicit = implicit + 1/3 * prev
- **PDCFR+**: alpha da DCFR (discount), predicted = cumulative * discount + instantaneous

### Storage4 Usage
- **SAPCFR+**: Memorizza "previous instantaneous regrets"
- **PDCFR+**: Memorizza "predicted regrets for next iteration"

## Validazione

Dopo implementazione:
1. ✅ PDCFR+ richiede storage4
2. ✅ Discount params corrispondono a DCFR
3. ✅ Strategy calculation usa predicted regrets (storage4)
4. ✅ Test compilation
5. ✅ Supporto Int16 quantization
