# Performance Regression Tests

Test nativi Rust per verificare che le performance dei tre algoritmi CFR (DCFR, DCFR+, SAPCFR+) non peggiorino durante lo sviluppo.

## Quick Start

```bash
# Esegui tutti i test di performance (raccomandato in release mode)
cargo test --release --test performance_regression -- --nocapture

# Esegui un singolo test
cargo test --release --test performance_regression test_performance_dcfr_16bit -- --nocapture

# Esegui il test comparativo (tutti e 3 gli algoritmi)
cargo test --release --test performance_regression --ignored -- --nocapture
```

## Test Disponibili

### 1. `test_performance_dcfr_16bit_node03_turn`
- **Algoritmo**: DCFR (dual discount factors)
- **Baseline**: 160 iterazioni, 4.42s
- **Tolleranza**: +5% sul tempo (miglioramenti sempre accettati ✅)

### 2. `test_performance_dcfrplus_16bit_node03_turn`
- **Algoritmo**: DCFR+ (single discount + clipping)
- **Baseline**: 290 iterazioni, 7.56s
- **Tolleranza**: +5% sul tempo (miglioramenti sempre accettati ✅)

### 3. `test_performance_sapcfrplus_16bit_node03_turn`
- **Algoritmo**: SAPCFR+ (asymmetric predictive CFR+)
- **Baseline**: 240 iterazioni, 8.77s
- **Tolleranza**: +5% sul tempo (miglioramenti sempre accettati ✅)

### 4. `test_compare_all_algorithms_node03_turn` (ignorato di default)
- Esegue tutti e 3 gli algoritmi e mostra una tabella comparativa
- Utile per verificare miglioramenti relativi

## Configurazione

I test usano la configurazione esatta da:
```
hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml
```

Questo garantisce consistenza con i risultati baseline.

## Interpretazione dei Risultati

### ✅ Test Passato
```
=== DCFR 16-bit Performance ===
Baseline: 160 iterations, 4.42s
Current:  Time 4.35s
Final exploitability: 17.057434 (target: 19.5)
```

### ❌ Performance Regression
```
=== DCFR 16-bit Performance ===
Baseline: 160 iterations, 4.42s
Current:  Time 5.80s
Final exploitability: 17.234567 (target: 19.5)

thread panicked: Performance regression! Time increased from 4.42s to 5.80s (31.2% increase)
```

### 🎉 Performance Migliorata
```
=== DCFR 16-bit Performance ===
Baseline: 160 iterations, 4.42s
Current:  Time 3.95s
Final exploitability: 16.543210 (target: 19.5)
✓ Performance IMPROVED! Time reduced by 0.47s (10.6%)
```

## Quando Eseguirli

- ✅ Prima di commit importanti
- ✅ Dopo modifiche agli algoritmi CFR
- ✅ Dopo ottimizzazioni delle performance
- ✅ Prima di merge in main
- ✅ In CI/CD pipeline

## Quando Aggiornare i Baseline

Aggiorna i baseline SOLO quando:
- ✅ Hai implementato un miglioramento documentato
- ✅ Cambi intenzionalmente il comportamento dell'algoritmo
- ✅ Le performance sono consistentemente migliori su più esecuzioni

NON aggiornare per:
- ❌ Far passare i test quando falliscono
- ❌ Variazioni casuali o noise
- ❌ Performance peggiorate senza spiegazione

## Note Tecniche

- I test usano precisione **16-bit** per tutte le misurazioni
- Il target exploitability è calcolato come `0.5% del pot = 19.5`
- La funzione `solve()` stampa l'output del progresso se eseguita con `--nocapture`
- I test caricano direttamente il file TOML per garantire consistenza

## Troubleshooting

### Il test è molto più veloce del baseline
Probabilmente stai eseguendo su hardware più veloce o con ottimizzazioni diverse. I baseline sono stati misurati su hardware specifico. Considera di aggiornare i baseline per il tuo hardware.

### Il test fallisce randomicamente
Aggiungi un margine di tolleranza maggiore o esegui il test 3 volte per verificare la consistenza:
```bash
for i in {1..3}; do
  cargo test --release test_performance_dcfr_16bit -- --nocapture
done
```

### Voglio tracciare anche le iterazioni
La funzione `solve()` attualmente non restituisce il numero di iterazioni, solo l'exploitability finale. Per tracciare le iterazioni, esegui con `--nocapture` e leggi l'output stampato.

## Vedi Anche

- [PERFORMANCE_REGRESSION_TESTS.md](../PERFORMANCE_REGRESSION_TESTS.md) - Report dettagliato con esempi di codice
- [examples/solve_from_config.rs](../examples/solve_from_config.rs) - Esempio di risoluzione da file TOML