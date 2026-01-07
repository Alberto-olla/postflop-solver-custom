# Performance Benchmark Guide

Sistema per tracciare le performance del solver e rilevare regressioni/miglioramenti.

## Quick Start

```bash
# Dopo ogni modifica al codice:
./benchmark.sh
./compare_benchmarks.sh benchmarks/BASELINE_LARGE.json benchmarks/baseline_*.json
```

## Baseline

Due baseline disponibili per scenari diversi:

### LARGE (default) - Albero complesso (~17s)

| Metrica | Valore |
|---------|--------|
| Config | `hand_0000007438_node_03_turn_DeepStack.toml` |
| IP Range | `22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32` |
| Mean | **17.589s** |
| Stddev | 0.423s (±2.4%) |
| Min/Max | 17.124s / 17.951s |
| File | `benchmarks/BASELINE_LARGE.json` |
| Data | 2025-01-07 |

### SMALL - Albero ridotto (~1.5s)

| Metrica | Valore |
|---------|--------|
| Config | `hand_0000007438_node_03_turn_DeepStack.toml` (con IP range ridotto) |
| IP Range | Range specifico con poche combo |
| Mean | **1.582s** |
| Stddev | 0.031s (±2%) |
| Min/Max | 1.537s / 1.633s |
| File | `benchmarks/BASELINE.json` |
| Data | 2025-01-07 |

### Switchare tra SMALL e LARGE

Nel file `hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml`:

```toml
# Per LARGE (default attuale):
ip = "22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32"
#ip = "AJs:0.00,QTs:0.00,..."  # commentato

# Per SMALL (swap i commenti):
#ip = "22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32"
ip = "AJs:0.00,QTs:0.00,..."  # decommentato
```

## Comandi

### 1. Eseguire un benchmark

```bash
./benchmark.sh [config] [runs] [nome]
```

**Esempi:**
```bash
# Default: config turn, 10 runs (ok per SMALL, lungo per LARGE)
./benchmark.sh

# LARGE: 5 runs raccomandati (~1.5 min)
./benchmark.sh hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml 5

# Con nome personalizzato
./benchmark.sh hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml 5 dopo_ottimizzazione
```

### 2. Confrontare con baseline

```bash
# Con LARGE (default):
./compare_benchmarks.sh benchmarks/BASELINE_LARGE.json benchmarks/<nuovo>.json

# Con SMALL:
./compare_benchmarks.sh benchmarks/BASELINE.json benchmarks/<nuovo>.json
```

**Output esempio:**
```
Baseline: 17.589s (±0.423s)
New:      15.234s (±0.312s)

Result: ✅ FASTER by 2.355s (-13.4%)
Confidence: HIGH (diff > 2σ)
```

## Workflow Sviluppo

### Dopo la modifica
```bash
# 1. Esegui benchmark (5 runs per LARGE, ~1.5 min)
./benchmark.sh hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml 5 post_modifica

# 2. Confronta con baseline
./compare_benchmarks.sh benchmarks/BASELINE_LARGE.json benchmarks/post_modifica_*.json
```

### Interpretare i risultati

| Risultato | Significato |
|-----------|-------------|
| ✅ FASTER + HIGH confidence | Miglioramento reale |
| ✅ FASTER + LOW confidence | Probabilmente rumore, non significativo |
| ❌ SLOWER + HIGH confidence | **Regressione!** Investigare |
| ❌ SLOWER + LOW confidence | Probabilmente rumore |

**Regola pratica:** differenze < 5% con LOW confidence sono rumore.

## Quante iterazioni usare?

### Con LARGE (~17s per run)
| Runs | Tempo totale | Quando usare |
|------|--------------|--------------|
| 3 | ~1 min | Quick check durante sviluppo |
| 5 | ~1.5 min | Default raccomandato |
| 10 | ~3 min | Conferma differenze piccole |

### Con SMALL (~1.5s per run)
| Runs | Tempo totale | Quando usare |
|------|--------------|--------------|
| 10 | ~20s | Default |
| 20 | ~35s | Conferma differenze piccole (3-5%) |
| 50 | ~90s | Misurazioni molto precise |

## Struttura file

```
benchmarks/
├── BASELINE_LARGE.json   # Riferimento permanente - albero grande (~17s)
├── BASELINE.json         # Riferimento permanente - albero piccolo (~1.5s)
├── *.json                # Benchmark con timestamp
└── *.md                  # Report markdown
```

## Aggiornare la baseline

Quando hai un miglioramento confermato e vuoi aggiornare la baseline:

```bash
# Per LARGE:
cp benchmarks/<nuovo_benchmark>.json benchmarks/BASELINE_LARGE.json

# Per SMALL:
cp benchmarks/<nuovo_benchmark>.json benchmarks/BASELINE.json
```

**Aggiorna anche questa documentazione** con i nuovi valori nella tabella corrispondente!

## Requisiti

- `hyperfine` (installato via `cargo install hyperfine`)
- `jq` per il confronto (opzionale, `brew install jq`)

## Note

- Il benchmark compila automaticamente in release prima di eseguire
- Include 1 warmup run (non contato nelle statistiche)
- I risultati JSON includono anche memory usage e exit codes
