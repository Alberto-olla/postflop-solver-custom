# Hand 7438

Questa cartella contiene le configurazioni e i risultati per la partita #7438.

## Struttura

```
7438/
├── configs/           File di configurazione (.toml)
│   ├── flop.toml     Analisi dal flop
│   ├── turn.toml     Analisi dal turn
│   └── river.toml    Analisi dal river
└── solved_games/     File risolti (.bin)
```

## Nomenclatura suggerita per i config

- `flop.toml` - Analisi completa dal flop
- `turn.toml` - Analisi dal turn
- `river.toml` - Analisi dal river
- `river_after_bet.toml` - Nodo specifico (dopo una bet al river)
- etc.

## Quick Commands

### Dry Run Node 01
```bash
# Ricompila e calcola dimensione albero (senza risolvere)
cargo build --release --example solve_from_config && \
./target/release/examples/solve_from_config hands/7438/configs/hand_0000007438_node_01_flop_DeepStack.toml
```

### Interactive Explorer
```bash
# Esplora un file risolto in modo interattivo
cargo build --release --example explore && \
./target/release/examples/explore hands/7438/solved_games/hand_0000007438_node_06_river_DeepStack-32-old.bin
```

**Comandi disponibili:**
- `actions` (o `a`) - Mostra azioni disponibili
- `strategy` (o `s`) - Mostra strategia GTO con percentuali
- `play <n>` (o `p <n>`) - Esegui azione n (0-indexed)
- `back` (o `b`) - Torna indietro di una mossa
- `info` (o `i`) - Mostra stato corrente (giocatore, profondità)
- `help` (o `h`) - Mostra aiuto
- `quit` (o `q`) - Esci

**Esempio sessione:**
```
explore> info          # OOP (player 0), depth 0
explore> actions       # [0] Check, [1] AllIn
explore> play 0        # OOP checks
explore> actions       # [0] Check, [1] Bet(5400), [2] AllIn
explore> play 1        # IP bets 5400
explore> strategy      # AhQh: Fold 10%, Call 60%, AllIn 30%
explore> back          # Torna a IP
explore> back          # Torna a OOP root
```

## Come usare

```bash
# Esegui un config specifico
cargo run --release --example solve_from_config -- hands/7438/configs/flop.toml

# I risultati verranno salvati in hands/7438/solved_games/
```

## Note

Ricorda di configurare nel file .toml:
```toml
[output]
directory = "hands/7438/solved_games"
```

## Tree Size Optimization Progress (Node 01)

| MEDIUM Preset Config | 32-bit | 16-bit | Status |
|---------------------|--------|--------|--------|
| flop: 2+2, turn: 2+2, river: 2+2 | 178.91 GB | 92.21 GB | Troppo grande |
| flop: 2+2, turn: 2+2, river: 1+1 | 167.54 GB | 86.35 GB | Ancora troppo |
| flop: 2+2, turn: 1+1, river: 1+1 | 61.70 GB | 31.81 GB | Migliorato ma alto |
| flop: 1+1, turn: 1+1, river: 1+1 | ~9 GB | ~4.6 GB | Target ragionevole |

Target: sotto 10 GB per permettere solving su macchine standard

Memory usage comparison:
32-bit float:    24.54 GB (25130.4 MB) (baseline)
16-bit integer:  12.71 GB (13015.6 MB) (-48.2%)
Current config:  12.71 GB (13015.6 MB) (-48.2%)
Using algorithm: DCFR (dual discount factors)

Memory precision configuration (estimated):
Strategy (storage1):      16-bit  ( 5729.17 MB,  44.0%)
Regrets (storage2/4):     16-bit  ( 5729.17 MB,  44.0%)
IP CFValues (storage_ip): 16-bit  (  640.35 MB,   4.9%)
Chance CFValues:          16-bit  (   16.17 MB,   0.1%)
Misc (node arena, etc):          (  900.74 MB,   6.9%)

Solving (starting from iter 0, max 2500 more iterations)...
Iteration 20: exploitability = 2309.154785
Iteration 40: exploitability = 457.511932
Iteration 60: exploitability = 214.704178
Iteration 80: exploitability = 174.502762
Iteration 100: exploitability = 112.085960
Iteration 120: exploitability = 90.321701
Iteration 140: exploitability = 75.284187
Iteration 160: exploitability = 61.026306
Iteration 180: exploitability = 48.187401
Iteration 200: exploitability = 37.070816
Iteration 220: exploitability = 40.530479
Iteration 240: exploitability = 28.631294
Iteration 260: exploitability = 63.217178
Iteration 280: exploitability = 28.592667
Iteration 300: exploitability = 19.425964
Iteration 320: exploitability = 17.058250
Iteration 340: exploitability = 16.905029
Iteration 360: exploitability = 9.338356
Target exploitability reached at iteration 360

================================================================================
STOP - Solving completed
Iterations: 361
Final exploitability: 9.338356
Target threshold:     15.000000
================================================================================

✓ Solving completed in 2289.84s

Using algorithm: DCFR (dual discount factors)

Memory precision configuration (estimated):
Strategy (storage1):      32-bit  (11458.33 MB,  74.5%)
Regrets (storage2/4):      8-bit  ( 2864.58 MB,  18.6%)
IP CFValues (storage_ip):  4-bit  (  160.09 MB,   1.0%)
Chance CFValues:           4-bit  (    4.04 MB,   0.0%)
Misc (node arena, etc):          (  900.74 MB,   5.9%)

Solving (starting from iter 0, max 2500 more iterations)...
Iteration 20: exploitability = 1322.594238
Iteration 40: exploitability = 886.040955
Iteration 60: exploitability = 421.675262
Iteration 80: exploitability = 238.594528
Iteration 100: exploitability = 129.573853
Iteration 120: exploitability = 102.216400
Iteration 140: exploitability = 93.078224
Iteration 160: exploitability = 66.087624
Iteration 180: exploitability = 47.932205
Iteration 200: exploitability = 42.666107
Iteration 220: exploitability = 31.759964
Iteration 240: exploitability = 26.650612
Iteration 260: exploitability = 182.104187
Iteration 280: exploitability = 82.012360
Iteration 300: exploitability = 57.521927
Iteration 320: exploitability = 44.582664
Iteration 340: exploitability = 31.170532
Iteration 360: exploitability = 16.590286
Iteration 380: exploitability = 15.839760
Iteration 400: exploitability = 13.808746
Target exploitability reached at iteration 400

================================================================================
STOP - Solving completed
Iterations: 401
Final exploitability: 13.808746
Target threshold:     15.000000
================================================================================

✓ Solving completed in 3178.97s


Memory usage comparison:
32-bit float:    24.54 GB (25130.4 MB) (baseline)
16-bit integer:  12.71 GB (13015.6 MB) (-48.2%)
Current config:  24.54 GB (25130.4 MB) (-0.0%)
Using algorithm: DCFR (dual discount factors)

Memory precision configuration (estimated):
Strategy (storage1):      32-bit  (11458.33 MB,  45.6%)
Regrets (storage2/4):     32-bit  (11458.33 MB,  45.6%)
IP CFValues (storage_ip): 32-bit  ( 1280.71 MB,   5.1%)
Chance CFValues:          32-bit  (   32.34 MB,   0.1%)
Misc (node arena, etc):          (  900.74 MB,   3.6%)

Solving (starting from iter 0, max 2500 more iterations)...
Iteration 20: exploitability = 1815.970337
Iteration 40: exploitability = 445.231995
Iteration 60: exploitability = 230.261475
Iteration 80: exploitability = 204.524490
Iteration 100: exploitability = 97.802261
Iteration 120: exploitability = 99.260712
Iteration 140: exploitability = 74.238991
Iteration 160: exploitability = 46.603813
Iteration 180: exploitability = 36.970718
Iteration 200: exploitability = 29.411446
Iteration 220: exploitability = 26.072624
Iteration 240: exploitability = 23.916626
Iteration 260: exploitability = 54.696220
Iteration 280: exploitability = 22.929214
Iteration 300: exploitability = 18.374275
Iteration 320: exploitability = 14.230850
Target exploitability reached at iteration 320

================================================================================
STOP - Solving completed
Iterations: 321
Final exploitability: 14.230850
Target threshold:     15.000000
================================================================================

✓ Solving completed in 3248.81s


Memory precision configuration (estimated):
Strategy (storage1):      16-bit  ( 5729.17 MB,  59.3%)
Regrets (storage2/4):      8-bit  ( 2864.58 MB,  29.7%)
IP CFValues (storage_ip):  4-bit  (  160.09 MB,   1.7%)
Chance CFValues:           4-bit  (    4.04 MB,   0.0%)
Misc (node arena, etc):          (  900.74 MB,   9.3%)

Solving (starting from iter 0, max 2500 more iterations)...
Iteration 20: exploitability = 1322.592529
Iteration 40: exploitability = 886.038940
Iteration 60: exploitability = 421.676453
Iteration 80: exploitability = 238.575348
Iteration 100: exploitability = 129.583588
Iteration 120: exploitability = 102.285080
Iteration 140: exploitability = 93.180588
Iteration 160: exploitability = 66.189224
Iteration 180: exploitability = 48.111488
Iteration 200: exploitability = 42.958473
Iteration 220: exploitability = 32.144882
Iteration 240: exploitability = 27.102509
Iteration 260: exploitability = 182.100830
Iteration 280: exploitability = 82.037064
Iteration 300: exploitability = 57.545212
Iteration 320: exploitability = 44.595795
Iteration 340: exploitability = 31.213531
Iteration 360: exploitability = 16.645447
Iteration 380: exploitability = 15.929626
Iteration 400: exploitability = 13.915115
Target exploitability reached at iteration 400

================================================================================
STOP - Solving completed
Iterations: 401
Final exploitability: 13.915115
Target threshold:     15.000000
================================================================================

✓ Solving completed in 2303.11s