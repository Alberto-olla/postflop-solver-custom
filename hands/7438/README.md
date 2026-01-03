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

Using algorithm: DCFR (dual discount factors)
Using 16-bit precision (compressed)

Solving (max 1000 iterations)...

iteration: 210 / 1000 (exploitability = 1.4992e1)
[STOP] Reached target exploitability at iteration 210
Current: 14.991928, Target: 15.000000



✓ Solving completed in 134.52s

Saving to: hands/7438/solved_games/hand_0000007438_node_02_flop_DeepStack-16.bin
✓ Game saved successfully!

File: hand_0000007438_node_02_flop_DeepStack-16.bin
Size: 122.91 MB
Compression: zstd level 0

Using algorithm: DCFR (dual discount factors)
Regret-based pruning: ENABLED (dynamic threshold)
Using 16-bit precision (compressed)

Solving (max 500 iterations)...

iteration: 210 / 500 (exploitability = 1.4992e1)
[STOP] Reached target exploitability at iteration 210
Current: 14.991928, Target: 15.000000



✓ Solving completed in 165.03s

Saving to: hands/7438/solved_games/hand_0000007438_node_02_flop_DeepStack-16.bin
✓ Game saved successfully!

File: hand_0000007438_node_02_flop_DeepStack-16.bin
Size: 122.91 MB
Compression: zstd level 0

Using algorithm: DCFR+ (single discount + clipping)
Using 16-bit precision (compressed)

Solving (max 500 iterations)...

iteration: 160 / 500 (exploitability = 1.4716e1)
[STOP] Reached target exploitability at iteration 160
Current: 14.715958, Target: 15.000000



✓ Solving completed in 106.60s

Saving to: hands/7438/solved_games/hand_0000007438_node_02_flop_DeepStack-16.bin
✓ Game saved successfully!

File: hand_0000007438_node_02_flop_DeepStack-16.bin
Size: 125.05 MB
Compression: zstd level 0

Using algorithm: SAPCFR+ (asymmetric predictive CFR+)
Using 16-bit precision (compressed)

Solving (max 500 iterations)...

iteration: 240 / 500 (exploitability = 1.4352e1)
[STOP] Reached target exploitability at iteration 240
Current: 14.352432, Target: 15.000000



✓ Solving completed in 312.12s

Saving to: hands/7438/solved_games/hand_0000007438_node_02_flop_DeepStack-16.bin
✓ Game saved successfully!

File: hand_0000007438_node_02_flop_DeepStack-16.bin
Size: 134.21 MB
Compression: zstd level 0

Using algorithm: PDCFR+ (predictive discounted CFR+)
Using 16-bit precision (compressed)

Solving (max 500 iterations)...

iteration: 440 / 500 (exploitability = 1.4421e1)
[STOP] Reached target exploitability at iteration 440
Current: 14.421402, Target: 15.000000



✓ Solving completed in 585.68s

Saving to: hands/7438/solved_games/hand_0000007438_node_02_flop_DeepStack-16.bin
✓ Game saved successfully!

File: hand_0000007438_node_02_flop_DeepStack-16.bin
Size: 124.52 MB
Compression: zstd level 0