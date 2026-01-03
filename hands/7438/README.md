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

Ottimizzazione: 

DCFR
[STOP] Reached target exploitability at iteration 210
Current: 14.991928, Target: 15.000000
Memory usage: 3494.19 MB
Strategy:     1544.14 MB
Regrets:      1544.14 MB
IP CFVs:       134.42 MB
Chance CFVs:     3.40 MB
Misc:          268.09 MB
✓ Solving completed in 136.57s

DCFR+
[STOP] Reached target exploitability at iteration 160
Current: 14.715958, Target: 15.000000
Memory usage: 3494.19 MB
Strategy:     1544.14 MB
Regrets:      1544.14 MB
IP CFVs:       134.42 MB
Chance CFVs:     3.40 MB
Misc:          268.09 MB
✓ Solving completed in 111.20s

DCFR+, Regrets8bit
[STOP] Reached target exploitability at iteration 230
Current: 13.969757, Target: 15.000000
Memory usage: 2722.12 MB
Strategy:     1544.14 MB
Regrets:       772.07 MB
IP CFVs:       134.42 MB
Chance CFVs:     3.40 MB
Misc:          268.09 MB
✓ Solving completed in 422.85s