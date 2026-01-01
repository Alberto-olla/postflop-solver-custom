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

32bit Solving (max 10000 iterations)...
iteration: 1030 / 10000 (exploitability = 3.0793e0)

✓ Solving completed in 6.56s

16bit
Solving (max 10000 iterations)...
iteration: 1030 / 10000 (exploitability = 2.9069e0)

✓ Solving completed in 6.76s

8bit
Solving (max 10000 iterations)...
iteration: 10000 / 10000 (exploitability = 7.8581e3)

✓ Solving completed in 67.87s

Saving to: hands/7438/solved_games/hand_0000007438_node_03_turn_DeepStack-8.bin
✓ Game saved successfully!

4bit
Solving (max 10000 iterations)...
iteration: 10000 / 10000 (exploitability = 7.9134e3)

✓ Solving completed in 67.87s
