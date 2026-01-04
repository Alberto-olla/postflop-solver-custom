# CS-CFR Warm-Start Implementation Plan

## Executive Summary

Implementare un sistema di warm-start per CS-CFR che trasferisce i **regret accumulati** (non le strategie) da un albero minimal a uno full, usando interpolazione lineare per le azioni parametriche (bet/raise) e normalizzazione per evitare shock numerici.

**Obiettivo**: Accelerare la convergenza del solver full tree di 1.5-3x usando l'esperienza del minimal tree.

---

## Architettura a 3 Fasi

```
Fase 1: Solve Minimal Tree
  ↓ (bet sizes: "50%" → ~40 iterations)
  ↓
Fase 2: Transfer Regrets (Warm-Start)
  ↓ extract from storage2 → decode → normalize → interpolate → encode
  ↓
Fase 3: Solve Full Tree (warm-started)
  ↓ (bet sizes: "25%,50%,75%,100%,a" → <160 iterations)
  ↓ Misurare: iters per raggiungere 0.5% exploitability
```

**Normalizzazione Critica**:
```
R_avg = R_minimal / T_minimal
R_warm = R_avg × W (W ≈ 10, warmstart weight)
T_full_start = W (non 0, non T_minimal!)
```

---

## File da Creare/Modificare

### 1. **src/warm_start.rs** (NEW - ~600 righe)
Modulo core con tutta la logica di warm-start.

**Funzioni Principali**:

```rust
/// Entry point per warm-start
pub fn apply_warm_start(
    small_game: &PostFlopGame,
    large_game: &mut PostFlopGame,
    small_iterations: u32,
    warmstart_weight: f32,  // Default: 10.0
) -> Result<(), String>

/// Traversal ricorsivo con transfer regrets
fn transfer_regrets_recursive(
    small_node: &PostFlopNode,
    large_node: &mut PostFlopNode,
    small_game: &PostFlopGame,
    large_game: &PostFlopGame,
    avg_factor: f32,  // = W / T_minimal
) -> Result<(), String>

/// Mapping azioni (cuore dell'algoritmo!)
enum ActionMatch {
    Direct(usize),           // Match esatto
    Interpolated {           // Interpolazione lineare
        low: usize,
        high: usize,
        weight: f32,         // [0,1] favoring high
    },
    Nearest(usize),          // Estrapolazione (nearest neighbor)
}

fn map_action(
    small_action: &Action,
    large_actions: &[Action],
    pot_size: i32,
) -> ActionMatch

/// Extract/Inject con supporto quantization (4/8/16/32 bit)
fn extract_regrets(node: &PostFlopNode, game: &PostFlopGame) -> Vec<f32>
fn inject_regrets(node: &mut PostFlopNode, regrets: &[f32], game: &PostFlopGame)

/// Interpolazione con accumulo
fn interpolate_regrets(
    small_regrets: &[f32],      // Singola azione
    action_match: ActionMatch,
    large_regrets: &mut [f32],  // Tutte le azioni
    num_hands: usize,
    avg_factor: f32,
)

/// Validazione compatibilità
fn validate_games(small: &PostFlopGame, large: &PostFlopGame) -> Result<(), String>
```

**Logica di Mapping** (Pseudocodice):

```
map_action(small_action, large_actions, pot_size):

  # Azioni fisse: direct copy
  IF small_action IN {Fold, Call, Check}:
    RETURN Direct(find_matching_index)

  # Azioni parametriche: interpolazione
  small_bet_pct = extract_bet(small_action) / pot_size
  large_bets = extract_all_bets(large_actions)  # [(idx, amt), ...]

  # 1. Exact match
  FOR (idx, amt) IN large_bets:
    IF amt == small_bet:
      RETURN Direct(idx)

  # 2. Bracketing (interpolazione tra due bet)
  FOR consecutive pairs (low, high) IN large_bets:
    IF low.amt < small_bet < high.amt:
      weight = (small_bet_pct - low_pct) / (high_pct - low_pct)
      RETURN Interpolated(low.idx, high.idx, weight)

  # 3. Extrapolazione sotto la prima bet
  IF small_bet < large_bets[0].amt:
    check_idx = find_check_action()
    weight = small_bet_pct / first_bet_pct
    RETURN Interpolated(check_idx, large_bets[0].idx, weight)

  # 4. Extrapolazione sopra l'ultima bet (nearest)
  RETURN Nearest(large_bets[last].idx)
```

**Gestione Quantization**:

```rust
// In extract_regrets():
match game.regret_bits() {
    32 => node.regrets().to_vec(),
    16 => {
        let scale = node.regret_scale();
        let decoder = scale / i16::MAX;
        node.regrets_compressed().iter()
            .map(|&r| r as f32 * decoder)
            .collect()
    }
    8 => {
        let scale = node.regret_scale();
        if cfr_algorithm == DCFRPlus {
            // Unsigned u8
            let decoder = scale / u8::MAX;
            node.regrets_u8().iter().map(|&r| r as f32 * decoder).collect()
        } else {
            // Signed i8
            let decoder = scale / i8::MAX;
            node.regrets_i8().iter().map(|&r| r as f32 * decoder).collect()
        }
    }
    4 => {
        // Decode 4-bit packed (nibbles)
        // Similar logic with scale / 7.0 (signed) or 15.0 (unsigned)
        ...
    }
}

// In inject_regrets(): usa le funzioni di encoding da utility.rs
// - encode_signed_slice() per 16-bit
// - encode_unsigned_regrets_u8() o encode_signed_i8() per 8-bit
// - encode_signed_i4_packed() per 4-bit
// Con stochastic rounding per 8/4 bit (seed deterministico)
```

---

### 2. **src/game/base.rs** (MODIFY - ~15 righe)

Aggiungere metodo pubblico a `PostFlopGame` (dopo `update_config`, linea ~340):

```rust
impl PostFlopGame {
    /// Apply warm-start from a solved minimal tree
    ///
    /// # Arguments
    /// * `source_game` - Solved minimal tree
    /// * `source_iterations` - Iterations completed on source
    /// * `warmstart_weight` - Normalization weight (default: 10.0)
    ///
    /// # Returns
    /// Starting iteration count (= warmstart_weight as u32)
    pub fn warm_start_from(
        &mut self,
        source_game: &PostFlopGame,
        source_iterations: u32,
        warmstart_weight: f32,
    ) -> Result<u32, String> {
        if !self.is_ready() {
            return Err("Target game must be ready (call allocate_memory first)".into());
        }
        if !source_game.is_ready() {
            return Err("Source game must be ready".into());
        }

        crate::warm_start::apply_warm_start(
            source_game,
            self,
            source_iterations,
            warmstart_weight,
        )?;

        Ok(warmstart_weight as u32)
    }
}
```

---

### 3. **src/lib.rs** (MODIFY - 2 righe)

Registrare il modulo (dopo gli altri mod, linea ~50):

```rust
pub mod warm_start;

// Re-export
pub use warm_start::apply_warm_start;
```

---

### 4. **tests/warm_start_test.rs** (NEW - ~400 righe)

Test suite completa con 3 test principali.

#### Test 1: `test_warm_start_acceleration()` (MAIN TEST)

```rust
#[test]
fn test_warm_start_acceleration() {
    // Config: Turn tree, pot=3900, stack=17600
    let card_config = CardConfig {
        range: [parse_range("AhQh"), parse_range("Q6o,T7o,...")],
        flop: flop_from_str("7h6d6h").unwrap(),
        turn: card_from_str("5s").unwrap(),
        river: NOT_DEALT,
    };

    // FASE 1: Solve minimal (50% bet only)
    let minimal_tree = build_tree_with_betsizes("50%", "2x");
    let mut game_minimal = PostFlopGame::with_config(card_config.clone(), minimal_tree)?;
    game_minimal.allocate_memory();

    for i in 0..40 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_expl = compute_exploitability(&game_minimal);
    println!("Minimal after 40 iters: {:.6}", minimal_expl);

    // FASE 2: Warm-start full tree (25%,50%,75%,100%,a)
    let full_tree = build_tree_with_betsizes("25%,50%,75%,100%,a", "2x");
    let mut game_full_warm = PostFlopGame::with_config(card_config.clone(), full_tree.clone())?;
    game_full_warm.allocate_memory();

    let start_iter = game_full_warm.warm_start_from(&game_minimal, 40, 10.0)?;
    assert_eq!(start_iter, 10);

    // Solve fino a target exploitability (0.5%)
    let target = 0.005;
    let mut warm_iters = 0;

    for i in start_iter..1000 {
        let expl = compute_exploitability(&game_full_warm);
        if expl <= target {
            warm_iters = i - start_iter;
            println!("Warm-start reached target at iter {}", i);
            break;
        }
        solve_step(&mut game_full_warm, i);
    }

    // FASE 3: Baseline cold-start
    let mut game_full_cold = PostFlopGame::with_config(card_config, full_tree)?;
    game_full_cold.allocate_memory();

    let mut cold_iters = 0;
    for i in 0..1000 {
        let expl = compute_exploitability(&game_full_cold);
        if expl <= target {
            cold_iters = i;
            println!("Cold-start reached target at iter {}", i);
            break;
        }
        solve_step(&mut game_full_cold, i);
    }

    // RISULTATI
    let total_warm = start_iter as usize + warm_iters;
    let speedup = cold_iters as f32 / total_warm as f32;

    println!("\n=== RESULTS ===");
    println!("Warm-start: {} iters (start={}, solve={})", total_warm, start_iter, warm_iters);
    println!("Cold-start: {} iters", cold_iters);
    println!("Speedup: {:.2}x\n", speedup);

    // ASSERTIONS
    assert!(total_warm < cold_iters, "Warm-start must be faster");
    assert!(speedup >= 1.2, "Expected ≥1.2x speedup, got {:.2}x", speedup);
}
```

#### Test 2: `test_warm_start_action_mapping()`

Verifica casi edge dell'interpolazione:
- Direct match (50% → 50%)
- Bracketing (50% → 25%/50%/75%)
- Extrapolazione (50% → 25%/100% senza 50%)
- All-in handling

#### Test 3: `test_warm_start_quantization_modes()`

Verifica compatibilità precision:
- 8-bit → 16-bit
- 16-bit → 16-bit
- 32-bit → 8-bit (con warning)
- DCFR+ (unsigned) vs DCFR (signed)

---

## Dettagli di Implementazione

### Traversal Strategy

**DFS parallelo tra small e large tree**:

```
transfer_regrets_recursive(small_node, large_node):

  # Base case: terminal/chance
  IF small_node.is_terminal() OR small_node.is_chance():
    FOR child IN small_node.children:
      matched = find_matching_child(large_node, child.action)
      RECURSE(child, matched)
    RETURN

  # Player node: transfer regrets
  small_regrets = extract_regrets(small_node)  # Decode from storage2
  num_hands = small_regrets.len / small_node.num_actions()

  large_regrets = ZEROS[large_node.num_actions() × num_hands]

  # Map ogni azione small → large
  FOR (small_idx, action) IN small_node.actions():
    action_regrets = small_regrets[small_idx*num_hands..(small_idx+1)*num_hands]

    match_result = map_action(action, large_node.actions(), pot_size)

    CASE match_result:
      Direct(large_idx):
        large_regrets[large_idx*num_hands..] = action_regrets × avg_factor

      Interpolated(low, high, weight):
        FOR hand IN 0..num_hands:
          r = action_regrets[hand] × avg_factor
          large_regrets[low*num_hands + hand] += r × (1 - weight)
          large_regrets[high*num_hands + hand] += r × weight

      Nearest(large_idx):
        # Same as Direct

  # Inject (encode back to storage2 con nuove scale)
  inject_regrets(large_node, large_regrets, large_game)

  # Recurse children
  FOR (small_idx, action) IN small_node.actions():
    large_idx = find_matching_child(large_node, action)
    RECURSE(small_node.child(small_idx), large_node.child(large_idx))
```

### Child Matching

Per trovare il child corrispondente:

```rust
fn find_matching_child(large_node, small_action) -> Result<usize> {
    // Chance nodes: match by card
    if large_node.is_chance() {
        if let Action::Chance(card) = small_action {
            return large_node.actions()
                .position(|a| matches!(a, Action::Chance(c) if c == card))
                .ok_or("Card not found");
        }
    }

    // Player nodes: match by action type + amount
    for (idx, large_action) in large_node.actions().enumerate() {
        if actions_equivalent(small_action, large_action) {
            return Ok(idx);
        }
    }

    Err(format!("No match for {:?}", small_action))
}

fn actions_equivalent(a1: &Action, a2: &Action) -> bool {
    match (a1, a2) {
        (Action::Fold, Action::Fold) => true,
        (Action::Check, Action::Check) => true,
        (Action::Call, Action::Call) => true,
        (Action::Bet(amt1), Action::Bet(amt2)) => amt1 == amt2,
        (Action::Raise(amt1), Action::Raise(amt2)) => amt1 == amt2,
        (Action::AllIn(amt1), Action::AllIn(amt2)) => amt1 == amt2,
        _ => false,
    }
}
```

### Pot Size Calculation

Per determinare le percentuali:

```rust
fn calculate_pot_size(node: &PostFlopNode) -> i32 {
    // Simplified: usa node.amount (committed by current player)
    // Pot totale ≈ starting_pot + 2 × amount (se symmetric)

    // TODO: tracciare pot_size durante traversal
    // Per ora: stima conservativa
    node.amount * 2
}
```

**Nota**: Potrebbe richiedere accesso a `tree_config.starting_pot`. Se necessario, passare come parametro extra.

---

## Edge Cases

### 1. **Quantization Mismatch**
- **8-bit → 16-bit**: OK (upgrade precision)
- **16-bit → 8-bit**: Warning + proceed (downgrade con stochastic rounding)
- **Soluzione**: Usare sempre `f32` come formato intermedio

### 2. **CFR Algorithm Mismatch**
- **DCFR → DCFR+**: OK (signed → unsigned, clip negatives during inject)
- **DCFR+ → DCFR**: OK (unsigned → signed)
- **SAPCFR+/PDCFR+**: Solo storage2 viene trasferito, storage4 inizializzato a zero
- **Warning**: Avvisare se algoritmi diversi

### 3. **Missing Actions in Small Tree**
- Se small ha bet non presente in large: **Ignora** (non ci sono regret da trasferire)
- È il caso opposto (large ha più azioni) che richiede interpolazione

### 4. **All-in Handling**
- Tratta `AllIn(stack)` come bet parametrica con `bet_amount = effective_stack`
- Se small e large hanno stesso stack → Direct match
- Se diversi → Nearest neighbor o interpolazione con largest bet

### 5. **Validation**
```rust
fn validate_games(small, large) -> Result<()> {
    // Must match:
    - card_config (ranges, flop, turn)
    - starting_pot
    - effective_stack

    // Can differ (with warning):
    - regret_bits
    - cfr_algorithm
    - bet_sizes (ovviamente!)
}
```

---

## Riferimenti Codebase

### File da Studiare:
- **src/utility.rs:212-399**: Funzioni encoding (encode_signed_slice, encode_unsigned_regrets_u8, stochastic_round)
- **src/game/node.rs:43-117**: Accessors storage (regrets(), regrets_compressed(), regrets_i8(), etc.)
- **src/solver.rs:589-1051**: Regret updates (pattern da seguire per decode/update/encode)
- **src/action_tree.rs:16-44**: Action enum e matching
- **tests/resume_test.rs**: Pattern per test structure

### Utilities Esistenti da Riusare:
```rust
// Da utility.rs
encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32
encode_unsigned_regrets_u8(dst: &mut [u8], slice: &[f32], seed: u32) -> f32
encode_signed_i8(dst: &mut [i8], slice: &[f32], seed: u32) -> f32
stochastic_round(val: f32, seed: &mut u32) -> i32

// Da game
PostFlopNode::regrets() -> &[f32]
PostFlopNode::regrets_compressed() -> &[i16]
PostFlopNode::regrets_i8() -> &[i8]
PostFlopNode::regret_scale() -> f32
PostFlopNode::set_regret_scale(f32)
```

---

## Expected Results

### Performance Target:
- **Cold-start full tree**: ~160 iterations per 0.5% exploitability
- **Warm-start full tree**: ~80-100 iterations (50% reduction)
- **Speedup**: 1.5x - 2x minimum

### Qualità Convergenza:
- Strategia finale identica (stesso Nash equilibrium)
- Exploitability finale: <0.5%
- Nessun overfitting al minimal tree

---

## Implementazione Step-by-Step

1. **Creare src/warm_start.rs** con strutture base (ActionMatch enum, function signatures)
2. **Implementare map_action()** con test isolato
3. **Implementare extract_regrets()** per tutti i quantization modes
4. **Implementare inject_regrets()** con encoding
5. **Implementare interpolate_regrets()** con accumulo
6. **Implementare transfer_regrets_recursive()** con DFS
7. **Implementare apply_warm_start()** entry point + validation
8. **Aggiungere warm_start_from() a PostFlopGame**
9. **Registrare modulo in lib.rs**
10. **Creare test suite** con i 3 test
11. **Debug & iteration** con toy examples
12. **Test finale** con configurazione reale (hand_0000007438)

---

## Test Configuration Reale

Usare: `hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml`

**Modifiche per il test**:

```toml
# Minimal variant
[bet_sizes.turn]
oop_bet = "50%"
oop_raise = "2x"
ip_bet = "50%"
ip_raise = "2x"

# Full variant (già presente)
[bet_sizes.turn]
oop_bet = "full"
oop_raise = "full"
ip_bet = "full"
ip_raise = "full"
```

**Metriche attese**:
- Minimal: 40 iterations → exploitability ~2-5%
- Full cold: 160 iterations → exploitability <0.5%
- Full warm: <100 iterations → exploitability <0.5%

---

## Note Finali

### Perché Interpolazione dei Regret (non Strategy)?

Se trasferisci solo la strategia (probabilità), il DCFR+ la sovrascriverà immediatamente perché i regret sottostanti sono zero. La strategia è **derivata** dai regret via regret matching:

```
strategy[action] ∝ max(regret[action], 0)
```

Trasferendo i regret, trasferisci l'**esperienza accumulata**: quali azioni sono state provate e quanto bene hanno funzionato. Questo attiva il pruning automatico: se "bet large" ha regret -1000 nel minimal, nel full tree "bet 90%" erediterà un regret negativo tramite interpolazione → viene immediatamente pruned.

### Normalizzazione Critica

Il fattore `avg_factor = W / T_minimal` serve a:
1. **Evitare dominio**: Regret accumulati per 1000 iter domineranno per sempre se T_full=1
2. **Mantenere direzione**: I segni (+/-) indicano quali azioni sono buone/cattive
3. **Permettere correzione**: Con T_full=10 e discount aggressivo, il solver può correggere errori di astrazione rapidamente

**Formula**:
```
R_avg = R_minimal / 1000    # Normalize per iteration
R_warm = R_avg × 10          # Scale to W iterations
T_start = 10                 # Start warm (not cold T=0, not hot T=1000)
```

Questo bilancia "esperienza del minimal tree" con "flessibilità per il full tree".
