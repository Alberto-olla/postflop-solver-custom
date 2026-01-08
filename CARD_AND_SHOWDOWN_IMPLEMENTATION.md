# Rappresentazione delle Carte e Valutazione Showdown

Questo documento descrive in dettaglio l'implementazione attuale del sistema di rappresentazione delle carte e della valutazione dello showdown nel postflop-solver.

---

## 1. Rappresentazione delle Carte

### 1.1 Tipo Base Card

**File:** `src/card.rs:8-17`

```rust
pub type Card = u8;
pub const NOT_DEALT: Card = Card::MAX;  // 255
```

Una carta e' rappresentata come un singolo byte (`u8`) con valore nell'intervallo `[0, 51]`. Il valore speciale `255` (`Card::MAX`) indica una carta non ancora distribuita.

### 1.2 Schema di Codifica

La formula di codifica e':

```
card_id = 4 * rank + suit
```

Dove:
- **Rank** (valore della carta): `2 => 0`, `3 => 1`, ..., `K => 11`, `A => 12`
- **Suit** (seme): `club => 0`, `diamond => 1`, `heart => 2`, `spade => 3`

**Esempi di mappatura:**

| Carta | Rank | Suit | card_id |
|-------|------|------|---------|
| 2c    | 0    | 0    | 0       |
| 2d    | 0    | 1    | 1       |
| 3d    | 1    | 1    | 5       |
| Th    | 8    | 2    | 34      |
| As    | 12   | 3    | 51      |

### 1.3 Estrazione di Rank e Suit

**File:** `src/range.rs:207-209` e `src/hand.rs:64-65`

```rust
let rank = card >> 2;    // Equivalente a card / 4
let suit = card & 3;     // Equivalente a card % 4
```

L'uso di operazioni bitwise (`>> 2` e `& 3`) invece di divisione e modulo e' un'ottimizzazione per velocita'.

### 1.4 Conversioni Carattere-Numero

**File:** `src/range.rs:136-189`

```rust
fn char_to_rank(c: char) -> Result<u8, String> {
    match c {
        'A' | 'a' => Ok(12),
        'K' | 'k' => Ok(11),
        'Q' | 'q' => Ok(10),
        'J' | 'j' => Ok(9),
        'T' | 't' => Ok(8),
        '2'..='9' => Ok(c as u8 - b'2'),
        _ => Err(...)
    }
}

fn char_to_suit(c: char) -> Result<u8, String> {
    match c {
        'c' => Ok(0),  // club
        'd' => Ok(1),  // diamond
        'h' => Ok(2),  // heart
        's' => Ok(3),  // spade
        _ => Err(...)
    }
}
```

### 1.5 Costruzione Card da Caratteri

**File:** `src/range.rs:271-280`

```rust
pub fn card_from_chars<T: Iterator<Item = char>>(chars: &mut T) -> Result<Card, String> {
    let rank_char = chars.next().ok_or_else(|| "Unexpected end".to_string())?;
    let suit_char = chars.next().ok_or_else(|| "Unexpected end".to_string())?;

    let rank = char_to_rank(rank_char)?;
    let suit = char_to_suit(suit_char)?;

    Ok((rank << 2) | suit)
}
```

La costruzione usa `(rank << 2) | suit` che e' equivalente a `4 * rank + suit`.

---

## 2. Indice delle Coppie di Carte (Card Pair Index)

### 2.1 Funzione card_pair_to_index

**File:** `src/card.rs:84-93`

```rust
pub(crate) fn card_pair_to_index(mut card1: Card, mut card2: Card) -> usize {
    if card1 > card2 {
        mem::swap(&mut card1, &mut card2);
    }
    card1 as usize * (101 - card1 as usize) / 2 + card2 as usize - 1
}
```

Questa funzione mappa una coppia ordinata di carte a un indice univoco nell'intervallo `[0, 1325]` (1326 combinazioni totali per hole cards).

**Formula matematica:**
- Prima ordina le carte in modo che `card1 < card2`
- L'indice e': `card1 * (101 - card1) / 2 + card2 - 1`

**Esempi:**
- `(0, 1)` -> `2d2c` -> indice `0`
- `(0, 2)` -> `2h2c` -> indice `1`
- `(50, 51)` -> `AsAh` -> indice `1325`

### 2.2 Funzione Inversa index_to_card_pair

**File:** `src/card.rs:98-103`

```rust
pub(crate) fn index_to_card_pair(index: usize) -> (Card, Card) {
    let card1 = (103 - (103.0 * 103.0 - 8.0 * index as f64).sqrt().ceil() as u16) / 2;
    let card2 = index as u16 - card1 * (101 - card1) / 2 + 1;
    (card1 as Card, card2 as Card)
}
```

Usa una formula quadratica inversa per ricostruire la coppia di carte dall'indice.

---

## 3. Struttura Hand per la Valutazione

### 3.1 Definizione della Struttura

**File:** `src/hand.rs:3-7`

```rust
#[derive(Clone, Copy, Default)]
pub(crate) struct Hand {
    cards: [usize; 7],      // Array fisso per 7 carte
    num_cards: usize,       // Numero di carte attualmente presenti
}
```

La struttura `Hand` rappresenta una mano da valutare (fino a 7 carte: 5 board + 2 hole).

**Nota:** Le carte sono memorizzate come `usize` (8 byte su sistemi a 64-bit) invece di `u8`. Questo e' per compatibilita' con gli indici degli array durante la valutazione.

### 3.2 Operazioni sulla Hand

**File:** `src/hand.rs:33-50`

```rust
impl Hand {
    pub fn new() -> Hand {
        Hand::default()
    }

    pub fn add_card(&self, card: usize) -> Hand {
        let mut hand = *self;
        hand.cards[hand.num_cards] = card;
        hand.num_cards += 1;
        hand
    }

    pub fn contains(&self, card: usize) -> bool {
        self.cards[0..self.num_cards].contains(&card)
    }
}
```

`add_card` e' **immutabile**: ritorna una nuova `Hand` invece di modificare quella esistente. Questo permette di costruire mani incrementalmente senza clonazioni esplicite.

---

## 4. Algoritmo di Valutazione delle Mani

### 4.1 Funzione evaluate_internal

**File:** `src/hand.rs:57-126`

L'algoritmo valuta una mano di 7 carte e produce un valore `i32` che codifica sia il **tipo** di mano che i **kicker**.

#### Fase 1: Costruzione dei Bitset

```rust
fn evaluate_internal(&self) -> i32 {
    let mut rankset = 0i32;              // Bitset dei ranghi presenti
    let mut rankset_suit = [0i32; 4];    // Bitset per ogni seme
    let mut rankset_of_count = [0i32; 5]; // Bitset per conteggio (1,2,3,4 carte)
    let mut rank_count = [0i32; 13];     // Conteggio per ogni rango

    for &card in &self.cards {
        let rank = card / 4;
        let suit = card % 4;
        rankset |= 1 << rank;
        rankset_suit[suit] |= 1 << rank;
        rank_count[rank] += 1;
    }

    for rank in 0..13 {
        rankset_of_count[rank_count[rank] as usize] |= 1 << rank;
    }
    // ...
}
```

Dopo questa fase:
- `rankset`: bit `i` e' 1 se il rango `i` e' presente nella mano
- `rankset_suit[s]`: bit `i` e' 1 se il rango `i` e' presente nel seme `s`
- `rank_count[r]`: quante carte del rango `r` sono presenti
- `rankset_of_count[n]`: bit `i` e' 1 se ci sono esattamente `n` carte del rango `i`

#### Fase 2: Rilevazione Flush

```rust
let mut flush_suit: i32 = -1;
for suit in 0..4 {
    if rankset_suit[suit as usize].count_ones() >= 5 {
        flush_suit = suit;
    }
}
```

Un flush richiede 5+ carte dello stesso seme.

#### Fase 3: Rilevazione Straight

**File:** `src/hand.rs:21-31`

```rust
fn find_straight(rankset: i32) -> i32 {
    const WHEEL: i32 = 0b1_0000_0000_1111;  // A-2-3-4-5
    let is_straight = rankset & (rankset << 1) & (rankset << 2)
                             & (rankset << 3) & (rankset << 4);
    if is_straight != 0 {
        keep_n_msb(is_straight, 1)
    } else if (rankset & WHEEL) == WHEEL {
        1 << 3  // Straight basso (5-high)
    } else {
        0
    }
}
```

**Spiegazione dell'algoritmo:**
- `rankset << n` shifta tutti i bit a sinistra di n posizioni
- L'AND tra 5 shift consecutivi produce un bit 1 solo dove ci sono 5 ranghi consecutivi
- `WHEEL` (`0b1_0000_0000_1111`) rappresenta A-2-3-4-5 (Asso come carta bassa)

#### Fase 4: Classificazione Gerarchica

```rust
if flush_suit >= 0 {
    let is_straight_flush = find_straight(rankset_suit[flush_suit as usize]);
    if is_straight_flush != 0 {
        (8 << 26) | is_straight_flush          // Straight Flush (tipo 8)
    } else {
        (5 << 26) | keep_n_msb(rankset_suit[flush_suit as usize], 5)  // Flush (tipo 5)
    }
} else if rankset_of_count[4] != 0 {
    // Four of a Kind (tipo 7)
    let remaining = keep_n_msb(rankset ^ rankset_of_count[4], 1);
    (7 << 26) | (rankset_of_count[4] << 13) | remaining
} else if rankset_of_count[3].count_ones() == 2 {
    // Full House con due tris (tipo 6)
    let trips = keep_n_msb(rankset_of_count[3], 1);
    let pair = rankset_of_count[3] ^ trips;
    (6 << 26) | (trips << 13) | pair
} else if rankset_of_count[3] != 0 && rankset_of_count[2] != 0 {
    // Full House normale (tipo 6)
    let pair = keep_n_msb(rankset_of_count[2], 1);
    (6 << 26) | (rankset_of_count[3] << 13) | pair
} else if is_straight != 0 {
    (4 << 26) | is_straight                    // Straight (tipo 4)
} else if rankset_of_count[3] != 0 {
    // Three of a Kind (tipo 3)
    let remaining = keep_n_msb(rankset_of_count[1], 2);
    (3 << 26) | (rankset_of_count[3] << 13) | remaining
} else if rankset_of_count[2].count_ones() >= 2 {
    // Two Pair (tipo 2)
    let pairs = keep_n_msb(rankset_of_count[2], 2);
    let remaining = keep_n_msb(rankset ^ pairs, 1);
    (2 << 26) | (pairs << 13) | remaining
} else if rankset_of_count[2] != 0 {
    // One Pair (tipo 1)
    let remaining = keep_n_msb(rankset_of_count[1], 3);
    (1 << 26) | (rankset_of_count[2] << 13) | remaining
} else {
    // High Card (tipo 0)
    keep_n_msb(rankset, 5)
}
```

#### Codifica del Valore di Ritorno

Il valore `i32` restituito e' codificato come:

```
| tipo (bit 26-31) | dati primari (bit 13-25) | kicker (bit 0-12) |
```

- **Tipo**: 0-8 (High Card, Pair, Two Pair, Trips, Straight, Flush, Full House, Quads, Straight Flush)
- **Dati primari**: ranghi delle carte che formano la combinazione principale
- **Kicker**: ranghi delle carte rimanenti per il tie-break

### 4.2 Funzione Helper keep_n_msb

**File:** `src/hand.rs:10-18`

```rust
fn keep_n_msb(mut x: i32, n: i32) -> i32 {
    let mut ret = 0;
    for _ in 0..n {
        let bit = 1 << (x.leading_zeros() ^ 31);
        x ^= bit;
        ret |= bit;
    }
    ret
}
```

Mantiene solo gli `n` bit piu' significativi (MSB) del valore. Usato per selezionare i kicker piu' alti.

---

## 5. Hand Table e Conversione a Ranking

### 5.1 La Tabella HAND_TABLE

**File:** `src/hand_table.rs`

```rust
pub(crate) const HAND_TABLE: [i32; 4824] = [
    236, 244, 364, 372, 376, ...
];
```

`HAND_TABLE` contiene **4824 valori** `i32` **ordinati in modo crescente**. Questi rappresentano tutti i possibili valori univoci di valutazione delle mani di 7 carte.

### 5.2 Conversione a Ranking

**File:** `src/hand.rs:52-55`

```rust
pub fn evaluate(&self) -> u16 {
    HAND_TABLE.binary_search(&self.evaluate_internal()).unwrap() as u16
}
```

La funzione pubblica `evaluate()`:
1. Calcola il valore raw con `evaluate_internal()`
2. Esegue una **binary search** nella `HAND_TABLE`
3. Ritorna l'indice trovato come `u16` (0-4823)

**Significato del ranking:**
- 0 = mano piu' debole (7-5-4-3-2 offsuit)
- 4823 = mano piu' forte (Royal Flush)
- Mani con lo stesso ranking sono equivalenti (split pot)

### 5.3 Distribuzione delle Mani

**File:** `src/hand.rs:166-174` (test)

```rust
assert_eq!(counter[8], 41584);     // Straight Flush
assert_eq!(counter[7], 224848);    // Four of a Kind
assert_eq!(counter[6], 3473184);   // Full House
assert_eq!(counter[5], 4047644);   // Flush
assert_eq!(counter[4], 6180020);   // Straight
assert_eq!(counter[3], 6461620);   // Three of a Kind
assert_eq!(counter[2], 31433400);  // Two Pair
assert_eq!(counter[1], 58627800);  // One Pair
assert_eq!(counter[0], 23294460);  // High Card
```

Il totale delle combinazioni di 7 carte e' C(52,7) = 133,784,560.

---

## 6. Struttura StrengthItem

### 6.1 Definizione

**File:** `src/card.rs:67-71`

```rust
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StrengthItem {
    pub(crate) strength: u16,  // Ranking della mano (0-4823)
    pub(crate) index: u16,     // Indice della mano in private_cards
}
```

`StrengthItem` associa un ranking di forza a un indice di mano. L'implementazione di `Ord` confronta prima `strength`, poi `index`.

### 6.2 Uso nelle Strutture Precomputate

**File:** `src/game/mod.rs:59-60`

```rust
hand_strength: Vec<[Vec<StrengthItem>; 2]>,
```

Per ogni possibile board (turn+river), viene precomputato un vettore ordinato di `StrengthItem` per ogni giocatore. L'ordinamento permette un'efficiente valutazione dello showdown.

---

## 7. Precomputazione della Hand Strength

### 7.1 Funzione hand_strength

**File:** `src/card.rs:183-245`

```rust
pub(crate) fn hand_strength(
    &self,
    private_cards: &PrivateCards,
) -> Vec<[Vec<StrengthItem>; 2]> {
    let mut ret = vec![Default::default(); 52 * 51 / 2];  // 1326 board possibili

    let mut board = Hand::new();
    for &card in &self.flop {
        board = board.add_card(card as usize);
    }

    for board1 in 0..52 {
        for board2 in board1 + 1..52 {
            // Salta board non validi
            if !board.contains(board1 as usize) && !board.contains(board2 as usize)
                && (self.turn == NOT_DEALT || board1 == self.turn || board2 == self.turn)
                && (self.river == NOT_DEALT || board1 == self.river || board2 == self.river)
            {
                let board = board.add_card(board1 as usize).add_card(board2 as usize);
                let mut strength = [
                    Vec::with_capacity(private_cards[0].len() + 2),
                    Vec::with_capacity(private_cards[1].len() + 2),
                ];

                for player in 0..2 {
                    // Aggiungi sentinelle (minimo e massimo)
                    strength[player].push(StrengthItem { strength: 0, index: 0 });
                    strength[player].push(StrengthItem { strength: u16::MAX, index: u16::MAX });

                    // Valuta ogni mano del giocatore
                    strength[player].extend(
                        private_cards[player].iter().enumerate().filter_map(
                            |(index, &(c1, c2))| {
                                if board.contains(c1 as usize) || board.contains(c2 as usize) {
                                    None  // Carta bloccata dal board
                                } else {
                                    let hand = board.add_card(c1 as usize).add_card(c2 as usize);
                                    Some(StrengthItem {
                                        strength: hand.evaluate() + 1,  // +1 per evitare 0
                                        index: index as u16,
                                    })
                                }
                            },
                        ),
                    );

                    strength[player].shrink_to_fit();
                    strength[player].sort_unstable();  // Ordina per forza
                }

                ret[card_pair_to_index(board1, board2)] = strength;
            }
        }
    }
    ret
}
```

**Punti chiave:**
1. Itera su tutti i possibili board turn+river (1326 combinazioni)
2. Per ogni board valido, valuta tutte le mani di ogni giocatore
3. Aggiunge due **sentinelle** con `strength = 0` e `strength = u16::MAX`
4. **Ordina** il vettore per forza crescente
5. Il `+1` nello strength evita conflitti con la sentinella minima

---

## 8. Valid Indices (Indici Validi)

### 8.1 Struttura

**File:** `src/game/mod.rs:54-57`

```rust
valid_indices_flop: [Vec<u16>; 2],
valid_indices_turn: Vec<[Vec<u16>; 2]>,
valid_indices_river: Vec<[Vec<u16>; 2]>,
```

Queste strutture contengono gli indici delle mani che non hanno carte in conflitto con il board corrente.

### 8.2 Calcolo degli Indici Validi

**File:** `src/card.rs:147-181`

```rust
fn valid_indices_internal(
    private_cards: &[Vec<(Card, Card)>; 2],
    board1: Card,
    board2: Card,
) -> [Vec<u16>; 2] {
    let mut ret = [
        Vec::with_capacity(private_cards[0].len()),
        Vec::with_capacity(private_cards[1].len()),
    ];

    let mut board_mask: u64 = 0;
    if board1 != NOT_DEALT {
        board_mask |= 1 << board1;
    }
    if board2 != NOT_DEALT {
        board_mask |= 1 << board2;
    }

    for player in 0..2 {
        ret[player].extend(private_cards[player].iter().enumerate().filter_map(
            |(index, &(c1, c2))| {
                let hand_mask: u64 = (1 << c1) | (1 << c2);
                if hand_mask & board_mask == 0 {  // Nessun conflitto
                    Some(index as u16)
                } else {
                    None  // Carta bloccata
                }
            },
        ));
        ret[player].shrink_to_fit();
    }
    ret
}
```

**Algoritmo:**
1. Crea una maschera `u64` per le carte del board
2. Per ogni mano, crea una maschera per le hole cards
3. Se l'AND delle maschere e' 0, non c'e' conflitto e la mano e' valida

---

## 9. Valutazione dello Showdown

### 9.1 Caso Base: Showdown senza Rake (2-Pass)

**File:** `src/game/evaluation.rs:94-151`

```rust
else if rake == 0.0 {
    let pair_index = card_pair_to_index(node.get_turn(), node.get_river());
    let hand_strength = &self.hand_strength[pair_index];
    let player_strength = &hand_strength[player];
    let opponent_strength = &hand_strength[player ^ 1];

    let valid_player_strength = &player_strength[1..player_strength.len() - 1];
    let mut i = 1;

    // PRIMO PASS: Calcola vittorie del giocatore
    for &StrengthItem { strength, index } in valid_player_strength {
        unsafe {
            // Avanza fino a trovare mani avversarie piu' deboli
            while opponent_strength.get_unchecked(i).strength < strength {
                let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                let cfreach_i = *cfreach.get_unchecked(opponent_index);
                if cfreach_i != 0.0 {
                    let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                    let cfreach_i_f64 = cfreach_i as f64;
                    cfreach_sum += cfreach_i_f64;
                    *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                    *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                }
                i += 1;
            }
            let (c1, c2) = *player_cards.get_unchecked(index as usize);
            let cfreach = cfreach_sum
                - cfreach_minus.get_unchecked(c1 as usize)
                - cfreach_minus.get_unchecked(c2 as usize);
            *result.get_unchecked_mut(index as usize) = (amount_win * cfreach) as f32;
        }
    }

    // Reset per il secondo pass
    cfreach_sum = 0.0;
    cfreach_minus.fill(0.0);
    i = opponent_strength.len() - 2;

    // SECONDO PASS: Calcola perdite del giocatore (iterazione inversa)
    for &StrengthItem { strength, index } in valid_player_strength.iter().rev() {
        unsafe {
            while opponent_strength.get_unchecked(i).strength > strength {
                // ... accumula mani avversarie piu' forti
                i -= 1;
            }
            let (c1, c2) = *player_cards.get_unchecked(index as usize);
            let cfreach = cfreach_sum
                - cfreach_minus.get_unchecked(c1 as usize)
                - cfreach_minus.get_unchecked(c2 as usize);
            *result.get_unchecked_mut(index as usize) += (amount_lose * cfreach) as f32;
        }
    }
}
```

**Algoritmo Two-Pass:**

1. **Primo pass (forward):** Per ogni mano del giocatore in ordine crescente di forza:
   - Accumula il cfreach di tutte le mani avversarie piu' deboli
   - Calcola il payoff di vittoria

2. **Secondo pass (backward):** Per ogni mano del giocatore in ordine decrescente:
   - Accumula il cfreach di tutte le mani avversarie piu' forti
   - Calcola il payoff di perdita

### 9.2 Principio di Inclusione-Esclusione

**File:** `src/game/evaluation.rs:86-90`

```rust
let cfreach = cfreach_sum + cfreach_same
    - *cfreach_minus.get_unchecked(c1 as usize)
    - *cfreach_minus.get_unchecked(c2 as usize);
```

Questo calcolo corregge il cfreach per le **card removal effects**:
- `cfreach_sum`: reach totale dell'avversario
- `cfreach_minus[c1]`: reach delle mani avversarie che contengono `c1`
- `cfreach_minus[c2]`: reach delle mani avversarie che contengono `c2`

Se la mano del giocatore ha carte `c1` e `c2`, l'avversario non puo' avere quelle carte, quindi dobbiamo sottrarre il reach delle mani che le contengono.

### 9.3 Caso con Rake (3-Pass)

**File:** `src/game/evaluation.rs:152-252`

Quando il rake e' > 0, e' necessario un terzo pass per gestire i pareggi, poiche' nel pareggio si paga comunque il rake.

```rust
// Variabili aggiuntive per tracciare i pareggi
let mut cfreach_sum_win = 0.0;
let mut cfreach_sum_tie = 0.0;
let mut cfreach_minus_win = [0.0; 52];
let mut cfreach_minus_tie = [0.0; 52];
```

Il payoff finale combina tre componenti:

```rust
let cfvalue = amount_win * cfreach_win
    + amount_tie * (cfreach_tie - cfreach_win + cfreach_same)
    + amount_lose * (cfreach_total - cfreach_tie);
```

---

## 10. Same Hand Index

### 10.1 Scopo

**File:** `src/game/mod.rs:52`

```rust
same_hand_index: [Vec<u16>; 2],
```

Per ogni mano, indica l'indice di una mano "equivalente" che potrebbe causare un pareggio per simmetria di semi. Usato principalmente nel calcolo del rake.

### 10.2 Esempio

Se il giocatore ha `AhKs` e nel range dell'avversario c'e' `AsKh`, queste mani sono equivalenti (stesso rank per entrambe le carte). In caso di pareggio, il rake viene diviso.

---

## 11. Isomorfismo dei Semi

### 11.1 Concetto

Due semi sono **isomorfi** se:
1. Hanno la stessa distribuzione di ranghi nel board
2. Hanno la stessa distribuzione nei range iniziali dei giocatori

### 11.2 Strutture Dati

**File:** `src/card.rs:73-82`

```rust
pub(crate) type SwapList = [Vec<(u16, u16)>; 2];

type IsomorphismData = (
    Vec<u8>,                    // isomorphism_ref_turn
    Vec<Card>,                  // isomorphism_card_turn
    [SwapList; 4],              // isomorphism_swap_turn
    Vec<Vec<u8>>,               // isomorphism_ref_river
    [Vec<Card>; 4],             // isomorphism_card_river
    [[SwapList; 4]; 4],         // isomorphism_swap_river
);
```

### 11.3 Calcolo dell'Isomorfismo

**File:** `src/card.rs:247-361`

```rust
pub(crate) fn isomorphism(&self, private_cards: &[Vec<(Card, Card)>; 2]) -> IsomorphismData {
    let mut suit_isomorphism = [0; 4];
    let mut next_index = 1;

    // Identifica semi isomorfi nel range
    'outer: for suit2 in 1..4 {
        for suit1 in 0..suit2 {
            if self.range[0].is_suit_isomorphic(suit1, suit2)
                && self.range[1].is_suit_isomorphic(suit1, suit2)
            {
                suit_isomorphism[suit2 as usize] = suit_isomorphism[suit1 as usize];
                continue 'outer;
            }
        }
        suit_isomorphism[suit2 as usize] = next_index;
        next_index += 1;
    }

    // Calcola rankset del flop per ogni seme
    let mut flop_rankset = [0; 4];
    for &card in &self.flop {
        let rank = card >> 2;
        let suit = card & 3;
        flop_rankset[suit as usize] |= 1 << rank;
    }

    // Turn isomorphism: due semi sono isomorfi se hanno lo stesso rankset nel flop
    for suit1 in 1..4 {
        for suit2 in 0..suit1 {
            if flop_rankset[suit1] == flop_rankset[suit2]
                && suit_isomorphism[suit1] == suit_isomorphism[suit2]
            {
                isomorphic_suit[suit1] = Some(suit2);
                // Calcola swap list per applicare l'isomorfismo
                // ...
            }
        }
    }
    // ...
}
```

### 11.4 SwapList

La `SwapList` contiene coppie di indici di mani che devono essere scambiate quando si applica un isomorfismo di seme. Ad esempio, se i semi heart e diamond sono isomorfi, le mani `AhKd` e `AdKh` sono equivalenti e le loro strategie devono essere swappate.

---

## 12. Strutture del Game per Carte e Showdown

### 12.1 PostFlopGame

**File:** `src/game/mod.rs:37-84`

```rust
pub struct PostFlopGame {
    // Configurazione
    card_config: CardConfig,

    // Mani private dei giocatori
    private_cards: [Vec<(Card, Card)>; 2],
    same_hand_index: [Vec<u16>; 2],

    // Indici validi per ogni stato del board
    valid_indices_flop: [Vec<u16>; 2],
    valid_indices_turn: Vec<[Vec<u16>; 2]>,
    valid_indices_river: Vec<[Vec<u16>; 2]>,

    // Hand strength precomputata
    hand_strength: Vec<[Vec<StrengthItem>; 2]>,

    // Dati isomorfismo
    isomorphism_ref_turn: Vec<u8>,
    isomorphism_card_turn: Vec<Card>,
    isomorphism_swap_turn: [SwapList; 4],
    isomorphism_ref_river: Vec<Vec<u8>>,
    isomorphism_card_river: [Vec<Card>; 4],
    isomorphism_swap_river: [[SwapList; 4]; 4],
    // ...
}
```

### 12.2 CardConfig

**File:** `src/card.rs:19-49`

```rust
pub struct CardConfig {
    pub range: [Range; 2],     // Range iniziali dei due giocatori
    pub flop: [Card; 3],       // Tre carte del flop (ordinate)
    pub turn: Card,            // Carta del turn (o NOT_DEALT)
    pub river: Card,           // Carta del river (o NOT_DEALT)
}
```

---

## 13. Range dei Giocatori

### 13.1 Struttura Range

**File:** `src/range.rs:40-44`

```rust
pub struct Range {
    data: [f32; 52 * 51 / 2],  // 1326 pesi per ogni coppia di carte
}
```

Ogni elemento contiene un peso `f32` nell'intervallo `[0.0, 1.0]` che indica la probabilita' che il giocatore abbia quella specifica mano.

### 13.2 Accesso ai Pesi

```rust
pub fn get_weight_by_cards(&self, card1: Card, card2: Card) -> f32 {
    self.data[card_pair_to_index(card1, card2)]
}
```

---

## 14. Riepilogo del Flusso di Valutazione

```
1. INIZIALIZZAZIONE
   CardConfig -> Range -> private_cards (mani valide per ogni giocatore)

2. PRECOMPUTAZIONE (per ogni board turn+river)
   - valid_indices: quali mani non hanno carte bloccate
   - hand_strength: ranking di ogni mano, ordinato

3. DURANTE IL SOLVING
   evaluate_internal() -> valuta nodo showdown
   - Usa hand_strength precomputata
   - Two-pass algorithm per calcolo efficiente
   - Applica card removal effects con inclusione-esclusione

4. ISOMORFISMO (opzionale)
   - Riduce lo spazio degli stati sfruttando simmetrie dei semi
   - SwapList per ricostruire strategie complete
```

---

## 15. Punti Chiave dell'Implementazione

1. **Card come u8**: Rappresentazione compatta che permette operazioni bitwise efficienti

2. **Formula card_id = 4*rank + suit**: Permette estrazione veloce con shift e mask

3. **Hand Table con Binary Search**: 4824 valori unici coprono tutte le possibili mani, O(log n) lookup

4. **Precomputazione hand_strength**: Evita valutazioni ripetute durante il solving

5. **Two-Pass Showdown**: Algoritmo O(n) invece di O(n^2) per confrontare tutte le mani

6. **Card Removal via Bitmask**: Calcolo efficiente degli effetti di blocco delle carte

7. **Isomorfismo dei Semi**: Riduzione dello spazio degli stati per semi equivalenti
