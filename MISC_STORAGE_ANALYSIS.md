# Analisi Dettagliata Memoria "MISC" (Node Arena & Auxiliary Data)

Questo documento analizza la categoria **"Misc"** riportata dal solver durante il solving. Nei grafici di utilizzo della memoria, "Misc" spesso rappresenta una porzione significativa (circa il 30-40%), e comprenderne la struttura è fondamentale per ottimizzare il consumo di RAM.

## 1. Cosa rappresenta "MISC"?

La voce "Misc" aggrega tutte le strutture dati necessarie per gestire l'albero di gioco, le valutazioni delle mani e le ottimizzazioni di simmetria (isomorfismo) che **non** sono i buffer di storage grezzi (Strategy, Regrets, CFValues).

I componenti principali sono:
1.  **Node Arena (`node_arena`)**: Il "corpo" dell'albero di gioco. Contiene i metadati di ogni nodo.
2.  **Hand Strength Data (`hand_strength`)**: Tabelle precalcolate per la forza delle mani su diverse board.
3.  **Isomorphism Tables**: Mappe per gestire i semi intercambiabili (es. Cuori vs Quadri in certe situazioni).
4.  **Indexing & Valid Indices**: Liste di mani valide per ogni possibile carta del Turn/River.

---

## 2. Struttura e Tipi di Dato

### A. Node Arena (`PostFlopNode`)
La `node_arena` è un `Vec<MutexLike<PostFlopNode>>`. Ogni nodo occupata una quantità fissa di memoria.

**Struttura di `PostFlopNode` (circa 80 bytes):**

| Campo | Tipo | Dimensione | Descrizione |
| :--- | :--- | :--- | :--- |
| `prev_action` | `Action` | 8 bytes | L'azione che ha portato a questo nodo (es. Bet, Check). |
| `player` | `u8` | 1 byte | Il giocatore che deve agire (OOP, IP, o Chance). |
| `turn`, `river` | `Card` | 2 bytes | Carte attuali associate al nodo. |
| `is_locked` | `bool` | 1 byte | Se la strategia è bloccata. |
| `amount` | `i32` | 4 bytes | Dimensione del pot/scommessa in questo punto. |
| `children_offset` | `u32` | 4 bytes | Offset per trovare i figli nell'arena. |
| `num_children` | `u16` | 2 bytes | Numero di azioni possibili. |
| `num_elements` | `u32` | 4 bytes | Numero di combinazioni di mani gestite. |
| `scale1..4` | `f32` (x4) | 16 bytes | Fattori di scala per la quantizzazione (es. 16-bit -> float). |
| `storage1..4` | `*mut u8` (x4) | 32 bytes | **Puntatori critici** ai dati di storage globali. |

**Impatto:** In un albero da 1 Milione di nodi, solo l'arena occupa **~80 MB**.

### B. Hand Strength (`StrengthItem`)
Il solver precalcola la forza di ogni combinazione di mani per accelerare i calcoli dei terminali (Showdown).

*   **`StrengthItem`**: Composto da `strength (u16)` e `index (u16)` = **4 bytes**.
*   **Struttura**: `Vec<[Vec<StrengthItem>; 2]>`.
*   **Esempio**: Per ogni possibile board (1326 combinazioni), memorizza la forza di ~1326 mani per ogni giocatore.
    *   *Calcolo*: `1326 boards * 2 players * 1326 hands * 4 bytes` $\approx$ **14 MB**.

### C. Isomorphism Data
Gestisce le simmetrie dei semi. Se il range è simmetrico (es. non abbiamo preferenze tra picche e fiori), il solver evita di calcolare rami ridondanti.
*   **Storage**: Vec di indici (`u8`) e mappe di swap (`u16`). Solitamente occupa pochi MB, ma cresce con la complessità dell'albero.

---

## 3. Strategie di Ottimizzazione per Risparmiare RAM

Per ridurre il peso della categoria "Misc", si possono adottare le seguenti tecniche:

### 1. Compattazione dei Nodi (Node Packing)
Attualmente `PostFlopNode` usa 80 bytes. Molti campi possono essere ridotti:
*   **Puntatori -> Offset**: Sostituire i 4 puntatori a 64-bit (`*mut u8`, 32 bytes) con offset a 32-bit (`u32`). **Risparmio: 16 bytes per nodo.**
*   **Bitfields**: Accorpare `player`, `is_locked`, `turn`, `river` in un unico set di bit.
*   **Scale Factors**: Usare un unico fattore di scala condiviso se la precisione lo permette.

### 2. Valutazione "Pigra" della Forza (Lazy Strength)
Invece di precalcolare tutte le 1326 board (14 MB), calcolare le forze solo per le board che effettivamente appaiono nell'albero.
*   In un'analisi di un singolo "node" turn/river, molte board non sono mai raggiunte.

### 3. Strutture ad Albero Implicite
Se l'albero ha una struttura regolare (es. stessi bet sizing ovunque), si possono derivare alcune informazioni algoritmicamente invece di memorizzarle in ogni nodo.

### 4. Condivisione dei Metadati
Nodi che rappresentano la stessa situazione ma in rami diversi (storia identica ma board isomorfiche) potrebbero condividere parte dei metadati.

---

## Esempio Pratico: Perché "Misc" è al 39%?

Nel log fornito:
- **Albero**: Turn DeepStack.
- **Strategia/Regret**: 16-bit (~23.5 MB totali).
- **Misc**: 15.40 MB.

Di questi 15.4 MB:
- Circa **10-14 MB** sono probabilmente dedicati alle tabelle `hand_strength` e `valid_indices` per il river (indispensabili per calcolare velocemente l'exploitability).
- Il resto è l'arena dei nodi (se il gioco ha ~50.000 nodi, l'arena pesa ~4 MB).

---

**Conclusione per lo Sviluppo:**
Per solving su larga scala (es. Full Trees con molti nodi), l'ottimizzazione di `PostFlopNode` (Puntatori -> Offset) è la via più efficace. Per solving rapidi o mirati, l'ottimizzazione delle tabelle di forza (`hand_strength`) tramite caching dinamico offrirebbe il risparmio maggiore.
