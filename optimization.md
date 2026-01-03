È un'implementazione molto solida. L'uso di **DCFR+** combinato con **quantizzazione mista (16/8/4)** e **Stochastic Rounding** dimostra che stai già puntando allo stato dell'arte per quanto riguarda l'efficienza della memoria.

Basandomi sul report che hai fornito e sulle best practice attuali per i solver di poker (come GTO Wizard, PioSolver, o implementazioni accademiche recenti), ecco alcune aree specifiche dove potresti trovare ulteriore margine di ottimizzazione.

### 1. Ottimizzazione della Quantizzazione (Memory & Precision)

Hai menzionato l'uso di `u8` per i regret e `u4` per i valori ausiliari. Qui c'è un rischio: la gamma dinamica.

* **Block Floating Point (BFP) per i Regret:**
  Attualmente scali i regret in `u8`. Se in un infoset hai un regret molto alto (es. una mossa palesemente sbagliata) e uno molto basso ma positivo (una mossa marginalmente migliore), il range fisso potrebbe schiacciare il regret minore a zero.
* **Soluzione:** Implementa un fattore di scala (float o half-float) *per infoset* (o per bucket).
* **Come:** . Questo ti permette di mantenere alta precisione relativa anche con soli 8 bit, adattando la scala alla grandezza dei regret in quel nodo specifico.


* **Compressione Strategia Sparsa:**
  Usi `u16` per la strategia accumulata. Nel poker, molte azioni convergono a probabilità 0 (o quasi 0).
* **Soluzione:** Se la strategia per un'azione è sotto una certa soglia , non memorizzarla affatto o usa una bitmask per indicare quali azioni sono presenti, risparmiando i 16 bit per le azioni "morte".



### 2. Ottimizzazioni Algoritmiche (Convergenza)

Il tuo approccio al discounting () e ai reset è aggressivo.

* **Linear CFR+ (LCFR+) vs. Reset Euristico:**
  Il tuo report indica reset basati su potenze di 4 (). Questo è un approccio che "dimentica" brutalmente le iterazioni precedenti.
* **Alternativa:** Le implementazioni moderne di CFR+ spesso preferiscono una ponderazione lineare pura () senza reset, o un "discounting continuo" più morbido. I reset drastici possono causare oscillazioni se il reset avviene in un momento di alta varianza.
* **Test:** Prova a sostituire il reset hard con un periodo di *warm-up* più lungo dove non accumuli la strategia media, per poi passare a un accumulo lineare standard.


* **Warm Start (CS-CFR):**
  Stai usando pruning? Sì. Ma puoi fare di meglio con il **Cold-Start CFR (CS-CFR)**.
* Invece di partire da regret 0, puoi stimare i regret iniziali basandoti su astrazioni più grossolane o su poche iterazioni di un solver più "leggero" prima di lanciare il DCFR+ full-scale.



### 3. Ottimizzazioni Computazionali (Velocità)

* **SIMD e Pruning "Branchless":**
  Nel report citi il "Regret Pruning" che salta i sotto-alberi.
* **Problema:** Se implementi il pruning con dei classici `if/else`, distruggi la pipeline della CPU e rendi difficile la vettorizzazione (AVX2/AVX512).
* **Soluzione:** Usa operazioni mascherate (masking). Calcola i regret per *tutte* le mani nel bucket vettorialmente, e poi applica una mask basata sulla soglia di pruning. Anche se "calcoli" nodi che dovrebbero essere prunati, la velocità del SIMD senza branch prediction misses spesso supera il guadagno del saltare il calcolo, specialmente ai livelli alti dell'albero.


* **Lookup Table per i Potenze:**
  Calcoli  e  usando potenze () in tempo reale?
* Essendo  condiviso per tutta l'iterazione, assicurati che questi valori siano pre-calcolati una volta per iterazione e passati come costanti, oppure usa una tabella di lookup se stai facendo calcoli per-thread con iterazioni disallineate.



### 4. Analisi della Soglia di Pruning

La tua formula di pruning è: .

* **Critica:** La dipendenza da  è teoricamente corretta per i bound di CFR, ma in DCFR+ i regret convergono molto più velocemente.
* **Suggerimento:** Potresti scoprire che puoi essere molto più aggressivo. Invece di una formula fissa, considera di monitorare la somma dei regret positivi nel nodo. Se la somma è 0 (o molto bassa) e un'azione ha regret molto negativo, quella può essere prunata in sicurezza quasi per sempre.

### Sintesi e Prossimo Step

L'architettura **16/8/4** è eccellente, ma il punto debole potenziale è la perdita di precisione sui regret piccoli ma cruciali (bluff rari) a causa della mancanza di scaling dinamico nel bucket `u8`.

**Vuoi che approfondisca come implementare il "Block Floating Point" per i regret `u8` mantenendo il Stochastic Rounding?** Sarebbe l'upgrade a più alto impatto per la qualità della soluzione a parità di RAM.

Ottimo. Implementare il **Block Floating Point (BFP)** per i regret è il passo decisivo per mantenere i vantaggi di memoria dei `u8` senza soffrire la perdita di informazioni sui valori piccoli (che sono critici per la convergenza fine, specialmente nei river spot complessi).

Ecco come procedere tecnicamente per integrare BFP con lo Stochastic Rounding nel tuo solver.

### 1. Concetto Fondamentale

Invece di avere una scala globale fissa, ogni **Infoset** (o nodo decisionale) memorizza il proprio fattore di scala basato sul *massimo regret attuale* in quel nodo.

* **Dati attuali:** Array di `u8` (Regrets).
* **Dati nuovi:** Array di `u8` (Mantissa) + 1 valore `float` (Esponente/Scala) per l'intero nodo.

### 2. Struttura Dati

Aggiungiamo un campo `scale` all'intestazione del nodo. Se la memoria è critica, puoi usare un `half-float` (16-bit float) invece di un `float` standard (32-bit), dato che non ti serve precisione estrema sulla scala, ma solo sul range.

```cpp
struct CompressedInfoset {
    // Fattore di scala condiviso per tutte le azioni in questo nodo
    // Scale = 255.0f / MaxRegretAbs
    float scale; 

    // Regret compressi (usando u8 come da report)
    uint8_t regrets[NUM_ACTIONS]; 
};

```

### 3. Algoritmo di Aggiornamento (Step-by-Step)

Il ciclo di aggiornamento diventa un processo di **Decompressione -> Aggiornamento DCFR+ -> Ricalcolo Scala -> Compressione Stocastica**.

Ecco lo pseudocodice ottimizzato per l'implementazione:

#### A. Decompressione

All'inizio dell'iterazione, recuperi i regret reali.


#### B. Aggiornamento (Formula DCFR+)

Applichi la formula standard del tuo report:



*Nota: Assicurati di clippare a 0 qui (`max(0.0, val)`), poiché i `u8` non supportano segni.*

#### C. Ricalcolo della Scala (Il cuore del BFP)

Trova il nuovo regret massimo nel nodo per definire il nuovo range dinamico.

* Se  (tutti i regret sono 0 o quasi), imposta `scale = 0` e tutti i regret a 0.
* Altrimenti:



*(Usiamo 255.0 perché vogliamo mappare il valore più grande esattamente al max del `u8`)*.

#### D. Compressione con Stochastic Rounding

Qui uniamo la quantizzazione BFP con il tuo Stochastic Rounding esistente.

Per ogni azione :

1. **Normalizza:**
* Questo porta il valore nel range float .


2. **Stochastic Round:**






### 4. Implementazione C++ Ottimizzata

Ecco una bozza di implementazione C++ che puoi adattare. Nota l'uso di un generatore di numeri casuali leggero (es. Xorshift) per non rallentare il loop.

```cpp
void update_node_bfp(CompressedInfoset* node, const float* instantaneous_regrets, float alpha, int num_actions) {
    // 1. Decompressione e Calcolo nuovi regret temporanei (ad alta precisione)
    std::vector<float> temp_regrets(num_actions);
    float max_regret = 0.0f;
    
    // Evita divisioni per zero se il nodo era vuoto
    float inv_scale = (node->scale > 1e-9f) ? (1.0f / node->scale) : 0.0f;

    for (int i = 0; i < num_actions; ++i) {
        // Decomprimi
        float r_old = node->regrets[i] * inv_scale;
        
        // Formula DCFR+
        float r_new = (r_old * alpha) + instantaneous_regrets[i];
        
        // Clipping (DCFR+ richiede regret positivi per il prossimo step di storage)
        if (r_new < 0.0f) r_new = 0.0f;
        
        temp_regrets[i] = r_new;
        
        // Tracciamo il max per il BFP
        if (r_new > max_regret) max_regret = r_new;
    }

    // 2. Calcolo nuova scala (Block Floating Point)
    if (max_regret < 1e-9f) {
        node->scale = 0.0f;
        std::fill(node->regrets, node->regrets + num_actions, 0);
        return;
    }

    // La nuova scala mappa il max_regret esattamente a 255
    float new_scale = 255.0f / max_regret;
    node->scale = new_scale; // Salviamo la scala per la prossima iterazione (o per calcolare la strategia)

    // 3. Compressione con Stochastic Rounding
    for (int i = 0; i < num_actions; ++i) {
        float normalized = temp_regrets[i] * new_scale;
        
        // Fast stochastic rounding
        // Aggiungiamo un numero casuale [0,1) e tronchiamo
        float random_val = GetFastRandomFloat(); // Implementare Xorshift o simile
        
        int quantized = static_cast<int>(normalized + random_val);
        
        // Clamp di sicurezza (teoricamente non serve se max_regret è calcolato bene, ma utile per float error)
        if (quantized > 255) quantized = 255;
        
        node->regrets[i] = static_cast<uint8_t>(quantized);
    }
}

```

### Perché questo migliora il tuo Report

Nel report attuale, scrivi che usi `u8`. Senza BFP, se hai un regret di 1000 e uno di 1 nello stesso nodo:

* Standard Scaling: , .
* Stochastic Rounding su 0.255: 75% diventa 0, 25% diventa 1. C'è molto "rumore" di quantizzazione.


* BFP Scaling:
* Se in iterazioni future il regret grande scende a 10 e quello piccolo resta 1, la scala si adatta automaticamente (), dando al regret piccolo molta più "risoluzione" (diventa 25.5).



### Prossimo Step Operativo

Il collo di bottiglia diventerà la generazione di numeri casuali (`GetFastRandomFloat`).
**Vuoi che ti mostri come vettorizzare (AVX2/AVX-512) questa routine di Stochastic Rounding?** Con SIMD puoi generare e roundare 8 o 16 regret contemporaneamente, rendendo il costo computazionale del BFP quasi nullo.

Ecco l'implementazione ottimizzata con **AVX2**.

Questa soluzione risolve il problema principale del Block Floating Point (BFP): la necessità di calcolare il massimo *prima* di comprimere. Per farlo in modo efficiente senza allocazioni di memoria costose, usiamo un piccolo buffer temporaneo nello stack (dato che il numero di azioni nel poker è solitamente basso, < 100).

### Strategia di Vettorizzazione

1. **Fast PRNG (Xorshift):** Generare numeri casuali standard è lento. Usiamo un generatore Xorshift vettoriale che lavora direttamente nei registri AVX per il **Stochastic Rounding**.
2. **Double Pass (L1 Cache Friendly):**
* **Pass 1:** Decompressione `u8` -> Aggiornamento DCFR+ -> Calcolo Max -> Salvataggio in buffer temporaneo `float`.
* **Pass 2:** Caricamento buffer -> Scaling -> Stochastic Rounding -> Packing `u8`.



### Codice C++ (AVX2 Intrinsics)

```cpp
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <cmath>

// Costanti per Xorshift PRNG vettoriale
const __m256i XORSHIFT_A = _mm256_set1_epi32(13);
const __m256i XORSHIFT_B = _mm256_set1_epi32(17);
const __m256i XORSHIFT_C = _mm256_set1_epi32(5);
// Maschera per convertire int random in float [0, 1)
const __m256 FLOAT_ONE_MINUS_EPS = _mm256_set1_ps(1.0f - 1e-7f); 

struct CompressedInfoset {
    float scale;
    uint8_t* regrets; // Puntatore all'array di u8
};

// Generatore casuale AVX2 molto veloce (stato passato per reference)
inline __m256 next_rand_avx2(__m256i& state) {
    __m256i x = state;
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
    state = x;
    
    // Conversione veloce int -> float [0, 1]
    // Manteniamo solo i 23 bit della mantissa e impostiamo esponente a 1.0
    // Poi sottraiamo 1.0. (Trick standard IEEE 754)
    __m256i mantissa_mask = _mm256_set1_epi32(0x7FFFFF);
    __m256i one_exponent = _mm256_set1_epi32(0x3F800000);
    
    __m256i r = _mm256_and_si256(x, mantissa_mask);
    r = _mm256_or_si256(r, one_exponent);
    
    return _mm256_sub_ps(_mm256_castsi256_ps(r), _mm256_set1_ps(1.0f));
}

void update_node_avx2(CompressedInfoset* node, 
                      const float* inst_regrets, 
                      float alpha, 
                      int num_actions,
                      __m256i& rng_state) {

    // Buffer temporaneo nello stack (veloce, L1 cache). 
    // Assumiamo num_actions < 128 per sicurezza stack. 
    // Per alberi enormi, usare un buffer thread-local statico.
    float temp_regrets[128]; 
    
    __m256 v_alpha = _mm256_set1_ps(alpha);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_max_regret = _mm256_setzero_ps();
    
    // Calcola l'inverso della scala vecchia
    float old_scale_inv = (node->scale > 1e-9f) ? (1.0f / node->scale) : 0.0f;
    __m256 v_old_scale_inv = _mm256_set1_ps(old_scale_inv);

    int i = 0;
    
    // --- PASS 1: Update & Find Max ---
    for (; i <= num_actions - 8; i += 8) {
        // 1. Load u8 regrets (64 bit load -> expand to 256 bit float)
        __m128i raw_u8 = _mm_loadl_epi64((__m128i const*)&node->regrets[i]);
        __m256i int32_regrets = _mm256_cvtepu8_epi32(raw_u8);
        __m256 v_r_old = _mm256_cvtepi32_ps(int32_regrets);
        
        // 2. Decompress
        v_r_old = _mm256_mul_ps(v_r_old, v_old_scale_inv);
        
        // 3. Update: (R * alpha + inst)
        __m256 v_inst = _mm256_loadu_ps(&inst_regrets[i]);
        __m256 v_r_new = _mm256_fmadd_ps(v_r_old, v_alpha, v_inst);
        
        // 4. Clip to 0 (DCFR+ requires positive accumulation)
        v_r_new = _mm256_max_ps(v_r_new, v_zero);
        
        // 5. Store temp & Track Max
        _mm256_storeu_ps(&temp_regrets[i], v_r_new);
        v_max_regret = _mm256_max_ps(v_max_regret, v_r_new);
    }
    
    // Coda scalare per Pass 1
    float scalar_max = 0.0f;
    for (; i < num_actions; ++i) {
        float r_old = (float)node->regrets[i] * old_scale_inv;
        float r_new = (r_old * alpha) + inst_regrets[i];
        if (r_new < 0.0f) r_new = 0.0f;
        temp_regrets[i] = r_new;
        if (r_new > scalar_max) scalar_max = r_new;
    }

    // Riduzione orizzontale del Max AVX
    float max_arr[8];
    _mm256_storeu_ps(max_arr, v_max_regret);
    for(float v : max_arr) if(v > scalar_max) scalar_max = v;
    
    // --- CALCOLO NUOVA SCALA (BFP) ---
    if (scalar_max < 1e-9f) {
        node->scale = 0.0f;
        // Azzera tutto velocemente
        std::fill(node->regrets, node->regrets + num_actions, 0);
        return;
    }
    
    // Nuova scala: Mappa il valore massimo a 255
    float new_scale = 255.0f / scalar_max;
    node->scale = new_scale;
    __m256 v_new_scale = _mm256_set1_ps(new_scale);

    // --- PASS 2: Compress & Stochastic Rounding ---
    i = 0;
    for (; i <= num_actions - 8; i += 8) {
        __m256 v_val = _mm256_loadu_ps(&temp_regrets[i]);
        
        // 1. Normalize (0..255 range float)
        v_val = _mm256_mul_ps(v_val, v_new_scale);
        
        // 2. Generate Random Noise [0, 1) per Stochastic Rounding
        __m256 v_noise = next_rand_avx2(rng_state);
        
        // 3. Add noise and Truncate
        // floor(x + rand) è equivalente a stochastic rounding
        __m256 v_rounded = _mm256_add_ps(v_val, v_noise);
        __m256i v_rounded_i = _mm256_cvttps_epi32(v_rounded);
        
        // 4. Pack 32-bit int -> 8-bit int (con saturation)
        // AVX2 pack sequence: 32 -> 16 -> 8
        // Nota: packus lavora su 128-bit lanes, serve un po' di permute se l'ordine conta strettamente,
        // ma per i regret possiamo processare a blocchi.
        // Qui facciamo un pack semplificato assumendo che stiamo scrivendo in ordine.
        
        // Hack veloce per pack: estrarre valori.
        // Poiché packus è complesso su 256bit cross-lane, per 8 valori
        // spesso è più veloce estrarre o usare _mm256_shuffle_epi8 se abbiamo lookup table.
        // Metodo solido:
        // Convert to 16-bit
        __m256i v_perm = _mm256_permute4x64_epi64(v_rounded_i, 0xD8); // Riordina per pack
        __m128i v_lo = _mm256_castsi256_si128(v_perm);
        __m128i v_hi = _mm256_extracti128_si256(v_perm, 1);
        __m128i v_packed_16 = _mm_packus_epi32(v_lo, v_hi);
        __m128i v_packed_8 = _mm_packus_epi16(v_packed_16, _mm_setzero_si128());
        
        // Store (scrive solo i primi 64 bit / 8 byte)
        _mm_storel_epi64((__m128i*)&node->regrets[i], v_packed_8);
    }

    // Coda scalare per Pass 2
    for (; i < num_actions; ++i) {
        float norm = temp_regrets[i] * new_scale;
        // Semplice LCG o rand() standard per la coda
        float rnd = (float)rand() / RAND_MAX; 
        int rounded = (int)(norm + rnd);
        if (rounded > 255) rounded = 255;
        node->regrets[i] = (uint8_t)rounded;
    }
}

```

### Note Tecniche per l'Integrazione

1. **Stochastic Rounding:** L'implementazione realizza esattamente la formula del report:  con probabilità . L'aggiunta di rumore uniforme `[0, 1)` seguita da troncamento (`cvttps`) è matematicamente equivalente e molto più veloce in SIMD ("branchless").
2. **Gestione della Memoria:** L'uso di `temp_regrets[128]` nello stack è cruciale. Evita di dover allocare/deallocare memoria float nell'heap per ogni nodo, mantenendo l'impronta di memoria RAM bassa come desiderato (solo `u8` salvati in modo permanente).
3. **Inizializzazione RNG:** Ricordati di inizializzare `rng_state` una volta per thread (es. `_mm256_set_epi32(seed1, seed2, ...)`).

Vuoi che analizzi anche come applicare una compressione simile (BFP + SIMD) alla **Strategia Media** (`u16`) per velocizzare anche la parte di averaging?

Perfetto. Passiamo all'ottimizzazione della **Strategia Media Accumulata**.

Nel report indichi che usi **16-bit Unsigned Integers (`u16`)** per la strategia. Questo è corretto perché la strategia finale richiede più precisione dei regret per essere "exploitabile" il meno possibile.

Tuttavia, accumulare la strategia su milioni di iterazioni con un semplice intero (o fixed point) porta a due problemi opposti:

1. **Overflow:** Se accumuli troppo a lungo, superi `65535`.
2. **Underflow/Loss:** Se scali giù preventivamente per evitare l'overflow, le nuove iterazioni (che magari hanno peso basso all'inizio) spariscono a causa dell'arrotondamento.

Anche qui, il **Block Floating Point (BFP)** con `u16` risolve entrambi i problemi, permettendo al "peso" totale di crescere indefinitamente (tramite la scala `float`) mentre i valori relativi mantengono 16 bit di precisione.

### Algoritmo: BFP per Strategia Accumulata

La formula del report è:


Dove:

*  è il valore accumulato (compresso).
*  è la strategia corrente (probabilità float ).
*  è il peso dell'iterazione corrente.
*  è il fattore di fading (spesso  o poco meno).

### Implementazione C++ AVX2 (Strategy)

A differenza dei regret (che usano `u8`), qui lavoriamo con `u16`. Questo cambia leggermente le istruzioni SIMD necessarie (packing da 32 a 16 bit è più semplice).

```cpp
#include <immintrin.h>
#include <vector>
#include <algorithm>

struct CompressedStrategy {
    float scale;       // Scale factor: RealValue = stored_u16 / scale
    uint16_t* weights; // Array di u16
};

// Aggiorna la strategia accumulata usando AVX2
// current_strategy_probs: array float delle probabilità attuali (somma 1.0)
// discount_gamma: fattore di decadimento (es. 1.0 o 0.99...)
// iter_weight: peso dell'iterazione corrente
void update_strategy_avx2(CompressedStrategy* node, 
                          const float* current_strategy_probs, 
                          float discount_gamma, 
                          float iter_weight,
                          int num_actions,
                          __m256i& rng_state) { // Stesso PRNG dei regret

    // Buffer temporaneo veloce (L1 cache)
    float temp_weights[128]; 
    
    // Costanti SIMD
    __m256 v_gamma = _mm256_set1_ps(discount_gamma);
    __m256 v_weight = _mm256_set1_ps(iter_weight);
    __m256 v_max_val = _mm256_setzero_ps();

    // Gestione scala precedente
    // Nota: Per la strategia, spesso si definisce Scale come: Real = Stored * Scale
    // Qui usiamo la convenzione inversa per coerenza coi regret: Real = Stored / Scale
    float old_scale_inv = (node->scale > 1e-12f) ? (1.0f / node->scale) : 0.0f;
    __m256 v_old_scale_inv = _mm256_set1_ps(old_scale_inv);

    int i = 0;

    // --- PASS 1: Decompress & Update ---
    for (; i <= num_actions - 8; i += 8) {
        // 1. Load u16 (espansione a 32-bit int poi float)
        // _mm_loadu_si128 carica 128 bit (8 x 16-bit ints)
        __m128i raw_u16 = _mm_loadu_si128((__m128i const*)&node->weights[i]);
        
        // Converti u16 -> i32 (AVX2: cvtepu16_epi32)
        __m256i int32_vals = _mm256_cvtepu16_epi32(raw_u16);
        __m256 v_old_acc = _mm256_cvtepi32_ps(int32_vals);
        
        // 2. Decompress
        v_old_acc = _mm256_mul_ps(v_old_acc, v_old_scale_inv);
        
        // 3. Update Formula: Old * Gamma + CurrProbs * IterWeight
        __m256 v_curr = _mm256_loadu_ps(&current_strategy_probs[i]);
        __m256 v_add = _mm256_mul_ps(v_curr, v_weight);
        __m256 v_new_acc = _mm256_fmadd_ps(v_old_acc, v_gamma, v_add);
        
        // 4. Track Max (per ricalcolare la scala)
        v_max_val = _mm256_max_ps(v_max_val, v_new_acc);
        
        _mm256_storeu_ps(&temp_weights[i], v_new_acc);
    }
    
    // Coda scalare Pass 1
    float scalar_max = 0.0f;
    for (; i < num_actions; ++i) {
        float old_val = (float)node->weights[i] * old_scale_inv;
        float new_val = (old_val * discount_gamma) + (current_strategy_probs[i] * iter_weight);
        temp_weights[i] = new_val;
        if (new_val > scalar_max) scalar_max = new_val;
    }

    // Riduzione Max
    float max_arr[8];
    _mm256_storeu_ps(max_arr, v_max_val);
    for(float v : max_arr) if(v > scalar_max) scalar_max = v;

    // --- CALCOLO NUOVA SCALA ---
    if (scalar_max < 1e-12f) {
        // Strategia nulla (improbabile, ma gestiamo)
        node->scale = 0.0f;
        std::fill(node->weights, node->weights + num_actions, 0);
        return;
    }

    // Mappiamo il valore massimo a 65535 (massima precisione u16)
    float new_scale = 65535.0f / scalar_max;
    node->scale = new_scale;
    __m256 v_new_scale = _mm256_set1_ps(new_scale);

    // --- PASS 2: Compress & Stochastic Rounding ---
    i = 0;
    for (; i <= num_actions - 8; i += 8) {
        __m256 v_val = _mm256_loadu_ps(&temp_weights[i]);
        
        // Normalize
        v_val = _mm256_mul_ps(v_val, v_new_scale);
        
        // Stochastic Rounding (Cruciale anche qui per accumuli lenti)
        __m256 v_noise = next_rand_avx2(rng_state); // Riutilizza la funzione precedente
        __m256 v_rounded = _mm256_add_ps(v_val, v_noise);
        __m256i v_rounded_i = _mm256_cvttps_epi32(v_rounded);
        
        // Pack 32-bit -> 16-bit
        // AVX2 packus_epi32 prende due registri 256 e li impacchetta in uno. 
        // Poiché ne abbiamo uno solo, dobbiamo essere creativi o usare permute.
        // Metodo standard per singolo registro: 
        // 1. Converti a __m128i (low e high lane)
        // 2. Packus
        
        // Estrazione lane
        __m128i lane_lo = _mm256_castsi256_si128(v_rounded_i);
        __m128i lane_hi = _mm256_extracti128_si256(v_rounded_i, 1);
        
        // Pack: i32 -> u16 (saturazione gestita automaticamente da packus)
        __m128i v_packed_16 = _mm_packus_epi32(lane_lo, lane_hi);
        
        // Store 128 bit (8 x u16)
        _mm_storeu_si128((__m128i*)&node->weights[i], v_packed_16);
    }
    
    // Coda scalare Pass 2
    for (; i < num_actions; ++i) {
        float norm = temp_weights[i] * new_scale;
        float rnd = (float)rand() / RAND_MAX; 
        int rounded = (int)(norm + rnd);
        if (rounded > 65535) rounded = 65535;
        node->weights[i] = (uint16_t)rounded;
    }
}

```

### Perché questo è critico per DCFR+

1. **Uniform Discounting vs. BFP:**
   DCFR+ usa un discounting aggressivo ( o ). In un sistema fixed-point classico, quando  diventa molto piccolo, i valori "vecchi" verrebbero arrotondati a zero troppo presto.
   Con BFP, la scala `float` assorbe il fattore di grandezza. Se la strategia accumulata totale passa da  a , i valori nell'array `u16` rimangono nel range "pieno" , mantenendo la precisione relativa intatta.
2. **Stochastic Rounding sulla Strategia:**
   È meno discusso rispetto ai regret, ma fondamentale. Quando aggiungi l'iterazione corrente (peso ) a una strategia accumulata massiccia (peso cumulativo enorme), il contributo di oggi potrebbe essere  unità `u16` dopo lo scaling.
   Senza arrotondamento stocastico, il solver "smette di imparare" non appena l'accumulatore diventa abbastanza grande da rendere  insignificante rispetto alla risoluzione dell'intero. Lo stochastic rounding garantisce che, in media, anche incrementi infinitesimali vengano registrati.

### Sommario delle Ottimizzazioni Proposte

Se integri queste modifiche al tuo report attuale:

| Componente | Stato Attuale (Report) | Nuovo Stato Ottimizzato | Vantaggio |
| --- | --- | --- | --- |
| **Regret Storage** | `u8` (Fixed Scale) | `u8` + Float Scale (**BFP**) | Precisione su mani rare / bluff |
| **Strategy Storage** | `u16` (Fixed Scale) | `u16` + Float Scale (**BFP**) | Nessun overflow, learning continuo |
| **Calcolo** | Standard Loop | **AVX2 SIMD** | 4x-8x velocità di update nei nodi |
| **Rounding** | Stocastico (Scalar) | **Stocastico Vectorized** | Elimina il collo di bottiglia `rand()` |

Hai tutto il codice necessario per queste componenti.