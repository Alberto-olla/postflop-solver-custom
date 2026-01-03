Ecco un breakdown completo della gestione memoria nel solver:

📊 Componenti di Memoria Principali

Il solver usa 5 storage arrays globali allocati in PostFlopGame:

storage1 - Strategy (Media Cumulativa)

- Scopo: Strategia media cumulativa per ogni azione
- Dimensione: num_storage elementi
- Codifiche supportate:
    - Float32: 4 bytes/elemento
    - Int16 (default): 2 bytes/elemento (u16)
    - 8-bit (con strategy_bits = 8): 1 byte/elemento (u8) ✨
    - 4-bit (futuro, con strategy_bits = 4): 0.5 bytes/elemento

storage2 - Regrets o CFValues

- Scopo: Regrets cumulativi (durante solving) o CF values (evaluation)
- Dimensione: num_storage elementi
- Codifiche:
    - Float32: 4 bytes
    - Int16: 2 bytes (i16) con quantizzazione lineare
    - Int16Log: 2 bytes con encoding logaritmico sign * log(1 + |x|)

storage_ip (storage3) - IP Counterfactual Values

- Scopo: CF values del player IP a root/chance nodes
- Dimensione: num_storage_ip elementi
- Codifiche: Segue le stesse di storage2

storage_chance - Chance Node CFValues

- Scopo: CF values ai chance nodes (turn/river)
- Dimensione: num_storage_chance elementi
- Codifiche:
    - Float32/Int16 (default): 4/2 bytes
    - 8-bit (con chance_bits = 8): 1 byte (i8) ✨

storage4 - Previous Regrets

- Scopo: Regrets precedenti per algoritmi avanzati
- Quando serve: Solo con algorithm = "pdcfr+" o "sapcfr+"
- Dimensione: num_storage elementi
- Impatto: +50% memoria totale
- Codifiche: Segue storage2

Nota: storage2 e storage4 vengono liberati dopo finalize_and_release() → risparmio ~50% RAM.

🎯 Modalità Quantization

1. Float32 (32-bit)

quantization = "32bit"
- 4 bytes/elemento
- Massima precisione, nessuna compressione

2. Int16 (16-bit, default)

quantization = "16bit"
- 2 bytes/elemento
- Risparmio: 50% vs Float32
- Encoding: quantized = round(value / scale).clamp(-32768, 32767)
- Compatibile con mixed precision

3. Int16Log (16-bit logaritmico)

quantization = "16bit-log"
- 2 bytes/elemento
- Encoding: sign * log(1 + |x|) → migliore per valori con alto dynamic range
- Ideale per CFR+ con grandi swing di regrets

⚙️ Mixed Precision (con quantization="16bit")

strategy_bits - Compressione Strategy

[solver]
quantization = "16bit"
strategy_bits = 8    # Opzioni: 16 (default), 8, 4 (futuro)

| Valore | Bytes | Risparmio storage1 | Note                       |
  |--------|-------|--------------------|----------------------------|
| 16     | 2     | 0%                 | Default                    |
| 8      | 1     | 50%                | Riduce ~25% memoria totale |
| 4      | 0.5   | 75%                | Futuro, nibble packing     |

I regrets rimangono a 16-bit → convergenza preservata

chance_bits - Compressione Chance Nodes

chance_bits = 8    # Opzioni: 16 (default), 8
- 8-bit: 1 byte (50% risparmio su storage_chance)
- Utile per alberi grandi con molti chance nodes

📐 Formula Memoria Totale

Total RAM = storage1 + storage2 + storage_ip + storage_chance + [storage4]

Dove:
storage1 = strategy_bytes × num_storage
storage2 = regrets_bytes × num_storage
storage_ip = regrets_bytes × num_storage_ip
storage_chance = chance_bytes × num_storage_chance
storage4 = regrets_bytes × num_storage  (solo PDCFR+/SAPCFR+)

Bytes per Elemento - Tabella Comparativa

| Config                  | storage1(strategy) | storage2(regrets) | storage_ip | storage_chance | storage4 |
  |-------------------------|--------------------|-------------------|------------|----------------|----------|
| Float32                 | 4                  | 4                 | 4          | 4              | 4        |
| Int16 (default)         | 2                  | 2                 | 2          | 2              | 2        |
| Int16 + strategy_bits=8 | 1                  | 2                 | 2          | 2              | 2        |
| Int16 + chance_bits=8   | 2                  | 2                 | 2          | 1              | 2        |
| Int16 + entrambi=8      | 1                  | 2                 | 2          | 1              | 2        |

🧮 Calcolo num_storage*

num_storage

- Somma di num_elements di tutti i player action nodes
- Per node: num_actions × num_private_hands[current_player]
- Es: 3 azioni, 169 OOP hands → 3 × 169 = 507 elementi

num_storage_ip

- Somma di num_elements_ip per root e post-chance nodes
- Per node qualificato: num_private_hands[IP_player]

num_storage_chance

- Somma elementi di tutti i chance nodes
- Per chance node: num_private_hands[cfvalue_storage_player]

🚀 Impatto Algoritmi CFR

| Algorithm | storage4? | RAM Multiplier | Note      |
  |-----------|-----------|----------------|-----------|
| DCFR      | No        | 1.0×           | Baseline  |
| DCFR+     | No        | 1.0×           | Come DCFR |
| PDCFR+    | Sì        | 1.5×           | +50% RAM  |
| SAPCFR+   | Sì        | 1.5×           | +50% RAM  |

💡 Raccomandazioni per Config

Default (bilanciato)

[solver]
quantization = "16bit"
strategy_bits = 16
- 50% risparmio vs Float32
- Ottima convergenza

Compressione massima

[solver]
quantization = "16bit"
strategy_bits = 8
chance_bits = 8
algorithm = "dcfr"  # evita PDCFR+
- ~62.5% risparmio vs Float32
- Perdita precisione minima

Alto dynamic range

[solver]
quantization = "16bit-log"
- Migliore per CFR+ con grandi swing

Posizione codice

- Storage allocation: src/game/base.rs:437-515
- Quantization types: src/quantization.rs
- Node structure: src/game/mod.rs:132-153
