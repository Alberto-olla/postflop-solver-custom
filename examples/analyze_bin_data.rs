use postflop_solver::*;

fn main() {
    // Carica il file .bin
    let path = "hands/7438/solved_games/hand_0000007438_node_03_turn_DeepStack-32.bin";

    println!("=== ANALISI FILE .BIN ===\n");
    println!("File: {}", path);

    let result = load_data_from_file::<PostFlopGame, _>(path, None);

    match result {
        Ok((mut game, memo)) => {
            println!("Memo: {}", memo);
            println!("\n=== CONFIGURAZIONE ===");
            println!("Quantization mode: {:?}", game.quantization_mode());
            println!("Strategy bits: {}", game.strategy_bits());
            println!("Compression enabled: {}", game.is_compression_enabled());
            println!("Storage mode: {:?}", game.storage_mode());
            
            println!("\n=== MEMORIA ===");
            println!("Estimated memory usage: {:.2} MB", game.estimated_memory_usage() as f64 / 1_048_576.0);
            let (mem_uncompressed, mem_compressed) = game.memory_usage();
            println!("Current memory usage: {:.2} MB (uncompressed), {:.2} MB (compressed)",
                     mem_uncompressed as f64 / 1_048_576.0,
                     mem_compressed as f64 / 1_048_576.0);
            println!("Actual storage usage: {:.2} MB", game.memory_usage_mb());

            // Analizza i dati delle strategie - usa l'API pubblica
            println!("\n=== ANALISI DATI STRATEGIE ===");

            // Calcola il numero totale di elementi basandosi sulla memoria allocata
            // Usiamo memory_usage_mb() per stimare
            let storage_mb = game.memory_usage_mb();
            let bytes_per_element = if game.is_compression_enabled() {
                if game.strategy_bits() == 8 {
                    1 + 2  // 1 byte strategy + 2 bytes regrets
                } else {
                    2 + 2  // 2 bytes strategy + 2 bytes regrets
                }
            } else {
                4 + 4  // 4 bytes strategy + 4 bytes regrets
            };

            let total_elements_estimate = (storage_mb * 1_048_576.0 / bytes_per_element as f64) as u64;

            println!("Estimated total strategy elements: ~{}", total_elements_estimate);
            
            println!("\n=== VALUTAZIONE TECNICHE DI COMPRESSIONE ===");

            // Analisi basata sul quantization mode attuale
            if game.quantization_mode() == QuantizationMode::Float32 {
                println!("\n⚠️  ATTENZIONE: Il file usa Float32 (NON COMPRESSO)");
                println!("   Questo è il formato meno efficiente in termini di memoria.");

                let current_bytes = total_elements_estimate * 8; // 4 bytes strategy + 4 bytes regrets
                let with_16bit = total_elements_estimate * 4;    // 2 bytes strategy + 2 bytes regrets
                let with_8bit = total_elements_estimate * 3;     // 1 byte strategy + 2 bytes regrets

                let saving_16bit = current_bytes - with_16bit;
                let saving_8bit = current_bytes - with_8bit;

                println!("\n1. QUANTIZZAZIONE 16-BIT (quantization = \"16bit\"):");
                println!("   Attuale (Float32): {} MB", current_bytes as f64 / 1_048_576.0);
                println!("   Con 16-bit: {} MB", with_16bit as f64 / 1_048_576.0);
                println!("   Risparmio: {} MB ({:.1}%)",
                         saving_16bit as f64 / 1_048_576.0,
                         saving_16bit as f64 / current_bytes as f64 * 100.0);
                println!("   ✓ FORTEMENTE RACCOMANDATO");
                println!("   ✓ GIÀ IMPLEMENTATO - basta cambiare config TOML");

                println!("\n2. MIXED PRECISION (quantization = \"16bit\" + strategy_bits = 8):");
                println!("   Attuale (Float32): {} MB", current_bytes as f64 / 1_048_576.0);
                println!("   Con 8-bit strategy: {} MB", with_8bit as f64 / 1_048_576.0);
                println!("   Risparmio: {} MB ({:.1}%)",
                         saving_8bit as f64 / 1_048_576.0,
                         saving_8bit as f64 / current_bytes as f64 * 100.0);
                println!("   ✓ MASSIMO RISPARMIO");
                println!("   ✓ GIÀ IMPLEMENTATO - basta cambiare config TOML");

            } else {
                // Caso Int16
                let current_strategy_bytes = if game.strategy_bits() == 16 {
                    total_elements_estimate * 2
                } else {
                    total_elements_estimate * 1
                };
                let current_regrets_bytes = total_elements_estimate * 2;
                let current_total = current_strategy_bytes + current_regrets_bytes;

                let with_8bit_strategy = total_elements_estimate * 1;
                let with_8bit_total = with_8bit_strategy + current_regrets_bytes;

                let saving_bytes = current_total - with_8bit_total;
                let saving_pct = saving_bytes as f64 / current_total as f64 * 100.0;

                println!("\n1. MIXED PRECISION (strategy_bits = 8):");
                println!("   Attuale: {} MB strategy + {} MB regrets = {} MB",
                         current_strategy_bytes as f64 / 1_048_576.0,
                         current_regrets_bytes as f64 / 1_048_576.0,
                         current_total as f64 / 1_048_576.0);
                println!("   Con 8-bit: {} MB strategy + {} MB regrets = {} MB",
                         with_8bit_strategy as f64 / 1_048_576.0,
                         current_regrets_bytes as f64 / 1_048_576.0,
                         with_8bit_total as f64 / 1_048_576.0);
                println!("   Risparmio: {} MB ({:.1}%)",
                         saving_bytes as f64 / 1_048_576.0, saving_pct);
                println!("   ✓ GIÀ IMPLEMENTATO nella tua codebase!");
            }
            
            println!("\n2. QUANTIZZAZIONE LINEARE:");
            println!("   ✓ GIÀ IMPLEMENTATA (encode_unsigned_strategy_u8)");
            println!("   - Formula: quantized = round(value * 255 / max)");
            println!("   - Precisione: ~0.39% (1/255)");
            
            println!("\n3. DELTA ENCODING:");
            println!("   Applicabilità: BASSA");
            println!("   - Le strategie sono probabilità normalizzate, non sequenze temporali");
            println!("   - I valori non hanno correlazione sequenziale");
            
            println!("\n4. VARINT:");
            println!("   Applicabilità: BASSA");
            println!("   - Efficace solo se la maggior parte dei valori è < 127");
            println!("   - Le strategie normalizzate usano tutto il range 0-255");
            println!("   ✗ NON CONSIGLIATO");
            
            println!("\n5. FIXED POINT:");
            println!("   Applicabilità: BASSA");
            println!("   - Le strategie sono già normalizzate (0-1)");
            println!("   - La quantizzazione lineare è più efficiente");
            
            println!("\n6. MINI-FLOATS (8-bit):");
            println!("   Applicabilità: MEDIA");
            println!("   - Utile per range dinamico molto ampio");
            println!("   - Le strategie hanno range limitato (0-1)");
            println!("   - Complessità implementativa alta");
            println!("   ✗ NON NECESSARIO per questo caso d'uso");
            
            println!("\n=== RACCOMANDAZIONI FINALI ===");

            if game.quantization_mode() == QuantizationMode::Float32 {
                println!("\n🎯 AZIONE IMMEDIATA RACCOMANDATA:");
                println!("   1. Cambia il file di configurazione TOML:");
                println!("      [solver]");
                println!("      quantization = \"16bit\"");
                println!("      strategy_bits = 8");
                println!("   ");
                println!("   2. Risparmio atteso: ~62.5% di memoria");
                println!("   3. Impatto sulla convergenza: MINIMO");
                println!("   4. Già implementato: SÌ - nessun codice da scrivere!");

            } else if game.strategy_bits() == 16 {
                println!("\n🎯 OTTIMIZZAZIONE DISPONIBILE:");
                println!("   Aggiungi al file TOML:");
                println!("   [solver]");
                println!("   strategy_bits = 8");
                println!("   ");
                println!("   Risparmio atteso: ~25% di memoria");

            } else {
                println!("\n✅ CONFIGURAZIONE OTTIMALE:");
                println!("   Stai già usando la configurazione più efficiente!");
                println!("   - quantization = \"16bit\"");
                println!("   - strategy_bits = 8");
            }

            println!("\n📊 COMPRESSIONE FILE (opzionale):");
            println!("   - Usa save_data_to_file(..., Some(3)) per compressione ZSTD");
            println!("   - Riduce ulteriormente la dimensione dei file .bin");
            println!("   - Trade-off: tempo di I/O vs spazio su disco");

            println!("\n❌ TECNICHE NON APPLICABILI (dal report):");
            println!("   ✗ Delta encoding: le strategie non sono sequenze temporali");
            println!("   ✗ Varint: beneficio marginale con dati normalizzati");
            println!("   ✗ Fixed point: già coperto dalla quantizzazione lineare");
            println!("   ✗ Mini-floats: complessità non giustificata per range 0-1");

            println!("\n📝 CONCLUSIONE:");
            println!("   La tua codebase ha GIÀ implementato le tecniche più efficaci");
            println!("   per questo tipo di dati (strategie poker normalizzate).");
            println!("   Le altre tecniche del report sono per casi d'uso diversi.");
        }
        Err(e) => {
            eprintln!("Errore nel caricamento del file: {}", e);
        }
    }
}

