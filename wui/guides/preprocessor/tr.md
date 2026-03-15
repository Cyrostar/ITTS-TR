### 👁️ Genel Bakış

Ön İşlemci (Özellik Çıkarımı - Feature Extraction) modülü, ham veri setiniz ile akustik eğitim aşaması arasında kritik bir köprüdür. Ham ses ve metin doğrudan TTS modeline beslenemez. Bunun yerine bu modül, verilerinizi birden fazla önceden eğitilmiş sinir ağından geçirerek yüksek boyutlu matematiksel temsiller (özellikler) çıkarır ve bunları `.npy` dizileri olarak kaydeder.

#### 📂 1. Veri Kaynağı ve Temel Ayarlar

Özellikleri çıkarmadan önce, hedef veri setini ve verilerin eğitim için nasıl yapılandırılması gerektiğini tanımlamalısınız.

- **Hedef Veri Seti (Target Dataset):** `metadata.csv` ve `wavs/` dizininizi içeren (Aşama 3'te oluşturulan) belirli veri seti klasörü.
- **Klasör Dili (Folder Language):** Veri seti açılır menüsünü (dropdown) yalnızca belirli bir dil etiketine ait veri setlerini gösterecek şekilde filtreler.
- **Dil İşaretçisi Enjekte Et (Inject Language Marker):** Metin simgelerinin (tokens) başına belirli bir dil kimliği simgesinin eklenip eklenmeyeceğini belirler.
  - **None (Hiçbiri):** Herhangi bir dil kimliği enjekte edilmez.
  - **TR (ID-3) / EN (ID-4):** Modelin dili açıkça tanımasını zorlar; bu, çok dilli modeller için çok önemlidir.
- **Doğrulama Ayrımı (Validation Split %):** Veri setinizin eğitimden ne kadarının ayrılacağını (yüzde olarak) belirler. Ayrılan bu veri (`val.jsonl`), eğitim aşamasında modelin daha önce görmediği veriler üzerindeki doğruluğunu test etmek için kullanılır.

#### ⚡ 2. Performans Ayarları

Özellik çıkarımı yüksek oranda kaynak tüketir. Bu ayarlar, donanımınızın sınırlarına karşı hızı dengelemenizi sağlar.

- **Yığın Boyutu (Batch Size):** GPU tarafından aynı anda işlenen ses kliplerinin sayısıdır. CUDA Bellek Yetersizliği (OOM) hatalarıyla karşılaşırsanız bu değeri düşürün.
- **CPU Çalışanları (CPU Workers):** Ses dosyalarını diskinizden yüklemek için ayrılmış paralel CPU iş parçacıklarının sayısıdır. Yüksek değerler veri hattını hızlandırır ancak daha fazla sistem RAM'i tüketir.

#### ⚙️ 3. Gelişmiş Yapılandırma

- **Göreceli Yollar Kullan (Use Relative Paths):** İşaretlendiğinde, oluşturulan bildirim dosyaları (`.jsonl`) mutlak (absolute) yollar yerine göreceli (relative) yolları depolar. Proje klasörünüzü veri seti bağlantılarını bozmadan başka bir sürücüye veya makineye taşımanıza olanak tanıdığı için bu şiddetle tavsiye edilir.
- **Birleştirilmiş Simgeleştirici Kullan (Use Merged Tokenizer):** Çıkarıcıya (extractor) standart simgeleştirici yerine `_bpe_merged.model` dosyasını kullanmasını söyler. Bunu yalnızca birden fazla simgeleştiriciyi açıkça birleştirdiyseniz kullanın.
- **Torch Compile:** Çıkarım modellerini optimize etmek için PyTorch 2.0+ `torch.compile()` işlevini kullanır. Bu, çıkarma işlemini önemli ölçüde hızlandırır ancak işlemin donmuş gibi görüneceği bir başlangıç "ısınma" süresi gerektirir.

#### 🧲 4. Kaputun Altında: Neler Çıkarılıyor?

Sistem, her geçerli ses-metin çifti için dört belirli özellik çıkarır ve bunları `extractions/<veri_seti_adi>` klasörüne kaydeder:

1. **Metin Kimlikleri (`text_ids/`):** Ham metin, özel SentencePiece modeliniz kullanılarak bir tamsayı dizisine dönüştürülür (tokenize edilir).
2. **Semantik Kodlar (`codes/`):** Ses, `W2V-BERT 2.0` özellik çıkarıcısından geçirilir ve ayrık ses simgeleri (discrete audio tokens) oluşturmak için `MaskGCT` semantik kodeği kullanılarak nicelleştirilir (quantized).
3. **Koşullandırma (`condition/`):** UnifiedVoice GPT modeli tarafından işlenen üst düzey akustik özellikler.
4. **Duygu Vektörleri (`emo_vec/`):** Semantik özelliklerden çıkarılan duygusal ve prozodik (tonlama/vurgu) gömülmeler (embeddings).

Son olarak modül, orijinal ses yollarını yeni oluşturulan bu `.npy` özellik dizileriyle eşleştiren iki bildirim dosyası (`train.jsonl` ve `val.jsonl`) oluşturur.
