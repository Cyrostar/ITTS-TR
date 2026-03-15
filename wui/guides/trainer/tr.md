### 👁️ Genel Bakış

Model Eğitimi (Model Training) modülü, Ön İşlemci (Preprocessor) tarafından oluşturulan özellikleri alır ve UnifiedVoice GPT akustik modelini ince ayar (fine-tune) yapmak için kullanır. Bu aşama, modele hedef dilinizin metin simgelerini karşılık gelen akustik semantik kodlarla nasıl eşleştireceğini öğreterek veri setinizin benzersiz sesini, ritmini ve duygusunu yakalar.

#### 📂 1. Proje ve Veri Keşfi

Diğer modüllerden farklı olarak eğitmen, yapılandırma uyuşmazlıklarını önlemek için büyük ölçüde otomasyona güvenir.

- **Proje Seç (Select Project):** Bir proje seçmek, projenin `extractions/` klasörünü otomatik olarak tarar.
- **Otomatik Keşif:** Sistem `config.yaml`, `bpe.model` (simgeleştirici) ve `train.jsonl` / `val.jsonl` bildirim dosyalarınızı (manifests) bulacaktır. Bu kritik dosyalardan herhangi biri eksikse veya uyuşmuyorsa başlamayı reddedecektir.
- **Çalıştırma Adı (Run Name):** Kontrol noktalarının (checkpoints) ve TensorBoard günlüklerinin kaydedileceği `trains/` dizini altındaki klasör adını tanımlar. Boş bırakılırsa, varsayılan olarak proje adını alır.

#### 🛠️ 2. Temel Hiperparametreler

Bu ayarlar sinir ağının nasıl öğreneceğini belirler.

- **Dönemler (Epochs):** Modelin tüm eğitim veri seti üzerinde kaç kez döneceğini (tekrarlayacağını) belirtir.
- **Yığın Boyutu (Batch Size):** GPU'ya aynı anda yüklenen ses kliplerinin sayısıdır. CUDA Bellek Yetersizliği (OOM) hatalarıyla karşılaşırsanız bunu azaltın.
- **Gradyan Birikimi (Gradient Accumulation):** Modelin ağırlıklarını güncellemeden önce birden fazla adım boyunca gradyanları biriktirir. *Mühendislik Notu: Gerçek Yığın Boyutu = Yığın Boyutu × Gradyan Birikimi.*
- **Öğrenme Oranı (Learning Rate):** Optimize edicinin ağırlıkları ayarlarken attığı adım boyutudur. İnce ayar için varsayılan değer (`2e-5`) şiddetle tavsiye edilir.
- **Doğrula ve Kaydet (Validate & Save Every):** Modelin doğrulama döngüsünü çalıştırmak ve bir `gpt_step_X.pth` kontrol noktası kaydetmek için (adım cinsinden) ne sıklıkla duraklayacağını belirler.

#### ⚙️ 3. Gelişmiş Kontroller

- **Süre Kontrolünü Kullan (Use Duration Control):** Etkinleştirildiğinde model, fonemlerin/kelimelerin süresini tahmin etmeyi açıkça öğrenir.
- **Süre Düşürme (Duration Dropout):** Eğitim sırasında süre bilgisini rastgele düşürür (Varsayılan: `0.3`). Bu, modeli yalnızca açık süre etiketlerine güvenmek yerine doğal tempoyu ve ritmi içsel olarak öğrenmeye zorlar ve daha doğal duyulan çıkarımlara (inference) yol açar.

#### 📦 4. Kontrol Noktası Ağırlık Dışa Aktarımı (Unwrapping)

Eğitim sırasında modeller, çıkarım (inference) için gereksiz olan meta veriler ve optimize edici durumları ile kaydedilir.

- **Paketi Aç ve Kaydet (Unwrap and Save):** Bu araç, optimize edici durumunu (optimizer state) soyar ve sözlük anahtarlarını temizler (örneğin, dağıtılmış eğitimden kalan `module.` öneklerini kaldırır).
- **Conformer Eşlemesi (Conformer Mapping):** Etkinleştirilirse, düzleştirilmiş iç içe katmanları düzeltir (örn. `.conv_pointwise_conv1`'i `.conv_module.pointwise_conv1`'e geri dönüştürür).
- **Sonuç:** Seçtiğiniz `gpt_step_X.pth` dosyasını alır ve çıkarım için kopyalanmaya hazır, temiz, hafif bir `gpt.pth` dosyasını doğrudan eğitim klasörünüze dışa aktarır.
