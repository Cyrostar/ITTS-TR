### 👁️ Genel Bakış

**Modeller** sekmesi, ITTS ardışık düzeni (pipeline) için gereken sinirsel ağırlıkların merkezi edinme ve dağıtım merkezi olarak hizmet verir. Yerel projeye özgü kontrol noktalarını, küresel paylaşılan mimarileri ve kritik ortam bağımlılıklarını işlemek için tasarlanmıştır.

### 📦 1. Proje Kontrol Noktaları (Yerel Depolama)

Bu bölüm, birincil Index-TTS model ağırlıklarına ayrılmıştır. Önceden eğitilmiş (pre-trained) depoların Hugging Face'ten doğrudan proje ortamınıza indirilmesini kolaylaştırır.

- **Depo Alımı:** En son model revizyonlarını getirmek için herhangi bir Hugging Face `Repo ID`'sini (örn. `IndexTeam/IndexTTS-2`) girebilirsiniz.

- **Otomatik Dağıtım:** Başarılı bir indirmenin ardından sistem, temel üçlü dosyayı—`bpe.model`, `gpt.pth` ve `config.yaml`—otomatik olarak tanımlayıp çıkarır ve Eğitim (Training) ve Çıkarım (Inference) aşamalarında anında kullanılabilmeleri için bunları küresel kontrol noktası dizinine (`ckpt`) kopyalar.

- **Dosya Tarayıcısı:** Gerçek zamanlı bir dizin görüntüleyici, yerel `indextts/checkpoints` yolundaki `.pth` ağırlıkları ve `.yaml` yapılandırmaları gibi temel dosyaların varlığını doğrulamanıza olanak tanır.

### 🌐 2. Küresel Önbellek Modelleri

Disk alanını optimize etmek ve gereksiz indirmeleri önlemek için, ağır temel modeller tüm projeler arasında paylaşılan küresel bir önbellekte saklanır. Bu modeller, TTS motoru için akustik ve mimari iskeleti sağlar:

- **W2V-BERT 2.0:** Üst düzey konuşma temsillerini çıkarmak için kullanılan devasa, öz-denetimli (self-supervised) bir ses kodlayıcı.

- **MaskGCT:** Özbağlanımlı olmayan (non-autoregressive) akustik üretim için özelleşmiş bir model.

- **CampPlus:** Yüksek doğruluklu konuşmacı gömme (speaker embedding) çıkarımı için kullanılır.

- **BigVGAN:** Matematiksel mel-spektrogramları yüksek sadakatli işitilebilir dalga formlarına dönüştürmek için kullanılan son teknoloji (state-of-the-art) vocoder.

### 🎙️ 3. Whisper Modelleri

Bu bölüm, ham sesin metne deşifre edilmesi (transcription) gereken "Derlem (Corpus)" aşaması için kritik olan **OpenAI Whisper** paketini yönetir.

- **Ayrıntılı Seçim:** `tiny`'den (hız için) `large-v3`'e (maksimum deşifre doğruluğu için) kadar tüm Whisper modelleri yelpazesinden seçim yapabilirsiniz.

- **Merkezi Model Yolu:** Tüm deşifre görevleri tarafından erişilebilir olmalarını sağlamak için modeller, kök sistem içinde özel bir dizine indirilir.

### 🛠️ 4. Bağımlılık ve Ortam Düzeltmeleri

Gelişmiş TTS kütüphaneleri, bazen bozuk üst akış (upstream) bağımlılıklarını veya eksik protokol arabelleklerini (protocol buffers) düzeltmek için manuel müdahale gerektirir.

- **SentencePiece Düzeltmesi:** `sentencepiece_model_pb2.py` dosyasını doğrudan resmi Google deposundan indirmek için özel bir yardımcı program.

- **Sistem Bütünlüğü:** Bu araç, kelime dağarcığı yeniden boyutlandırması sırasında `.model` dosyaları üzerinde cerrahi işlemler gerçekleştirmek için BPE tokenizer mantığının gerekli Python bağlamalarına (bindings) sahip olmasını sağlar.
