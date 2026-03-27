### 👁️ Genel Bakış

**Modeller** sekmesi, ITTS ardışık düzeni (pipeline) için gereken sinirsel ağırlıkların merkezi edinme ve dağıtım merkezi olarak hizmet verir. Yerel projeye özgü kontrol noktalarını, küresel paylaşılan mimarileri ve kritik ortam bağımlılıklarını işlemek için tasarlanmıştır.

### 📦 1. Proje Kontrol Noktaları (Yerel Depolama)

Bu bölüm, birincil Index-TTS model ağırlıklarına ayrılmıştır. Önceden eğitilmiş (pre-trained) depoların Hugging Face'ten doğrudan proje ortamınıza indirilmesini kolaylaştırır.

- **Depo Alımı:** En son model revizyonlarını getirmek için herhangi bir Hugging Face `Repo ID`'sini (örn. `IndexTeam/IndexTTS-2`) girebilirsiniz.
- **Otomatik Dağıtım:** Başarılı bir indirmenin ardından sistem, temel üçlü dosyayı—`bpe.model`, `gpt.pth` ve `config.yaml`—otomatik olarak tanımlayıp çıkarır ve Bağımsız (Standalone) TTS ve Eğitim (Training) aşamalarında anında kullanılabilmeleri için bunları küresel kontrol noktası dizinine (`ckpt/itts`) kopyalar.
- **Dosya Tarayıcısı:** Gerçek zamanlı bir dizin görüntüleyici, yerel yol içindeki temel dosyaların varlığını doğrulamanıza olanak tanır.

### 🌐 2. Küresel Önbellek Modelleri

Disk alanını optimize etmek ve gereksiz indirmeleri önlemek için ağır temel modeller, tüm projeler arasında paylaşılan küresel bir önbellekte saklanır. Bu modeller, TTS motoru için akustik ve mimari iskeleti sağlar:

- **W2V-BERT 2.0:** Üst düzey konuşma temsillerini (representations) çıkarmak için kullanılan devasa, kendi denetimli (self-supervised) bir ses kodlayıcı.
- **MaskGCT:** Özbağlanımlı olmayan (non-autoregressive) akustik üretim için özelleşmiş bir model.
- **CampPlus:** Yüksek doğruluklu konuşmacı gömme (speaker embedding) çıkarımı için kullanılır.
- **BigVGAN:** Matematiksel mel-spektrogramları yüksek sadakatli işitilebilir dalga formlarına dönüştürmek için kullanılan son teknoloji (state-of-the-art) vocoder.

### 🎙️ 3. Whisper Modelleri

Bu bölüm, ham sesin metne deşifre edilmesi (transcription) gereken "Derlem (Corpus)" aşaması için kritik olan **OpenAI Whisper** paketini yönetir.

- **Ayrıntılı Seçim:** `tiny`'den (hız için) `large-v3`'e (maksimum deşifre doğruluğu için) kadar tüm Whisper modelleri yelpazesinden seçim yapabilirsiniz.
- **Merkezi Model Yolu:** Tüm deşifre görevleri tarafından erişilebilir olmalarını sağlamak için modeller, kök sistem içinde özel bir dizine indirilir.

### 🎤 4. RVC Önkoşulları (Applio Altyapısıyla)

Bu bölüm, Geri Getirim Tabanlı Ses Dönüştürme (RVC) ardışık düzeni için gerekli temel modellerin edinilmesini yönetir.

- **Applio Entegrasyonu:** Önkoşul indirme mantığı cömertçe **Applio RVC deposu** tarafından sağlanmakta ve desteklenmektedir; bu da sağlam ve güncel bir ortam hazırlığı sağlar.
- **Temel Modeller:** İndirme işleminin başlatılması, RVC modülü içinde doğru perde (pitch) çıkarımı ve ses dönüştürme yetenekleri için gerekli olan temel HuBERT ve RMVPE modellerini getirir.

### 🛠️ 5. Bağımlılık ve Ortam Düzeltmeleri

Gelişmiş TTS kütüphaneleri, bazen bozuk üst akış (upstream) bağımlılıklarını veya eksik protokol arabelleklerini (protocol buffers) düzeltmek için manuel müdahale gerektirir.

- **SentencePiece Düzeltmesi:** `sentencepiece_model_pb2.py` dosyasını doğrudan resmi Google deposundan indirmek için özel bir yardımcı program.
- **Sistem Bütünlüğü:** Bu araç, BPE belirteçleyici (tokenizer) mantığının, kelime dağarcığını (vocabulary) yeniden boyutlandırma sırasında `.model` dosyaları üzerinde cerrahi işlemler gerçekleştirmek için gerekli Python bağlamalarına (bindings) sahip olmasını sağlar.

### ♨️ 6. Türkçe Ağırlıklar

Bu özel modül, son derece optimize edilmiş bir tokenizasyon (belirteçleme) stratejisi kullanarak özellikle Türkçe dili için eğitilmiş önceden yapılandırılmış ağırlıkları getirir.

- **Doğrudan Edinme:** `tr_bpe.model`, `tr_config.yaml` ve `tr_gpt.pth` dosyalarını doğrudan `ruygar/itts_tr_lex` Hugging Face deposundan indirir.
- **Merkezi Yönlendirme:** İndirilen ağırlıklar otomatik olarak küresel `ckpt/itts` dizinine yönlendirilir, böylece yalıtılmış bir proje oluşturmaya gerek kalmadan bağımsız (standalone) TTS motoru tarafından anında kullanılabilir.
- **Hibrit Grafem Tokenizasyonu:** Bu model, son derece verimli bir karma belirteçleme (mixed-tokenizer) stratejisi kullanılarak eğitilmiştir. Orijinal İngilizce kelime dağarcığı korunmuş ve küçük harfli bir Türkçe kelime dağarcığı ile birleştirilmiştir. Ancak, sistemin normalleştiricisi (yapılandırmadan okuyarak) gelen tüm metni **büyük harfe** zorladığı için, standart Türkçe küçük harfli BPE belirteçleme tamamen atlanır.
- **Hızlı Yakınsama:** Bu zorunlu büyük harf yönlendirmesinin bir sonucu olarak, belirteçleyici, enjekte edilen büyük harfli Türkçe özel karakterlerin yanı sıra sağlam, önceden eğitilmiş İngilizce büyük harfleri kullanarak metni sıkı bir şekilde işler. Bu, Türkçe için grafem benzeri (karakter karakter) bir belirteçleme hattı oluşturur. İngilizce büyük harflerin yerleşik fonetik temsillerini ödünç alarak, model ince ayar (fine-tuning) sırasında olağanüstü hızlı bir yakınsama elde eder.