### 👁️ Genel Bakış

TTS motorunun kalbine hoş geldiniz. `config.yaml` dosyası, model eğitimi ve çıkarımı (inference) sırasında kullanılan kesin mimari boyutları, dizi (sequence) sınırlarını ve ses işleme matematiğini belirler. **Uyarı:** Gelişmiş parametreleri eğitimin ortasında değiştirmek checkpoint dosyanızı bozar. Bu ayarları yalnızca yeni bir eğitime başlamadan *önce* değiştirin.

#### 🧠 1. Temel Hiperparametreler (Core Hyperparameters)

Bunlar en kritik ayarlardır. Doğrudan modelinizin "beyin boyutunu" kontrol eder ve ne kadar GPU VRAM'ine ihtiyacınız olacağını belirler.

- **Kelime Dağarcığı Boyutu (number_text_tokens):** GPT modelinin anlayabileceği maksimum benzersiz token sayısını belirler. Bu sayı, 4. Aşamada eğiteceğiniz Tokenizer'ın (`.model` dosyası) kelime dağarcığı boyutuyla tam olarak eşleşmelidir.
- **Ses Örnekleme Hızı (Audio Sample Rate):** Ses işlemeniz için hedef frekans. Standart değerler 22050 Hz veya 24000 Hz'dir. Daha yüksek hızlar daha net ses üretir ancak çok daha fazla işlem gücü gerektirir.
- **Maksimum Metin/Mel Token'ları:** Girdi metninin veya üretilen ses dizilerinin maksimum uzunluğu. Metni 600 olarak ayarlarsanız, daha uzun olan her şey kırpılır. Bu uzunlukları artırmak, GPU VRAM kullanımınızı katlanarak artıracaktır.
- **Model Boyutu (model_dim):** GPT transformer'ının dahili gizli boyutu. Yüksek kaliteli modeller için 1024 veya 1280 standarttır.
- **Katmanlar ve Başlıklar (Layers & Heads):** Üst üste yığılmış transformer bloklarının (Layers) ve paralel dikkat mekanizmalarının (Heads) sayısı. Daha fazla katman daha iyi akıl yürütme ve prozodi anlamına gelir, ancak eğitimi yavaşlatır.



#### 🔤 2. Gelişmiş Tokenizer ve Metin Ön Ucu (Advanced Tokenizer & Text Front-End)

Bu bölüm, ham girdi metninin akustik modele girmeden önce nasıl temizleneceğini, normalleştirileceğini ve dilbilimsel tokenlara dönüştürüleceğini belirleyen metin işleme ardışık düzenini yapılandırır.

- **Dil (Language):** Hedef dil kodu (örn. `en`, `tr`, `es`). Bu, uygun karakter beyaz listelerini (whitelists) ve noktalama bölme kurallarını uygulamak için metni dinamik olarak doğru dile özgü normalleştiriciye yönlendirir.
- **Tokenizer Türü ve Vocab Türü (Tokenizer Type & Vocab Type):** Metni sindirilebilir ağ girdilerine bölmek için kullanılan algoritmik yaklaşımı tanımlar (örneğin, Bayt Çifti Kodlaması - Byte-Pair Encoding için `bpe` veya karakter düzeyinde ayrıştırma).
- **Harf Büyüklüğü Formatı (Case Format):** Tokenizer'ın eğitildiği tam harf durumuyla (casing) eşleşmesi için metnin belirli bir büyük/küçük harf formatına standartlaştırılıp standartlaştırılmayacağını belirler (örneğin, tüm karakterleri küçük veya büyük harfe zorlamak).
- **Kelimeye Dönüştür (Wordify):** Etkinleştirildiğinde sistem; sayıları, tarihleri, saatleri, para birimlerini ve matematiksel sembolleri tam sözlü kelime karşılıklarına genişletir (örn. "$5", "five dollars" olur veya "5 ₺", "beş lira" olur).
- **Kısaltmalar (Abbreviations):** Yaygın, dile özgü kısaltmaların genişletilmesini açıp kapatır (örn. "Dr." ifadesini "Doktor" veya "Mah." ifadesini "Mahallesi" olarak değiştirmek).
- **Çıkarım (Extract / Grapheme Extraction):** Etkinleştirildiğinde bu işlev, her bir karakterin arasına zorla boşluk ekler (örn. "merhaba", "m e r h a b a" olur). Bu, BPE alt kelime birleştirmesini (sub-word merging) kullanmayan belirli karakter düzeyindeki akustik hizalama modelleri için genellikle gereklidir.

#### 🎛️ 3. Veri Seti ve Mel Ayarları

Bu bölüm, ham `.wav` ses dosyalarınızın yapay sinir ağının okuyabileceği spektrogramlara matematiksel olarak nasıl dönüştürüleceğini kontrol eder.

- **BPE Modeli:** Veri seti yükleyicisinin araması gereken tokenizer modelinin (örn. `bpe.model`) tam dosya adı.
- **N FFT (n_fft):** Hızlı Fourier Dönüşümü (Fast Fourier Transform) penceresinin boyutu. 22kHz-24kHz sesler için endüstri standardı 1024'tür.
- **Atlama Uzunluğu (Hop Length):** Ardışık STFT kareleri arasındaki ses örneklerinin sayısı. 256 değeri, modelin her 256 örnekte bir sesin "anlık görüntüsünü" aldığı anlamına gelir.
- **Pencere Uzunluğu (Win Length):** Sese uygulanan pencere fonksiyonunun boyutu. Genellikle `n_fft` (1024) ile eşleşir.
- **N Mels:** Üretilecek Mel-frekans bantlarının sayısı. 80 veya 100 standarttır.
- **Mel Normalize Et:** Spektrogram değerlerinin istatistiksel olarak normalize edilip edilmeyeceği. Özel eğitim komut dosyanız açıkça normalize edilmiş girdiler gerektirmedikçe bunu `False` olarak bırakın.

#### 🧩 4. GPT Token Mantığı

Bu bölüm, üretken metinden sese süreci için yapısal sınırları ve koşullandırma mantığını tanımlar.

- **Mel Kodlarını Girdi Olarak Kullan:** `True` olduğunda, model eğitim sırasında akustik token'ları kendi içine otoregresif olarak geri besler.
- **Solo Embedding Eğit:** Belirli ince ayar (fine-tuning) aşamalarında embedding katmanlarını izole etmek için özelleştirilmiş bir bayrak.
- **Koşul Türü (Condition Type):** Metin ve ses arasında köprü kurmak için kullanılan mimari modülü tanımlar. `conformer_perceiver` son derece gelişmiş, verimli bir çapraz dikkat (cross-attention) mekanizmasıdır.
- **Başlangıç/Bitiş Token'ları:** Bunlar modele bir ses dizisinin veya metin dizisinin ne zaman başlayıp bittiğini söyleyen kesin kimlik (ID) numaralarıdır (örn. `start_text_token` varsayılan olarak 0'dır).
- **Mel Kodları Sayısı:** Semantik ses codec'inizin toplam kelime dağarcığı boyutu.

#### 🔗 5. Checkpoint'ler ve Vocoder

Ardışık düzeninizin (pipeline) dayandığı harici ağırlıklar ve bileşenler için yollar ve tanımlar.

- **Checkpoint'ler (gpt.pth, s2mel.pth):** Sistemin GPT ve Semantic-to-Mel model ağırlıklarını kaydedeceği (veya oradan devam edeceği) göreceli dosya adları.
- **W2V Stat ve Matrisler:** Konuşmacı ve duygu koşullandırması için kullanılan önceden hesaplanmış istatistiksel tensörlerin (örneğin `wav2vec2bert_stats.pt`, `feat1.pt`) yolları.
- **Qwen Emo Yolu:** Duygu çıkarımı için kullanılan temel LLM'nin (Büyük Dil Modeli) dizin yolu.
- **Vocoder Türü ve Adı:** Vocoder, yapay zeka tarafından oluşturulan spektrogramı alıp duyulabilir bir `.wav` dosyasına dönüştürmekten sorumlu yapay sinir ağıdır. `bigvgan`, son derece net ve insansı ses kalıntıları üreten son teknoloji bir vocoder'dır.

#### ⬡ → ◯ Index-TTS'te Model Boyutlandırma ve Ağırlıkların Korunması

UnifiedVoice mimarisi için yeni bir yapılandırma oluştururken, sistem önceden eğitilmiş ağırlıklarınızın gereksiz yere kaybolmamasını sağlamak için akıllı bir ağırlık aktarım algoritması kullanır. Bu kılavuz, bu sürecin arkasındaki matematiksel mekaniği açıklamaktadır.

### 1. Temel Dilimleme Mekanizması (Slicing Mechanism)

Eğitilmiş ağırlıkların korunması, orijinal önceden eğitilmiş tensör ile yeni yapılandırdığınız tensör arasındaki örtüşen matematiksel sınırın hesaplanmasına dayanır. 

Sistem bunu, eski ve yeni şekiller arasındaki minimum boyutu (`min(ds, ts)`) değerlendiren dinamik bir dilimleme işlemi kullanarak gerçekleştirir. Bu, yalnızca yeni hesaplama grafiğine (computational graph) mükemmel bir şekilde uyan veri kesişimini aktarmamızı sağlar.

### 2. Yeniden Boyutlandırma Senaryoları

Kullanıcı arayüzündeki (Gradio UI) parametreleri nasıl ayarladığınıza bağlı olarak, model önceden eğitilmiş ağırlıkları üç farklı şekilde işler:

* **Aynı Katmanlar (Değişiklik Yok):** Eğer `model_dim`, `layers` ve `heads` gibi temel yapısal parametrelere dokunulmazsa, derin ağ bileşenleri (Transformer attention blokları, feed-forward katmanları ve normalizasyon katmanları) 1:1 olarak eşleşir. Önceden eğitilmiş ağırlıklar hiçbir değişikliğe uğramadan mükemmel bir şekilde kopyalanır.
* **Genişletme (Örn. Sözlük Dağarcığını Artırma):** `number_text_tokens` değerini (örneğin 10.000'den 12.000'e) artırırsanız, orijinal 10.000 eğitilmiş yerleştirme (embedding) doğrudan yeni tensöre kopyalanır. Yeni eklenen 2.000 yuva rastgele, eğitilmemiş ağırlıklarla başlatılır; böylece modelin temel karakterlere ilişkin sahip olduğu temel bilgisi korunmuş olur.
* **Kırpma (Örn. Bağlamı Azaltma):** Bir parametreyi küçültürseniz, sistem tensörü matematiksel olarak kırpar. 0 indeksinden yeni kesme sınırınıza kadar olan eğitilmiş ağırlıkları korur. Tokenizer'lar (BPE) genellikle token'ları frekansa göre sıraladığından, bu işlem en kritik ve en çok eğitilmiş karakterleri güvenli bir şekilde muhafaza eder.

### 3. İstisna: Boyut Uyuşmazlıkları (Dimension Mismatches)

Eğitilmiş ağırlıkların kaybedildiği/atıldığı tek senaryo, `model_dim` değerini 1280'den 512'ye düşürmek gibi temel bir mimari boyutu köklü bir şekilde değiştirerek uyumsuz bir şekil (shape) yaratmanızdır. 

Bu gibi durumlarda sistem, kontrollü bir geri çekilme (graceful degradation) mekanizmasına güvenir:

* Boyut uyuşmazlığını tespit eder.
* İlgili katmanı güvenli bir şekilde atlar.
* Kullanıcı arayüzünde (UI loglarında) kesin bir uyarı kaydeder (örn. `Skipped layer... Dimension mismatch`).
* Uyumlu ağ bileşenlerinin geri kalanını kurtarırken, yalnızca o uyumsuz katmanı sıfırdan başlatır.
