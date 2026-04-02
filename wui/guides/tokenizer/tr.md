### 👁️ Genel Bakış

Tokenizer modülü, ham metninizi TTS akustik modelinin anlayabileceği sayısal dizilere çeviren bir Byte Pair Encoding (BPE) modelini SentencePiece kullanarak eğitmekten kesin olarak sorumludur. Kelime dağarcığı (vocabulary) oluşturma, karakter kapsamı, metin normalizasyonu ve özel jeton (token) enjeksiyonunu yönetir.

#### 📂 1. Veri Seçimi

Eğitime başlamadan önce, tokenizer'ın öğreneceği metinsel temeli tanımlamalısınız.

- **Dil ve Veri Seti Seçin:** Hedef dilinizi ve belirli veri seti klasörünü seçin. Arayüz, bu veri setiyle ilişkili `metadata.csv` dosyasını otomatik olarak ayrıştıracaktır.

- **Meta Veri Kapsamı:** Eğitim için veri setinizin meta verilerinin belirli bir yüzdesini örneklemenizi sağlayan bir kaydırıcı (%10 - %100). Devasa veri setlerinde hızlı prototipleme için kullanışlıdır.

- **Birleştirilmiş Korpus Metnini Dahil Et:** `corpus.db` veritabanınızın içeriğini veri seti meta verilerinize ekler.

- **Sadece Korpus Metni ile Eğit:** Veri seti meta verilerini tamamen yoksayar ve yalnızca `corpus.db` veritabanı üzerinde eğitim yapar.

#### 🧠 2. Kelime Dağarcığı ve Kapsam Yapılandırması

- **Kelime Dağarcığı Boyutu:** Modelin ezberleyebileceği maksimum benzersiz jeton (alt kelime/kelime) sayısını belirler. Kaydırıcı 2.000 ile 30.000 arasında değişir, varsayılan değer 12.000'dir. *Mühendislik Notu: Bir veri seti seçildiğinde, sistem bu değeri otomatik olarak projenizin `config.yaml` dosyasında tanımlanan `number_text_tokens` değeriyle senkronize etmeye çalışacaktır.*

- **Karakter Kapsamı:** Modelin içine dahil edilecek ham karakter varyasyonlarının yüzdesini tanımlar. Varsayılan değer `1.0`'dır (%100).

#### 🏷️ 3. Özel Jetonlar ve İşaretler (Tags)
- **Stil ve Duygu İşaretleri:** Akustik modele son derece etkileyici ifade sınırlarını öğretmek için önceden tanımlanmış konuşma, anlatım ve duygusal durum işaretlerini (ör. `[happy]`, `[whisper]`, `[podcast]`) otomatik olarak enjekte eder.
- **Alfabe Uzantıları:** Tokenizer'ı standart İngilizce harfleri, Türki genişletilmiş karakterleri, Türkçe uzun ünlüleri (ör. `â`, `î`) ve standart noktalama işaretlerini ezberlemeye zorlayan onay kutuları.
- **Özel Jetonlar Ekle:** Kelime dağarcığına manuel olarak kilitlemek istediğiniz belirli sembolleri, para birimlerini veya karakterleri (`|` ile ayırarak) tanımlayabileceğiniz bir metin alanı.

#### 💉 4. Fonetik ve Dilbilimsel Enjeksiyonlar
Bu temel özellikler istatistiksel BPE algoritmasını atlar. Tokenizer'ı belirli dilbilimsel birimleri kelime dağarcığı matrisine kalıcı olarak kilitlemeye zorlayarak TTS sentezi sırasında mükemmel akustik hizalamayı garanti eder.

- **Yüksek Frekanslı Heceleri Enjekte Et:** Derlenmiş `corpus.db` dosyanızı veri setlerinizdeki en yaygın heceler için (**Hece Sayısı** değerine göre) sorgular ve bunları kilitler. Bu, yutulmuş veya atlanmış ses artefaktlarını büyük ölçüde azaltan kesin fonetik çapalar sağlar.
- **Yüksek Frekanslı Kelimeleri Enjekte Et:** Veritabanını en sık kullanılan tam kelimeler için (**Kelime Sayısı** değerine göre) sorgular. Sık kullanılan kelimeleri donanımsal olarak kodlamak, modelin bunları robotik bir şekilde bir araya getirmek yerine tek bir akustik yerleştirme olarak doğal ritim ve tonlamalarıyla (prosodi) öğrenmesini sağlar.
- **Kelime Dağarcığı Kapasite Motoru:** Sistem, zorunlu enjeksiyonlarınızın (işaretler + heceler + kelimeler) temel dilbilimsel alfabe ve kontrol baytları için yeterli zorunlu alan (en az 256 yuva) bırakıp bırakmadığını dinamik olarak hesaplar. Bir kelime dağarcığı taşma riski tespit ederse işlemi güvenli bir şekilde durdurur.

#### ⚙️ 5. Gelişmiş Eğitim Kuralları
- **Normalizasyon ve Harf Durumu:** SPM dahili normalleştiricisini atlayıp atlamayacağınızı (`identity` kuralı) ve kelime dağarcığınızı kesin olarak büyük harf veya küçük harf formatlarına zorlayıp zorlamayacağınızı belirleyin.
- **Maksimum Cümle (Örneklem Boyutu):** Ayrıştırılan satırları sınırlayarak devasa veri setlerinde RAM kullanımını azaltır. Mevcut tüm cümleleri kullanmak için 0 yapın.
- **Son Derece Büyük Korpus Eğit:** Multi-gigabaytlık eğitim akışlarını ayrıştırmak için C++ bellek optimizasyonlarını devreye sokar.
- **Korpusu Karıştır:** Tekdüze bir dilsel dağılım sağlamak için ayrıştırılmış girdi akışlarını rastgele karıştırır.
- **Kesin Vocab Sınırı:** İstenen kelime dağarcığı boyutunu sonuna dolgu (padding) yapmadan kesin olarak uygular.

### 🧰 Araçlar (Utilities)

Bu teşhis araçları, akustik model eğitimine başlamadan önce metin işleme hattınızı doğrulamanıza olanak tanır.

#### 🎗️ Tokenizer Güvenlik Kontrolü

TTS için tokenizer uygunluğunu kontrol eden otomatik bir doğrulama paketi.

- Standart karakterleri başarıyla yakaladığını doğrulamak için eğitilmiş SentencePiece `.model` dosyanızı yükleyin.
- Normalizasyon ve harf durumu kurallarınıza göre özel karakterlerin eşlemesini açıkça test eder.
- Zararlı byte-fallback jetonlarının varlığını kontrol eder.
- Hiçbir `<unk>` jetonu üretilmediğinden emin olmak için karmaşık kelimeler üzerinde örnek jetonlaştırma gerçekleştirir.

#### 💱 Tokenizer Test Aracı

Modelinizin metni nasıl parçaladığını görselleştirmek için doğrudan bir çıkarım (inference) aracı.

- Aktif projenin tokenizer'ının metni alt kelimelere tam olarak nasıl böldüğünü görmek için ham metin girin.
- Hem Standart (eğitilmiş) hem de Birleştirilmiş (merged) model durumlarını test edin.
- Toplam jeton sayısını ve detaylı bir `[ID] Parça` dizisini çıkarır.

#### 📚 Çok Dilli Kelimeleştirici (Wordifier)

Sayı/tarih genişletme ve benzersiz kelime çıkarma mantığını test eder.

- Sayılar, tarihler veya kısaltmalar (ör. "19.05.1919" veya "2.500") gibi karmaşık yapılar içeren metinler girin.
- **Dönüş Formatı:** "Tam Blok" (genişletilmiş cümle) veya "Kelime Listesi" (çıkarılan kelimelerin virgülle ayrılmış bir dizisi) arasında geçiş yapın.

#### 🫧 Çok Dilli Normalleştirici

Ham metin üzerindeki ön işleme mantığını test eder.

- Karışık harf büyüklüğü, noktalama hataları veya özel semboller içeren karmaşık metinler girin.
- Çıktı, normalizasyon kuralları ve kısaltma genişletmeleri uygulandıktan sonra akustik modelin metni tam olarak nasıl "okuyacağını" ortaya koyar.

#### ✂️ Türkçe Heceleyici

Türkçe heceleme, vurgu işaretleme ve ünlü uyumu algoritmalarını test eder.

- Sistemin metni programatik olarak nasıl belirgin fonetik hecelere ayırdığını görmek için Türkçe metin girin.
- Vurgu işaretleri, ünlü uyumu doğrulaması ve detaylı kelime kelime analiz modu gibi gelişmiş dilbilimsel kontrolleri açıp kapatın.

#### 🎨 Özel Model Tasarla

Orijinal (Kaynak) ve eğitilmiş (Hedef) tokenizer modellerinden özel bir model tasarlamak için güçlü bir görsel arayüz.

- Resmi temel modele birleştirmek için bir hedef model yükleyin.
- **Kaynak Yapılandırması:** Dil işaretleri, CJK jetonları, İngilizce jetonları veya kaynak noktalama işaretleri gibi istenmeyen unsurları temel modelden temizleyin. Ayrıca küçük harfe dönüştürmeye zorlayabilir ve gerekli yapısal jetonları enjekte edebilirsiniz.
- **Hedef Yapılandırması:** Yeni modelin temelle tam olarak nasıl birleşeceğini tanımlayın. Bağımsız harfleri ve noktalama işaretlerini koruyup korumayacağınızı seçin ve jeton harf durumu kurallarını kesin olarak uygulayın.
- İşlenmiş Kaynak Dağarcığı, işlenmiş Hedef Dağarcığı ve nihai Birleştirilmiş Çıktının doğrudan yan yana görünümünü sağlar.