### 👁️ Genel Bakış

Tokenizer modülü, ham metninizi TTS akustik modelinin anlayabileceği sayısal dizilere çeviren SentencePiece kullanarak bir Byte Pair Encoding (BPE) modelini eğitmekten kesinlikle sorumludur. Kelime dağarcığı oluşturma, karakter kapsamı, metin normalleştirme ve özel token enjeksiyon işlemlerini yönetir.

#### 📂 1. Veri Seçimi

Eğitimden önce, tokenizer'ın öğreneceği metinsel temeli belirlemelisiniz.

- **Dil ve Veri Seti Seçimi (Select Language & Dataset):** Hedef dilinizi ve belirli veri seti klasörünü seçin. Arayüz, bu veri setiyle ilişkili `metadata.csv` dosyasını otomatik olarak ayrıştıracaktır.

- **Meta Veri Kapsamı (Metadata Coverage):** Eğitim için veri setinizin meta verilerinden belirli bir yüzdeyi örneklemenize olanak tanıyan bir kaydırıcıdır (%10 - %100). Devasa veri setlerinde hızlı prototipleme için kullanışlıdır.

- **Birleştirilmiş Derlem Metnini Dahil Et (Include Unified Corpus Text):** `corpus/corpus.txt` dosyasının içeriğini veri seti meta verilerinize ekler.

- **Sadece Derlem Metni İle Eğit (Train Only With Corpus Text):** Veri seti meta verilerini tamamen yoksayar ve yalnızca `corpus.txt` dosyası üzerinde eğitim yapar.

#### 🧠 2. Kelime Dağarcığı ve Kapsam Yapılandırması

- **Kelime Dağarcığı Boyutu (Vocabulary Size):** Modelin ezberleyebileceği maksimum benzersiz token (alt kelime/kelime) sayısını belirler. Kaydırıcı 2.000 ile 30.000 arasında değişir ve varsayılan değer 12.000'dir. *Mühendislik Notu: Bir veri seti seçildiğinde, bu değer otomatik olarak projenizin `config.yaml` dosyasında tanımlanan `number_text_tokens` ile senkronize edilmeye çalışılır.*

- **Karakter Kapsamı (Character Coverage):** Ham metinden korunacak karakterlerin yüzdesini belirler (0.99 ile 1.0 arası). Bunu 1.0 olarak ayarlamak, tüm nadir karakterlerin (Türkçe veri setlerindeki Q, W, X gibi) tutulmasını sağlar.

#### 🔣 3. Özel Tokenlar ve Karakterler

Sağlam bir TTS tokenizer'ı, noktalama işaretleri, yapısal etiketler ve belirli alfabetik uç durumlar hakkında açık bir farkındalık gerektirir.

- **Özel Tokenlar (Special Tokens):** Özel sembolleri (örn. para birimleri, matematik operatörleri) aralarına dikey çizgi `|` karakteri koyarak manuel olarak ekleyebileceğiniz bir metin alanı.

- **Karakter Enjeksiyon Onay Kutuları (Character Injection Checkboxes):** Belirli karakter setlerinin `<unk>` (bilinmeyen) tokenlara dönüşmesini önlemek için tokenizer'ı bu setleri tanımaya zorlar:
  
  - Türkçe karakterler (ç, ğ, ö, ş, ü).
  
  - İngilizce karakterler (q, w, x).
  
  - Türki karakterler (ə, ұ, қ).
  
  - Uzun ünlüler (â, î, û).
  
  - Noktalama işaretleri (. , ? ! ' : ; ...).

- **Otomatik Etiket Enjeksiyonu (Automated Tag Injection):** Arka planda sistemin; kapsamlı bir stil etiketleri dizisini (örn. `[casual]`, `[podcast]`) ve duygu etiketlerini (örn. `[happy]`, `[whisper]`) otomatik olarak doğrudan tokenizer kelime dağarcığına enjekte ettiğini unutmayın.

#### ⚙️ 4. Normalleştirme ve Artırma

- **Normalleştirme Kuralı (Normalization Rule):** SentencePiece'in dahili normalleştiricisini yapılandırır (`nmt_nfkc`, `nfkc`, `none`, vb.). *Not: Metin, SentencePiece'e ulaşmadan önce dahili `TurkishWalnutNormalizer` tarafından zaten yoğun bir şekilde normalleştirilmektedir.*

- **💉 Kelime Örnekleri Enjekte Et (Inject Word Samples):** Bu seçenek etkinleştirildiğinde, BPE algoritmasını alt kelime parçalarından ziyade tam kelime tokenları oluşturmaya yönlendirmek için eğitim verisine yaygın Türkçe kelimeler enjekte edilir. Bu etkinin yoğunluğu **Enjeksiyon Çarpanı (Injection Multiplier)** kaydırıcısı (1-100) ile kontrol edilir.

- **♻️ Veriyi Tekilleştir (Deduplicate Data):** Eğitim havuzundan birebir aynı olan tekrarlı satırları kaldırarak, tokenizer'ın tekrarlanan ifadelere karşı aşırı bir eğilim (bias) göstermesini engeller.

### 🧰 Araçlar

Bu teşhis araçları, akustik model eğitimine başlamadan önce metin işleme hattınızı doğrulamanızı sağlar.

#### 🖊️ Türkçe Tokenizer Güvenlik Kontrolü

Eğitilmiş SentencePiece `.model` dosyaları için otomatik bir doğrulama paketidir.

- Standart `a-z` karakterlerini başarılı bir şekilde kapsadığını doğrulamak için eğitilmiş modelinizi yükleyin.

- Normalleştirme kurallarınıza dayalı olarak özel Türkçe karakterlerin (örn. `Ç` → `ç`) açık eşlemesini test eder.

- Zararlı byte-fallback (bayta geri dönüş) tokenlarının varlığını kontrol eder.

- Hiçbir `<unk>` tokeninin üretilmediğinden emin olmak için karmaşık kelimeler üzerinde örnek tokenizasyon (parçalama) gerçekleştirir.

#### 💱 Tokenizer Test Aracı

Modelinizin metni nasıl parçaladığını görselleştirmek için doğrudan bir çıkarım (inference) aracıdır.

- Aktif projenin tokenizer'ının ham metni alt kelimelere tam olarak nasıl böldüğünü görmek için metin girin.

- Toplam token sayısını ve ayrıntılı bir `[ID] Piece` (Parça) çiftleri dizisini çıktı olarak verir.

#### 🫧 Turkish Walnut Normalizer

Ham metin ön işleme mantığını test eder.

- Karışık büyük/küçük harf, noktalama hataları veya özel semboller içeren düzensiz bir metin girin.

- Çıktı, normalleştirme kuralları uygulandıktan sonra akustik modelin metni tam olarak nasıl "okuyacağını" gösterir.

#### 📚 Türkçe Metin Bloğu Kelimeleştirici (Wordifier)

Sayısal ifadelerin ve tarihlerin genişletilmesini doğrular.

- Sayılar içeren bir metin girin (örn. "19.05.1919" veya "2.500").

- **Dönüş Formatı (Return Format):** "Tam Blok" (genişletilmiş cümle) veya "Kelime Listesi" (çıkarılan kelimelerin virgülle ayrılmış bir dizisi) arasında geçiş yapın.
