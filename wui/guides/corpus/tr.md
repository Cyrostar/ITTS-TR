### 👁️ Genel Bakış

Dil Derlemi (Language Corpus) modülü, TTS işlem hattının temel veri toplama aşamasıdır. Yüksek kaliteli bir akustik model, hedef dilin yapısını anlamak için milyonlarca kelime görmüş bir simgeleştiriciye (tokenizer) ihtiyaç duyar. Bu modül, PDF ve TXT dosyalarından ham metin toplamak için yüksek performanslı, çok çekirdekli bir SQLite veritabanı (`corpus.db`) kullanacak şekilde devasa bir güncelleme almış olup, ham ses kaynaklarından yeni metin verileri oluşturmak için de güçlü bir ses çıkarma ve transkripsiyon araçları paketi sunmaya devam etmektedir.

#### 🗄️ 1. Veritabanı İşleme Motoru

Bu bölüm, eski `corpus.txt` sisteminin yerini alan ve kelime dağarcığı veritabanınızı oluşturmak için kullanılan temel sekmeleri adım adım açıklamaktadır.

- **PDF Derlem Oluşturucu (PDF Corpus Builder):** Sistemi PDF'lerle dolu bir klasöre yönlendirin. Metni paralel olarak çıkarmak için mevcut tüm CPU çekirdeklerini kullanır, metni okunabilir parçalara (chunks) böler ve veritabanına güvenli bir şekilde kaydeder.
- **Metin Normalleştirici (Text Normalizer):** Hata toleranslı B-Tree sayfalama kullanarak ham metin parçalarını okur ve kalıcı bir CPU havuzu üzerinden Çok Dilli Normalleştirici (Multilingual Normalizer) ile işler. Benzersiz metin parçalarını ve bunların tam geçme/bulunma sayılarını (occurrence counts) toplayarak kaydeder.
- **Kelime Çıkarıcı (Word Extractor):** Benzersiz (farklı) bireysel kelimeleri çıkarmak için normalleştirilmiş veritabanını tarar ve tüm veri setinizdeki kesin frekanslarını hesaplar.
- **Heceleyici (Syllabifier):** Metni fonetik hecelere ayırmak için normalleştirilmiş metni Türkçe Heceleyici (veya dil eşdeğerleri) aracılığıyla işler. Mutlak en yaygın fonetik sesleri bulmak için hece frekanslarını o hecenin içinde bulunduğu metin parçasının geçme sayısıyla çarpar.
- **Kelime Dağarcığı İstatistikleri (Vocabulary Statistics):** Veritabanınızdaki İlk 10 kelimeyi ve heceyi kontrol etmek için analitik bir görünüm sunar. Ayrıca harici kullanım için İlk 2000 listelerini doğrudan JSON dosyaları olarak dışa aktarabilirsiniz.
- **Simgeleştirici (Tokenizer):** Doğrudan veritabanınızdaki önceden normalleştirilmiş metin parçalarından bir Byte Pair Encoding (BPE) SentencePiece modeli eğitir. Fonetik kararlılığı sağlamak için en yüksek frekanslı ilk 1000 kelimeyi ve heceyi otomatik olarak kelime dağarcığına (vocabulary) zorla dahil eder.

#### 🧰 2. Çalışma Alanı ve Yükleme Araçları

- **Belge Ekle (Add Documents):** Ham metin belgelerinizi buraya bırakın. Bunları yerel proje klasörlerinize kaydedebilir veya "İşle ve Veritabanına Birleştir" (Process & Merge to DB) düğmesini kullanarak anında parçalara ayırabilir, normalleştirebilir ve doğrudan veritabanınıza enjekte edebilirsiniz.
- **Dosya Depoları (File Repositories):** Şu anda projenizin çalışma alanında bulunan, başarıyla işlenmiş PDF ve TXT dosyalarını görüntüler.

#### 🧽 3. Ses Edinimi ve Temizleme

Ham metniniz yoksa ancak konuşma seslerine erişiminiz varsa, bu araçlar sesi çıkarmanıza ve transkripsiyona hazırlamanıza yardımcı olur.

- **YouTube İndirici (YouTube Downloader):** Bir videodan en yüksek kaliteli ses parçasını anında getirmek ve çıkarmak için bir URL yapıştırın. Podcast veya röportaj verileri toplamak için mükemmeldir.
- **Ses Temizleyici (Audio Cleaner - Demucs):** Ham sesler genellikle transkripsiyonu ve akustik eğitimi bozan arka plan müziği veya gürültü içerir. Bu araç, insan ses (vokal) kanalını matematiksel olarak izole etmek ve arka plan gürültüsünü atmak için `htdemucs` sinir ağını kullanır.

#### 🎙️ 4. Transkripsiyon ve Konuşmacı Ayrıştırma (Diarization)

Temiz sesinizi kullanılabilir metne ve ayrılmış konuşmacı dosyalarına dönüştürün.

- **Ses Transkriptörü (Audio Transcriptor - Whisper):** Son derece doğru ve noktalama işaretli metin oluşturmak için sesinizi OpenAI'ın Whisper modeline (`large-v3`'e kadar) besler.
  - *Normalleştirici Seçeneği (Normalizer Toggle):* Metnin TTS eğitimi için mükemmel şekilde biçimlendirilmesi amacıyla Whisper çıktısını otomatik olarak normalleştiriciden geçirir.
- **Konuşmacı Ayrıştırma (Diarization):** Tek bir ses dosyasındaki birden fazla konuşmacıyı (kullanıcı tanımlı maksimum sayıya kadar) algılamak için `pyannote/speaker-diarization-3.1` modelini kullanır.
  - *Sessizliği Kırp (Trim Silence):* Boşluk (gap) ayarlarınıza bağlı olarak algılanan bölümleri otomatik olarak birbirine diker/birleştirir.
  - *Konuşmacı Dosyaları:* Hedefli veri seti oluşturmak amacıyla konuşmalarını tamamen izole ederek, algılanan her benzersiz konuşmacı için bağımsız bir `.wav` dosyası dışa aktarır.

#### 🏷️ 5. Dosya Standardizasyonu

Makine öğrenimi işlem hatları katı, öngörülebilir dosya yolları gerektirir.

- **Belge İsimlendirici & Sesli Kitap İsimlendirici (Document & Audiobook Namer):** Bu araçlar, girdi dizelerinizi temizler (örn. 'ç' gibi Türkçe karakterleri 'c'ye dönüştürür, kısa çizgileri boşluklarla değiştirir ve alt çizgileri zorunlu kılar) ve katı bir adlandırma kuralı oluşturur (`Tür-Yazar-Başlık` veya `Audiobook-Kaynak-Seslendiren-Tür-Yazar-Başlık`). Çalışma alanınızı yapısal olarak sağlam tutmak için dosyaları yüklemeden *önce* bunları kullanın.