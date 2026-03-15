### 👁️ Genel Bakış

Dil Derlemi (Language Corpus) modülü, TTS işlem hattının temel veri toplama aşamasıdır. Yüksek kaliteli bir akustik model, hedef dilin yapısını anlamak için milyonlarca kelime görmüş bir simgeleştiriciye (tokenizer) ihtiyaç duyar. Bu modül, PDF ve TXT dosyalarından ham metin toplamanıza olanak tanır ve ham ses kaynaklarından yeni metin verileri oluşturmak için güçlü bir ses çıkarma ve transkripsiyon araçları paketi sunar.

#### 📘 1. Çalışma Alanı ve Depolar (Repositories)

Bu bölüm, daha sonra SentencePiece Simgeleştirici eğitim aşamasına beslenecek olan ana `corpus.txt` dosyasını oluşturmaya adanmıştır.

- **PDF/Metin Yükle:** Ham metin belgelerinizi buraya bırakın. Sistem metni ayrıştırır, dosyayı şişirmeden kelime dağarcığı verimliliğini en üst düzeye çıkarmak için isteğe bağlı olarak benzersiz kelimeleri (unique words) filtreler ve bunları proje depolarında saklar.
- **Depolar (Repositories):** Şu anda projenizin çalışma alanında bulunan, başarıyla işlenmiş PDF ve TXT dosyalarını görüntüler.
- **Tüm Mix Dosyalarını Birleştir:** Depolarınızdaki işlenmiş her belgeyi tek ve devasa bir `corpus.txt` dosyasında derler.

#### 🧰 2. Ses Edinimi ve Temizleme

Ham metniniz yoksa ancak konuşma seslerine erişiminiz varsa, bu araçlar sesi çıkarmanıza ve transkripsiyona hazırlamanıza yardımcı olur.

- **YouTube İndirici:** Bir videonun ses parçasını anında getirmek ve çıkarmak için bir URL yapıştırın. Podcast veya röportaj verileri toplamak için mükemmeldir.
- **Ses Temizleyici (Demucs):** Ham sesler genellikle transkripsiyonu ve akustik eğitimi bozan arka plan müziği veya gürültü içerir. Bu araç, insan ses (vokal) kanalını matematiksel olarak izole etmek ve arka plan gürültüsünü atmak için `Demucs` sinir ağını kullanır.

#### 🎙️ 3. Transkripsiyon ve Konuşmacı Ayrıştırma (Diarization)

Temiz sesinizi kullanılabilir metne ve ayrılmış konuşmacı dosyalarına dönüştürün.

- **Ses Transkriptörü (Whisper):** Son derece doğru ve noktalama işaretli metin oluşturmak için sesinizi OpenAI'ın Whisper modeline besler.
  - *Walnut Normalleştirici:* Metnin TTS eğitimi için mükemmel şekilde biçimlendirilmesi amacıyla Whisper çıktısını otomatik olarak özel normalleştiricimizden geçirir.
- **Konuşmacı Ayrıştırma (Diarization):** Tek bir ses dosyasındaki birden fazla konuşmacıyı algılamak için `pyannote/speaker-diarization-3.1` modelini kullanır.
  - *Sessizliği Kırp (Trim Silence):* Algılanan bölümleri otomatik olarak birleştirerek uzun duraklamaları (sessizlikleri) ortadan kaldırır.
  - *Konuşmacı Dosyaları:* Hedefli veri seti oluşturmak amacıyla konuşmalarını tamamen izole ederek, algılanan her benzersiz konuşmacı için bağımsız bir `.wav` dosyası dışa aktarır.

#### 🏷️ 4. Dosya Standardizasyonu

Makine öğrenimi işlem hatları katı, öngörülebilir dosya yolları gerektirir.

- **Belge İsimlendirici & Sesli Kitap İsimlendirici:** Bu araçlar, girdi dizelerinizi temizler (örn. 'ç' gibi Türkçe karakterleri 'c'ye dönüştürür, boşlukları alt çizgi ile değiştirir) ve katı bir adlandırma kuralını zorlar (`Tür_Yazar_Başlık.txt` veya `Kaynak_Seslendiren_Tür_Yazar_Başlık.wav`). Çalışma alanınızı yapısal olarak sağlam tutmak için dosyaları yüklemeden *önce* bunları kullanın.
