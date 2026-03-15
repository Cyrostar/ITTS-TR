### 👁️ Genel Bakış

Veri Seti Hazırlama (Dataset Preparation) modülü, ses-metin çiftlerini elde etmek, standartlaştırmak ve akustik model eğitimi için birleşik bir formata sokmaktan kesinlikle sorumludur. Girdi kaynağı ne olursa olsun, bu modül ses kliplerini içeren bir `wavs/` dizini ve her dosyayı normalleştirilmiş metin transkriptiyle eşleştiren yapılandırılmış bir `metadata.csv` dosyasından oluşan standart bir çıktı üretir.

#### 🤗 Yöntem 1: Hugging Face Veri Seti

Bu yöntem, Hugging Face Hub'daki mevcut konuşma veri setlerini doğrudan çekmenize ve işlemenize olanak tanır.

- **Hugging Face Veri Seti (Dataset):** Depo kimliği (örn. `erenfazlioglu/turkishvoicedataset`). Sistem otomatik olarak `train` (eğitim) bölümünü indirir ve iç yapısını ayrıştırır.
- **Hedef Klasör Adı (Target Folder Name):** Veri setinin `datasets/<dil>/<hedef_klasör_adı>` altında saklanacağı yerel dizin adını tanımlar.
- **Otomatik Çıkarım:** İşlem hattı (pipeline), farklı veri seti şemalarındaki metinleri (`transcription`, `text` veya `sentence` gibi anahtarları arayarak) güvenli bir şekilde çıkarır ve ham ses baytlarını doğrudan `.wav` dosyalarına ayıklar.

#### ✂️ Yöntem 2: Özel Ses Dilimleyici (Custom Audio Slicer)

Bu yöntem, tek ve uzun formatlı bir ses dosyasını (örn. podcast veya sesli kitap) binlerce kısa, transkripti çıkarılmış eğitim klibine dönüştürür.

- **Ses Yükle (Upload Audio):** Yerel uzun formatlı ses dosyanızı seçin.
- **Maksimum Klip Süresi (Max Clip Duration):** Tek bir ses klibi için kesin sınırı belirler. Bir konuşma bölümü bu süreyi aşarsa, sistem Whisper tarafından sağlanan zaman damgası (timestamp) sınırlarına göre onu akıllıca daha da dilimler.
- **VAD ve Konuşmacı Ayrıştırma (Diarization):** Arka planda sistem, Ses Aktivite Algılaması (Voice Activity Detection - VAD) gerçekleştirmek, gerçek konuşmayı izole etmek ve uzun sessizlikleri yok saymak için `pyannote/speaker-diarization-3.1` modelini başlatır. *(Not: Bu işlem geçerli bir `HF_TOKEN` ortam değişkeni gerektirir).*
- **Whisper Transkripsiyonu:** İzole edilen her konuşma bölümü, hedef dilinizde son derece doğru bir metin transkripti oluşturmak için `large-v3` Whisper modeline beslenir.

#### ⚙️ Temel Yapılandırma ve Kontroller

Her iki işleme yöntemi de ses verilerinizi standartlaştırmak için kritik parametreleri paylaşır.

- **Dil (Language):** Hem çıktı dizin yapısını hem de Whisper transkripsiyon modeline zorlanan dili belirleyen dil etiketini (örn. `tr`, `en`) atar.
- **Yeniden Örnekle (Resample To):** Sesi belirli bir örnekleme hızına (16kHz, 22.05kHz, 24kHz, 44.1kHz veya 48kHz) zorlar. Eğer `None` olarak ayarlanırsa, kaynak sesin orijinal örnekleme hızı korunur.
- **Her X Klipte Bir Kaydet (Save Every X Clips):** `metadata.csv` dosyasının diske ne sıklıkla yazılacağını kontrol eder. Düşük sayılar çökmelere karşı güvenlik sağlarken, yüksek sayılar işlem hızını bir miktar artırır.

#### 🔄 İşleme ve Normalizasyon Hattı

Akustik modelin matematiksel olarak temiz veri aldığından emin olmak için, tüm metin ve ses kayıt edilmeden önce katı bir işlem hattından geçer:

1. **Ses Formatlama:** Ses dizileri dönüştürülür ve belirlenen örnekleme hızında tek kanallı (mono) `.wav` dosyaları olarak kaydedilir.
2. **Metin Kelimeleştirme (Wordification):** Ham metin, sayıları, tarihleri ve sembolleri sözlü kelime karşılıklarına (örn. "1919" -> "bin dokuz yüz on dokuz") genişletmek için `TurkishWordifier` modülünden geçirilir.
3. **Metin Normalizasyonu:** Genişletilmiş metin daha sonra büyük/küçük harf durumunu, noktalama işaretleri standardizasyonunu ve dile özgü yazım kurallarını ele almak için `TurkishWalnutNormalizer` modülüne beslenir.
4. **Durum Yönetimi:** Modül işlenen dosya adlarını izler. Eğer kesintiye uğrarsa, **♻️ İşlemi Sürdür (Resume Process)** düğmesine tıklamak mevcut `metadata.csv` dosyasını tarayacak ve halihazırda formatlanmış olan ses kliplerini otomatik olarak atlayarak tekrarlanan çabayı önleyecektir.
