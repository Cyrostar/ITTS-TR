### 👁️ Genel Bakış

Çıkarım (Inference) modülü, eğitilmiş akustik modelinizin metinden konuşma ürettiği işlem hattının son aşamasıdır. Bu arayüz, özel olarak ince ayar yapılmış (fine-tuned) seslerinizi test etmenize, bunları resmi temel modelle karşılaştırmanıza ve üretilen sesin duygusal sunumunu derinlemesine manipüle etmenize olanak tanır.

#### 📂 1. Model ve Metin Yapılandırması

Konuşma üretmeden önce modeli seçmeli ve hedef metni sağlamalısınız.

- **Klasör Seçimi:** Açılır menüden eğitilmiş projenizi seçin. Sistem, projenizin `trains/` dizininden en son açılmış (unwrapped) kontrol noktasını (`gpt.pth` veya `latest.pth`) otomatik olarak yükleyecektir.
- **Orijinal Modeli Kullan (Use Original Model):** Özel eğitim klasörünüzü atlar ve değiştirilmemiş resmi temel modeli yükler. Bu, temel kaliteyi ince ayar yapılmış sonuçlarınızla karşılaştırmak için mükemmeldir.
- **Dil (Language):** Modelin telaffuz kurallarını yönlendirmek için metnin başına belirli bir dil simgesini (`TR` veya `EN`) zorlar. Bunu `Auto` (Otomatik) olarak ayarlamak, modelin dahili dil tahminine güvenir.
- **Girdi Metni (Input Text):** Modelin okumasını istediğiniz metin.

#### 📢 2. Ses ve Duygu Kontrolü

UnifiedVoice mimarisi, konuşmacının akustik kimliğini (Ses) sunumundan (Duygu) ayırır.

- **Referans Ses (Ses Tonu):** Kısa (3-10 saniyelik) temiz bir ses klibi yükleyin. Model, bu konuşmacının akustik özelliklerini (ses rengi, perde, ortam) kopyalayacaktır (klonlama).
- **Kontrol Modu (Control Mode):** Modelin metnin duygusal sunumunu nasıl belirleyeceğini belirler:
  - **Referansla Aynı (Same as Reference):** Model, duyguyu doğrudan Ses Referans Sesinizden çıkarır ve kopyalar.
  - **Referans Ses (Reference Audio):** Sadece duyguyu çıkarmak için *ikinci* bir ses dosyası yüklemenize olanak tanır. (Örn. A Kişisinin sesi, ancak B Kişisi gibi ağlıyor).
  - **Duygu Vektörleri (Emotion Vectors):** Kesin duygu ağırlıklarını ayarlamanıza olanak tanıyan manuel kaydırıcıların (Mutlu, Kızgın, Üzgün, Melankolik vb.) kilidini açar.
  - **Açıklama Metni (Description Text):** Bir metin istemini (örn. "acilen fısıldayarak") yorumlamak ve onu matematiksel olarak duygu vektörlerine dönüştürmek için yerleşik bir LLM (Qwen) kullanır.
- **Duygu Yoğunluğu (Emotion Intensity):** Seçilen duygunun nötr temel çizgiyi (baseline) ne kadar güçlü bir şekilde geçersiz kılacağını ölçeklendiren bir çarpan (0.0'dan 1.0'a kadar).

#### ⚙️ 3. Gelişmiş Parametreler

Bu ayarlar, altta yatan matematiksel üretim sürecini kontrol eder.

- **BigVGAN CUDA Çekirdeğini Etkinleştir:** Ortamınız destekliyorsa (`ninja` ve bir C++ derleyicisi gerektirir), bu son dalga formu (waveform) üretimini önemli ölçüde hızlandırır.
- **Örneklemeyi Etkinleştir (Enable Sampling):** İşaretlendiğinde model hafif bir rastgelelik katarak, tekrarlanan üretimlerin biraz farklı ve daha doğal duyulmasını sağlar.
- **Sıcaklık (Temperature) & Top P:** Örneklemenin rastgeleliğini kontrol eder. Düşük değerler konuşmayı oldukça öngörülebilir ve kararlı hale getirirken, yüksek değerler daha etkileyici ancak yapaylıklara (artifacts) eğilimli hale getirir.
- **Maksimum Simgeler (Max Tokens):** Son derece uzun metin parçalarında sonsuz üretim döngülerini (infinite loops) önlemek için kesin sınırlar.
