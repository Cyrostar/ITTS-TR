<h1 align="center">ITTS-TR</h1> 
<div align="center"> 
  <a href="README.md"><img src="img/flags/gb.svg" alt="GB" width="24"/></a>  | 
  <a href="README-TR.md"><img src="img/flags/tr.svg" alt="TR" width="24"/></a>
</div>

---

Index-TTS metinden sese (text-to-speech) modelini yönetmek, eğitmek ve çıkarım (inference) yapmak için Gradio tabanlı kapsamlı bir Web Kullanıcı Arayüzü. Bu arayüz, veri hazırlığından nihai ses üretimine kadar tüm makine öğrenimi (ML) iş akışını kolaylaştırır.

**ORİJİNAL DEPO:** [INDEX-TTS Resmi Deposu](https://github.com/index-tts/index-tts)

**Dil Desteği Üzerine Not:** Bu proje özel olarak Türkçe model eğitimi için tasarlanmıştır; ancak Latin alfabesi tabanlı diğer dilleri eğitmek için de kullanılabilir. Latin alfabesi kullanmayan diller için kod üzerinde değişiklik yapılması gerekebilir.

## ✨ Özellikler

Bu WebUI, modüler ve sekmeli bir iş akışı sunar:

* **Ana Sayfa (Home):** Proje yönetimi ve gerçek zamanlı donanım izleme (CPU, RAM, VRAM, Sıcaklıklar).
* **Modeller (Models):** Model kontrol noktası (checkpoint) seçimi ve yönetimi.
* **Derlem ve Veri Seti (Corpus & Dataset):** Ses ve metin verilerinin alınması, biçimlendirilmesi ve veri setinin derlenmesi.
* **Simgeleştirici ve Ön İşlemci (Tokenizer & Preprocessor):** Modelin işleyebilmesi için metin simgeleştirme ve ses ön işleme adımları.
* **Eğitmen (Trainer):** Index-TTS model eğitimini/ince ayarını (fine-tuning) yapılandırma ve izleme arayüzü.
* **Çıkarım (Inference):** Eğitilmiş kontrol noktalarını kullanarak metinden yüksek kaliteli (high-fidelity) ses üretimi.
* **TTS:** Proje ayarlarını atlayarak doğrudan model yükleme, zero-shot (sıfır-atış) kontrolleri ve hızlı üretim sağlayan bağımsız bir çıkarım motoru.
* **Ses Dönüştürme (RVC):** Zero-shot ses dönüştürme ve yüksek doğruluklu ses rengi (timbre) modifikasyonu için entegre Applio RVC mimarisi.

## 🧩 Önkoşullar

* NVIDIA GPU (Eğitim ve Çıkarım için şiddetle tavsiye edilir)
* PyTorch kurulumunuzla uyumlu CUDA Toolkit
* Windows 10+

## 🚀 Kurulum

ITTS-TR ortamını düzgün bir şekilde kurmak için lütfen şu adımları izleyin:

1. **Depoyu İndirin:** Bu depoyu klonlayın veya yerel makinenize indirin.
2. **Yükleyiciyi Çalıştırın:** Kurulum komut dosyalarını içeren **bat** klasörüne gidin ve `install.bat` dosyasına çift tıklayın. 
3. **Ekrandaki Yönergeleri İzleyin:** Toplu iş dosyası (batch script) sizi aşağıdaki otomatik kurulum aşamalarından geçirecektir:
   * **Git Kurulumu:** Sisteminizde yüklü değilse, taşınabilir (portable) bir GitHub sürümü kurmanız istenecektir.
   * **Python Kurulumu:** İstendiğinde Python sürümü olarak **3.11.9** girin. Komut dosyası izole bir Python ortamı indirecek, çıkaracak ve yapılandıracaktır (gerekli C++ başlıkları ve kütüphaneleri NuGet aracılığıyla dahil edilir).
   * **Temel Bağımlılıklar:** Komut dosyası modern derleme arka uçlarını (`uv` ve `setuptools`) kurar ve `requirments.txt` içinde tanımlanan temel Python gereksinimlerini otomatik olarak yükler.
   * **PyTorch & CUDA Yapılandırması:** Komut dosyası önerilen Torch sürümünü (örn. 2.8.0) otomatik olarak algılayacaktır. CUDA desteği ile kurmak isteyip istemediğiniz sorulacaktır. Devam ederseniz, uygun GPU hızlandırmasını sağlamak için tercih ettiğiniz CUDA sürümünü (12.6, 12.8 veya 13.0) seçebilirsiniz. **12.8** sürümü şiddetle tavsiye edilir.
   * **FFmpeg Kurulumu:** FFmpeg kurmanız istenecektir; Kararlı (Stable - v7.1.1) veya En Son Sürüm (Latest Release) arasında seçim yapma seçenekleri sunulacaktır.
   * **yt-dlp:** Medya indirme işlemleri için isteğe bağlı olarak `yt-dlp` aracını kurmayı seçebilirsiniz.
   * **Çekirdek Modeli Klonlama:** Seyrek (sparse) `index-tts` deposunu klonlamanız istenecektir.
   * **RVC Entegrasyonu:** RVC özelliklerini işlem hattına (pipeline) entegre etmek için seyrek `Applio` deposunu klonlamanız istenecektir.
   * **Sonlandırma ve Yamalar:** Son olarak, komut dosyası WebUI çalışma alanı klasörlerini (`uix` ve `wui`) otomatik olarak başlatacak ve Index-TTS, SpeechBrain ve RVC kod tabanlarına zorunlu bağımlılık düzeltmelerini uygulayacaktır.
   
### 🔑 Hugging Face Token Yapılandırması (`HF_TOKEN`)

`paths.bat` yapılandırma dosyası bir `HF_TOKEN` ortam değişkeni içerir. Bu token, Hugging Face Hub'daki bazı kısıtlı modellere ve ağırlıklara erişim sağlamak ve indirmek için kesinlikle gereklidir. 

Windows sisteminizde genel bir ortam değişkeni olarak yapılandırılmış bir `HF_TOKEN` yoksa, WebUI üzerinden model indirmeye çalışmadan önce `paths.bat` dosyasını bir metin düzenleyici ile açmalı ve Hugging Face erişim token'ınızı manuel olarak eklemelisiniz.

---

## 📚 Kullanım

Arayüzü başlatmak için kök dizinden webui betiğini çalıştırın:

```bat
webui.bat
```

Uygulama, çalışma alanı verilerinizi depolamak için bir `projects/` dizini ve genel kullanıcı arayüzü tercihlerinizi (dil ayarları gibi) kaydetmek için bir `wui.json` dosyası oluşturacaktır. Terminalinizde sağlanan yerel URL'yi (genellikle `http://127.0.0.1:7860`) tarayıcınızda açın.

Tensorboard'u başlatmak için bat klasörünün içindeki tensorboard betiğini çalıştırın:

```bat
bat\tensorboard.bat
```

---

## ⚡ Triton

Dinamik olarak derlenen GPU çekirdeklerini kullanarak maksimum eğitim ve çıkarım hızına ulaşmak için OpenAI'nin Triton derleyicisini etkinleştirebilirsiniz. Triton çekirdekleri çalışma zamanında yerel olarak derlediğinden, Windows kullanıcılarının katı bir derleme ortamı yapılandırması gerekir.

**Triton için Sistem Gereksinimleri:** 1. **Visual Studio C++ Derleme Araçları (Build Tools):** Visual Studio Yükleyicisini (Installer) indirin ve **"C++ ile masaüstü geliştirme" (Desktop development with C++)** iş yükünü yükleyin. Bu, gerekli MSVC derleyicisini (`cl.exe`) sağlar.

2. **NVIDIA CUDA Toolkit:** Resmi bağımsız (standalone) CUDA Toolkit'i yükleyin. Sürüm, `install.bat` aşamasında PyTorch için seçtiğiniz CUDA sürümüyle tamamen aynı olmalıdır (örn. 12.6, 12.8 veya 13.0). 

3. **Katı Yol Yapılandırması (Strict Path Configuration):** Dinamik derleyici, sistem yollarına sabitlenmiş olarak (hardcoded) dayanır. `paths.bat` dosyanızın, dizin değişkenleri yerel MSVC ve CUDA Toolkit kurulum yollarınızla tam olarak eşleşecek şekilde yapılandırıldığından emin olun. Betiğin dahili yönlendirmesi tarafından `nvcc` veya `cl.exe` bulunamazsa, Triton çekirdekleri derleyemeyecektir.

---

## 🛡️ Lisans ve Yasal Uyarı

Bu depo ikili bir lisanslama yapısı (dual-licensing) kullanır:

**1. Kullanıcı Arayüzü ve Sarıcı Kod (Apache 2.0)**
Kök dizinde bulunan kapsayıcı Gradio arayüzü, proje yönetim mantığı ve yardımcı betikler **Apache Lisansı 2.0** altında lisanslanmıştır. Tüm detaylar için kök dizindeki `LICENSE` dosyasına bakabilirsiniz.

**2. Index-TTS Çekirdek Modeli (Bilibili Model Kullanım Lisans Sözleşmesi)**
`indextts/` dizini içinde bulunan çekirdek metinden sese modeli, model ağırlıkları ve belirli eğitim kodları Bilibili'ye aittir ve kesinlikle Bilibili Model Kullanım Lisans Sözleşmesi'ne tabidir. Bu sözleşmeyi index-tts'nin [resmi GitHub deposunda](https://github.com/index-tts/index-tts) bulabilirsiniz. Bu yazılımı kullanarak, yüksek riskli dağıtım yasakları da dahil olmak üzere bu şartlara uymayı kabul etmiş olursunuz.

### Zorunlu Yasal Uyarı

*Bu Türev Çalışmada orijinal model üzerinde yapılan hiçbir değişiklik, orijinal modelin asıl hak sahibi tarafından onaylanmamakta, garanti edilmemekte veya taahhüt edilmemektedir ve orijinal hak sahibi, bu Türev Çalışma ile ilgili tüm sorumluluğu reddetmektedir.*