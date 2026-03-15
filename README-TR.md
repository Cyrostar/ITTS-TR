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

## 🧩 Önkoşullar

* NVIDIA GPU (Eğitim ve Çıkarım için şiddetle tavsiye edilir)
* PyTorch kurulumunuzla uyumlu CUDA Toolkit
* Windows 10+

## 🚀 Kurulum

ITTS-TR ortamını düzgün bir şekilde kurmak için lütfen aşağıdaki adımları izleyin:

1. **Depoyu İndirin:** Bu depoyu klonlayın veya yerel makinenize indirin.
2. **Kurulum Aracını Çalıştırın:** Kurulum betiklerini içeren **bat** klasörüne gidin ve `install.bat` dosyasına çift tıklayın. 
3. **Ekrandaki Yönergeleri İzleyin:** Toplu iş betiği (batch script) sizi aşağıdaki otomatik kurulum aşamalarından geçirecektir:
   * **Git Kurulumu:** Eğer yüklü değilse, taşınabilir (portable) bir GitHub sürümünü yüklemeniz istenecektir.
   * **Python Kurulumu:** İstendiğinde **3.11.9** Python sürümünü girin. Betik, izole edilmiş bir Python ortamını indirecek, çıkaracak ve yapılandıracaktır.
   * **Temel Bağımlılıklar:** Betik, `requirements.txt` dosyasında tanımlanan temel Python gereksinimlerini otomatik olarak yükler.
   * **PyTorch ve CUDA Yapılandırması:** PyTorch'u CUDA desteğiyle kurmak isteyip istemediğiniz sorulacaktır. Devam ederseniz, uygun GPU hızlandırmasını sağlamak için tercih ettiğiniz CUDA sürümünü (12.6, 12.8 veya 13.0) seçebilirsiniz. Sürüm **12.8** şiddetle tavsiye edilir.
   * **FFmpeg Kurulumu:** Kararlı (Stable - v7.1.1) veya En Son Sürüm (Latest Release) seçenekleriyle FFmpeg kurmanız istenecektir.
   * **yt-dlp:** Medya indirme işlemleri için isteğe bağlı olarak `yt-dlp` çalıştırılabilir dosyasını kurmayı seçebilirsiniz.
   * **Çekirdek Modeli Klonlama ve Yamalama:** Son olarak betik, seyreltilmiş (sparse) `index-tts` deposunu onayınızdan sonra klonlayacak ve zorunlu bağımlılık düzeltmelerini uygulayacaktır.

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