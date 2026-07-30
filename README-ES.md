<h1 align="center">ITTS-TR</h1> 
<div align="center"> 
  <a href="README-TR.md"><img src="img/flags/tr.svg" alt="TR" width="24"/></a> | 
  <a href="README.md"><img src="img/flags/gb.svg" alt="GB" width="24"/></a> |
  <a href="README-ES.md"><img src="img/flags/es.svg" alt="ES" width="24"/></a>
</div>

---

Una interfaz de usuario web completa basada en Gradio para gestionar, entrenar y ejecutar inferencias en el modelo de conversión de texto a voz Index-TTS. Esta interfaz simplifica todo el flujo de trabajo de aprendizaje automático, desde la preparación de los datos hasta la generación final del audio.

**REPOSITORIO ORIGINAL :** [Repositorio Oficial de INDEX-TTS](https://github.com/index-tts/index-tts)

**Nota sobre el soporte de idiomas:** Este proyecto está diseñado específicamente para entrenar en el idioma turco; sin embargo, se puede utilizar para entrenar otros idiomas basados en caracteres latinos. Para idiomas no latinos, es posible que se requieran modificaciones en el código.

## ✨ Características

Esta WebUI proporciona un flujo de trabajo modular basado en pestañas:

* **Inicio (Home):** Gestión de proyectos y monitorización de hardware en tiempo real (CPU, RAM, VRAM, Temperaturas).
* **Modelos (Models):** Selección y gestión de puntos de control (checkpoints) del modelo.
* **Corpus y Conjunto de Datos (Corpus & Dataset):** Ingestión de datos de audio y texto, formateo y compilación de conjuntos de datos.
* **Tokenizador y Preprocesador (Tokenizer & Preprocessor):** Tuberías (pipelines) de tokenización de texto y preprocesamiento de audio para la ingesta del modelo.
* **Entrenador (Trainer):** Interfaz para configurar y supervisar el entrenamiento y ajuste fino (fine-tuning) del modelo Index-TTS.
* **Inferencia (Inference):** Generación de audio de alta fidelidad a partir de texto utilizando puntos de control entrenados.
* **TTS:** Un motor de inferencia independiente que ignora la configuración del proyecto para cargar directamente el modelo, con controles zero-shot y generación rápida.
* **Conversión de Voz (Voice Conversion - RVC):** Arquitectura RVC de Applio integrada para conversión de voz zero-shot y modificación precisa del timbre.

## 🧩 Requisitos Previos

* GPU NVIDIA (Altamente recomendada para Entrenamiento e Inferencia)
* Kit de herramientas CUDA compatible con su instalación de PyTorch
* Windows 10+

## 🚀 Instalación

Para configurar el entorno ITTS-TR correctamente, por favor siga estos pasos:

1. **Obtenga el Repositorio:** Clone o descargue este repositorio en su máquina local.
2. **Ejecute el Instalador:** Navegue a la carpeta **bat** que contiene los scripts de configuración y haga doble clic en el archivo `install.bat`. 
3. **Siga las Instrucciones en Pantalla:** El script por lotes lo guiará a través de las siguientes fases de configuración automatizada:
   * **Instalación de Git:** Se le pedirá que instale una versión portátil de GitHub si aún no la tiene.
   * **Configuración de Python:** Ingrese la versión **3.11.9** de Python cuando se le solicite. El script descargará, extraerá y configurará un entorno Python aislado, incluyendo los encabezados y bibliotecas C++ necesarios a través de NuGet.
   * **Dependencias Base:** El script instala herramientas de compilación modernas (`uv` y `setuptools`) e instala automáticamente los requisitos principales de Python definidos en `requirements.txt`.
   * **Configuración de PyTorch y CUDA:** El script detectará automáticamente la versión de Torch recomendada (ej. 2.8.0). Se le preguntará si desea instalarla con soporte para CUDA. Si continúa, podrá seleccionar su versión preferida de CUDA (12.6, 12.8 o 13.0) para garantizar una aceleración de GPU adecuada. Se recomienda encarecidamente la versión **12.8**.
   * **Instalación de FFmpeg:** Se le pedirá que instale FFmpeg, con opciones para elegir la versión Estable (v7.1.1) o la Última Versión.
   * **yt-dlp:** Opcionalmente puede elegir instalar el ejecutable `yt-dlp` para la descarga de medios.
   * **Clonación del Modelo Principal:** Se le pedirá que clone el repositorio disperso (sparse) de `index-tts`.
   * **Integración de RVC:** Se le pedirá que clone el repositorio disperso de `Applio` para integrar las características de RVC en el flujo de trabajo.
   * **Finalización y Parcheo:** Por último, el script inicializará automáticamente las carpetas del espacio de trabajo de la WebUI (`uix` y `wui`) y aplicará las correcciones de dependencias obligatorias a los códigos base de Index-TTS, SpeechBrain y RVC.

### 🔑 Configuración del Token de Hugging Face (`HF_TOKEN`)

El archivo de configuración `paths.bat` contiene una variable de entorno `HF_TOKEN`. Este token es estrictamente necesario para autenticar y descargar ciertos modelos y pesos restringidos desde el Hugging Face Hub.

Si aún no tiene un `HF_TOKEN` configurado como variable de entorno global en su sistema Windows, debe abrir `paths.bat` en un editor de texto e insertar manualmente su token de acceso de Hugging Face antes de intentar descargar modelos en la WebUI.

---

## 📚 Uso

Para iniciar la interfaz, ejecute el script por lotes de webui desde el directorio raíz:

```bat
webui.bat
```

La aplicación generará un directorio `projects/` para almacenar los datos de su espacio de trabajo y un archivo `wui.json` para sus preferencias globales de interfaz (como la configuración del idioma). Abra la URL local proporcionada en su terminal (generalmente `http://127.0.0.1:7860`) en su navegador.

Para lanzar tensorboard, ejecute el script por lotes de tensorboard ubicado dentro de la carpeta bat:

```bat
bat\tensorboard.bat
```

---

## ⚡ Triton

Para lograr la máxima velocidad de entrenamiento e inferencia mediante el uso de núcleos GPU compilados dinámicamente, puede habilitar Triton de OpenAI. Dado que Triton compila núcleos de forma nativa en tiempo de ejecución, los usuarios de Windows deben configurar un entorno de compilación estricto.

**Requisitos del Sistema para Triton:** 
1. **Herramientas de Compilación de Visual Studio C++:** Descargue el Instalador de Visual Studio e instale la carga de trabajo **"Desarrollo para el escritorio con C++"**. Esto proporciona el compilador esencial de MSVC (`cl.exe`).

2. **Kit de Herramientas de NVIDIA CUDA:** Instale el Kit de Herramientas CUDA oficial e independiente. La versión debe coincidir exactamente con la versión de CUDA que seleccionó para PyTorch durante la fase `install.bat` (ej. 12.6, 12.8 o 13.0). 

3. **Configuración Estricta de Rutas:** El compilador dinámico depende de rutas del sistema codificadas. Asegúrese de que su archivo `paths.bat` esté configurado para que sus variables de directorio coincidan estrictamente con las rutas de instalación locales de su MSVC y Kit de Herramientas CUDA. Si el enrutamiento interno del script por lotes no puede localizar `nvcc` o `cl.exe`, Triton no podrá compilar los núcleos.

---

## 🛡️ Licencia y Descargo de Responsabilidad Legal

Este repositorio utiliza una estructura de doble licencia:

**1. Interfaz de Usuario y Código Envolvente (Apache 2.0)**
La interfaz general de Gradio, la lógica de gestión de proyectos y los scripts de utilidades ubicados en el directorio raíz están autorizados bajo la **Licencia Apache 2.0**. Consulte el archivo `LICENSE` en el directorio raíz para obtener todos los detalles.

**2. Modelo Principal Index-TTS (Acuerdo de Licencia de Uso de Modelo de Bilibili)**
El modelo central de conversión de texto a voz, los pesos del modelo y el código de entrenamiento específico ubicados dentro del directorio `indextts/` son propiedad de Bilibili y están estrictamente regidos por el Acuerdo de Licencia de Uso de Modelo de Bilibili. Puede encontrar este acuerdo en el [repositorio oficial de gitHub](https://github.com/index-tts/index-tts) de index-tts. Al usar este software, usted acepta cumplir con sus términos, incluidas las prohibiciones sobre implementaciones de alto riesgo.

### Descargo de Responsabilidad Requerido

*Cualquier modificación realizada al modelo original en este Trabajo Derivado no está respaldada, garantizada ni asegurada por el titular de los derechos originales del modelo original, y el titular de los derechos originales renuncia a toda responsabilidad relacionada con este Trabajo Derivado.*
