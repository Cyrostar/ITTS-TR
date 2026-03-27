### 👁️ Descripción General

La pestaña de **Modelos** (Models) sirve como el centro principal de adquisición y distribución de los pesos neuronales requeridos por el pipeline de ITTS. Está diseñada para manejar checkpoints locales específicos del proyecto, arquitecturas globales compartidas y dependencias críticas del entorno.

### 📦 1. Checkpoints del Proyecto (Almacenamiento Local)

Esta sección está dedicada a los pesos principales del modelo Index-TTS. Facilita la descarga directa de repositorios preentrenados desde Hugging Face hacia el entorno de su proyecto.

- **Ingesta de Repositorios:** Puede introducir cualquier `Repo ID` de Hugging Face (ej., `IndexTeam/IndexTTS-2`) para obtener las últimas revisiones del modelo.
- **Despliegue Automatizado:** Tras una descarga exitosa, el sistema identifica y extrae automáticamente el trío central de archivos—`bpe.model`, `gpt.pth` y `config.yaml`—y los copia al directorio global de checkpoints (`ckpt/itts`) para su disponibilidad inmediata en las fases de TTS Autónomo (Standalone) y Entrenamiento (Training).
- **Navegador de Archivos:** Un visor de directorios en tiempo real le permite verificar la presencia de archivos esenciales dentro de la ruta de destino local.

### 🌐 2. Modelos de Caché Global

Para optimizar el espacio en disco y prevenir descargas redundantes, los modelos fundacionales pesados se almacenan en una caché global compartida entre todos los proyectos. Estos modelos proporcionan el andamiaje acústico y arquitectónico para el motor TTS:

- **W2V-BERT 2.0:** Un codificador de audio masivo autosupervisado utilizado para extraer representaciones de voz de alto nivel.
- **MaskGCT:** Un modelo especializado para la generación acústica no autorregresiva (non-autoregressive).
- **CampPlus:** Utilizado para la extracción de embeddings de hablantes (speaker embeddings) de alta precisión.
- **BigVGAN:** El vocoder de última generación (state-of-the-art) utilizado para transformar mel-espectrogramas matemáticos en formas de onda audibles de alta fidelidad.

### 🎙️ 3. Modelos Whisper

Esta sección administra la suite **OpenAI Whisper**, la cual es crítica para la fase de "Corpus", donde el audio sin procesar debe ser transcrito a texto.

- **Selección Granular:** Puede elegir entre toda la gama de modelos Whisper, desde `tiny` (para mayor velocidad) hasta `large-v3` (para máxima precisión de transcripción).
- **Ruta de Modelos Centralizada:** Los modelos se descargan en un directorio dedicado dentro del sistema raíz para asegurar que sean accesibles por todas las tareas de transcripción.

### 🎤 4. Requisitos Previos de RVC (Potenciado por Applio)

Esta sección maneja la adquisición de los modelos base necesarios para el pipeline de Conversión de Voz Basada en Recuperación (Retrieval-based Voice Conversion o RVC).

- **Integración de Applio:** La lógica de descarga de los requisitos previos es proporcionada generosamente y potenciada por el **repositorio RVC de Applio**, asegurando una preparación del entorno robusta y actualizada.
- **Modelos Base:** Iniciar la descarga obtiene los modelos esenciales HuBERT y RMVPE requeridos para la extracción precisa del tono (pitch) y las capacidades de conversión de voz dentro del módulo RVC.

### 🛠️ 5. Correcciones de Dependencias y Entorno

Las bibliotecas TTS avanzadas a veces requieren intervención manual para corregir dependencias upstream rotas o protocol buffers faltantes.

- **Corrección de SentencePiece:** Una utilidad dedicada para descargar el archivo `sentencepiece_model_pb2.py` directamente desde el repositorio oficial de Google.
- **Integridad del Sistema:** Esta herramienta garantiza que la lógica del tokenizador BPE tenga los bindings de Python necesarios para realizar operaciones quirúrgicas en los archivos `.model` durante el redimensionamiento del vocabulario.

### ♨️ 6. Pesos en Turco

Este módulo especializado obtiene pesos preconfigurados entrenados específicamente para el idioma turco, utilizando una estrategia de tokenización altamente optimizada.

- **Adquisición Directa:** Descarga los archivos `tr_bpe.model`, `tr_config.yaml` y `tr_gpt.pth` directamente desde el repositorio de Hugging Face `ruygar/itts_tr_lex`.
- **Enrutamiento Centralizado:** Los pesos descargados se enrutan automáticamente al directorio global `ckpt/itts`, lo que permite su utilización inmediata por el motor TTS autónomo sin requerir la creación de un proyecto aislado.
- **Tokenización Híbrida de Grafemas:** Este modelo fue entrenado utilizando una estrategia de tokenización mixta altamente eficiente. El vocabulario original en inglés fue preservado y fusionado con un vocabulario turco en minúsculas. Sin embargo, debido a que el normalizador del sistema (leyendo desde la configuración) fuerza todo el texto entrante a **mayúsculas**, la tokenización estándar BPE en minúsculas para el turco se omite por completo.
- **Convergencia Rápida:** Como resultado de este enrutamiento forzado a mayúsculas, el tokenizador procesa el texto estrictamente utilizando las letras mayúsculas del inglés robustas y preentrenadas, junto con caracteres especiales turcos en mayúsculas inyectados. Esto crea efectivamente un pipeline de tokenización tipo grafema (carácter por carácter) para el turco. Al tomar prestadas las representaciones fonéticas establecidas de las letras mayúsculas del inglés, el modelo logra una convergencia excepcionalmente rápida durante el ajuste fino (fine-tuning).