### 👁️ Descripción General

La pestaña de **Modelos** (Models) sirve como el centro principal de adquisición y distribución de los pesos neuronales requeridos por el pipeline de ITTS. Está diseñada para manejar checkpoints locales específicos del proyecto, arquitecturas globales compartidas y dependencias críticas del entorno.

### 📦 1. Checkpoints del Proyecto (Almacenamiento Local)

Esta sección está dedicada a los pesos principales del modelo Index-TTS. Facilita la descarga directa de repositorios preentrenados desde Hugging Face hacia el entorno de su proyecto.

- **Ingesta de Repositorios:** Puede introducir cualquier `Repo ID` de Hugging Face (ej., `IndexTeam/IndexTTS-2`) para obtener las últimas revisiones del modelo.

- **Despliegue Automatizado:** Tras una descarga exitosa, el sistema identifica y extrae automáticamente el trío central de archivos—`bpe.model`, `gpt.pth` y `config.yaml`—y los copia al directorio global de checkpoints (`ckpt`) para su disponibilidad inmediata en las fases de Entrenamiento (Training) e Inferencia (Inference).

- **Navegador de Archivos:** Un visor de directorios en tiempo real le permite verificar la presencia de archivos esenciales como los pesos `.pth` y las configuraciones `.yaml` dentro de la ruta local `indextts/checkpoints`.

### 🌐 2. Modelos de Caché Global

Para optimizar el espacio en disco y prevenir descargas redundantes, los modelos fundacionales pesados se almacenan en una caché global compartida entre todos los proyectos. Estos modelos proporcionan el andamiaje acústico y arquitectónico para el motor TTS:

- **W2V-BERT 2.0:** Un codificador de audio masivo autosupervisado (self-supervised) utilizado para extraer representaciones del habla de alto nivel.

- **MaskGCT:** Un modelo especializado para la generación acústica no autorregresiva (non-autoregressive).

- **CampPlus:** Utilizado para la extracción de embeddings de hablantes (speaker embeddings) de alta precisión.

- **BigVGAN:** El vocoder de última generación (state-of-the-art) utilizado para transformar mel-espectrogramas matemáticos en formas de onda audibles de alta fidelidad.

### 🎙️ 3. Modelos Whisper

Esta sección administra la suite **OpenAI Whisper**, la cual es crítica para la fase de "Corpus", donde el audio sin procesar debe ser transcrito a texto.

- **Selección Granular:** Puede elegir entre toda la gama de modelos Whisper, desde `tiny` (para mayor velocidad) hasta `large-v3` (para máxima precisión de transcripción).

- **Ruta de Modelos Centralizada:** Los modelos se descargan en un directorio dedicado dentro del sistema raíz para asegurar que sean accesibles por todas las tareas de transcripción.

### 🛠️ 4. Correcciones de Dependencias y Entorno

Las bibliotecas TTS avanzadas a veces requieren intervención manual para corregir dependencias upstream rotas o protocol buffers faltantes.

- **Corrección de SentencePiece:** Una utilidad dedicada para descargar el archivo `sentencepiece_model_pb2.py` directamente desde el repositorio oficial de Google.

- **Integridad del Sistema:** Esta herramienta garantiza que la lógica del tokenizador BPE tenga los bindings de Python necesarios para realizar operaciones quirúrgicas en los archivos `.model` durante el redimensionamiento del vocabulario.
