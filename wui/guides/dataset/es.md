### 👁️ Descripción General

El módulo de Preparación de Datasets (Dataset Preparation) es estrictamente responsable de adquirir, estandarizar y estructurar pares de audio-texto en un formato unificado para el entrenamiento del modelo acústico. Independientemente de la fuente de entrada, este módulo genera una salida estandarizada que consiste en un directorio `wavs/` que contiene clips de audio y un `metadata.csv` estructurado que mapea cada archivo a su transcripción de texto normalizada.

#### 🤗 Método 1: Dataset de Hugging Face

Este método le permite extraer y procesar directamente datasets de habla preexistentes desde el Hugging Face Hub.

- **Dataset de Hugging Face:** El ID del repositorio (ej., `google/fleurs`). El sistema descarga automáticamente la partición (split) `train` y analiza la estructura interna.
- **Nombre de Carpeta Objetivo (Target Folder Name):** Define el nombre del directorio local donde se almacenará el dataset bajo `datasets/<language>/<target_folder_name>`.
- **Extracción Automatizada:** El pipeline extrae texto de manera segura a través de diferentes esquemas de datasets (buscando claves como `transcription`, `text` o `sentence`) y extrae los bytes de audio sin procesar directamente en archivos `.wav`.

#### ✂️ Método 2: Cortador de Audio Personalizado (Custom Audio Slicer)

Este método procesa un archivo de audio largo único (ej., un podcast o audiolibro) en miles de clips de entrenamiento cortos y transcritos.

- **Cargar Audio (Upload Audio):** Seleccione su archivo de audio local de larga duración.
- **Duración Máxima del Clip en seg (Max Clip Duration):** Define el límite estricto para un clip de audio individual. Si un segmento de habla excede esta duración, el sistema lo corta de forma inteligente basándose en los límites de las marcas de tiempo proporcionadas por Whisper.
- **VAD y Diarización:** Detrás de escena, el sistema inicializa `pyannote/speaker-diarization-3.1` para realizar Detección de Actividad de Voz (Voice Activity Detection), aislando el habla real e ignorando los silencios largos. *(Nota: Esto requiere una variable de entorno `HF_TOKEN` válida)*.
- **Transcripción Whisper:** Cada segmento de habla aislado se introduce en el modelo Whisper `large-v3` para generar una transcripción de texto de alta precisión en su idioma objetivo.

#### ⚙️ Configuración y Controles Centrales

Ambos métodos de procesamiento comparten parámetros críticos para estandarizar sus datos de audio.

- **Idioma (Language):** Asigna la etiqueta de idioma (ej., `tr`, `en`) que dicta tanto la estructura del directorio de salida como el idioma forzado en el modelo de transcripción Whisper.
- **Remuestrear a (Resample To):** Fuerza el audio a una frecuencia de muestreo específica (16kHz, 22.05kHz, 24kHz, 44.1kHz, o 48kHz). Si se establece en `None`, se conserva la frecuencia de muestreo original del audio fuente.
- **Guardar Cada X Clips (Save Every X Clips):** Controla con qué frecuencia se vuelca el `metadata.csv` al disco. Los números más bajos proporcionan seguridad contra bloqueos, mientras que los números más altos mejoran ligeramente la velocidad de procesamiento.

#### 🔄 Pipeline de Procesamiento y Normalización

Para asegurar que el modelo acústico reciba datos matemáticamente limpios, todo el texto y el audio pasan a través de un pipeline estricto antes de guardarse:

1. **Formateo de Audio:** Las matrices de audio se convierten y guardan como archivos `.wav` de un solo canal (mono) a la frecuencia de muestreo designada.
2. **Wordificación de Texto (Text Wordification):** El texto sin procesar pasa por el `TurkishWordifier` para expandir números, fechas y símbolos a sus equivalentes de palabras habladas (ej., "1919" se convierte en "bin dokuz yüz on dokuz").
3. **Normalización de Texto:** El texto expandido luego se introduce en el `TurkishWalnutNormalizer` para manejar mayúsculas/minúsculas, estandarización de puntuación y reglas de ortografía específicas del idioma.
4. **Gestión de Estado (State Management):** El módulo rastrea los nombres de archivos procesados. Si se interrumpe, hacer clic en **♻️ Reanudar Proceso (Resume Process)** escaneará el `metadata.csv` existente y omitirá automáticamente los clips de audio que ya han sido formateados, evitando la duplicación de esfuerzos.
