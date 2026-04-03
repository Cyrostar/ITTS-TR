### 👁️ Descripción General

El módulo de Corpus de Lenguaje (Language Corpus) es la fase fundamental de recopilación de datos del pipeline de TTS. Un modelo acústico de alta calidad requiere un tokenizador que haya visto millones de palabras para comprender la estructura del idioma objetivo. Este módulo ha sido actualizado masivamente para utilizar una base de datos SQLite multinúcleo de alto rendimiento (`corpus.db`) para agregar texto sin procesar de archivos PDF y TXT, junto con un potente conjunto de herramientas de extracción y transcripción de audio para generar nuevos datos de texto.

#### 🗄️ 1. Motor de Procesamiento de Base de Datos

Esta sección detalla las pestañas principales utilizadas para construir su base de datos de vocabulario, la cual reemplaza el antiguo sistema `corpus.txt`.

- **Constructor de Corpus PDF (PDF Corpus Builder):** Apunte el sistema a una carpeta de PDFs. Utiliza todos los núcleos de CPU disponibles para extraer texto en paralelo, dividiéndolo en fragmentos legibles y almacenándolos de forma segura en la base de datos.
- **Normalizador de Texto (Text Normalizer):** Lee los fragmentos sin procesar utilizando paginación B-Tree tolerante a fallos y los procesa a través del Normalizador Multilingüe mediante un grupo (pool) persistente de CPU. Guarda fragmentos únicos y agrega sus recuentos de aparición exactos.
- **Extractor de Palabras (Word Extractor):** Escanea la base de datos normalizada para extraer palabras individuales distintas, calculando sus frecuencias exactas en todo su conjunto de datos.
- **Silabificador (Syllabifier):** Procesa el texto normalizado a través del Silabificador Turco (o equivalentes según el idioma) para desglosar el texto en sílabas fonéticas. Multiplica las frecuencias por la aparición del fragmento para encontrar los sonidos fonéticos absolutos más comunes.
- **Estadísticas de Vocabulario (Vocabulary Statistics):** Una vista analítica para comprobar las 10 palabras y sílabas principales en su base de datos. También puede exportar las listas de los 2000 principales directamente a archivos JSON para uso externo.
- **Tokenizador (Tokenizer):** Entrena un modelo SentencePiece de Codificación de Pares de Bytes (BPE) directamente desde los fragmentos de texto prenormalizados en su base de datos. Fuerza automáticamente las 1000 palabras y sílabas de mayor frecuencia en el vocabulario para garantizar la estabilidad fonética.

#### 🧰 2. Espacio de Trabajo y Utilidades de Carga

- **Agregar Documentos (Add Documents):** Suelte sus documentos de texto sin procesar aquí. Puede guardarlos en las carpetas de su proyecto local o usar el botón "Procesar y Fusionar a BD" (Process & Merge to DB) para dividirlos en fragmentos, normalizarlos e inyectarlos instantáneamente en su base de datos.
- **Repositorios de Archivos (File Repositories):** Muestra los archivos PDF y TXT procesados exitosamente que residen actualmente en el espacio de trabajo de su proyecto.

#### 🧽 3. Adquisición y Limpieza de Audio

Si carece de texto sin procesar pero tiene acceso a voz, estas herramientas le ayudan a extraer y preparar el audio para la transcripción.

- **Descargador de YouTube (YouTube Downloader):** Pegue una URL para obtener y extraer inmediatamente la pista de audio de la más alta calidad de un video. Excelente para recopilar datos de podcasts o entrevistas.
- **Limpiador de Audio (Audio Cleaner - Demucs):** El audio sin procesar a menudo contiene música de fondo o ruido que arruina la transcripción y el entrenamiento acústico. Esta herramienta utiliza la red neuronal `htdemucs` para aislar matemáticamente la pista vocal humana y descartar el ruido de fondo.

#### 🎙️ 4. Transcripción y Diarización

Convierta su audio limpio en texto utilizable y archivos de hablantes separados.

- **Transcriptor de Audio (Audio Transcriptor - Whisper):** Introduce su audio en el modelo Whisper de OpenAI (hasta `large-v3`) para generar texto puntuado de alta precisión.
  - *Opción de Normalizador (Normalizer Toggle):* Pasa automáticamente la salida de Whisper a través del normalizador para que el texto esté perfectamente formateado para el entrenamiento TTS.
- **Diarización (Separación de Hablantes):** Utiliza el modelo `pyannote/speaker-diarization-3.1` para detectar múltiples hablantes (hasta un máximo definido por el usuario) en un solo archivo de audio.
  - *Recortar Silencio (Trim Silence):* Une automáticamente los segmentos detectados según su configuración de espacios (gaps).
  - *Archivos de Hablantes:* Exporta un archivo `.wav` independiente para cada hablante único detectado, aislando completamente su habla para la creación de conjuntos de datos específicos.

#### 🏷️ 5. Estandarización de Archivos

Los pipelines de aprendizaje automático requieren rutas de archivos estrictas y predecibles.

- **Nombrador de Documentos y Nombrador de Audiolibros (Document & Audiobook Namer):** Estas utilidades sanean sus cadenas de entrada (por ejemplo, convirtiendo caracteres turcos como 'ç' a 'c', reemplazando guiones con espacios y forzando guiones bajos) para crear una convención de nomenclatura estricta (`Genero-Autor-Titulo` o `Audiobook-Fuente-Narrador-Genero-Autor-Titulo`). Utilícelos *antes* de cargar archivos para mantener la estructura de su espacio de trabajo intacta.