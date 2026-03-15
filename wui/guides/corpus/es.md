### 👁️ Descripción General

El módulo de Corpus de Lenguaje (Language Corpus) es la fase fundamental de recopilación de datos del pipeline de TTS. Un modelo acústico de alta calidad requiere un tokenizador que haya visto millones de palabras para comprender la estructura del idioma objetivo. Este módulo le permite agregar texto sin procesar de archivos PDF y TXT, y proporciona un potente conjunto de herramientas de extracción de audio y transcripción para generar nuevos datos de texto a partir de fuentes de audio sin procesar.

#### 📘 1. Espacio de Trabajo y Repositorios

Esta sección está dedicada a la construcción del archivo maestro `corpus.txt`, que posteriormente alimentará la fase de entrenamiento del Tokenizador SentencePiece.

- **Cargar PDF/Texto:** Suelte sus documentos de texto sin procesar aquí. El sistema analiza el texto, opcionalmente filtra palabras únicas para maximizar la eficiencia del vocabulario sin inflar el archivo, y los almacena en los repositorios del proyecto.
- **Repositorios:** Muestra los archivos PDF y TXT procesados exitosamente que residen actualmente en el espacio de trabajo de su proyecto.
- **Combinar Todos los Archivos de Mezcla:** Compila cada documento procesado en sus repositorios en un único y masivo archivo `corpus.txt`.

#### 🧰 2. Adquisición y Limpieza de Audio

Si carece de texto sin procesar pero tiene acceso a voz, estas herramientas le ayudan a extraer y preparar el audio para la transcripción.

- **Descargador de YouTube:** Pegue una URL para obtener y extraer inmediatamente la pista de audio de un video. Excelente para recopilar datos de podcasts o entrevistas.
- **Limpiador de Audio (Demucs):** El audio sin procesar a menudo contiene música de fondo o ruido que arruina la transcripción y el entrenamiento acústico. Esta herramienta utiliza la red neuronal `Demucs` para aislar matemáticamente la pista vocal humana y descartar el ruido de fondo.

#### 🎙️ 3. Transcripción y Diarización

Convierta su audio limpio en texto utilizable y archivos de hablantes separados.

- **Transcriptor de Audio (Whisper):** Introduce su audio en el modelo Whisper de OpenAI para generar texto puntuado de alta precisión.
  - *Normalizador Walnut:* Pasa automáticamente la salida de Whisper a través de nuestro normalizador personalizado para que el texto esté perfectamente formateado para el entrenamiento TTS.
- **Diarización (Separación de Hablantes):** Utiliza el modelo `pyannote/speaker-diarization-3.1` para detectar múltiples hablantes en un solo archivo de audio.
  - *Recortar Silencio:* Une automáticamente los segmentos detectados, eliminando pausas largas.
  - *Archivos de Hablantes:* Exporta un archivo `.wav` independiente para cada hablante único detectado, aislando completamente su habla para la creación de conjuntos de datos específicos.

#### 🏷️ 4. Estandarización de Archivos

Los pipelines de aprendizaje automático requieren rutas de archivos estrictas y predecibles.

- **Nombrador de Documentos y Nombrador de Audiolibros:** Estas utilidades sanean sus cadenas de entrada (por ejemplo, convirtiendo caracteres turcos como 'ç' a 'c', reemplazando espacios con guiones bajos) y aplican una convención de nomenclatura estricta (`Genero_Autor_Titulo.txt` o `Fuente_Narrador_Genero_Autor_Titulo.wav`). Utilícelos *antes* de cargar archivos para mantener la estructura de su espacio de trabajo intacta.
