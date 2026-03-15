### 👁️ Descripción General

Bienvenido al corazón del motor TTS. El archivo `config.yaml` dicta las dimensiones arquitectónicas exactas, los límites de secuencia y las matemáticas de procesamiento de audio utilizadas durante el entrenamiento del modelo y la inferencia. **Advertencia:** Cambiar parámetros avanzados a mitad del entrenamiento romperá su checkpoint. Solo ajuste esta configuración *antes* de comenzar una nueva ejecución de entrenamiento.

#### 🧠 1. Hiperparámetros Centrales (Core Hyperparameters)

Estas son las configuraciones más críticas. Controlan directamente el "tamaño del cerebro" de su modelo y determinan cuánta VRAM de la GPU necesitará.

- **Tamaño del Vocabulario (number_text_tokens):** Esto dicta el número máximo de tokens únicos que el modelo GPT puede entender. Este número *debe* coincidir exactamente con el tamaño del vocabulario del Tokenizador (archivo `.model`) que entrenará en la Fase 4.

- **Frecuencia de Muestreo de Audio (Audio Sample Rate):** La frecuencia objetivo para su procesamiento de audio. Los valores estándar son 22050 Hz o 24000 Hz. Las tasas más altas producen un audio más nítido pero requieren una potencia de cálculo significativamente mayor.

- **Tokens Máximos de Texto/Mel (Max Text/Mel Tokens):** La longitud máxima del texto de entrada o de las secuencias de audio generadas. Si establece el texto en 600, cualquier cosa más larga se trunca. Aumentar estas longitudes incrementará exponencialmente su uso de VRAM de la GPU.

- **Dimensión del Modelo (model_dim):** El tamaño oculto interno del transformador GPT. 1024 o 1280 son estándar para modelos de alta calidad. * **Capas y Cabezales (Layers & Heads):** El número de bloques de transformadores apilados juntos (Capas) y los mecanismos de atención paralela (Cabezales). Más capas significan mejor razonamiento y prosodia, pero un entrenamiento más lento.

#### 🔤 2. Tokenizador Avanzado y Front-End de Texto

Esta sección configura el pipeline de procesamiento de texto, determinando cómo el texto de entrada sin procesar se sanea, normaliza y convierte en tokens lingüísticos antes de entrar al modelo acústico.

- **Idioma (Language):** El código del idioma objetivo (ej., `en`, `tr`, `es`). Esto enruta dinámicamente el texto a través del normalizador específico del idioma correcto para aplicar las listas blancas de caracteres y las reglas de división de puntuación adecuadas.
- **Tipo de Tokenizador y Tipo de Vocab (Tokenizer Type & Vocab Type):** Define el enfoque algorítmico utilizado para dividir el texto en entradas de red digeribles (como `bpe` para Byte-Pair Encoding o análisis a nivel de caracteres).
- **Formato de Mayúsculas/Minúsculas (Case Format):** Determina si el texto debe estandarizarse a un formato específico (ej., forzando todos los caracteres a minúsculas o mayúsculas) para que coincida con el estado exacto con el que se entrenó el Tokenizador.
- **Palabrizar (Wordify):** Cuando está habilitado, el sistema expande números, fechas, horas, monedas y símbolos matemáticos en sus equivalentes completos de palabras habladas (ej., "$5" se convierte en "cinco dólares").
- **Abreviaturas (Abbreviations):** Alterna la expansión de abreviaturas comunes específicas del idioma (ej., convertir "Dr." en "Doctor" o "Av." en "Avenida").
- **Extraer (Extract / Grapheme Extraction):** Cuando está habilitado, esta función inserta forzosamente espacios entre cada carácter individual (ej., "hola" se convierte en "h o l a"). Esto suele ser necesario para modelos específicos de alineación acústica a nivel de caracteres que no utilizan la fusión de subpalabras (sub-word merging) BPE.

#### 🎛️ 3. Configuración de Dataset y Mel

Esta sección controla cómo sus archivos de audio `.wav` sin procesar se convierten matemáticamente en espectrogramas que la red neuronal puede leer.

- **Modelo BPE:** El nombre de archivo exacto del modelo tokenizador (ej., `bpe.model`) que debe buscar el cargador del dataset.

- **N FFT (n_fft):** El tamaño de la ventana de la Transformada Rápida de Fourier (Fast Fourier Transform). 1024 es el estándar de la industria para audio de 22kHz-24kHz.

- **Longitud de Salto (Hop Length):** El número de muestras de audio entre tramas STFT sucesivas. Un valor de 256 significa que el modelo toma una "instantánea" del audio cada 256 muestras.

- **Longitud de Ventana (Win Length):** El tamaño de la función de ventana aplicada al audio. Generalmente coincide con `n_fft` (1024).

- **N Mels:** El número de bandas de frecuencia Mel a generar. 80 o 100 son estándar.

- **Normalizar Mel (Normalize Mel):** Si se deben normalizar estadísticamente los valores del espectrograma. Mantenga esto en `False` a menos que su script de entrenamiento específico requiera explícitamente entradas normalizadas.

#### 🧩 4. Lógica de Tokens GPT

Esta sección define los límites estructurales y la lógica de condicionamiento para el proceso generativo de texto a audio.

- **Usar Códigos Mel como Entrada (Use Mel Codes as Input):** Cuando es `True`, el modelo retroalimenta los tokens acústicos en sí mismo de forma autorregresiva durante el entrenamiento.

- **Entrenar Embeddings Solos (Train Solo Embeddings):** Una bandera especializada para aislar las capas de incrustación (embedding) durante etapas específicas de ajuste fino (fine-tuning).

- **Tipo de Condición (Condition Type):** Define el módulo arquitectónico utilizado para unir texto y audio. `conformer_perceiver` es un mecanismo de atención cruzada altamente avanzado y eficiente.

- **Tokens de Inicio/Parada (Start/Stop Tokens):** Estos son números de identificación estrictos que le dicen al modelo cuándo comienza y termina una secuencia de audio o una secuencia de texto (por ejemplo, `start_text_token` predeterminado a 0).

- **Número de Códigos Mel (Number Mel Codes):** El tamaño total del vocabulario de su códec de audio semántico.

#### 🔗 5. Checkpoints y Vocoder

Rutas y definiciones para los pesos externos y componentes en los que confía su pipeline.

- **Checkpoints (gpt.pth, s2mel.pth):** Los nombres de archivo relativos donde el sistema guardará (o reanudará desde) los pesos del modelo GPT y Semántico-a-Mel.

- **Estadísticas y Matrices W2V:** Rutas a tensores estadísticos precalculados (como `wav2vec2bert_stats.pt`, `feat1.pt`) utilizados para el condicionamiento de hablante y emoción.

- **Ruta de Emoción Qwen (Qwen Emo Path):** Ruta del directorio para el LLM subyacente utilizado para la extracción de emociones.

- **Tipo y Nombre de Vocoder:** El Vocoder es la red neuronal responsable de tomar el espectrograma generado por IA y convertirlo de nuevo en un archivo `.wav` audible. `bigvgan` es un vocoder de última generación que produce artefactos de voz excepcionalmente nítidos y parecidos a los humanos.

#### ⬡ → ◯ Redimensionamiento del Modelo y Preservación de Pesos en Index-TTS

Al generar una nueva configuración para la arquitectura UnifiedVoice, el sistema emplea un algoritmo inteligente de transferencia de pesos para asegurar que sus pesos preentrenados nunca se pierdan innecesariamente. Esta guía explica la mecánica matemática detrás de este proceso.

### 1. El Mecanismo de Corte Central (The Core Slicing Mechanism)

La preservación de los pesos entrenados depende de calcular el límite matemático superpuesto entre el tensor preentrenado original y su tensor recién configurado.

El sistema logra esto utilizando una operación de corte dinámica que evalúa el tamaño de dimensión mínimo entre las formas antiguas y nuevas (`min(ds, ts)`). Esto asegura que solo transfiramos la intersección exacta de datos que encaja perfectamente en el nuevo gráfico computacional.

### 2. Escenarios de Redimensionamiento

Dependiendo de cómo ajuste los parámetros en la interfaz de configuración, el modelo maneja los pesos preentrenados de tres maneras distintas:

- **Capas Idénticas (Sin Cambio):** Si los parámetros estructurales centrales como `model_dim`, `layers` y `heads` permanecen intactos, los componentes profundos de la red (bloques de atención del Transformador, capas feed-forward y capas de normalización) se mapean 1:1. Los pesos preentrenados se copian perfectamente sin alteraciones.

- **Expansión (ej., Aumento de Vocabulario):** Si aumenta el `number_text_tokens` (por ejemplo, de 10,000 a 12,000), los 10,000 embeddings entrenados originales se copian directamente en el nuevo tensor. Los 2,000 espacios recién agregados se inicializan con pesos aleatorios no entrenados, asegurando que el modelo retenga su conocimiento fundamental de los caracteres base.

- **Truncamiento (ej., Disminución de Contexto):** Si reduce un parámetro, el script recorta matemáticamente el tensor. Conserva los pesos entrenados desde el índice 0 hasta su nuevo límite de corte. Debido a que los tokenizadores generalmente se ordenan por frecuencia, esto retiene de manera segura los tokens más críticos y altamente entrenados.

### 3. La Excepción: Discrepancias de Dimensión

El único escenario donde se descartan los pesos entrenados es si altera drásticamente una dimensión arquitectónica central, como cambiar `model_dim` de 1280 a 512, lo que crea una forma incompatible.

En tales casos, el sistema se basa en un mecanismo de degradación elegante (graceful degradation fallback):

- Detecta la discrepancia de dimensión.

- Omite de forma segura esa capa específica.

- Registra una advertencia precisa en la interfaz de usuario (ej., `Skipped layer... Dimension mismatch`).

- Inicializa solo esa capa incompatible desde cero mientras rescata el resto de los componentes de red compatibles.
