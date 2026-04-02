### 👁️ Visión General

El módulo Tokenizador es estrictamente responsable de entrenar un modelo de Codificación de Pares de Bytes (BPE) utilizando SentencePiece, el cual traduce su texto sin procesar a secuencias numéricas comprensibles por el modelo acústico TTS. Gestiona la generación de vocabulario, la cobertura de caracteres, la normalización de texto y la inyección de tokens especiales.

#### 📂 1. Selección de Datos

Antes de entrenar, debe definir la base textual de la que aprenderá el tokenizador.

- **Seleccionar Idioma y Conjunto de Datos:** Elija su idioma de destino y la carpeta de datos específica. La interfaz de usuario (UI) analizará automáticamente el archivo `metadata.csv` asociado con este conjunto de datos.

- **Cobertura de Metadatos:** Un control deslizante (10% - 100%) que le permite muestrear un porcentaje específico de los metadatos de su conjunto de datos para el entrenamiento. Útil para la creación rápida de prototipos en conjuntos de datos masivos.

- **Incluir Texto del Corpus Unificado:** Agrega el contenido de su base de datos `corpus.db` a los metadatos de su conjunto de datos.

- **Entrenar Solo con Texto del Corpus:** Ignora los metadatos del conjunto de datos por completo y entrena exclusivamente en la base de datos `corpus.db`.

#### 🧠 2. Configuración de Vocabulario y Cobertura

- **Tamaño del Vocabulario:** Determina el número máximo de tokens únicos (subpalabras/palabras) que el modelo puede memorizar. El control deslizante varía de 2,000 a 30,000, con un valor predeterminado de 12,000. *Nota de ingeniería: Seleccionar un conjunto de datos intentará sincronizar automáticamente este valor con el `number_text_tokens` definido en el archivo `config.yaml` de su proyecto.*

- **Cobertura de Caracteres:** Define el porcentaje de variaciones de caracteres sin procesar a abarcar dentro del modelo. El valor predeterminado es `1.0` (100%).

#### 🏷️ 3. Tokens Especiales y Etiquetas (Tags)
- **Etiquetas de Estilo y Emoción:** Inyecta automáticamente etiquetas predefinidas de conversación, narración y estado emocional (ej. `[happy]`, `[whisper]`, `[podcast]`) para enseñar al modelo acústico límites de entrega altamente expresivos.
- **Extensiones de Alfabeto:** Casillas de verificación para forzar explícitamente al tokenizador a memorizar letras inglesas estándar, caracteres extendidos túrquicos, vocales largas turcas (ej. `â`, `î`) y signos de puntuación estándar.
- **Tokens Especiales Personalizados:** Un campo de texto donde puede definir símbolos específicos, monedas o caracteres (separados por `|`) para bloquearlos manualmente en el vocabulario.

#### 💉 4. Inyecciones Fonéticas y Lingüísticas
Estas funciones principales eluden el algoritmo estadístico BPE. Fuerzan al Tokenizador a bloquear permanentemente unidades lingüísticas específicas en su matriz de vocabulario, garantizando una alineación acústica perfecta durante la síntesis TTS.

- **Inyectar Sílabas de Alta Frecuencia:** Consulta su `corpus.db` compilado para obtener las sílabas absolutamente más comunes en sus conjuntos de datos (gobernado por el valor **Recuento de Sílabas**) y las bloquea. Esto proporciona anclajes fonéticos estrictos que reducen drásticamente la pronunciación arrastrada y los artefactos de audio omitidos.
- **Inyectar Palabras de Alta Frecuencia:** Consulta la base de datos para obtener las palabras enteras más frecuentes (gobernado por el valor **Recuento de Palabras**). Codificar de forma fija las palabras frecuentes permite al modelo aprender su prosodia natural y distintiva (ritmo y entonación) como una incrustación acústica única, en lugar de unirlas de manera robótica.
- **Motor de Capacidad de Vocabulario:** El sistema calcula dinámicamente si sus inyecciones forzadas (etiquetas + sílabas + palabras) dejan suficiente espacio obligatorio (al menos 256 ranuras) para el alfabeto lingüístico básico y los bytes de control. Detendrá la canalización de forma segura si detecta un riesgo de desbordamiento de vocabulario.

#### ⚙️ 5. Reglas de Entrenamiento Avanzadas
- **Normalización y Mayúsculas/Minúsculas:** Determine si desea eludir el normalizador interno de SPM (regla `identity`) y si desea forzar su vocabulario a formatos estrictos de mayúsculas o minúsculas.
- **Máximo de Oraciones (Tamaño de Muestra):** Limita el uso de RAM en conjuntos de datos masivos al limitar las líneas analizadas. Ponga 0 para usar todas las oraciones disponibles.
- **Entrenar Corpus Extremadamente Grande:** Activa optimizaciones de memoria de C++ para analizar flujos de entrenamiento de varios gigabytes.
- **Mezclar Corpus:** Aleatoriza los flujos de entrada analizados para garantizar una distribución lingüística uniforme.
- **Límite Estricto de Vocabulario:** Aplica estrictamente el tamaño de vocabulario solicitado sin rellenar el espacio final.

### 🧰 Utilidades

Estas herramientas de diagnóstico le permiten validar su canalización de procesamiento de texto antes de comprometerse con el entrenamiento del modelo acústico.

#### 🎗️ Comprobación de Seguridad del Tokenizador

Un conjunto de validación automatizado para comprobar la idoneidad del tokenizador para TTS.

- Suba su archivo `.model` de SentencePiece entrenado para verificar que captura con éxito caracteres estándar.
- Prueba el mapeo explícito de caracteres especiales según sus reglas de normalización y mayúsculas/minúsculas.
- Comprueba la presencia de tokens de retroceso de bytes (byte-fallback) perjudiciales.
- Ejecuta una tokenización de muestra en palabras complejas para asegurar que no se generen tokens `<unk>`.

#### 💱 Probador de Tokenizador

Una herramienta de inferencia directa para visualizar cómo su modelo desglosa el texto.

- Introduzca texto sin procesar para ver exactamente cómo el tokenizador del proyecto activo lo divide en subpalabras.
- Pruebe los estados de modelo Estándar (entrenado) y Fusionado (merged).
- Muestra el recuento total de tokens y una matriz detallada de pares `[ID] Fragmento`.

#### 📚 Wordifier Multilingüe

Prueba la expansión numérica/de fechas y la lógica de extracción de palabras únicas.

- Introduzca texto que contenga estructuras complejas como números, fechas o abreviaturas (ej. "19.05.1919" o "2.500").
- **Formato de Retorno:** Alterne entre "Bloque Completo" (la oración expandida) o "Lista de Palabras" (una matriz separada por comas de las palabras extraídas).

#### 🫧 Normalizador Multilingüe

Prueba la lógica de preprocesamiento en el texto sin procesar.

- Introduzca texto desordenado que contenga una mezcla de mayúsculas/minúsculas, errores de puntuación o símbolos especiales.
- La salida revela exactamente cómo el modelo acústico "leerá" el texto después de que se apliquen las reglas de normalización y las expansiones de abreviaturas.

#### ✂️ Silabificador Turco

Prueba la silabificación turca, el marcado de acento y los algoritmos de armonía.

- Introduzca texto turco para ver cómo el sistema lo desglosa programáticamente en distintas sílabas fonéticas.
- Alterne comprobaciones lingüísticas avanzadas como marcadores de acento, validación de armonía vocálica y un modo de análisis detallado palabra por palabra.

#### 🎨 Diseñar Modelo Personalizado

Una potente interfaz visual para diseñar un modelo personalizado a partir de los modelos de tokenizador original (Origen) y entrenado (Destino).

- Suba un modelo de destino para fusionarlo con el modelo base oficial.
- **Configuración de Origen:** Elimine elementos no deseados del modelo base como marcadores de idioma, tokens CJK, tokens en inglés o puntuación de origen. También puede forzar la conversión a minúsculas e inyectar tokens estructurales requeridos.
- **Configuración de Destino:** Defina exactamente cómo el nuevo modelo se fusiona con la base. Elija si desea preservar letras independientes y puntuación, y aplique estrictamente las reglas de mayúsculas/minúsculas de tokens.
- Proporciona una salida directa lado a lado del Vocabulario de Origen procesado, el Vocabulario de Destino procesado y la Salida Fusionada final.