### 👁️ Descripción General

El módulo de Preprocesador (Extracción de Características) es un puente crítico entre su conjunto de datos sin procesar y la fase de entrenamiento acústico. El audio y el texto sin procesar no pueden introducirse directamente en el modelo TTS. En su lugar, este módulo pasa sus datos a través de múltiples redes neuronales preentrenadas para extraer representaciones matemáticas de alta dimensión (características) y las guarda como matrices `.npy`.

#### 📂 1. Fuente de Datos y Configuración Principal

Antes de extraer características, debe definir el conjunto de datos de destino y cómo deben estructurarse los datos para el entrenamiento.

- **Conjunto de Datos de Destino (Target Dataset):** La carpeta específica del conjunto de datos (creada en la Fase 3) que contiene su directorio `wavs/` y `metadata.csv`.
- **Idioma de la Carpeta (Folder Language):** Filtra el menú desplegable de conjuntos de datos para mostrar los que pertenecen a una etiqueta de idioma específica.
- **Inyectar Marcador de Idioma (Inject Language Marker):** Determina si se debe anteponer un token de ID de idioma específico a los tokens de texto.
  - **Ninguno (None):** No se inyecta ningún ID de idioma.
  - **TR (ID-3) / EN (ID-4):** Fuerza al modelo a reconocer el idioma explícitamente, lo cual es crucial para modelos multilingües.
- **División de Validación (%) (Validation Split):** Determina el porcentaje de su conjunto de datos que se reservará del entrenamiento. Estos datos retenidos (`val.jsonl`) se utilizan para probar la precisión del modelo en datos no vistos durante la fase de entrenamiento.

#### ⚡ 2. Configuración de Rendimiento

La extracción de características consume muchos recursos. Estas configuraciones le permiten equilibrar la velocidad con los límites de su hardware.

- **Tamaño de Lote (Batch Size):** El número de clips de audio procesados simultáneamente por la GPU. Disminuya este valor si encuentra errores de falta de memoria (OOM) de CUDA.
- **Trabajadores de CPU (CPU Workers):** El número de hilos de CPU paralelos dedicados a cargar archivos de audio desde su disco. Los valores más altos aceleran la canalización de datos pero consumen más memoria RAM del sistema.

#### ⚙️ 3. Configuración Avanzada

- **Usar Rutas Relativas (Use Relative Paths):** Cuando está marcado, los archivos de manifiesto generados (`.jsonl`) almacenarán rutas relativas en lugar de rutas absolutas. Esto es muy recomendable ya que le permite mover la carpeta de su proyecto a otra unidad o máquina sin romper los enlaces del conjunto de datos.
- **Usar Tokenizador Fusionado (Use Merged Tokenizer):** Instruye al extractor a usar el modelo `_bpe_merged.model` en lugar del tokenizador estándar. Utilice esto solo si ha fusionado explícitamente múltiples tokenizadores.
- **Compilar con Torch (Torch Compile):** Utiliza `torch.compile()` de PyTorch 2.0+ para optimizar los modelos de extracción. Esto acelera significativamente el proceso de extracción, pero requiere un período de "calentamiento" inicial en el que el proceso parecerá congelado.

#### 🧲 4. Bajo el Capó: ¿Qué se Extrae?

Por cada par válido de audio y texto, el sistema extrae cuatro características específicas y las guarda en la carpeta `extractions/<dataset_name>`:

1. **IDs de Texto (`text_ids/`):** El texto sin procesar se tokeniza utilizando su modelo SentencePiece personalizado en una secuencia de números enteros.
2. **Códigos Semánticos (`codes/`):** El audio se pasa a través del extractor de características `W2V-BERT 2.0` y se cuantifica utilizando el códec semántico `MaskGCT` para crear tokens de audio discretos.
3. **Condicionamiento (`condition/`):** Características acústicas de alto nivel procesadas por el modelo GPT UnifiedVoice.
4. **Vectores de Emoción (`emo_vec/`):** Incrustaciones (embeddings) emocionales y prosódicas extraídas de las características semánticas.

Finalmente, el módulo genera dos archivos de manifiesto (`train.jsonl` y `val.jsonl`) que mapean las rutas de audio originales a estas matrices de características `.npy` recién generadas.