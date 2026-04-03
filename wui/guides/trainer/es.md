### 👁️ Descripción General

El módulo de Entrenamiento del Modelo (Model Training) toma las características generadas por el Preprocesador y las utiliza para ajustar (fine-tune) el modelo acústico UnifiedVoice GPT. Esta fase enseña al modelo cómo mapear los tokens de texto de su idioma objetivo a los códigos semánticos acústicos correspondientes, capturando la voz, el ritmo y la emoción únicos de su conjunto de datos.

#### 📂 1. Proyecto y Descubrimiento de Datos

A diferencia de otros módulos, el entrenador depende en gran medida de la automatización para evitar desajustes en la configuración.

- **Seleccionar Proyecto (Select Project):** Al elegir un proyecto, se escanea automáticamente su carpeta `extractions/`.
- **Autodescubrimiento (Auto-Discovery):** El sistema localizará sus archivos `config.yaml`, `bpe.model` (tokenizador) y los manifiestos `train.jsonl` / `val.jsonl`. Se negará a iniciar si falta alguno de estos archivos críticos o si no coinciden.
- **Nombre de la Ejecución (Run Name):** Define el nombre de la carpeta bajo el directorio `trains/` donde se guardarán los puntos de control (checkpoints) y los registros de TensorBoard. Si se deja en blanco, toma por defecto el nombre del proyecto.

#### 🛠️ 2. Hiperparámetros Principales

Estas configuraciones dictan cómo aprende la red neuronal.

- **Épocas (Epochs):** El número de veces que el modelo iterará sobre todo el conjunto de datos de entrenamiento.
- **Tamaño de Lote (Batch Size por paso):** El número de clips de audio cargados en la GPU simultáneamente. Reduzca este valor si encuentra errores de falta de memoria (OOM) de CUDA.
- **Acumulación de Gradientes (Gradient Accumulation):** Acumula gradientes a lo largo de múltiples pasos antes de actualizar los pesos del modelo. *Nota de Ingeniería: Tamaño de Lote Real = Tamaño de Lote × Acumulación de Gradientes.*
- **Tasa de Aprendizaje (Learning Rate):** El tamaño del paso que da el optimizador al ajustar los pesos. El valor predeterminado (`2e-5`) es altamente recomendado para el ajuste fino (fine-tuning).
- **Validar y Guardar Cada (Validate & Save Every):** Dicta con qué frecuencia (en pasos) el modelo hace una pausa para ejecutar el bucle de validación y guardar un punto de control `gpt_step_X.pth`.

#### ⚙️ 3. Controles Avanzados

- **Usar Control de Duración (Use Duration Control):** Cuando está habilitado, el modelo aprende explícitamente a predecir la duración de los fonemas/palabras.
- **Abandono de Duración (Duration Dropout):** Omite aleatoriamente información de duración durante el entrenamiento (Predeterminado: `0.3`). Esto obliga al modelo a aprender un ritmo y cadencia naturales internamente, en lugar de depender únicamente de etiquetas de duración explícitas, lo que lleva a una inferencia que suena más natural.

#### 📦 4. Exportación de Pesos del Punto de Control (Unwrapping)

Durante el entrenamiento, los modelos se guardan con metadatos y estados del optimizador que son innecesarios para la inferencia.

- **Desempaquetar y Guardar (Unwrap and Save):** Esta herramienta elimina el estado del optimizador y limpia las claves del diccionario (por ejemplo, eliminando los prefijos `module.` que quedan del entrenamiento distribuido).
- **Mapeo Conformer (Conformer Mapping):** Si está habilitado, corrige las capas anidadas aplanadas (por ejemplo, convirtiendo `.conv_pointwise_conv1` de nuevo a `.conv_module.pointwise_conv1`).
- **Resultado:** Toma su `gpt_step_X.pth` seleccionado y exporta un archivo `gpt.pth` limpio y ligero directamente en su carpeta de entrenamiento, listo para ser copiado para la inferencia.