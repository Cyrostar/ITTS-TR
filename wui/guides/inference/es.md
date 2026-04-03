### 👁️ Descripción General

El módulo de Inferencia es la etapa final del pipeline, donde su modelo acústico entrenado genera habla a partir de texto. Esta interfaz le permite probar sus voces personalizadas ajustadas (fine-tuned), compararlas con el modelo base oficial y manipular profundamente la entrega emocional del audio generado.

#### 📂 1. Configuración de Modelo y Texto

Antes de generar voz, debe seleccionar el modelo y proporcionar el texto de destino.

- **Selección de Carpeta (Folder Selection):** Elija su proyecto entrenado en el menú desplegable. El sistema cargará automáticamente el último punto de control desempaquetado (`gpt.pth` o `latest.pth`) desde el directorio `trains/` de su proyecto.
- **Usar Modelo Original (Use Original Model):** Omite su carpeta de entrenamiento personalizada y carga el modelo base oficial sin modificaciones. Esto es excelente para comparar la calidad de referencia con sus resultados ajustados.
- **Idioma (Language):** Fuerza un token de idioma específico (`TR` o `EN`) al principio del texto para guiar las reglas de pronunciación del modelo. Configurar esto en `Auto` confía en la suposición interna de idioma del modelo.
- **Texto de Entrada (Input Text):** El texto que desea que lea el modelo. 

#### 📢 2. Control de Voz y Emoción

La arquitectura UnifiedVoice separa la identidad acústica del hablante (Voz) de su entrega (Emoción).

- **Audio de Referencia o Voz (Reference Audio):** Suba un clip de audio limpio y corto (3-10 segundos). El modelo clonará las características acústicas (timbre, tono, entorno) de este hablante.
- **Modo de Control (Control Mode):** Dicta cómo el modelo determina la entrega emocional del texto:
  - **Igual que la Referencia (Same as Reference):** El modelo extrae y copia la emoción directamente de su Audio de Referencia de Voz.
  - **Audio de Referencia (Reference Audio):** Le permite cargar un *segundo* archivo de audio específicamente para extraer emoción. (Por ejemplo, la voz de la Persona A, pero llorando como la Persona B).
  - **Vectores de Emoción (Emotion Vectors):** Desbloquea controles deslizantes manuales (Feliz, Enojado, Triste, Melancolía, etc.) que le permiten marcar pesos emocionales exactos.
  - **Texto de Descripción (Description Text):** Aprovecha un LLM integrado (Qwen) para interpretar una instrucción de texto (por ejemplo, "susurrando urgentemente") y convertirla matemáticamente en vectores de emoción.
- **Intensidad de Emoción (Emotion Intensity):** Un multiplicador (0.0 a 1.0) que escala con qué fuerza la emoción seleccionada anula la línea base neutral.

#### ⚙️ 3. Parámetros Avanzados

Estas configuraciones controlan el proceso matemático de generación subyacente.

- **Habilitar Kernel CUDA BigVGAN (Enable BigVGAN CUDA Kernel):** Si su entorno lo admite (requiere `ninja` y un compilador de C++), esto acelera significativamente la generación de la forma de onda final.
- **Habilitar Muestreo (Enable Sampling):** Cuando está marcado, el modelo introduce una ligera aleatoriedad, haciendo que las generaciones repetidas suenen ligeramente diferentes y más naturales. 
- **Temperatura y Top P (Temperature & Top P):** Controla la aleatoriedad del muestreo. Los valores más bajos hacen que el habla sea altamente predecible y estable, mientras que los valores más altos la hacen más expresiva pero propensa a artefactos.
- **Tokens Máximos (Max Tokens):** Límites estrictos para evitar bucles de generación infinitos en segmentos excepcionalmente largos.