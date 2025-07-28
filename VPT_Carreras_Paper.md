# Video PreTraining (VPT) para Juegos de Carreras de Coches: Aprendiendo a Conducir Observando Videos sin Etiquetas

## Resumen
Presentamos un enfoque de preentrenamiento de video (VPT) adaptado a juegos de carreras de coches, que permite entrenar agentes de inteligencia artificial capaces de conducir vehículos en simuladores o videojuegos de carreras, utilizando videos no etiquetados de partidas humanas. Inspirado en el método original aplicado a Minecraft, nuestro enfoque se basa en dos etapas principales: (1) entrenamiento de un modelo de dinámica inversa (IDM) con un pequeño conjunto de datos etiquetados, y (2) clonación de comportamiento a gran escala usando videos no etiquetados.

## 1. Introducción
El aprendizaje por observación es una vía prometedora para entrenar agentes en tareas complejas donde la recolección de datos etiquetados es costosa. En los juegos de carreras, existen grandes cantidades de videos de partidas humanas disponibles en línea, pero carecen de información explícita sobre las acciones tomadas (dirección, acelerador, freno, etc.). Nuestro objetivo es aprovechar estos videos para entrenar agentes que puedan aprender a conducir de manera competente en entornos de carreras.

## 2. Metodología
### 2.1. Modelo de Dinámica Inversa (IDM)
Entrenamos un modelo IDM utilizando un conjunto reducido de datos donde cada frame de video está asociado a las acciones humanas reales (por ejemplo, telemetría de simuladores, datos de hardware, etc.). El IDM aprende a predecir las acciones (dirección, acelerador, freno, marcha, etc.) a partir de secuencias de imágenes y, opcionalmente, otros sensores (velocidad, posición en pista).

### 2.2. Etiquetado Automático de Videos
El IDM se utiliza para inferir las acciones en grandes volúmenes de videos no etiquetados, generando así un dataset sintético de pares (video, acción) para entrenamiento masivo.

### 2.3. Clonación de Comportamiento
Entrenamos un agente de conducción mediante clonación de comportamiento sobre el dataset generado, permitiendo que el agente aprenda a imitar el comportamiento humano observado en los videos.


## 3. Espacio de Acciones y Observaciones
### 3.1. Espacio de Acciones
- **Dirección (Steering):** valor continuo en [-1, 1] o discretizado en varios niveles.
- **Acelerador (Throttle):** valor continuo en [0, 1] o discretizado.
- **Freno (Brake):** valor continuo en [0, 1] o discretizado.
- **Cambio de marcha (Gear):** entero o acción discreta (subir/bajar marcha).
- **Otros controles:** embrague, freno de mano, DRS, etc., según el juego.

### 3.2. Espacio de Observaciones
- **Imagen de la cámara frontal:** secuencia de frames RGB.
- **Velocidad:** escalar o vectorial.
- **Posición en pista:** coordenadas, distancia al borde, etc.
- **Estado del vehículo:** daños, temperatura, etc.
- **Sensores adicionales:** LIDAR, mapas de pista, etc. (opcional).

### 3.3. Formato de Datos
Los datos se almacenan como pares (video, acción) en archivos `.mp4` y `.jsonl`, donde cada línea del `.jsonl` contiene las acciones y observaciones asociadas a cada frame o intervalo de tiempo.


## 4. Arquitectura de los Modelos
### 4.1. Modelo de Dinámica Inversa (IDM)
- **Entrada:** secuencia de imágenes (y sensores opcionales).
- **Red neuronal:** CNN (por ejemplo, ResNet, EfficientNet) para extracción de características visuales, seguida de capas recurrentes (LSTM/GRU) o transformadores para modelar la secuencia temporal.
- **Salida:** predicción de las acciones humanas correspondientes a cada frame.

### 4.2. Agente de Clonación de Comportamiento
- **Entrada:** secuencia de imágenes y observaciones.
- **Red neuronal:** arquitectura similar al IDM, pero optimizada para inferir la acción óptima dada la observación.
- **Cabeza de política:** puede ser regresión (para acciones continuas) o clasificación (para acciones discretas).

### 4.3. Consideraciones de Implementación
- Uso de normalización de entradas (imágenes y sensores).
- Sincronización temporal entre video y acciones.
- Manejo de datos faltantes o inconsistentes.


## 5. Preprocesamiento y Pipeline de Datos
### 5.1. Extracción de Frames
- Conversión de videos a secuencias de imágenes a la frecuencia deseada (por ejemplo, 10-30 FPS).

### 5.2. Sincronización y Alineación
- Alinear cada frame con la acción correspondiente usando marcas de tiempo o interpolación.

### 5.3. Normalización y Augmentación
- Normalización de imágenes (media, desviación estándar).
- Augmentación de datos: recortes, rotaciones, cambios de brillo/contraste, ruido, etc.
- Normalización de señales de sensores.

### 5.4. División de Datos
- Separar conjuntos de entrenamiento, validación y prueba, asegurando diversidad de pistas y condiciones.


## 6. Entrenamiento y Evaluación
### 6.1. Entrenamiento del IDM
- Optimización con función de pérdida adecuada (MSE para acciones continuas, cross-entropy para discretas).
- Early stopping y regularización para evitar sobreajuste.

### 6.2. Etiquetado Masivo
- Uso del IDM para inferir acciones en videos no etiquetados y construir un gran dataset sintético.

### 6.3. Entrenamiento del Agente
- Clonación de comportamiento usando el dataset generado.
- Técnicas de validación cruzada y ajuste de hiperparámetros.

### 6.4. Evaluación
- Métricas: error medio de acción, éxito en completar vueltas, tiempos de vuelta, robustez ante condiciones nuevas.
- Evaluación en simuladores y, si es posible, en hardware real o entornos de validación externos.

## 7. Extensiones y Futuro
- Integración con aprendizaje por refuerzo para refinar la política tras el preentrenamiento.
- Uso de sensores adicionales (LIDAR, radar, mapas de pista).
- Transferencia a diferentes juegos o simuladores.
- Aprendizaje multitarea (diferentes tipos de circuitos, condiciones climáticas, etc.).
- Generación de datos sintéticos para aumentar la diversidad.

## 8. Conclusión
El preentrenamiento de video para juegos de carreras permite aprovechar grandes volúmenes de datos no etiquetados para entrenar agentes de conducción, reduciendo la necesidad de datos manualmente anotados y abriendo nuevas posibilidades para el aprendizaje por observación en simuladores y videojuegos.

---

_Este documento es una adaptación conceptual del enfoque VPT de OpenAI, orientado a la aplicación en juegos de carreras de coches. Para detalles técnicos y experimentales, se recomienda consultar el paper original y adaptar la metodología a las particularidades del entorno de carreras seleccionado._

---

_Este documento es una adaptación conceptual del enfoque VPT de OpenAI, orientado a la aplicación en juegos de carreras de coches. Para detalles técnicos y experimentales, se recomienda consultar el paper original y adaptar la metodología a las particularidades del entorno de carreras seleccionado._

## 9. Ejemplos de Código y Diagramas

### 9.1. Ejemplo de Formato de Acción y Observación
```json
{
  "frame": 1234,
  "timestamp": 12.34,
  "observation": {
    "image": "frame_1234.png",
    "speed": 87.2,
    "track_position": 0.12,
    "gear": 3
  },
  "action": {
    "steering": -0.15,
    "throttle": 0.8,
    "brake": 0.0,
    "gear_change": 0
  }
}
```

### 9.2. Ejemplo de Preprocesamiento de Imágenes en Python
```python
import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalización
    return img.astype(np.float32)
```

### 9.3. Ejemplo de Red Neuronal para el IDM (PyTorch)
```python
import torch
import torch.nn as nn

class IDMNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(3136, 256, batch_first=True)
        self.fc = nn.Linear(256, num_actions)

    def forward(self, x):
        # x: (batch, seq, C, H, W)
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)
        x, _ = self.lstm(x)
        return self.fc(x)
```

### 9.4. Diagrama Conceptual del Pipeline

```mermaid
flowchart TD
    A[Videos de carreras humanos] --> B[Extracción de frames y telemetría]
    B --> C[Entrenamiento IDM con datos etiquetados]
    A --> D[Etiquetado automático con IDM]
    D --> E[Dataset masivo (video, acción)]
    E --> F[Entrenamiento del agente por clonación de comportamiento]
    F --> G[Evaluación y despliegue]
```

### 9.5. Ejemplo de Entrenamiento de Clonación de Comportamiento
```python
import torch.optim as optim

def train_bc(agent, dataloader, epochs=10):
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for obs, action in dataloader:
            pred = agent(obs)
            loss = loss_fn(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```