


# Video-Pre-Training para Juegos de Carreras de Coches
Preentrenamiento de Video (VPT): Aprendiendo a conducir observando videos de carreras sin etiquetar

> :page_facing_up: [Paper original VPT](https://cdn.openai.com/vpt/Paper.pdf) \
  :mega: [Blog OpenAI](https://openai.com/blog/vpt)


# Ejecución de modelos de agente en juegos de carreras

Instala los prerrequisitos para el entorno de simulación de carreras que vayas a utilizar (por ejemplo, [CARLA](https://carla.org/), [TORCS](https://sourceforge.net/projects/torcs/), [Gym](https://www.gymlibrary.dev/), etc.).
Luego instala los requisitos del repositorio con:

```
pip install -r requirements.txt
```

> ⚠️ Nota: Asegúrate de que la versión de PyTorch y las dependencias sean compatibles con tu entorno y hardware.

Para ejecutar el código, usa:

```
python run_agent.py --model [ruta al archivo .model] --weights [ruta al archivo .weight] --env [nombre_del_entorno]
```

Después de cargar, deberías ver una ventana o consola con el agente conduciendo en el simulador de carreras seleccionado.


# Modelos y Pesos de Agente
Incluye aquí los enlaces a los modelos y pesos preentrenados para juegos de carreras (puedes añadir tus propios modelos o enlaces a modelos públicos de referencia):

* [:arrow_down: Modelo base de carreras]([enlace_a_modelo])
* [:arrow_down: Pesos base de carreras]([enlace_a_pesos])

### Ejemplo de formato de datos
Cada episodio debe estar compuesto por un video (o secuencia de imágenes) y un archivo `.jsonl` con las acciones y observaciones:

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

# Entrenamiento y Fine-tuning

Para realizar fine-tuning de un modelo en tus propios datos de carreras:

```
python behavioural_cloning.py --data-dir data --in-model modelo-carreras.model --in-weights modelo-carreras.weights --out-weights finetuned-carreras.weights
```

Puedes modificar los hiperparámetros y la arquitectura en la parte superior de `behavioural_cloning.py`.

# Consideraciones y Adaptaciones

- Asegúrate de adaptar el espacio de acciones y observaciones en `agent.py` y `lib/actions.py` para reflejar los controles de un coche de carreras (dirección, acelerador, freno, marcha, etc.).
- El preprocesamiento de imágenes y señales debe ajustarse a la resolución y sensores del simulador.
- Si usas un entorno diferente a los soportados originalmente, adapta las interfaces de entorno y carga de datos.
- Consulta el archivo `CONTEXT.md` para una guía detallada de los cambios necesarios.

# Créditos y Referencias
Este repositorio es una adaptación del enfoque VPT de OpenAI, orientado a juegos de carreras de coches. Consulta el [paper original](https://cdn.openai.com/vpt/Paper.pdf) para detalles teóricos y experimentales.


# Ejecución de modelos de agente

Instala los prerrequisitos para [MineRL](https://minerl.readthedocs.io/en/latest/tutorials/index.html).
Luego instala los requisitos con:

```
pip install git+https://github.com/minerllabs/minerl
pip install -r requirements.txt
```

> ⚠️ Nota: Por razones de reproducibilidad, la versión de PyTorch está fijada en `torch==1.9.0`, la cual es incompatible con Python 3.10 o superior. Si usas Python 3.10 o superior, instala una [versión más reciente de PyTorch](https://pytorch.org/get-started/locally/) (usualmente, `pip install torch`). Sin embargo, ten en cuenta que esto *podría* cambiar sutilmente el comportamiento del modelo (por ejemplo, seguirá funcionando pero puede que no alcance el rendimiento reportado).

Para ejecutar el código, usa

```
python run_agent.py --model [ruta al archivo .model] --weights [ruta al archivo .weight]
```

Después de cargar, deberías ver una ventana con el agente jugando Minecraft.



# Zoológico de Modelos de Agente
A continuación se encuentran los archivos de modelo y pesos para varios modelos de Minecraft preentrenados.
Los archivos de modelo 1x, 2x y 3x corresponden a su respectivo ancho de pesos.

* [:arrow_down: Modelo 1x](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model)
* [:arrow_down: Modelo 2x](https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model)
* [:arrow_down: Modelo 3x](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.model)

These models are trained on video demonstrations of humans playing Minecraft
using behavioral cloning (BC) and are more general than later models which
use reinforcement learning (RL) to further optimize the policy.
Foundational models are trained across all videos in a single training run
while house and early game models refine their respective size foundational
model further using either the housebuilding contractor data or early game video
sub-set. See the paper linked above for more details.

#### Foundational Model :chart_with_upwards_trend:
  * [:arrow_down: 1x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights)
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-2x.weights)
  * [:arrow_down: 3x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.weights)

#### Fine-Tuned from House :chart_with_upwards_trend:
  * [:arrow_down: 3x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-house-3x.weights)

#### Fine-Tuned from Early Game :chart_with_upwards_trend:
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-early-game-2x.weights)
  * [:arrow_down: 3x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-early-game-3x.weights)

### Models With Environment Interactions
These models further refine the above demonstration based models with a reward
function targeted at obtaining diamond pickaxes. While less general then the behavioral
cloning models, these models have the benefit of interacting with the environment
using a reward function and excel at progressing through the tech tree quickly.
See the paper for more information
on how they were trained and the exact reward schedule.

#### RL from Foundation :chart_with_upwards_trend:
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-foundation-2x.weights)

#### RL from House :chart_with_upwards_trend:
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-house-2x.weights)

#### RL from Early Game :chart_with_upwards_trend:
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.weights)

# Running Inverse Dynamics Model (IDM)

IDM aims to predict what actions player is taking in a video recording.

Setup:
* Install requirements: `pip install -r requirements.txt`
* Download the IDM model [.model :arrow_down:](https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model) and [.weight :arrow_down:](https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights) files
* For demonstration purposes, you can use the contractor recordings shared below to. For this demo we use
  [this .mp4](https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4)
  and [this associated actions file (.jsonl)](https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl).

To run the model with above files placed in the root directory of this code:
```
python run_inverse_dynamics_model.py --weights 4x_idm.weights --model 4x_idm.model --video-path cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4 --jsonl-path cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl
```

A window should pop up which shows the video frame-by-frame, showing the predicted and true (recorded) actions side-by-side on the left.

Note that `run_inverse_dynamics_model.py` is designed to be a demo of the IDM, not code to put it into practice.

# Using behavioural cloning to fine-tune the models

**Disclaimer:** This code is a rough demonstration only and not an exact recreation of what original VPT paper did (but it contains some preprocessing steps you want to be aware of)! As such, do not expect replicate the original experiments with this code. This code has been designed to be run-able on consumer hardware (e.g., 8GB of VRAM).

Setup:
* Install requirements: `pip install -r requirements.txt`
* Download `.weights` and `.model` file for model you want to fine-tune.
* Download contractor data (below) and place the `.mp4` and `.jsonl` files to the same directory (e.g., `data`). With default settings, you need at least 12 recordings.

```

You can then use `finetuned-1x.weights` when running the agent. You can change the training settings at the top of `behavioural_cloning.py`.

