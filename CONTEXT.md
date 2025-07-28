# Contexto para la adaptación a juegos de carreras de coches

Este repositorio fue originalmente diseñado para el preentrenamiento de agentes en Minecraft usando videos y clonación de comportamiento. Para adaptarlo a juegos de carreras de coches, se deben considerar los siguientes puntos clave:

## Cambios necesarios

1. **Datos de entrada y formato**
   - Los videos de entrada deben ser grabaciones de partidas de juegos de carreras (por ejemplo, Gran Turismo, Forza, Mario Kart, etc.).
   - Los archivos de acciones asociadas deben reflejar controles típicos de carreras: dirección (steering), acelerador (throttle), freno (brake), marcha (gear), velocidad, posición en pista, etc.
   - El formato de los archivos `.jsonl` debe adaptarse para registrar estas acciones y observaciones relevantes.

2. **Espacio de acciones y observaciones**
   - Redefinir el espacio de acciones en el código (por ejemplo, en `agent.py` y `lib/actions.py`) para incluir:
     - Dirección (valor continuo o discretizado)
     - Acelerador (valor continuo o discretizado)
     - Freno (valor continuo o discretizado)
     - Cambio de marcha (opcional)
     - Otros controles específicos del juego de carreras
   - Las observaciones pueden incluir la imagen de la cámara frontal, velocidad, posición en pista, etc.

3. **Modelos y scripts**
   - Modificar los modelos de política y dinámica inversa para que trabajen con el nuevo espacio de acciones y observaciones.
   - Ajustar los scripts de entrenamiento y evaluación (`run_agent.py`, `run_inverse_dynamics_model.py`, `behavioural_cloning.py`) para cargar y procesar datos de carreras.

4. **Preprocesamiento y utilidades**
   - Adaptar el preprocesamiento de imágenes y datos para el formato y resolución de los juegos de carreras.
   - Actualizar utilidades de carga de datos (`data_loader.py`) para manejar los nuevos archivos y estructuras.

5. **Documentación**
   - Cambiar la documentación (README.md) para reflejar el nuevo dominio, ejemplos de uso y datasets de carreras.

## Consideraciones adicionales

- Es posible que se requiera ajustar la arquitectura de los modelos para aprovechar mejor las características de los juegos de carreras (por ejemplo, secuencias temporales más largas, sensores adicionales, etc.).
- Se recomienda definir claramente el formato de los datos de entrada y salida antes de modificar el código.
- Si se usan entornos de simulación (Gym, CARLA, TORCS, etc.), adaptar las interfaces de entorno en consecuencia.
