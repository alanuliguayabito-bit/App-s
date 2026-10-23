# Analizador multilingüe de audio (prototipo)

Aplicación web experimental para transcribir voz y mostrar observaciones acústicas y elementos del contenido que requieren verificación.

## Archivos
- `app.py`: aplicación Flask y su interfaz web en un solo archivo.
- `requirements.txt`: dependencias.
- `Procfile`: comando de inicio.
- `render.yaml`: configuración de ejemplo para Render.

## Importante
- No detecta ni demuestra mentiras.
- La transcripción puede equivocarse.
- El audio se procesa temporalmente en el servidor y el archivo temporal se elimina al finalizar la solicitud, pero el proveedor de alojamiento puede tener sus propios registros.
- Antes de publicarla, completa una política de privacidad real y agrega controles contra abuso.
- El modelo de voz se descarga en la primera ejecución. Los límites de memoria/CPU del alojamiento gratuito pueden impedir que funcione o hacerlo lento.
