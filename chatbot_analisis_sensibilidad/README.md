

# Chatbot de Análisis de Sensibilidad - SistemaOptimizacion

## Descripción
El Chatbot de Análisis de Sensibilidad es un submódulo del sistema SistemaOptimizacion que permite realizar un análisis detallado de los resultados obtenidos en Programación Lineal. Su función principal es interpretar las soluciones óptimas y generar recomendaciones para maximizar beneficios o minimizar costos.

Este chatbot se encuentra integrado dentro del sistema general y permite interactuar con el usuario para:

🔍 Analizar la sensibilidad de soluciones óptimas.
📊 Evaluar el impacto de cambios en restricciones y coeficientes.
📷 Extraer datos desde imágenes con resultados de optimización.
🤖 Brindar respuestas en lenguaje natural con recomendaciones claras.
🚀 Integrarse con otros módulos del sistema (Programación Lineal, Transporte y Redes).


## Características
💬 Interfaz web interactiva para realizar consultas en tiempo real.
📈 Optimización avanzada basada en Programación Lineal.
🏗 Análisis detallado del impacto de restricciones y coeficientes.
🔥 Extracción automática de datos desde imágenes (OCR).
📂 Historial de Conversaciones para seguimiento de análisis previos.
🤖 Uso de OpenRouter GPT para respuestas inteligentes y contextualizadas.

📋 ## Requisitos
Este submódulo forma parte del repositorio principal SistemaOptimizacion, pero requiere instalar dependencias específicas para su correcto funcionamiento:

pip install flask openai numpy opencv-python pytesseract pillow python-dotenv

Además, es necesario contar con una clave API de OpenRouter.

Para configurarla, crea un archivo .env dentro de chatbot_analisis_sensibilidad/ y agrega lo siguiente:

OPENROUTER_API_KEY=tu_clave_api

## Instalación e Integración con el Sistema General
1️⃣ Clonar el repositorio principal

git clone https://github.com/alwxav98/SistemaOptimizacion.git
cd SistemaOptimizacion
2️⃣ Acceder al submódulo del chatbot

cd chatbot_analisis_sensibilidad
3️⃣ Ejecutar el chatbot

python app.py
4️⃣ Acceder a la interfaz
Abre tu navegador y dirígete a:

http://localhost:5000/chatbot
Si deseas regresar al menú principal del sistema, accede a:

http://localhost:5000/

## Estructura del Proyecto
El chatbot es un submódulo dentro del sistema SistemaOptimizacion y sigue la siguiente estructura:

SistemaOptimizacion/
│-- chatbot_analisis_sensibilidad/   # Submódulo del chatbot
│   │-- app.py          # Aplicación Flask del chatbot
│   │-- static/
│   │   └── style.css   # Estilos del chatbot
│   │-- templates/
│   │   ├── chat.html   # Interfaz del chatbot
│   │-- .env            # Variables de entorno (clave API)
│   │-- README.md       # Documentación del chatbot
│-- programacion_linear_solver/  # Otro submódulo
│-- transporte_solver_web/       # Otro submódulo
│-- network_solver/              # Otro submódulo
│-- main.py           # Menú principal del sistema
│-- requirements.txt  # Dependencias del proyecto
│-- README.md         # Documentación general del sistema


## API Endpoints
Método	Endpoint	Descripción
- GET	/chatbot	Página principal del chatbot
- POST	/chatbot/chat	Enviar consulta o imagen para análisis
El chatbot está diseñado para recibir los datos de la solución óptima generada en programacion_linear_solver y realizar un análisis detallado.

## Mejoras Futuras
📊 Visualización gráfica del análisis de sensibilidad.
🧠 Uso de modelos de Machine Learning para predicciones.
🔄 Integración más avanzada con otros módulos del sistema.
