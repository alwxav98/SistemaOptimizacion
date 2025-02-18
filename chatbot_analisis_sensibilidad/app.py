from flask import Blueprint, render_template, request, redirect, url_for, session
import openai
import os
import pytesseract
import re
import numpy as np
import json
from PIL import Image
from dotenv import load_dotenv
import cv2

# Cargar variables de entorno
load_dotenv()

# API Key de OpenRouter
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("❌ ERROR: No se encontró la clave API de OpenRouter. Verifica tu archivo .env.")

# Configurar cliente OpenRouter
client = openai.OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Crear Blueprint para chatbot
chatbot = Blueprint("chatbot", __name__, template_folder="templates", static_folder="static")

# Expresiones regulares para capturar números en la tabla
NUMERIC_PATTERN = r"\d+\.\d+|\d+"

# Palabras clave relacionadas exclusivamente con Programación Lineal
KEYWORDS = [
    "solución óptima", "valor óptimo", "variable", "reduced cost", "slack",
    "surplus", "dual price", "maximizar", "restricción", "holgura", "costo",
    "artificial", "base", "coeficiente", "análisis de sensibilidad"
]

def es_pregunta_valida(texto):
    """Verifica si el texto contiene términos clave o estructuras numéricas relevantes."""
    texto = texto.lower()
    coincidencias = sum(1 for kw in KEYWORDS if kw in texto)
    numeros_encontrados = re.findall(NUMERIC_PATTERN, texto)
    return coincidencias >= 2 or len(numeros_encontrados) >= 3

def extraer_texto_desde_imagen(image_file):
    """Extrae texto de una imagen utilizando OCR."""
    try:
        image = Image.open(image_file)
        image = np.array(image)

        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar umbral para mejorar la detección de caracteres
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Extraer texto con OCR
        extracted_text = pytesseract.image_to_string(thresh)

        # Filtrar solo valores numéricos y palabras clave importantes
        texto_filtrado = []
        for line in extracted_text.split("\n"):
            if any(kw in line.lower() for kw in KEYWORDS) or re.search(NUMERIC_PATTERN, line):
                texto_filtrado.append(line.strip())

        texto_final = "\n".join(texto_filtrado)

        if not texto_final.strip():
            return None  # No se detectó información útil

        print(f"📸 Texto extraído:\n{texto_final}")
        return texto_final

    except Exception as e:
        print(f"❌ Error procesando la imagen: {str(e)}")
        return None

@chatbot.route("/")
def index():
    return redirect(url_for("chatbot.chat"))

@chatbot.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        session.pop("historial", None)  # 🔥 Borra la sesión al entrar nuevamente

    if "historial" not in session:
        session["historial"] = []  # Si no existe, inicializar historial vacío

    full_prompt = ""  # Inicializar la variable que contendrá la consulta del usuario

    # 🔹 **Si los datos vienen desde `solucion.html`**
    if request.method == "POST" and "solucion" in request.form:
        try:
            solucion = json.loads(request.form["solucion"])  # ✅ Convertir JSON a diccionario
            variables_holgura_exceso = json.loads(request.form["variables_holgura_exceso"])

            # 🔹 **Construir mensaje para el chatbot**
            full_prompt = "**📊 Análisis de Sensibilidad**\n\n"
            full_prompt += f"**Solución Óptima:**\n"
            for key, val in solucion.items():
                full_prompt += f"{key} = {val}\n"

            full_prompt += "\n**Variables de Holgura/Exceso:**\n"
            for key, val in variables_holgura_exceso.items():
                full_prompt += f"{key} = {val}\n"

            full_prompt += "\n🔍 Analiza cómo cambios en los coeficientes de la función objetivo y restricciones afectan la solución."

        except json.JSONDecodeError as e:
            print(f"❌ Error al procesar JSON: {e}")
            return render_template("chat.html", historial=session["historial"], bot_respuesta="❌ Error en los datos recibidos.")

    # 🔹 **Si el usuario hace una consulta directa en el chat**
    if request.method == "POST" and "message" in request.form:
        user_input = request.form.get("message", "").strip()
        image_file = request.files.get("image")

        extracted_text = ""
        if image_file:
            texto_extraido = extraer_texto_desde_imagen(image_file)
            extracted_text = f"Datos extraídos:\n{texto_extraido}" if texto_extraido else "⚠️ No se pudo extraer texto válido de la imagen."

        full_prompt = f"{user_input}\n{extracted_text}".strip()

    # 🔹 **Si no hay consulta válida, regresar**
    if not full_prompt:
        return render_template("chat.html", historial=session["historial"], bot_respuesta="Por favor, ingrese un mensaje o suba una imagen válida.")

    # 📌 **Agregar contexto y consulta al historial**
    session["historial"].append({"user": full_prompt, "bot": "Procesando análisis..."})
    session.modified = True  # Asegurar que Flask guarde cambios en sesión

    # 🔹 **Enviar datos a OpenRouter para análisis**
    mensajes_previos = [{"role": "system", "content": "Eres un experto en Programación Lineal y Análisis de Sensibilidad."}]
    mensajes_previos.append({"role": "user", "content": full_prompt})

    try:
        print("📨 Enviando solicitud a OpenRouter...")  # DEPURACIÓN
        respuesta = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=800,
            messages=mensajes_previos
        )
        print("✅ Respuesta recibida.")  # DEPURACIÓN

        if not respuesta or not respuesta.choices or not respuesta.choices[0].message:
            bot_respuesta = "⚠️ No se recibió respuesta del modelo. Intente de nuevo."
            print("⚠️ Error: No se recibió respuesta del modelo.")  # DEPURACIÓN
        else:
            bot_respuesta = respuesta.choices[0].message.content
            print(f"🤖 Respuesta generada:\n{bot_respuesta}")  # DEPURACIÓN

        session["historial"][-1]["bot"] = bot_respuesta  # ✅ Reemplaza "Procesando análisis..."
        session.modified = True  

    except Exception as e:
        print(f"❌ Error al procesar la solicitud: {str(e)}")
        return render_template("chat.html", historial=session["historial"], bot_respuesta=f"❌ Error en la solicitud: {str(e)}")

    return render_template("chat.html", historial=session["historial"])
