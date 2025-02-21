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
#openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

#if not openrouter_api_key:
    #raise ValueError("❌ ERROR: No se encontró la clave API de OpenRouter. Verifica tu archivo .env.")
openrouter_api_key = "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"
# Configurar cliente OpenRouter
client = openai.OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Crear Blueprint para chatbot
chatbot = Blueprint("chatbot", __name__, template_folder="templates", static_folder="static")

# Expresiones regulares para capturar números en la tabla
NUMERIC_PATTERN = r"\d+\.\d+|\d+"

# Palabras clave relacionadas con Programación Lineal
KEYWORDS = [
    "solución óptima", "valor óptimo", "variable", "reduced cost", "slack",
    "surplus", "dual price", "maximizar", "restricción", "holgura", "costo",
    "artificial", "base", "coeficiente", "análisis de sensibilidad"
]

def extraer_texto_desde_imagen(image_file):
    """Extrae texto de una imagen utilizando OCR."""
    try:
        image = Image.open(image_file)
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        extracted_text = pytesseract.image_to_string(thresh)
        
        texto_filtrado = []
        for line in extracted_text.split("\n"):
            if any(kw in line.lower() for kw in KEYWORDS) or re.search(NUMERIC_PATTERN, line):
                texto_filtrado.append(line.strip())
        
        texto_final = "\n".join(texto_filtrado)
        return texto_final if texto_final.strip() else None
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
        session["historial"] = []
    
    full_prompt = ""

    # 📌 Datos desde solucion.html
    if request.method == "POST" and "solucion" in request.form:
        try:
            solucion = json.loads(request.form["solucion"])
            variables_holgura_exceso = json.loads(request.form["variables_holgura_exceso"])
            nombres_variables = json.loads(request.form.get("nombres_variables", "{}"))  # Cargar nombres personalizados
            
            full_prompt = "*📊 Análisis de Sensibilidad*\n\n"
            full_prompt += "*Solución Óptima:*\n"
            for key, val in solucion.items():
                nombre_variable = nombres_variables.get(key, key)
                full_prompt += f"{nombre_variable} = {val}\n"
            
            full_prompt += "\n*Variables de Holgura/Exceso:*\n"
            for key, val in variables_holgura_exceso.items():
                full_prompt += f"{key} = {val}\n"
            
            full_prompt += """
            🔍 *Instrucciones para el Análisis:*
            - Analiza el impacto de las restricciones en la solución óptima.
            - Identifica qué restricciones limitan más la producción.
            - Sugiere cómo modificar restricciones o coeficientes para maximizar beneficios.
            - Explica si aumentar recursos en una restricción mejoraría la solución.
            """
        except json.JSONDecodeError as e:
            print(f"❌ Error al procesar JSON: {e}")
            return render_template("chat.html", historial=session["historial"], bot_respuesta="❌ Error en los datos recibidos.")
    
    # 📌 Consulta manual del usuario
    if request.method == "POST" and "message" in request.form:
        user_input = request.form.get("message", "").strip()
        image_file = request.files.get("image")
        
        extracted_text = ""
        if image_file:
            texto_extraido = extraer_texto_desde_imagen(image_file)
            extracted_text = f"Datos extraídos:\n{texto_extraido}" if texto_extraido else "⚠️ No se pudo extraer texto válido de la imagen."
        
        full_prompt = f"{user_input}\n{extracted_text}".strip()
    
    if not full_prompt:
        return render_template("chat.html", historial=session["historial"], bot_respuesta="Por favor, ingrese un mensaje o suba una imagen válida.")
    
    session["historial"].append({"user": full_prompt, "bot": "Procesando análisis..."})
    session.modified = True
    
    # 📌 Configuración del modelo
    mensajes_previos = [
        {"role": "system", "content": """
        Eres un experto en Programación Lineal y Análisis de Sensibilidad. 
        Analiza los datos proporcionados y genera recomendaciones detalladas sobre la optimización del sistema productivo. 
        Explica el impacto de las restricciones y coeficientes en la solución óptima.
        """}
    ]
    mensajes_previos.append({"role": "user", "content": full_prompt})
    
    try:
        print("📨 Enviando solicitud a OpenRouter...")
        respuesta = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=800,
            messages=mensajes_previos
        )
        
        bot_respuesta = respuesta.choices[0].message.content if respuesta.choices else "⚠️ No se recibió respuesta del modelo."
        session["historial"][-1]["bot"] = bot_respuesta
        session.modified = True  
    except Exception as e:
        print(f"❌ Error al procesar la solicitud: {str(e)}")
        return render_template("chat.html", historial=session["historial"], bot_respuesta=f"❌ Error en la solicitud: {str(e)}")
    
    return render_template("chat.html", historial=session["historial"])

def chat_analyze(problem_statement, solution, cost):
    """
    Envía la solución y el contexto al chatbot para análisis de sensibilidad.
    """
    analysis_prompt = f"""
    {problem_statement}
    La solución obtenida es {solution} con un costo de {cost}.
    ¿Puedes realizar un análisis de sensibilidad sobre esta solución?
    """

    response = chatbot_response(analysis_prompt)
    return response

def chatbot_response(prompt):
    """
    Llama a OpenAI para generar la respuesta.
    """
    mensajes_previos = [
        {"role": "system", "content": """
        Eres un experto en redes y transporte. Evalúa la estabilidad de la solución ante cambios en los costos o capacidades.
        """},
        {"role": "user", "content": prompt}
    ]

    try:
        respuesta = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
            messages=mensajes_previos
        )
        return respuesta.choices[0].message.content if respuesta.choices else "⚠️ No se recibió respuesta del modelo."
    except Exception as e:
        return f"❌ Error en la solicitud: {str(e)}"
