import os
from flask import Blueprint, request, jsonify, render_template, session
import pulp
import pandas as pd
import openai  # Integración con OpenRouter para IA
from dotenv import load_dotenv  # Para cargar la API Key desde .env

# 📌 Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener API Key de OpenRouter desde el entorno
#OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

#if not OPENROUTER_API_KEY:
    #raise ValueError("❌ ERROR: No se encontró la API Key de OpenRouter en el archivo .env")

# Configuracion cliente OpenRouter
OPENROUTER_API_KEY = "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"

client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Crear Blueprint
transporte_solver_web = Blueprint(
    "transporte_solver_web", __name__, template_folder="templates", static_folder="static"
)

@transporte_solver_web.route('/')
def index():
    return render_template('inicio.html')

@transporte_solver_web.route('/solve', methods=['POST'])
def solve_transportation():
    try:
        data = request.json
        costos = data['costos']
        oferta = data['oferta']
        demanda = data['demanda']
        
        total_oferta = sum(oferta)
        total_demanda = sum(demanda)

        mensaje_equilibrado = "El problema estaba equilibrado."
        equilibrado = True  
        tabla_equilibrada = None  

        if total_oferta < total_demanda:
            costos.append([0] * len(demanda))  
            oferta.append(total_demanda - total_oferta)  
            mensaje_equilibrado = "El problema no estaba equilibrado. Se agregó un proveedor ficticio."
            equilibrado = False
        elif total_oferta > total_demanda:
            for fila in costos:
                fila.append(0)  
            demanda.append(total_oferta - total_demanda)
            mensaje_equilibrado = "El problema no estaba equilibrado. Se agregó un destino ficticio."
            equilibrado = False

        if sum(oferta) != sum(demanda):
            return jsonify({
                "status": "error",
                "message": "El problema no pudo equilibrarse correctamente. Revisa los datos."
            })

        if not equilibrado:
            tabla_equilibrada = {
                "costos": [[costos[i][j] for j in range(len(demanda))] for i in range(len(oferta))],
                "oferta": oferta,
                "demanda": demanda
            }

        n = len(oferta)
        m = len(demanda)

        prob = pulp.LpProblem("Problema de Transporte", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous') for j in range(m)] for i in range(n)]

        prob += pulp.lpSum(costos[i][j] * x[i][j] for i in range(n) for j in range(m))

        for i in range(n):
            prob += pulp.lpSum(x[i][j] for j in range(m)) == oferta[i]

        for j in range(m):
            prob += pulp.lpSum(x[i][j] for i in range(n)) == demanda[j]

        prob.solve()

        resultado = [[pulp.value(x[i][j]) for j in range(m)] for i in range(n)]
        df = pd.DataFrame(resultado, columns=[f"Destino {j + 1}" for j in range(m)],
                          index=[f"Proveedor {i + 1}" for i in range(n)])

        costo_total = pulp.value(prob.objective)

        session['solucion'] = df.to_dict()
        session['costo_total'] = costo_total

        return jsonify({
            "status": "success",
            "mensaje": mensaje_equilibrado,
            "solucion": df.to_dict(),
            "tabla_equilibrada": tabla_equilibrada,
            "costo_total": costo_total,
            "equilibrado": equilibrado
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@transporte_solver_web.route('/mostrar_analisis')
def mostrar_analisis():
    solucion = session.get('solucion', {})
    costo_total = session.get('costo_total', 0)
    return render_template('analisis_sensibilidad.html', solucion=solucion, costo_total=costo_total)

@transporte_solver_web.route('/procesar_analisis', methods=['POST'])
def procesar_analisis():
    try:
        data = request.json
        print("📥 Datos recibidos en el backend:", data)  # Debug

        contexto = data.get('contexto', '').strip()
        solucion = data.get('solucion', None)
        costo_total = data.get('costo_total', None)

        if not solucion or not costo_total:
            return jsonify({"status": "error", "analisis": "❌ Error: Datos no válidos recibidos."})

        analisis = generar_analisis_de_sensibilidad(contexto, solucion, costo_total)

        return jsonify({"status": "success", "analisis": analisis})

    except Exception as e:
        print("❌ Error en el análisis de sensibilidad:", str(e))
        return jsonify({"status": "error", "analisis": f"❌ Error en el análisis: {str(e)}"})


def generar_analisis_de_sensibilidad(contexto, solucion, costo_total):
    """
    Genera un análisis de sensibilidad utilizando IA con OpenRouter.
    """
    
    # Formatear la solución para una mejor interpretación
    resumen_solucion = []
    for destino, proveedores in solucion.items():
        for proveedor, valor in proveedores.items():
            resumen_solucion.append(f"Desde {proveedor} hasta {destino}: {valor} unidades.")

    # Construcción del prompt para la IA
    prompt = f"""
    Se te proporciona un problema de transporte con los siguientes detalles:

    **Contexto del problema:**
    {contexto}

    **Resultados obtenidos:**
    - Costo total del transporte: {costo_total}
    - Distribución óptima del transporte:
    {chr(10).join(resumen_solucion)}

    **Tareas para el análisis de sensibilidad:**
    1. Identifica restricciones activas y con holgura.
    2. Evalúa qué rutas son críticas y cuáles podrían ajustarse.
    3. Analiza cómo cambios en oferta/demanda afectarían el resultado.
    4. Brinda recomendaciones específicas para mejorar costos o eficiencia.

    **Importante:** El análisis debe basarse únicamente en los resultados obtenidos y el contexto del problema, sin información predefinida.
    """

    try:
        print("📨 Enviando solicitud a OpenRouter...")  # Debugging
        respuesta = client.chat.completions.create(
            model="openai/gpt-4-turbo",  # O el modelo que uses en OpenRouter
            temperature=0.7,
            max_tokens=800,
            messages=[{"role": "system", "content": "Eres un experto en optimización y análisis de sensibilidad."},
                      {"role": "user", "content": prompt}]
        )

        analisis = respuesta.choices[0].message.content

    except Exception as e:
        analisis = f"Error al generar el análisis con OpenRouter: {str(e)}"

    return analisis
