import os
from flask import Blueprint, request, jsonify, render_template, session
import pulp  # Biblioteca para resolver problemas de optimización lineal
import pandas as pd  # Para manejar la solución en formato tabular
import openai  # Integración con OpenRouter para IA
from dotenv import load_dotenv  # Para cargar la API Key desde un archivo .env

# 📌 Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configuración de la API Key de OpenRouter
# Se recomienda no exponer la API Key en el código por razones de seguridad
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Si la API Key no está configurada correctamente, lanzar un error
# if not OPENROUTER_API_KEY:
#    raise ValueError("❌ ERROR: No se encontró la API Key de OpenRouter en el archivo .env")

# 🔴 ⚠ API Key expuesta (no recomendado en producción)
OPENROUTER_API_KEY = "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"

# Configuración del cliente OpenRouter para generar análisis de sensibilidad con IA
client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# 🔹 Crear un Blueprint de Flask para manejar las rutas del módulo de transporte
transporte_solver_web = Blueprint(
    "transporte_solver_web", __name__, template_folder="templates", static_folder="static"
)

# 📌 Ruta principal de la aplicación (Página de inicio)
@transporte_solver_web.route('/')
def index():
    return render_template('inicio.html')

# 📌 Endpoint para resolver el problema de transporte
@transporte_solver_web.route('/solve', methods=['POST'])
def solve_transportation():
    try:
        # Obtener los datos del problema desde la solicitud JSON
        data = request.json
        costos = data['costos']
        oferta = data['oferta']
        demanda = data['demanda']
        
        total_oferta = sum(oferta)
        total_demanda = sum(demanda)

        # Variables para manejar la información sobre el equilibrio del problema
        mensaje_equilibrado = "El problema estaba equilibrado."
        equilibrado = True  
        tabla_equilibrada = None  

        # 📌 Verificar si el problema está equilibrado (oferta = demanda)
        if total_oferta < total_demanda:
            # Si la oferta es menor, agregar un proveedor ficticio con costo 0
            costos.append([0] * len(demanda))  
            oferta.append(total_demanda - total_oferta)  
            mensaje_equilibrado = "El problema no estaba equilibrado. Se agregó un proveedor ficticio."
            equilibrado = False
        elif total_oferta > total_demanda:
            # Si la demanda es menor, agregar un destino ficticio con costo 0
            for fila in costos:
                fila.append(0)  
            demanda.append(total_oferta - total_demanda)
            mensaje_equilibrado = "El problema no estaba equilibrado. Se agregó un destino ficticio."
            equilibrado = False

        # Si después de los ajustes aún no se equilibró, devolver un error
        if sum(oferta) != sum(demanda):
            return jsonify({
                "status": "error",
                "message": "El problema no pudo equilibrarse correctamente. Revisa los datos."
            })

        # Guardar la tabla equilibrada si se realizaron ajustes
        if not equilibrado:
            tabla_equilibrada = {
                "costos": [[costos[i][j] for j in range(len(demanda))] for i in range(len(oferta))],
                "oferta": oferta,
                "demanda": demanda
            }

        # 📌 Crear el modelo de optimización lineal con PuLP
        n = len(oferta)  # Número de proveedores
        m = len(demanda)  # Número de destinos

        prob = pulp.LpProblem("Problema de Transporte", pulp.LpMinimize)

        # Crear variables de decisión: x_ij (cantidad transportada de i a j)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous') for j in range(m)] for i in range(n)]

        # 📌 Definir la función objetivo (minimizar costos de transporte)
        prob += pulp.lpSum(costos[i][j] * x[i][j] for i in range(n) for j in range(m))

        # 📌 Restricciones de oferta (cada proveedor solo puede enviar su cantidad disponible)
        for i in range(n):
            prob += pulp.lpSum(x[i][j] for j in range(m)) == oferta[i]

        # 📌 Restricciones de demanda (cada destino debe recibir su cantidad exacta)
        for j in range(m):
            prob += pulp.lpSum(x[i][j] for i in range(n)) == demanda[j]

        # Resolver el problema de optimización
        prob.solve()

        # 📌 Obtener los resultados
        resultado = [[pulp.value(x[i][j]) for j in range(m)] for i in range(n)]
        df = pd.DataFrame(resultado, columns=[f"Destino {j + 1}" for j in range(m)],
                          index=[f"Proveedor {i + 1}" for i in range(n)])

        costo_total = pulp.value(prob.objective)  # Obtener el costo mínimo encontrado

        # Guardar resultados en sesión para futuras consultas
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

# 📌 Ruta para mostrar el análisis de sensibilidad
@transporte_solver_web.route('/mostrar_analisis')
def mostrar_analisis():
    solucion = session.get('solucion', {})
    costo_total = session.get('costo_total', 0)
    return render_template('analisis_sensibilidad.html', solucion=solucion, costo_total=costo_total)

# 📌 Ruta para procesar el análisis de sensibilidad usando IA
@transporte_solver_web.route('/procesar_analisis', methods=['POST'])
def procesar_analisis():
    try:
        data = request.json
        contexto = data.get('contexto', '').strip()
        solucion = data.get('solucion', None)
        costo_total = data.get('costo_total', None)

        if not solucion or not costo_total:
            return jsonify({"status": "error", "analisis": "❌ Error: Datos no válidos recibidos."})

        analisis = generar_analisis_de_sensibilidad(contexto, solucion, costo_total)

        return jsonify({"status": "success", "analisis": analisis})

    except Exception as e:
        return jsonify({"status": "error", "analisis": f"❌ Error en el análisis: {str(e)}"})

# 📌 Función para generar el análisis de sensibilidad usando IA con OpenRouter
def generar_analisis_de_sensibilidad(contexto, solucion, costo_total):
    resumen_solucion = []
    for destino, proveedores in solucion.items():
        for proveedor, valor in proveedores.items():
            resumen_solucion.append(f"Desde {proveedor} hasta {destino}: {valor} unidades.")

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
    """

    try:
        respuesta = client.chat.completions.create(
            model="openai/gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return respuesta.choices[0].message.content

    except Exception as e:
        return f"Error al generar el análisis con OpenRouter: {str(e)}"
