from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import numpy as np
import pulp  # Biblioteca para resolver problemas de Programación Lineal
import openai  # Integración con OpenRouter para IA
import json
import os
from urllib.parse import quote
from dotenv import load_dotenv  # Para cargar variables de entorno desde un archivo .env

# 📌 Cargar variables de entorno desde el archivo .env
load_dotenv()

# 📌 Obtener la clave API de OpenRouter desde las variables de entorno (comentado para evitar exposición de la clave)
# openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# 📌 Manejo de error si no se encuentra la clave API
# if not openrouter_api_key:
#     raise ValueError("\u274c ERROR: No se encontró la clave API de OpenRouter. Verifica tu archivo .env.")

# ⚠ API Key expuesta (no recomendado en producción)
openrouter_api_key = "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"

# 📌 Configuración del cliente OpenRouter para la integración de análisis de sensibilidad con IA
client = openai.OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# 📌 Crear un Blueprint para el módulo de programación lineal en Flask
programacion_linear_solver = Blueprint("programacion_linear_solver", __name__, template_folder="templates", static_folder="static")

# 📌 Página principal donde el usuario introduce el contexto del problema
@programacion_linear_solver.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 📌 Captura de datos ingresados por el usuario
        contexto_problema = request.form['contexto_problema']  # Descripción del problema
        num_variables = int(request.form['num_variables'])  # Número de variables de decisión
        num_restricciones = int(request.form['num_restricciones'])  # Número de restricciones
        es_dualidad = request.form.get('es_dualidad', 'no') == 'si'  # Verificación de dualidad

        # 📌 Redirige al formulario para ingresar los coeficientes y restricciones
        return redirect(url_for('programacion_linear_solver.formulario',
                                contexto_problema=contexto_problema.replace('%20', ' ').replace('%0A', ''),
                                num_variables=num_variables, 
                                num_restricciones=num_restricciones, 
                                es_dualidad=es_dualidad))

    return render_template('index.html')

# 📌 Página donde el usuario ingresa coeficientes y restricciones del problema
@programacion_linear_solver.route('/formulario', methods=['GET', 'POST'])
def formulario():
    # 📌 Recuperar parámetros enviados en la URL
    contexto_problema = request.args.get('contexto_problema', 'No definido')
    num_variables = int(request.args.get('num_variables', 2))
    num_restricciones = int(request.args.get('num_restricciones', 2))
    es_dualidad = request.args.get('es_dualidad', 'False') == 'True'

    if request.method == 'POST':
        # 📌 Capturar datos ingresados por el usuario
        objetivo_tipo = request.form['objetivo_tipo']  # Maximizar o Minimizar
        coef_objetivo = [float(request.form[f'obj_{i}']) for i in range(num_variables)]  # Coeficientes de la función objetivo
        restricciones = []
        signos = []
        rhs = []

        # 📌 Captura de restricciones y valores de lado derecho
        for i in range(num_restricciones):
            restricciones.append([float(request.form[f'res_{i}_{j}']) for j in range(num_variables)])
            signos.append(request.form[f'signo_{i}'])  # <=, >= o =
            rhs.append(float(request.form[f'const_{i}']))  # Constante del lado derecho de la restricción

        # 📌 Resolver el problema usando el método Simplex
        solucion, iteraciones, variables_holgura_exceso = resolver_simplex(
            objetivo_tipo, coef_objetivo, restricciones, signos, rhs, es_dualidad
        )

        # 📌 Determinar el nombre de la función objetivo (Z para primal, W para dual)
        obj_name = 'W' if es_dualidad else 'Z'

        # 📌 Renderizar la página de solución con los resultados obtenidos
        return render_template('solucion.html', 
                               contexto_problema=contexto_problema,
                               solucion=solucion, 
                               iteraciones=iteraciones, 
                               variables_holgura_exceso=variables_holgura_exceso, 
                               es_dualidad=es_dualidad,
                               obj_name=obj_name)

    return render_template('formulario.html', contexto_problema=contexto_problema, 
                           num_variables=num_variables, num_restricciones=num_restricciones, es_dualidad=es_dualidad)

# 📌 Función para resolver el problema de Programación Lineal con el método Simplex
def resolver_simplex(objetivo_tipo, coef_objetivo, restricciones, signos, rhs, es_dualidad):
    # 📌 Crear modelo de optimización (Minimizar o Maximizar)
    prob = pulp.LpProblem("Problema PL", pulp.LpMinimize if objetivo_tipo == 'Minimizar' else pulp.LpMaximize)

    # 📌 Definir prefijo de variables según el tipo de problema (X para primal, Y para dual)
    var_prefix = 'Y' if es_dualidad else 'X'
    obj_name = 'W' if es_dualidad else 'Z'

    # 📌 Crear variables de decisión
    variables = [pulp.LpVariable(f'{var_prefix}{i+1}', lowBound=0) for i in range(len(coef_objetivo))]
    holgura_exceso_vars = {}

    # 📌 Agregar restricciones al modelo
    for i in range(len(restricciones)):
        holgura_exceso = pulp.LpVariable(f'S{i+1}', lowBound=0)
        holgura_exceso_vars[f'S{i+1}'] = holgura_exceso
        if signos[i] == '<=':
            prob += pulp.lpSum(restricciones[i][j] * variables[j] for j in range(len(variables))) + holgura_exceso == rhs[i]
        elif signos[i] == '>=':
            prob += pulp.lpSum(restricciones[i][j] * variables[j] for j in range(len(variables))) - holgura_exceso == rhs[i]
        else:
            prob += pulp.lpSum(restricciones[i][j] * variables[j] for j in range(len(variables))) == rhs[i]

    # 📌 Definir la función objetivo
    prob += pulp.lpSum(coef_objetivo[i] * variables[i] for i in range(len(coef_objetivo)))

    # 📌 Resolver el problema utilizando PuLP
    prob.solve()

    # 📌 Extraer valores óptimos de las variables
    solucion = {var.name: var.varValue for var in variables}
    solucion[obj_name] = pulp.value(prob.objective)

    # 📌 Extraer valores de variables de holgura y exceso
    variables_holgura_exceso = {var_name: var.varValue for var_name, var in holgura_exceso_vars.items()}

    iteraciones = []  # Se pueden almacenar iteraciones intermedias si es necesario

    return solucion, iteraciones, variables_holgura_exceso

# 📌 Endpoint para realizar análisis de sensibilidad con OpenRouter AI


@programacion_linear_solver.route("/analizar_sensibilidad_ajax", methods=["POST"])
def analizar_sensibilidad_ajax():
    """
    Procesa la solución óptima y el contexto del problema para realizar el análisis de sensibilidad con IA.
    """
    data = request.get_json()
    
    contexto_problema = data.get('contexto_problema', 'No especificado')
    solucion = data.get('solucion', {})
    variables_holgura_exceso = data.get('variables_holgura_exceso', {})
    
    # Determinar si el problema es dual por la presencia de W e Y
    es_dualidad = any(key.startswith("W") or key.startswith("Y") for key in solucion.keys())
    
    if es_dualidad:
        prompt = f"""
        Contexto del problema:
        {contexto_problema}

        Solución dual óptima:
        {json.dumps(solucion, indent=2)}

        Variables de holgura y exceso:
        {json.dumps(variables_holgura_exceso, indent=2)}

        Realiza el análisis de sensibilidad en función de los valores duales. 
        Explica cómo afectan los valores duales a los coeficientes de las restricciones y 
        proporciona recomendaciones sobre los cambios en los costos y recursos,
        relacionándolo con el contexto del problema.

        *Justificación matemática:*
        - Explica la relación entre los valores duales y los coeficientes de restricción.
        - Discute el impacto de cambios en los valores de los coeficientes del problema dual.
        - Relaciona los resultados con el contexto específico del problema.
        
        *Interpretación final:*
        - Proporciona un análisis en términos de interpretación de negocio.
        - Explica cómo estos valores pueden afectar la toma de decisiones considerando el contexto del problema.
        """
    else:
        prompt = f"""
        Contexto del problema:
        {contexto_problema}

        Solución óptima:
        {json.dumps(solucion, indent=2)}

        Variables de holgura y exceso:
        {json.dumps(variables_holgura_exceso, indent=2)}

        Con base en estos datos, proporciona el análisis en el siguiente formato, siempre relacionándolo con el contexto del problema:

        *Justificación matemática:*
        - Explica los cálculos realizados para obtener la solución óptima.
        - Muestra las restricciones activas e inactivas y su impacto.
        - Relaciona los resultados con el contexto del problema.
        
        *Interpretación final:*
        - Explica la solución en un lenguaje claro y conciso.
        - Relaciona los resultados con el contexto del problema y proporciona recomendaciones basadas en él.
        """

    try:
        print("📨 Enviando solicitud a OpenRouter...")
        respuesta = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
            messages=[{"role": "system", "content": "Eres un experto en optimización matemática y análisis de sensibilidad."},
                      {"role": "user", "content": prompt}]
        )

        resultado_ia = respuesta.choices[0].message.content if respuesta.choices else "⚠ No se recibió respuesta del modelo."
        return jsonify({"analisis": resultado_ia})
    
    except Exception as e:
        print(f"❌ Error en la solicitud: {str(e)}")
        return jsonify({"analisis": "❌ Error al generar el análisis de sensibilidad."})
