from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import numpy as np
import pulp
import openai
import json
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Cargar API Key desde .env
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("❌ ERROR: No se encontró la clave API de OpenRouter. Verifica tu archivo .env.")

# Configurar cliente OpenRouter
client = openai.OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Crear Blueprint
programacion_linear_solver = Blueprint("programacion_linear_solver", __name__, template_folder="templates", static_folder="static")

from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import numpy as np
import pulp
import openai
import json
import os
from urllib.parse import quote
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Cargar API Key desde .env
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("\u274c ERROR: No se encontró la clave API de OpenRouter. Verifica tu archivo .env.")

# Configurar cliente OpenRouter
client = openai.OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Crear Blueprint
programacion_linear_solver = Blueprint("programacion_linear_solver", __name__, template_folder="templates", static_folder="static")

from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import numpy as np
import pulp
import openai
import json
import os
from urllib.parse import quote
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Cargar API Key desde .env
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("\u274c ERROR: No se encontró la clave API de OpenRouter. Verifica tu archivo .env.")

# Configurar cliente OpenRouter
client = openai.OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Crear Blueprint
programacion_linear_solver = Blueprint("programacion_linear_solver", __name__, template_folder="templates", static_folder="static")

@programacion_linear_solver.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        contexto_problema = request.form['contexto_problema']  # Captura el contexto
        num_variables = int(request.form['num_variables'])
        num_restricciones = int(request.form['num_restricciones'])
        es_dualidad = request.form.get('es_dualidad', 'no') == 'si'

        # Pasa el contexto como query parameter para evitar problemas de URL
        return redirect(url_for('programacion_linear_solver.formulario',
                                contexto_problema=contexto_problema.replace('%20', ' ').replace('%0A', ''),
                                num_variables=num_variables, 
                                num_restricciones=num_restricciones, 
                                es_dualidad=es_dualidad))

    return render_template('index.html')

@programacion_linear_solver.route('/formulario', methods=['GET', 'POST'])
def formulario():
    contexto_problema = request.args.get('contexto_problema', 'No definido')
    num_variables = int(request.args.get('num_variables', 2))
    num_restricciones = int(request.args.get('num_restricciones', 2))
    es_dualidad = request.args.get('es_dualidad', 'False') == 'True'

    if request.method == 'POST':
        objetivo_tipo = request.form['objetivo_tipo']
        coef_objetivo = [float(request.form[f'obj_{i}']) for i in range(num_variables)]
        restricciones = []
        signos = []
        rhs = []

        for i in range(num_restricciones):
            restricciones.append([float(request.form[f'res_{i}_{j}']) for j in range(num_variables)])
            signos.append(request.form[f'signo_{i}'])
            rhs.append(float(request.form[f'const_{i}']))

        solucion, iteraciones, variables_holgura_exceso = resolver_simplex(
            objetivo_tipo, coef_objetivo, restricciones, signos, rhs, es_dualidad
        )

        obj_name = 'W' if es_dualidad else 'Z'
        return render_template('solucion.html', 
                               contexto_problema=contexto_problema,
                               solucion=solucion, 
                               iteraciones=iteraciones, 
                               variables_holgura_exceso=variables_holgura_exceso, 
                               es_dualidad=es_dualidad,
                               obj_name=obj_name)

    return render_template('formulario.html', contexto_problema=contexto_problema, 
                           num_variables=num_variables, num_restricciones=num_restricciones, es_dualidad=es_dualidad)




def resolver_simplex(objetivo_tipo, coef_objetivo, restricciones, signos, rhs, es_dualidad):
    prob = pulp.LpProblem("Problema PL", pulp.LpMinimize if objetivo_tipo == 'Minimizar' else pulp.LpMaximize)
    
    var_prefix = 'Y' if es_dualidad else 'X'
    obj_name = 'W' if es_dualidad else 'Z'
    variables = [pulp.LpVariable(f'{var_prefix}{i+1}', lowBound=0) for i in range(len(coef_objetivo))]
    holgura_exceso_vars = {}

    for i in range(len(restricciones)):
        holgura_exceso = pulp.LpVariable(f'S{i+1}', lowBound=0)
        holgura_exceso_vars[f'S{i+1}'] = holgura_exceso
        if signos[i] == '<=':
            prob += pulp.lpSum(restricciones[i][j] * variables[j] for j in range(len(variables))) + holgura_exceso == rhs[i]
        elif signos[i] == '>=':
            prob += pulp.lpSum(restricciones[i][j] * variables[j] for j in range(len(variables))) - holgura_exceso == rhs[i]
        else:
            prob += pulp.lpSum(restricciones[i][j] * variables[j] for j in range(len(variables))) == rhs[i]

    prob += pulp.lpSum(coef_objetivo[i] * variables[i] for i in range(len(coef_objetivo)))

    prob.solve()
    solucion = {var.name: var.varValue for var in variables}
    solucion[obj_name] = pulp.value(prob.objective)

    variables_holgura_exceso = {var_name: var.varValue for var_name, var in holgura_exceso_vars.items()}
    
    iteraciones = []  # Aquí se pueden agregar pasos intermedios si se requiere

    return solucion, iteraciones, variables_holgura_exceso

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

        **Justificación matemática:**
        - Explica la relación entre los valores duales y los coeficientes de restricción.
        - Discute el impacto de cambios en los valores de los coeficientes del problema dual.
        - Relaciona los resultados con el contexto específico del problema.
        
        **Interpretación final:**
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

        **Justificación matemática:**
        - Explica los cálculos realizados para obtener la solución óptima.
        - Muestra las restricciones activas e inactivas y su impacto.
        - Relaciona los resultados con el contexto del problema.
        
        **Interpretación final:**
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

        resultado_ia = respuesta.choices[0].message.content if respuesta.choices else "⚠️ No se recibió respuesta del modelo."
        return jsonify({"analisis": resultado_ia})
    
    except Exception as e:
        print(f"❌ Error en la solicitud: {str(e)}")
        return jsonify({"analisis": "❌ Error al generar el análisis de sensibilidad."})
