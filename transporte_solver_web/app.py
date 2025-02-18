from flask import Blueprint, request, jsonify, render_template
import pulp
import pandas as pd

# Crear Blueprint
transporte_solver_web = Blueprint(
    "transporte_solver_web",
    __name__,
    template_folder="templates",static_folder="static"
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
        n = len(oferta)
        m = len(demanda)

        # Definir el problema de optimización
        prob = pulp.LpProblem("Problema de Transporte", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous') for j in range(m)] for i in range(n)]

        # Función objetivo (minimizar costos)
        prob += pulp.lpSum(costos[i][j] * x[i][j] for i in range(n) for j in range(m))

        # Restricciones de oferta
        for i in range(n):
            prob += pulp.lpSum(x[i][j] for j in range(m)) <= oferta[i]

        # Restricciones de demanda
        for j in range(m):
            prob += pulp.lpSum(x[i][j] for i in range(n)) >= demanda[j]

        # Resolver el problema
        prob.solve()

        # Extraer los resultados
        resultado = [[pulp.value(x[i][j]) for j in range(m)] for i in range(n)]
        df = pd.DataFrame(resultado, columns=[f"Destino {j + 1}" for j in range(m)],
                          index=[f"Proveedor {i + 1}" for i in range(n)])

        return jsonify({"status": "success", "solucion": df.to_dict()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
