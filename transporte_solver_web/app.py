from flask import Blueprint, request, jsonify, render_template
import pulp
import pandas as pd

# Crear Blueprint
transporte_solver_web = Blueprint(
    "transporte_solver_web",__name__,template_folder="templates",static_folder="static"
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
        equilibrado = True  # ✅ Variable para verificar si hubo ajuste
        tabla_equilibrada = None  # ✅ Inicializar sin datos

        # **Paso 1: Ajustar la tabla si no está equilibrada**
        if total_oferta < total_demanda:
            costos.append([0] * len(demanda))  # Nueva fila con costos 0
            oferta.append(total_demanda - total_oferta)  # Nueva oferta ficticia
            mensaje_equilibrado = "El problema no estaba equilibrado. Se agregó un proveedor ficticio."
            equilibrado = False
        elif total_oferta > total_demanda:
            for fila in costos:
                fila.append(0)  # Nueva columna con costos 0
            demanda.append(total_oferta - total_demanda)
            mensaje_equilibrado = "El problema no estaba equilibrado. Se agregó un destino ficticio."
            equilibrado = False

        # **Paso 2: Verificar que el problema ya está equilibrado**
        if sum(oferta) != sum(demanda):
            return jsonify({
                "status": "error",
                "message": "El problema no pudo equilibrarse correctamente. Revisa los datos."
            })

        # **Paso 3: Solo guardar la tabla equilibrada si el problema se modificó**
        if not equilibrado:
            tabla_equilibrada = {
                "costos": [[costos[i][j] for j in range(len(demanda))] for i in range(len(oferta))],
                "oferta": oferta,
                "demanda": demanda
            }

        n = len(oferta)
        m = len(demanda)

        # **Paso 4: Definir el problema de optimización**
        prob = pulp.LpProblem("Problema de Transporte", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous') for j in range(m)] for i in range(n)]

        # **Paso 5: Función objetivo**
        prob += pulp.lpSum(costos[i][j] * x[i][j] for i in range(n) for j in range(m))

        # **Paso 6: Restricciones de oferta**
        for i in range(n):
            prob += pulp.lpSum(x[i][j] for j in range(m)) == oferta[i]

        # **Paso 7: Restricciones de demanda**
        for j in range(m):
            prob += pulp.lpSum(x[i][j] for i in range(n)) == demanda[j]

        # **Paso 8: Resolver el problema**
        prob.solve()

        # **Paso 9: Extraer los resultados**
        resultado = [[pulp.value(x[i][j]) for j in range(m)] for i in range(n)]
        df = pd.DataFrame(resultado, columns=[f"Destino {j + 1}" for j in range(m)],
                          index=[f"Proveedor {i + 1}" for i in range(n)])

        costo_total = pulp.value(prob.objective)

        return jsonify({
            "status": "success",
            "mensaje": mensaje_equilibrado,
            "solucion": df.to_dict(),
            "tabla_equilibrada": tabla_equilibrada,  # ✅ Solo enviamos la tabla si el problema fue equilibrado
            "costo_total": costo_total,
            "equilibrado": equilibrado  # ✅ Enviar si la tabla fue equilibrada
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})



