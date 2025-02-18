from flask import Blueprint, render_template, request, redirect, url_for
import numpy as np
import pulp

# Crear Blueprint
programacion_linear_solver = Blueprint("programacion_linear_solver", __name__, template_folder="templates",static_folder="static")

@programacion_linear_solver.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_variables = int(request.form['num_variables'])
        num_restricciones = int(request.form['num_restricciones'])
        return redirect(url_for('programacion_linear_solver.formulario', num_variables=num_variables, num_restricciones=num_restricciones))
    return render_template('index.html')

@programacion_linear_solver.route('/formulario/<int:num_variables>/<int:num_restricciones>', methods=['GET', 'POST'])
def formulario(num_variables, num_restricciones):
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

        solucion, iteraciones, variables_holgura_exceso = resolver_simplex(objetivo_tipo, coef_objetivo, restricciones, signos, rhs)
        return render_template('solucion.html', solucion=solucion, iteraciones=iteraciones, variables_holgura_exceso=variables_holgura_exceso)

    return render_template('formulario.html', num_variables=num_variables, num_restricciones=num_restricciones)

def resolver_simplex(objetivo_tipo, coef_objetivo, restricciones, signos, rhs):
    prob = pulp.LpProblem("Problema PL", pulp.LpMinimize if objetivo_tipo == 'Minimizar' else pulp.LpMaximize)
    variables = [pulp.LpVariable(f'X{i+1}', lowBound=0) for i in range(len(coef_objetivo))]
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
    solucion['Z'] = pulp.value(prob.objective)

    variables_holgura_exceso = {var_name: var.varValue for var_name, var in holgura_exceso_vars.items()}

    iteraciones = []  # Aquí se pueden agregar pasos intermedios si se requiere

    return solucion, iteraciones, variables_holgura_exceso
