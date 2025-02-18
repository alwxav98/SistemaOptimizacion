from flask import Flask, render_template
from network_solver.app import network_solver
from programacion_linear_solver.app import programacion_linear_solver
from transporte_solver_web.app import transporte_solver_web

app = Flask(__name__)

# Registrar los Blueprints
app.register_blueprint(programacion_linear_solver, url_prefix='/programacion_linear_solver')
app.register_blueprint(network_solver, url_prefix='/network_solver')
app.register_blueprint(transporte_solver_web, url_prefix='/transporte_solver_web')

@app.route('/')
def home():
    return render_template('menu.html')

if __name__ == '__main__':
    app.run(debug=True)
