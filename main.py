from flask import Flask, render_template
from network_solver.app import network_solver  # 📌 Importa el Blueprint del módulo Network Solver
from programacion_linear_solver.app import programacion_linear_solver  # 📌 Importa el Blueprint del módulo Programación Lineal Solver
from transporte_solver_web.app import transporte_solver_web  # 📌 Importa el Blueprint del módulo Transporte Solver Web
from chatbot_analisis_sensibilidad.app import chatbot  # ✅ Importa el Blueprint del chatbot para análisis de sensibilidad

# 🔹 Inicializa la aplicación Flask
app = Flask(__name__)
app.secret_key = "clave_secreta_segura"  # 🔑 Necesario para el uso de sesiones en Flask (por ejemplo, para almacenar información del usuario)

# 🔹 Registrar los Blueprints para cada módulo, asignando un prefijo de URL correspondiente
app.register_blueprint(programacion_linear_solver, url_prefix='/programacion_linear_solver')  # Ruta para Programación Lineal Solver
app.register_blueprint(network_solver, url_prefix='/network_solver')  # Ruta para Network Solver
app.register_blueprint(transporte_solver_web, url_prefix='/transporte_solver_web')  # Ruta para Transporte Solver Web
app.register_blueprint(chatbot, url_prefix='/chatbot')  # ✅ Ruta para el chatbot de análisis de sensibilidad

# 🔹 Define la ruta principal del sistema, que carga el menú principal
@app.route('/')
def home():
    return render_template('menu.html')  # 📌 Renderiza la plantilla del menú principal

# 🔹 Ejecuta la aplicación en modo debug si el script se ejecuta directamente
if __name__ == '__main__':
    app.run(debug=True)  # ⚙️ Activa el modo debug para facilitar el desarrollo y depuración
