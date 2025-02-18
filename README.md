# Sistema de Programación Lineal, Redes y Transporta + IA con Flask

Este proyecto es un sistema modular basado en Flask que permite resolver problemas de **Programación Lineal**, **Transporte** y **Redes** para la optimización matemática.

## 📂 Estructura del Proyecto

```
📦 SistemaOptimizacion
│── .idea/                        # Configuración del proyecto (IDE)
│── chatbot_analisis_sensibilidad/ # Análisis de sensibilidad con chatbot
│   ├── static/
│   │   ├── chatbot.jpeg
│   │   ├── scripts.js
│   │   ├── style.css
│   ├── templates/
│   │   ├── chat.html
│   ├── app.py                     # API del chatbot
│   ├── requirements.txt
│
│── network_solver/                # Solver de redes (flujos, caminos, etc.)
│   ├── static/
│   │   ├── styles.css
│   ├── templates/
│   │   ├── indexN.html
│   ├── app.py                      # Lógica de resolución
│   ├── README.md
│
│── programacion_linear_solver/     # Solver de Programación Lineal
│   ├── static/
│   │   ├── style.css
│   ├── templates/
│   │   ├── formulario.html
│   │   ├── index.html
│   │   ├── solucion.html
│   ├── app.py                      # API del solver
│   ├── readme.md
│
│── transporte_solver_web/          # Solver de Transporte
│   ├── static/
│   │   ├── style.css
│   ├── templates/
│   │   ├── inicio.html
│   ├── app.py                      # API del solver
│
├── main.py
├── requirements.txt
│── static/
│   ├── styles.css
│
│── templates/
│   ├── menu.html
```

## 🚀 Funcionalidades
Este sistema está compuesto por varios módulos independientes:

### 🔹 **Chatbot de Análisis de Sensibilidad**
- Analiza soluciones óptimas y proporciona información de sensibilidad.
- Interfaz visual basada en Flask con integración de JavaScript.

### 🔹 **Solver de Redes**
- Resuelve problemas de flujo máximo, caminos más cortos y árboles de expansión mínima.
- Interfaz en Flask para definir grafos y restricciones.

### 🔹 **Solver de Programación Lineal**
- Permite modelar y resolver problemas de optimización lineal con PuLP.
- Utiliza el **Método Simplex** para obtener la solución óptima.

### 🔹 **Solver de Transporte**
- Implementa algoritmos para problemas de asignación y distribución óptima.
- Interfaz en Flask con definición de costos y restricciones.

## 🛠 Instalación y Ejecución
### 🔹 **Requisitos Previos**
- Python 3.x
- Flask
- PuLP

### 🔹 **Instalación de Dependencias**
Ejecuta el siguiente comando en la raíz del proyecto:
```sh
pip install -r requirements.txt
```

### 🔹 **Ejecutar un Módulo**
Para iniciar el solver de programación lineal:
```sh
cd programacion_linear_solver
python app.py
```
Accede a `http://127.0.0.1:5000/` en tu navegador.

## 📌 Uso del Proyecto
1. Accede a la interfaz de usuario en el navegador.
2. Define el problema de optimización (variables, restricciones, etc.).
3. Ejecuta la solución y revisa los resultados en la página de solución óptima.

## 📖 Funcionamiento Interno
Cada solver implementa la lógica de optimización de acuerdo con su tipo:

### 🔹 **Solver de Programación Lineal**
Utiliza **PuLP** para definir variables de decisión, restricciones y objetivos:

### 🔹 **Solver de Redes**
Se basa en **NetworkX** para modelar grafos y resolver problemas de optimización en redes:


## 📄 Licencia
Este proyecto está bajo la licencia **MIT**.



