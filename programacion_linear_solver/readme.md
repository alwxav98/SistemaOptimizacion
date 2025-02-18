# Modelo de Programación Lineal con Flask

Este proyecto es una aplicación web desarrollada con Flask que permite resolver problemas de Programación Lineal utilizando el método de Dos Fases. La aplicación permite ingresar la función objetivo y las restricciones para encontrar la solución óptima.

## Características
- Interfaz web para ingresar los datos del problema.
- Resolución automática de problemas de Programación Lineal con `PuLP`.
- Visualización de la solución óptima, incluyendo valores de variables de decisión y variables de holgura/exceso.
- Análisis de sensibilidad de los resultados.

## Estructura del Proyecto
```
MODELO PROGRAMACION LINEAL
│── templates/
│   ├── formulario.html    # Página para ingresar la función objetivo y restricciones.
│   ├── index.html         # Página de inicio para definir el número de variables y restricciones.
│   ├── solucion.html      # Página que muestra la solución óptima.
│── static/                # Archivos de estilos CSS.
│   ├── style.css
│── app.py                 # Código principal de la aplicación con Flask y PuLP.
```

## Instalación y Ejecución
### Prerrequisitos
- Python 3.x
- Flask
- PuLP

### Instalación

Instalar dependencias:
   ```sh
   pip install flask pulp
   ```

### Ejecución
Ejecutar el servidor Flask:
```sh
python app.py
```
Luego, abre tu navegador y accede a `http://127.0.0.1:5000/`.

## Uso
1. Ingresa el número de variables y restricciones en la página principal.
2. Define la función objetivo y las restricciones.
3. Visualiza la solución óptima en la página de resultados.
4. Puedes analizar la sensibilidad de la solución con el chatbot de análisis.

## Tecnologías Utilizadas
- **Flask**: Framework de Python para la creación de aplicaciones web.
- **PuLP**: Librería de Python para optimización lineal.
- **HTML, CSS**: Para la interfaz de usuario.

## Funcionamiento de la Librería PuLP
PuLP es una librería de optimización en Python que permite modelar y resolver problemas de programación lineal de manera sencilla. Se basa en la formulación estándar de optimización lineal y emplea el **Método Simplex** o **algoritmos de puntos interiores** según el solver utilizado.

### **Modelo Matemático**
Dado un problema de programación lineal en su forma estándar:

**Función Objetivo:**
\[
Z = c_1X_1 + c_2X_2 + ... + c_nX_n
\]

**Sujeto a restricciones:**
\[
a_{11}X_1 + a_{12}X_2 + ... + a_{1n}X_n (≤, =, ≥) b_1
\]
\[
a_{21}X_1 + a_{22}X_2 + ... + a_{2n}X_n (≤, =, ≥) b_2
\]
\[
...
\]
\[
a_{m1}X_1 + a_{m2}X_2 + ... + a_{mn}X_n (≤, =, ≥) b_m
\]

**Con restricciones de no negatividad:**
\[
X_1, X_2, ..., X_n ≥ 0
\]

### **Cómo lo Resuelve PuLP con el Método Simplex**
1. **Definir el problema:** Se establece si es de maximización o minimización con `pulp.LpMaximize` o `pulp.LpMinimize`.
2. **Declarar las variables de decisión:** Se crean variables con restricciones de no negatividad mediante `pulp.LpVariable`.
3. **Agregar restricciones:** Se incluyen ecuaciones de restricciones como expresiones algebraicas en `pulp.LpProblem`.
4. **Transformación a forma estándar:** PuLP agrega automáticamente variables de holgura y exceso cuando es necesario para convertir desigualdades en ecuaciones.
5. **Resolver con Simplex:** Se llama al solver de PuLP, que ejecuta internamente el método Simplex para encontrar la solución óptima:
   - Construcción de la tabla simplex.
   - Identificación de la variable entrante (columna pivote) basada en el coeficiente más positivo de la función objetivo.
   - Determinación de la variable saliente (fila pivote) mediante la regla del mínimo cociente.
   - Actualización iterativa de la tabla hasta que no haya coeficientes positivos en la función objetivo (para maximización).
6. **Obtener la solución:** Se acceden a los valores óptimos de las variables de decisión y de la función objetivo.

## Contribuciones
Las contribuciones son bienvenidas. Para ello:
1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature-nueva-funcionalidad`).
3. Realiza tus cambios y haz un commit (`git commit -m 'Agrega nueva funcionalidad'`).
4. Haz push a la rama (`git push origin feature-nueva-funcionalidad`).
5. Abre un Pull Request.

## Licencia
Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo `LICENSE`.










