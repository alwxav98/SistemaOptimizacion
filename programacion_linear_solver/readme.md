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

## Contribuciones
Las contribuciones son bienvenidas. Para ello:
1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature-nueva-funcionalidad`).
3. Realiza tus cambios y haz un commit (`git commit -m 'Agrega nueva funcionalidad'`).
4. Haz push a la rama (`git push origin feature-nueva-funcionalidad`).
5. Abre un Pull Request.

## Licencia
Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo `LICENSE`.


