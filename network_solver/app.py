from flask import Blueprint, render_template, request, jsonify
import heapq
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
# Se define un Blueprint para la aplicación Flask que maneja la lógica de resolución de problemas de redes.
network_solver = Blueprint("network_solver", __name__, template_folder="templates",
                           static_folder="static")

# Clase que representa un grafo con métodos para resolver diferentes problemas de optimización.
class Graph:
    def __init__(self):
        self.nodes = set()  # Conjunto de nodos
        self.edges = {}  # Diccionario para almacenar las aristas y sus pesos
        self.capacity = {}  # Diccionario para capacidades en el problema de flujo máximo

    def add_node(self, value):
        """Agrega un nodo al grafo."""
        if value not in self.nodes:
            self.nodes.add(value)
            self.edges[value] = {}
            self.capacity[value] = {}

     def add_edge(self, from_node, to_node, weight):
        """Agrega una arista con un peso entre dos nodos."""
        if from_node in self.nodes and to_node in self.nodes:
            self.edges[from_node][to_node] = int(weight)
            self.edges[to_node][from_node] = int(weight)
            self.capacity[from_node][to_node] = int(weight)
            self.capacity[to_node][from_node] = 0 # Capacidad inversa para el algoritmo de flujo máximo

    def dijkstra(self, start, end):
        """Calcula la ruta más corta entre dos nodos usando el algoritmo de Dijkstra."""
        queue = [(0, start)]  # Cola de prioridad que almacena (distancia, nodo)
        distances = {node: float('inf') for node in self.nodes}  # Diccionario de distancias inicializado en infinito
        distances[start] = 0  # La distancia al nodo de inicio es 0
        previous_nodes = {node: None for node in self.nodes}  # Diccionario para rastrear el camino más corto

        while queue:
            current_distance, current_node = heapq.heappop(queue)  # Extrae el nodo con menor distancia acumulada
  
            # Si se llega al nodo destino, se reconstruye la ruta más corta
            if current_node == end:
                path = []
                while previous_nodes[current_node] is not None:
                    path.insert(0, current_node)  # Inserta en orden inverso
                    current_node = previous_nodes[current_node]
                path.insert(0, start)  # Se agrega el nodo de inicio al camino
                return path, distances[end]  # Retorna la ruta y la distancia total

            # Explora los nodos vecinos
            for neighbor, weight in self.edges[current_node].items():
                distance = current_distance + weight  # Calcula la nueva distancia acumulada
                
                # Si la nueva distancia es menor, se actualiza
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node  # Se registra el nodo previo en el camino
                    heapq.heappush(queue, (distance, neighbor))  # Se añade a la cola de prioridad

        return None, float('inf') # Retorna None si no hay camino posible

    def bfs(self, source, sink, parent):
      """Algoritmo de búsqueda en anchura (BFS) utilizado en Edmonds-Karp para encontrar caminos aumentantes."""
        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            node = queue.popleft()
            for neighbor, capacity in self.capacity[node].items():
                # Si el nodo vecino no ha sido visitado y hay capacidad en la arista, se avanza en la búsqueda
                if neighbor not in visited and capacity > 0:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = node  # Se almacena el camino
                    if neighbor == sink:
                        return True  # Se encontró un camino hasta el nodo sumidero
        return False

    def edmonds_karp(self, source, sink):
      """Implementa el algoritmo de Edmonds-Karp para calcular el flujo máximo en un grafo."""
        parent = {}
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float('inf')
            s = sink
            # Se encuentra el flujo mínimo en el camino aumentado
            while s != source:
                path_flow = min(path_flow, self.capacity[parent[s]][s])
                s = parent[s]

            # Se actualizan las capacidades residuales de las aristas y sus inversa
            v = sink
            while v != source:
                u = parent[v]
                self.capacity[u][v] -= path_flow
                self.capacity[v][u] += path_flow
                v = parent[v]

            max_flow += path_flow # Se suma el flujo encontrado al flujo total

        return max_flow

    def prim(self):
         """Algoritmo de Prim para calcular el árbol de expansión mínima."""
        mst = []
        visited = set()
        start_node = next(iter(self.nodes)) # Se elige un nodo arbitrario como inicio
        edges = [(weight, start_node, to) for to, weight in self.edges[start_node].items()]
        heapq.heapify(edges)
        total_weight = 0
        added_edges = set()

        while edges:
            weight, frm, to = heapq.heappop(edges)
            if to not in visited:
                visited.add(to)
                
                # Se asegura que la arista no se repita en ambas direcciones
                if (frm, to) not in added_edges and (to, frm) not in added_edges:
                    mst.append({"from": frm, "to": to, "weight": weight})
                    total_weight += weight
                    added_edges.add((frm, to))

               # Se agregan nuevas aristas al heap si su nodo destino no ha sido visitado
                for neighbor, weight in self.edges[to].items():
                    if neighbor not in visited:
                        heapq.heappush(edges, (weight, to, neighbor))

        return {"mst": mst, "total_weight": total_weight}

# Se inicializa un objeto de la clase Graph
graph = Graph()

# Definición de rutas en la API Flask para manejar las operaciones del grafo
@network_solver.route("/")
def index():
     """Renderiza la página principal con los nodos y aristas actuales del grafo."""
    return render_template("indexN.html", nodes=list(graph.nodes),
                           edges=[(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v])


@network_solver.route("/add_node", methods=["POST"])
def add_node():
    """API para agregar un nodo al grafo."""
    node = request.form["node"]
    graph.add_node(node)
    return jsonify({"nodes": list(graph.nodes)})


@network_solver.route("/add_edge", methods=["POST"])
def add_edge():
    """API para agregar una arista entre dos nodos con un peso determinado."""
    from_node = request.form["from"]
    to_node = request.form["to"]
    weight = request.form["weight"]
    graph.add_edge(from_node, to_node, int(weight))
    return jsonify({"edges": [(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v]})


@network_solver.route("/shortest_path", methods=["POST"])
def shortest_path():
    """API para calcular la ruta más corta entre dos nodos usando Dijkstra."""
    start = request.form["start"]
    end = request.form["end"]
    path, cost = graph.dijkstra(start, end)
    return jsonify({"path": path, "cost": cost})


@network_solver.route("/max_flow", methods=["POST"])
def max_flow():
    """API para calcular el flujo máximo entre dos nodos usando Edmonds-Karp."""
    source = request.form["source"]
    sink = request.form["sink"]
    flow = graph.edmonds_karp(source, sink)
    return jsonify({"max_flow": flow})


@network_solver.route("/minimum_spanning_tree", methods=["POST"])
def minimum_spanning_tree():
    """API para calcular el árbol de expansión mínima usando Prim."""
    result = graph.prim()
    return jsonify(result)


@network_solver.route("/generate_graph")
def generate_graph():
   """Genera una representación visual del grafo actual."""
    G = nx.Graph()  # Se crea un objeto de grafo vacío con NetworkX

    # Se agregan los nodos al grafo
    for node in graph.nodes:
        G.add_node(node)

    # Se agregan las aristas con sus respectivos pesos
    for from_node, connections in graph.edges.items():
        for to_node, weight in connections.items():
            G.add_edge(from_node, to_node, weight=weight)

    # Se configura la figura y la disposición del grafo    
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G) # Se utiliza el layout de resorte para posicionar los nodos
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10)
    # Se generan las etiquetas de las aristas con sus pesos
    edge_labels = {(from_node, to_node): f"{weight}" for from_node, to_node, weight in
                   [(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    # Se guarda la imagen en un buffer y se convierte a formato base64 para ser enviada en la respuesta JSON
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
    plt.close()

    return jsonify({"graph_img": f"data:image/png;base64,{img_base64}"})
