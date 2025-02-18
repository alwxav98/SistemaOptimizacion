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
        queue = [(0, start)]
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        previous_nodes = {node: None for node in self.nodes}

        while queue:
            current_distance, current_node = heapq.heappop(queue)
            if current_node == end:
                path = []
                while previous_nodes[current_node] is not None:
                    path.insert(0, current_node)
                    current_node = previous_nodes[current_node]
                path.insert(0, start)
                return path, distances[end]

            for neighbor, weight in self.edges[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))

        return None, float('inf')

    def bfs(self, source, sink, parent):
        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            node = queue.popleft()
            for neighbor, capacity in self.capacity[node].items():
                if neighbor not in visited and capacity > 0:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = node
                    if neighbor == sink:
                        return True
        return False

    def edmonds_karp(self, source, sink):
        parent = {}
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.capacity[parent[s]][s])
                s = parent[s]

            v = sink
            while v != source:
                u = parent[v]
                self.capacity[u][v] -= path_flow
                self.capacity[v][u] += path_flow
                v = parent[v]

            max_flow += path_flow

        return max_flow

    def prim(self):
        mst = []
        visited = set()
        start_node = next(iter(self.nodes))
        edges = [(weight, start_node, to) for to, weight in self.edges[start_node].items()]
        heapq.heapify(edges)
        total_weight = 0
        added_edges = set()

        while edges:
            weight, frm, to = heapq.heappop(edges)
            if to not in visited:
                visited.add(to)

                if (frm, to) not in added_edges and (to, frm) not in added_edges:
                    mst.append({"from": frm, "to": to, "weight": weight})
                    total_weight += weight
                    added_edges.add((frm, to))

                for neighbor, weight in self.edges[to].items():
                    if neighbor not in visited:
                        heapq.heappush(edges, (weight, to, neighbor))

        return {"mst": mst, "total_weight": total_weight}

# Se inicializa un objeto de la clase Graph
graph = Graph()

# Definición de rutas en la API Flask para manejar las operaciones del grafo
@network_solver.route("/")
def index():
    return render_template("indexN.html", nodes=list(graph.nodes),
                           edges=[(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v])


@network_solver.route("/add_node", methods=["POST"])
def add_node():
    node = request.form["node"]
    graph.add_node(node)
    return jsonify({"nodes": list(graph.nodes)})


@network_solver.route("/add_edge", methods=["POST"])
def add_edge():
    from_node = request.form["from"]
    to_node = request.form["to"]
    weight = request.form["weight"]
    graph.add_edge(from_node, to_node, int(weight))
    return jsonify({"edges": [(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v]})


@network_solver.route("/shortest_path", methods=["POST"])
def shortest_path():
    start = request.form["start"]
    end = request.form["end"]
    path, cost = graph.dijkstra(start, end)
    return jsonify({"path": path, "cost": cost})


@network_solver.route("/max_flow", methods=["POST"])
def max_flow():
    source = request.form["source"]
    sink = request.form["sink"]
    flow = graph.edmonds_karp(source, sink)
    return jsonify({"max_flow": flow})


@network_solver.route("/minimum_spanning_tree", methods=["POST"])
def minimum_spanning_tree():
    result = graph.prim()
    return jsonify(result)


@network_solver.route("/generate_graph")
def generate_graph():
    G = nx.Graph()

    for node in graph.nodes:
        G.add_node(node)

    for from_node, connections in graph.edges.items():
        for to_node, weight in connections.items():
            G.add_edge(from_node, to_node, weight=weight)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10)
    edge_labels = {(from_node, to_node): f"{weight}" for from_node, to_node, weight in
                   [(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
    plt.close()

    return jsonify({"graph_img": f"data:image/png;base64,{img_base64}"})
