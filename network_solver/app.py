from flask import Blueprint, render_template, request, jsonify
import heapq
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64

network_solver = Blueprint("network_solver", __name__, template_folder="templates",
                           static_folder="static")

shortest_path_edges = set()


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.capacity = {}
        self.cost = {}

    def add_node(self, value):
        if value not in self.nodes:
            self.nodes.add(value)
            self.edges[value] = {}
            self.capacity[value] = {}

    def add_edge(self, from_node, to_node, weight):
        if from_node in self.nodes and to_node in self.nodes:
            self.edges[from_node][to_node] = int(weight)
            self.edges[to_node][from_node] = int(weight)
            self.capacity[from_node][to_node] = int(weight)
            self.capacity[to_node][from_node] = 0

            # ✅ Guardar los costos de la arista
            if from_node not in self.cost:
                self.cost[from_node] = {}
            if to_node not in self.cost:
                self.cost[to_node] = {}

            self.cost[from_node][to_node] = int(weight)
            self.cost[to_node][from_node] = int(weight)  # Si el grafo es no dirigido

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

    def min_cost_max_flow(self, source, sink):
        G = nx.DiGraph()

        # ✅ Agregar nodos
        for node in self.nodes:
            G.add_node(node)

        # ✅ Agregar aristas con capacidad y costo
        for from_node in self.edges:
            for to_node in self.edges[from_node]:
                if self.capacity[from_node][to_node] > 0:
                    G.add_edge(
                        from_node, to_node, 
                        capacity=self.capacity[from_node][to_node], 
                        weight=self.cost[from_node][to_node]  # ✅ Agregar costo correctamente
                    )

        # ✅ Verificar si el grafo tiene flujo positivo antes de ejecutar
        if not nx.has_path(G, source, sink):
            return 0  # No hay camino entre source y sink

        try:
            # ✅ Resolver el problema de flujo de costo mínimo
            flow_dict = nx.max_flow_min_cost(G, source, sink)
            min_cost = nx.cost_of_flow(G, flow_dict)

            return min_cost
        except Exception as e:
            print(f"Error en min_cost_max_flow: {e}")
            return 0  # Si hay error, devolver 0

    def remove_node(self, node):
        """Elimina un nodo y todas sus conexiones."""
        if node in self.nodes:
            self.nodes.remove(node)

            # Eliminar todas las aristas conectadas a este nodo
            del self.edges[node]
            del self.capacity[node]
            if node in self.cost:
                del self.cost[node]

            # Eliminar este nodo de las conexiones de los otros nodos
            for other_node in list(self.edges.keys()):
                if node in self.edges[other_node]:
                    del self.edges[other_node][node]
                    del self.capacity[other_node][node]
                    if node in self.cost[other_node]:
                        del self.cost[other_node][node]


    def remove_edge(self, from_node, to_node):
        """Elimina una arista entre dos nodos."""
        if from_node in self.edges and to_node in self.edges[from_node]:
            del self.edges[from_node][to_node]
            del self.capacity[from_node][to_node]
            if to_node in self.cost[from_node]:
                del self.cost[from_node][to_node]

        if to_node in self.edges and from_node in self.edges[to_node]:
            del self.edges[to_node][from_node]
            del self.capacity[to_node][from_node]
            if from_node in self.cost[to_node]:
                del self.cost[to_node][from_node]






graph = Graph()


@network_solver.route("/")
def index():
    return render_template("indexN.html", nodes=list(graph.nodes),
                           edges=[(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v])


@network_solver.route("/add_node", methods=["POST"])
def add_node():
    node = request.form["node"]
    if node in graph.nodes:
        return jsonify({"error": "El nodo ya existe", "nodes": list(graph.nodes)})
    
    graph.add_node(node)
    return jsonify({"nodes": list(graph.nodes)})

@network_solver.route("/get_nodes", methods=["GET"])
def get_nodes():
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
    global shortest_path_edges
    start = request.form["start"]
    end = request.form["end"]
    path, cost = graph.dijkstra(start, end)

    # Almacenar la última ruta más corta
    shortest_path_edges = set()
    if path:
        for i in range(len(path) - 1):
            shortest_path_edges.add((path[i], path[i + 1]))
            shortest_path_edges.add((path[i + 1], path[i]))  # Para grafos no dirigidos

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

@network_solver.route("/min_cost_max_flow", methods=["POST"])
def min_cost_max_flow():
    source = request.form["source"]
    sink = request.form["sink"]

    try:
        result = graph.min_cost_max_flow(source, sink)
        return jsonify({"status": "success", "cost": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@network_solver.route("/remove_node", methods=["POST"])
def remove_node():
    node = request.form["node"]
    graph.remove_node(node)
    
    return jsonify({
        "nodes": list(graph.nodes),
        "edges": [(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v]
    })



@network_solver.route("/remove_edge", methods=["POST"])
def remove_edge():
    edge = request.form["edge"]
    from_node, to_node = edge.split(" - ")  # Separar los nodos de la arista
    graph.remove_edge(from_node, to_node)

    return jsonify({
        "edges": [(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v]
    })

@network_solver.route("/get_edges", methods=["GET"])
def get_edges():
    return jsonify({
        "edges": [(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v]
    })


@network_solver.route("/generate_graph")
def generate_graph():
    global shortest_path_edges
    G = nx.Graph()

    # Agregar nodos al grafo
    for node in graph.nodes:
        G.add_node(node)

    # Agregar aristas al grafo
    for from_node, connections in graph.edges.items():
        for to_node, weight in connections.items():
            G.add_edge(from_node, to_node, weight=weight)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G)

    # Dibujar los nodos
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10)

    # Dibujar las aristas con colores: gris para normales, rojo para la ruta más corta
    for from_node, to_node in G.edges:
        edge_color = "red" if (from_node, to_node) in shortest_path_edges else "gray"
        nx.draw_networkx_edges(G, pos, edgelist=[(from_node, to_node)], edge_color=edge_color, width=2)

    # Etiquetas de las aristas
    edge_labels = {(from_node, to_node): f"{weight}" for from_node, to_node, weight in
                   [(k, v, w) for k, v_w in graph.edges.items() for v, w in v_w.items() if k < v]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black")

    # Agregar texto de leyenda dentro del gráfico
    plt.text(0.05, 0.02, "Líneas en rojo = Ruta más corta", fontsize=12, color="red",
             transform=plt.gcf().transFigure, bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    # Guardar la imagen en base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
    plt.close()

    return jsonify({"graph_img": f"data:image/png;base64,{img_base64}"})


