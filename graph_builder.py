import networkx as nx
import numpy as np

def create_distance_graph(unique_distances, transport_graphs):
    """
    Находит кратчайшие расстояния от случайно выбранной стартовой остановки до ближайших
    остановок в графе, находящихся на расстоянии менее max_distance метров, с использованием алгоритма Дейкстры.

    Параметры:
    - graph (networkx.Graph): Граф, представляющий собой сеть общественного транспорта, в котором
      вершины представляют остановки, а ребра - маршруты между ними.
    - max_distance (int): Максимальное расстояние до остановки, которое нужно рассматривать (по умолчанию 2500 м).

    Возвращает:
    - distances (dict): Словарь, где ключи - идентификаторы остановок, а значения - кратчайшие расстояния
      от стартовой остановки до каждой ближайшей остановки.
      Если расстояние до остановки не найдено, значение будет None.
    - num_nearby_stops (int): Количество других ближайших остановок ,
      находящихся на расстоянии менее max_distance метров.
    """
    distances = {}  # Словарь для хранения расстояний
    num_nearby_stops = 0  # Счетчик других ближайших остановок

    # Получаем все остановки общественного транспорта
    public_transport_nodes = [node for node, attr in graph.nodes(data=True) if 'layer' in attr]
    # Выбираем случайную стартовую остановку
    source_node = random.choice(public_transport_nodes)

    # Алгоритм Дейкстры
    pq = [(0, source_node)]  # Приоритетная очередь для обработки вершин
    while pq:
        dist, node = heapq.heappop(pq)
        if node in distances:
            continue
        distances[node] = dist
        for neighbor, attrs in graph[node].items():
            if 'weight' in attrs:
                new_dist = dist + attrs['weight']
                if new_dist <= max_distance:
                    heapq.heappush(pq, (new_dist, neighbor))
                    if neighbor in public_transport_nodes:
                        num_nearby_stops += 1

    # Проверяем, найдены ли расстояния до всех остановок
    for stop in public_transport_nodes:
        if stop not in distances:
            distances[stop] = None

    return distances, num_nearby_stops

def build_multilayer_graph(transport_graphs, G_transfer):
    """
    Добавляет вершины из списка вершин в граф, обеспечивая уникальность вершин по комбинации (stop_id, layer).

    Параметры:
    - graph (networkx.Graph): Граф, в который будут добавлены вершины.
    - vertices (list): Список словарей с информацией о вершинах.

    Возвращает:
    - None
    """
    # Создаем словарь для хранения вершин по уникальной комбинации (stop_id, layer)
    unique_vertices = {}
    
    # Заполняем словарь уникальными вершинами
    for vertex in vertices:
        stop_id = vertex['stop_id']
        utm_x = vertex['utm_x']
        utm_y = vertex['utm_y']
        layer = vertex['layer']
        key = (stop_id, layer)
        
        # Проверяем, есть ли уже такая вершина в словаре
        if key not in unique_vertices:
            unique_vertices[key] = {'stop_id': stop_id, 'utm_x': utm_x, 'utm_y': utm_y, 'layer': layer}
    
    # Добавляем вершины из словаря в граф
    for key, attributes in unique_vertices.items():
        graph.add_node(key, **attributes)

def add_edges_to_graph(graph, edges):
    """
    Добавляет ребра из списка ребер в граф.

    Параметры:
    - graph (networkx.Graph): Граф, в который будут добавлены ребра.
    - edges (list): Список словарей с информацией о ребрах.

    Возвращает:
    - None
    """
    for edge in edges:
        id_1 = edge['id_1']
        id_2 = edge['id_2']
        layer = edge['layer']
        weight = edge['weight']
        
        # Находим вершины с соответствующими id и layer
        vertices_with_id_1 = [v for v in graph.nodes(data=True) if v[0][0] == id_1 and v[0][1] == layer]
        vertices_with_id_2 = [v for v in graph.nodes(data=True) if v[0][0] == id_2 and v[0][1] == layer]
            
        # Если найдены соответствующие вершины, добавляем ребро
        if vertices_with_id_1 and vertices_with_id_2:
            graph.add_edge(vertices_with_id_1[0][0], vertices_with_id_2[0][0], layer=layer, weight=weight)
            def add_dismount_edges_to_graph(graph, edges):
    """
    Добавляет ребра спешивания из списка ребер в граф.

    Параметры:
    - graph (networkx.Graph): Граф, в который будут добавлены ребра.
    - edges (list): Список словарей с информацией о ребрах.

    Возвращает:
    - None
    """
    for edge in edges:
        if edge['layer'] == 'dismount':
            id_1 = edge['id_1']
            id_2 = edge['id_2']
            weight = edge['weight']
            
            # Находим вершины с stop_id равным id_1
            vertices_with_id_1 = [v for v in graph.nodes(data=True) if v[0][0] == id_1]
            
            # Должно быть только две такие вершины
            if len(vertices_with_id_1) == 2:
                vertex_1 = vertices_with_id_1[0][0]
                vertex_2 = vertices_with_id_1[1][0]
                
                # Добавляем ребро между найденными вершинами
                graph.add_edge(vertex_1, vertex_2, layer='dismount', weight=weight)