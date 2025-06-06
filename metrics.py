import networkx as nx
import pandas as pd
import csv

def compute_aggregated_degree(G):
    """
    Вычисляет нормализованную агрегированную степень для вершин в многослойном графе,
    нормализуя по количеству вершин в слое 'transfer'.
    """
    aggregated_degrees = {}

    # Считаем количество вершин в слое 'transfer'
    num_transfer_nodes = sum(1 for node in G_multilayer.nodes() if G_multilayer.nodes[node]['layer'] == 'transfer')

    # Проверка на случай деления на ноль
    if num_transfer_nodes == 0:
        raise ValueError("Нет ни одной вершины с layer='transfer' для нормализации.")

    for node in G_multilayer.nodes():
        stop_id, layer = node
        utm_x = G_multilayer.nodes[node]['utm_x']
        utm_y = G_multilayer.nodes[node]['utm_y']

        # Подсчёт рёбер только в пределах своего слоя
        edge_count = sum(
            1 for edge in G_multilayer.edges(node)
            if G_multilayer.edges[edge]['layer'] == layer
        )

        if stop_id in aggregated_degrees:
            aggregated_degrees[stop_id]['edge_count'] += edge_count
        else:
            aggregated_degrees[stop_id] = {
                'utm_x': utm_x,
                'utm_y': utm_y,
                'edge_count': edge_count
            }

    # Сохраняем нормализованные значения
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['stop_id', 'utm_x', 'utm_y', 'aggregated_degree_norm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for stop_id, data in aggregated_degrees.items():
            normalized_degree = data['edge_count'] / num_transfer_nodes
            writer.writerow({
                'stop_id': stop_id,
                'utm_x': data['utm_x'],
                'utm_y': data['utm_y'],
                'aggregated_degree_norm': normalized_degree
            })

    return aggregated_degrees
def compute_cross_layer_betweenness_centrality(G_multilayer, all_shortest_paths, node_id, layer_1, layer_2):
    node_layer = G_multilayer.nodes[node_id]['layer']
    cross_layer_betweenness = 0
    total_cross_layer_paths = 0

    for path in all_shortest_paths.values():
        layers = {G_multilayer.nodes[node]['layer'] for node in path}
        if node_layer in layers and layer_1 in layers and layer_2 in layers:
            total_cross_layer_paths += 1
            if node_id in path:
                cross_layer_betweenness += 1

    # Нормализация по общему числу путей через 3 слоя
    if total_cross_layer_paths > 0:
        cross_layer_betweenness /= total_cross_layer_paths
    else:
        cross_layer_betweenness = 0

    return cross_layer_betweenness

def compute_cross_layer_betweenness_for_all_vertices(G_multilayer, all_shortest_paths, layer_1, layer_2):
    cross_layer_betweenness = {}

    for node_id in G_multilayer.nodes():
        vertex_layer = G_multilayer.nodes[node_id]['layer']
        if vertex_layer != layer_1 and vertex_layer != layer_2:
            centrality = compute_cross_layer_betweenness_centrality(
                G_multilayer, all_shortest_paths, node_id, layer_1, layer_2
            )
            cross_layer_betweenness[node_id] = centrality

    return cross_layer_betweenness

def save_cross_layer_betweenness_to_csv_with_coordinates(G_multilayer, cross_layer_betweenness, output_file):
    data = []
    for node_id, centrality in cross_layer_betweenness.items():
        layer = G_multilayer.nodes[node_id]['layer']
        utm_x = G_multilayer.nodes[node_id]['utm_x']
        utm_y = G_multilayer.nodes[node_id]['utm_y']
        data.append({
            'Node': node_id,
            'Layer': layer,
            'Cross_layer_betweenness': centrality,
            'UTM_X': utm_x,
            'UTM_Y': utm_y
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

def cross_layer_betweenness_centrality(G_multilayer, all_shortest_paths, layer_1, layer_2, output_file):
    cross_layer_betweenness_all_vertices = compute_cross_layer_betweenness_for_all_vertices(
        G_multilayer, all_shortest_paths, layer_1, layer_2
    )

    save_cross_layer_betweenness_to_csv_with_coordinates(
        G_multilayer, cross_layer_betweenness_all_vertices, output_file
    )

    return cross_layer_betweenness_all_vertices

# Пример вызова:
layer_1 = "0"  # Первый слой
layer_2 = "3"  # Второй слой
output_file = "cross_betweenness_centrality.csv"

# all_shortest_paths должен быть предварительно вычислен
cross_layer_betweenness_all_vertices = cross_layer_betweenness_centrality(
    G_multilayer, all_shortest_paths, layer_1, layer_2, output_file
)

#3 Взаимозависимость
def cross_layer_dependence(G_multilayer, all_shortest_paths, output_file):
    """
    Вычисляет, визуализирует и сохраняет показатель взаимозависимости для каждой вершины в графе в CSV файл.

    Параметры:
    - G_multilayer (networkx.Graph): Многослойный граф.
    - all_shortest_paths (dict): Словарь с кратчайшими путями для каждой пары вершин.
    - output_file (str): Путь к выходному файлу.

    Возвращает:
    - cross_layer_dependence (dict): Словарь, содержащий значения показателя взаимозависимости для каждой вершины.
    """
    # Функция вычисления показателя взаимозависимости
    def compute_cross_layer_dependence(G_multilayer, all_shortest_paths):
        cross_layer_dependence = {}
    
        for node in G_multilayer.nodes():
            single_layer_paths = 0
            multiple_layer_paths = 0
        
            for source, target in all_shortest_paths.keys():
                if source != node and target != node:
                    shortest_path = all_shortest_paths[(source, target)]
                
                    if node in shortest_path:
                        layers_in_path = set(G_multilayer.nodes[n]['layer'] for n in shortest_path)
                        if len(layers_in_path) == 2:
                            single_layer_paths += 1
                        else:
                            multiple_layer_paths += 1
        
            total_paths = single_layer_paths + multiple_layer_paths
            if total_paths > 0:
                cross_layer_dependence[node] = multiple_layer_paths / total_paths
            else:
                cross_layer_dependence[node] = 0  # Или np.nan, если хочешь специально отметить

        return cross_layer_dependence


    # Функция визуализации показателя взаимозависимости
    def visualize_cross_layer_dependence(G, cross_layer_dependence):
        pos = {node: (G.nodes[node]['utm_x'], G.nodes[node]['utm_y']) for node in G.nodes()}
        node_size = [cross_layer_dependence[node] * 500 for node in G.nodes()]
        node_color = list(cross_layer_dependence.values())
        cmap = plt.cm.plasma
        vmin = min(node_color)
        vmax = max(node_color)
        node_color = [cmap((x - vmin) / (vmax - vmin)) for x in node_color]

        scalebar = ScaleBar(1, location='lower left')
        plt.gca().add_artist(scalebar)

        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title("Показатель взаимозависимости")
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca(), label="Значение показателя")
        plt.axis('off')
        plt.show()

    # Функция сохранения показателя взаимозависимости в CSV файл
    def save_cross_layer_dependence_to_csv_with_coordinates(G, cross_layer_dependence, output_file):
        data = []

        for node, dependence in cross_layer_dependence.items():
            layer = G.nodes[node]['layer']
            utm_x = G.nodes[node]['utm_x']
            utm_y = G.nodes[node]['utm_y']
            data.append({'Вершина': node, 'Слой': layer, 'Показатель взаимозависимости': dependence, 'UTM_X': utm_x, 'UTM_Y': utm_y})

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

    # Вычисление показателя взаимозависимости
    comp_cross_layer_dependence = compute_cross_layer_dependence(G_multilayer, all_shortest_paths)

    # Визуализация показателя взаимозависимости
    visualize_cross_layer_dependence(G_multilayer, comp_cross_layer_dependence)

    # Сохранение показателя взаимозависимости в CSV
    save_cross_layer_dependence_to_csv_with_coordinates(G_multilayer, comp_cross_layer_dependence, output_file)

    return cross_layer_dependence
cross_layer_dependence(G_multilayer, all_shortest_paths, 'cross_dependence.csv')

#4 Среднее кросс-расстояние
def compute_cross_average_distance_for_all_layers(G_multilayer, all_shortest_paths):
    """
    Вычисляет нормализованное среднее кросс-расстояние между всеми парами слоев,
    нормализуя его на среднюю длину всех кратчайших путей в сети.
    """
    
    # Вычисляем глобальную среднюю длину всех кратчайших путей
    all_lengths = [len(path) - 1 for path in all_shortest_paths.values() if len(path) > 1]
    if all_lengths:
        global_avg_length = sum(all_lengths) / len(all_lengths)
    else:
        raise ValueError("Невозможно рассчитать среднюю длину кратчайших путей: пустой список.")

    # Вложенная функция для расчета кросс-расстояния между двумя слоями
    def compute_cross_average_distance(layer_1, layer_2):
        cross_layer_distances = []

        for node_1 in G_multilayer.nodes():
            if G_multilayer.nodes[node_1]['layer'] == layer_1:
                for node_2 in G_multilayer.nodes():
                    if G_multilayer.nodes[node_2]['layer'] == layer_2:
                        if (node_1, node_2) in all_shortest_paths:
                            path_length = len(all_shortest_paths[(node_1, node_2)]) - 1
                            cross_layer_distances.append(path_length)

        if cross_layer_distances:
            raw_avg = sum(cross_layer_distances) / len(cross_layer_distances)
            norm_avg = raw_avg / global_avg_length
        else:
            norm_avg = 0

        return norm_avg

    # Главный словарь результатов
    cross_average_distances = {}
    layers = set(nx.get_node_attributes(G_multilayer, 'layer').values())

    for layer_1 in layers:
        for layer_2 in layers:
            if layer_1 != layer_2 and (layer_2, layer_1) not in cross_average_distances:
                norm_avg = compute_cross_average_distance(layer_1, layer_2)
                cross_average_distances[(layer_1, layer_2)] = norm_avg

    return cross_average_distances

def save_cross_avg_distance_to_csv(cross_average_distances, filename="cross_average_distance.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Layer_1', 'Layer_2', 'Normalized_Cross_Avg_Distance'])
        for (layer_1, layer_2), value in cross_average_distances.items():
            writer.writerow([layer_1, layer_2, value])

# Вызываем функцию
cross_average_distances = compute_cross_average_distance_for_all_layers(G_multilayer, all_shortest_paths)
# Печатаем результаты в читаемом виде
print("Нормализованные средние кросс-расстояния между слоями:")
for (layer_1, layer_2), value in cross_average_distances.items():
    print(f"{layer_1} ↔ {layer_2}: {value:.4f}")
save_cross_avg_distance_to_csv(cross_average_distances)

# 5 Межслойная близость
def compute_interlayer_closeness_centrality(G_multilayer, all_shortest_paths):
    """
    Вычисляет нормализованную межслойную близость для всех узлов,
    деля локальную величину на среднюю длину всех межслоевых кратчайших путей.
    """
    interlayer_closeness = {}
    interlayer_distances = []

    # 1. Собираем все межслоевые расстояния для глобального среднего
    for (source, target), path in all_shortest_paths.items():
        if G_multilayer.nodes[source]['layer'] != G_multilayer.nodes[target]['layer']:
            interlayer_distances.append(len(path) - 1)

    # 2. Глобальное среднее расстояние
    if interlayer_distances:
        global_avg_interlayer_distance = sum(interlayer_distances) / len(interlayer_distances)
    else:
        raise ValueError("Не удалось вычислить среднюю межслоевую длину: отсутствуют межслоевые пути.")

    # 3. Локальный расчёт для каждой вершины
    for v in G_multilayer.nodes():
        v_layer = G_multilayer.nodes[v]['layer']
        total_distance = 0
        count = 0

        for (source, target), path in all_shortest_paths.items():
            if source == v and G_multilayer.nodes[target]['layer'] != v_layer:
                total_distance += len(path) - 1
                count += 1

        if total_distance > 0:
            local_ratio = count / total_distance
            norm_closeness = local_ratio / (1 / global_avg_interlayer_distance)
            interlayer_closeness[v] = norm_closeness
        else:
            interlayer_closeness[v] = 0

    return interlayer_closeness

def save_interlayer_closeness_to_csv(G_multilayer, closeness_dict, output_file):
    data = []

    for node, closeness in closeness_dict.items():
        layer = G_multilayer.nodes[node]['layer']
        utm_x = G_multilayer.nodes[node].get('utm_x', None)
        utm_y = G_multilayer.nodes[node].get('utm_y', None)
        data.append({'Node': node, 'Layer': layer, 'Closeness': closeness, 'UTM_X': utm_x, 'UTM_Y': utm_y})

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

interlayer_closeness = compute_interlayer_closeness_centrality(G_multilayer, all_shortest_paths)
save_interlayer_closeness_to_csv(G_multilayer, interlayer_closeness, "interlayer_closeness.csv")


