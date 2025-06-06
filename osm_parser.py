import osmnx as ox
import networkx as nx

def load_pedestrian_graph(city_name):
    """
    Загружает пешеходный граф из OSM для указанного города.

    Параметры:
    city_name (str): Название города для загрузки пешеходного графа.

    Возвращает:
    G_walk (networkx.Graph): Пешеходный граф для указанного города (не направленный и без параллельных рёбер)
                             с атрибутами длины рёбер и координатами узлов.
    """
    # Загрузка графа пешеходных дорог из OSM с дополнительной информацией о длине рёбер и координатах узлов
    G_osm = ox.graph_from_place(city_name, network_type='walk')

    # Преобразование графа в формат, подходящий для networkx
    G_walk = nx.Graph(G_osm)

    return G_walk

def convert_pedestrian_graph_coordinates_to_utm(G_pedestrian):
    """
    Пересчитывает координаты вершин пешеходного графа в UTM с заданной зоной.

    Параметры:
    - G_pedestrian (networkx.Graph): Пешеходный граф.
    - zone (int): Номер UTM зоны.

    Возвращает:
    - G_pedestrian_utm (networkx.Graph): Пешеходный граф с пересчитанными координатами в UTM.
    """
    G_pedestrian_utm = G_pedestrian.copy()  # Создаем копию пешеходного графа

    # Проходимся по всем вершинам пешеходного графа
    for node, data in G_pedestrian_utm.nodes(data=True):
        # Получаем текущие координаты вершины
        latitude = data['y']
        longitude = data['x']
        # Пересчитываем координаты в UTM
        utm_x, utm_y = convert_coordinates_to_utm_same_zone(latitude, longitude)
        # Обновляем атрибуты вершины с новыми координатами в UTM
        data['utm_x'] = utm_x
        data['utm_y'] = utm_y

    return G_pedestrian_utm
