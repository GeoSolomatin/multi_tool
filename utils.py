from pyproj import Proj
import networkx as nx
import numpy as np

def convert_coordinates_to_utm(lat, lon):
    """
    Преобразует координаты широты и долготы в систему координат UTM.

    Параметры:
    - latitude (float): Широта в градусах.
    - longitude (float): Долгота в градусах.

    Возвращает:
    - utm_x (float): Координата X в системе UTM.
    - utm_y (float): Координата Y в системе UTM.
    """
    utm_zone = int((longitude + 180) / 6) + 1
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    utm_x, utm_y = utm_proj(longitude, latitude)
    return utm_x, utm_y

def check_graph_connectivity(graph):
    result = {}

    # Проверка на изолированные вершины
    isolated_nodes = list(nx.isolates(graph))
    if isolated_nodes:
        result['isolated_nodes'] = isolated_nodes
    else:
        result['isolated_nodes'] = None

    # Проверка на связность
    if nx.is_connected(graph):
        result['connectivity'] = 'Связный'
    else:
        result['connectivity'] = 'Не связный'
        components = list(nx.connected_components(graph))
        result['num_components'] = len(components)
        result['components_size'] = [len(comp) for comp in components]

    return result
