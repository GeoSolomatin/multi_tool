import os
import pandas as pd
import networkx as nx
from pyproj import Proj

def load_gtfs_data(gtfs_folder):
    """
    Загружает данные GTFS из папки.
    
    Параметры:
    ----------
    gtfs_folder : str
        Путь к папке с файлами GTFS
        
    Возвращает:
    -----------
    dict
        Словарь с DataFrame для каждого файла GTFS
    """
    gtfs_data = {}
    files = ['agency.txt', 'routes.txt', 'trips.txt', 'stops.txt', 'stop_times.txt']
    for file in files:
        gtfs_data[file[:-4]] = pd.read_csv(os.path.join(gtfs_folder, file))
    return gtfs_data

def build_transport_graphs(gtfs_data):
    """
    Строит графы для всех типов транспорта на основе данных GTFS.

    Параметры:
    - gtfs_data (dict): словарь с данными GTFS. Ключи словаря:
        * 'routes': DataFrame Pandas с информацией о маршрутах.
        * 'trips': DataFrame Pandas с информацией о поездках.
        * 'stop_times': DataFrame Pandas с информацией об остановках.

    Возвращает:
    - transport_stops (dict): словарь с информацией об остановках транспорта. Ключи словаря:
        * ID остановки (строка): информация остановки.
    - transport_graphs (dict): словарь с графами транспорта. Ключи словаря:
        * Тип транспорта (строка): граф транспорта.
    """
    transport_stops = {}
    transport_edges = {}
    transport_graphs = {}

    # Проходимся по всем маршрутам
    for _, route in gtfs_data['routes'].iterrows():
        route_id = route['route_id']
        route_type = str(route['route_type'])  # Преобразуем в строку

        # Получаем поездки для текущего маршрута
        trips = gtfs_data['trips'][gtfs_data['trips']['route_id'] == route_id]

        # Создаем словарь для текущего типа транспорта, если его нет
        if route_type not in transport_edges:
            transport_edges[route_type] = []

        # Проходимся по всем поездкам для текущего маршрута
        for _, trip in trips.iterrows():
            trip_id = trip['trip_id']

            # Получаем время остановок для текущей поездки
            stop_times = gtfs_data['stop_times'][gtfs_data['stop_times']['trip_id'] == trip_id]

            # Проходимся по всем остановкам для текущей поездки
            prev_stop = None
            for _, stop_time in stop_times.iterrows():
                stop_id = stop_time['stop_id']

                # Добавляем остановку в словарь, если ее нет
                if stop_id not in transport_stops:
                    transport_stops[stop_id] = {'transport_type': route_type}

                # Добавляем ребро между предыдущей и текущей остановкой
                if prev_stop is not None:
                    transport_edges[route_type].append((prev_stop, stop_id, {'route_type': route_type}))  # Используем route_type как строку

                prev_stop = stop_id

    # Создаем графы для каждого типа транспорта
    for transport_type, edges in transport_edges.items():
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Добавляем атрибуты utm_x и utm_y для каждой вершины
        for node in G.nodes():
            if 'utm_x' not in G.nodes[node]:
                G.nodes[node]['utm_x'] = 0
            if 'utm_y' not in G.nodes[node]:
                G.nodes[node]['utm_y'] = 0
            if 'layer' not in G.nodes[node]:
                G.nodes[node]['layer'] = transport_type
        
        # Добавляем атрибуты layer и weight для каждого ребра
        for edge in G.edges(data=True):
            if 'layer' not in edge[2]:
                edge[2]['layer'] = edge[2]['route_type']
            if 'weight' not in edge[2]:
                edge[2]['weight'] = 0
        
        transport_graphs[transport_type] = G

    return transport_stops, transport_graphs
    
def get_stops_with_coordinates(gtfs_folder):
    """
    Получение информации о всех остановках с их координатами в UTM и принадлежности к типам транспорта.

    Параметры:
    - gtfs_folder (str): Путь к папке с файлами GTFS.

    Возвращает:
    - stops_info (dict): Словарь, содержащий информацию об остановках. Ключи - идентификаторы остановок, значения - словари с информацией о координатах в UTM и типах транспорта.
    - num_stops (int): Количество остановок.
    """
    # Загрузка данных GTFS
    gtfs_data = load_gtfs_data(gtfs_folder)
    
    # Загрузка координат остановок
    stops_df = gtfs_data['stops']
    routes_df = gtfs_data['routes']
    stop_times_df = gtfs_data['stop_times']

    # Создание словаря для хранения остановок с их координатами и типами транспорта
    stops_info = {}

    # Проход по каждой остановке
    for _, stop in stops_df.iterrows():
        note_id = stop['stop_id']  # Изменяем наименование поля на note_id
        stop_lat = stop['stop_lat']
        stop_lon = stop['stop_lon']
        
        # Определение маршрутов, проходящих через эту остановку
        routes = stop_times_df[stop_times_df['stop_id'] == note_id]['trip_id'].unique()
        route_types = routes_df[routes_df['route_id'].isin(gtfs_data['trips'][gtfs_data['trips']['trip_id'].isin(routes)]['route_id'])]['route_type'].unique().tolist()
        
        # Преобразование координат в UTM
        utm_x, utm_y = convert_coordinates_to_utm(stop_lat, stop_lon)
        
        # Преобразование route_type в строку
        route_types_str = ','.join(map(str, route_types))
        
        # Добавление информации об остановке в словарь
        stops_info[note_id] = {'utm_x': utm_x, 'utm_y': utm_y, 'route_type': route_types_str}  # Изменяем stop_id на note_id
        
    return stops_info, len(stops_info)

