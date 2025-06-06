"""
Модуль для анализа транспортных сетей с использованием мультислойного подхода

Содержит инструменты для:
- Загрузки и обработки данных GTFS
- Извлечения пешеходной сети из OSM
- Построения мультислойного графа
- Расчета метрик интеграции транспортных сетей
"""
from .gtfs_parser import *
from .osm_parser import *
from .graph_builder import *
from .metrics import *
from .utils import *

__version__ = "1.0.0"
__author__ = "Solomatin Mikhail"