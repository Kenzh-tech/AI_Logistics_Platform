"""
Диспетчер заказов: приём заказов, согласование стоимости, отслеживание доставки.
"""
import pandas as pd
from datetime import datetime, timedelta
import routing_engine as re

try:
    import company_config as cfg
except ImportError:
    cfg = None

SPEED_KMH = 70  # Средняя скорость грузовика


def get_pending_orders(df, limit=15):
    """
    Заявки на доставку.

    Для диспетчера "заказ" представляем как агрегированную заявку по паре:
    (город назначения, товар).
    """
    if df.empty:
        return pd.DataFrame()

    city_product_orders = df.groupby(['city', 'product']).agg(
        sales=('sales', 'sum'),
        orders_count=('city', 'size'),
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        region=('region', 'first'),
        avg_days_sched=('days_sched', 'mean'),
        avg_days_real=('days_real', 'mean'),
        risk_rate=('risk', 'mean'),
        mode=('mode', 'first'),
    ).reset_index()

    city_product_orders = city_product_orders.nlargest(limit, 'sales')
    return city_product_orders


def estimate_delivery_time_hrs(km):
    """Оценка времени в часах (с учётом 10% задержек)."""
    return (km / SPEED_KMH) * 1.1


def get_truck_position(route_coords, progress_0_to_1):
    """
    Позиция грузовика на маршруте (0 = старт, 1 = конец).
    progress_0_to_1: от 0 до 1.
    """
    if not route_coords or len(route_coords) < 2:
        return None, None
    n = len(route_coords) - 1
    seg = progress_0_to_1 * n
    idx = min(int(seg), n - 1)
    t = seg - idx
    lat1, lon1 = route_coords[idx]['lat'], route_coords[idx]['lon']
    lat2, lon2 = route_coords[idx + 1]['lat'], route_coords[idx + 1]['lon']
    lat = lat1 + (lat2 - lat1) * t
    lon = lon1 + (lon2 - lon1) * t
    return lat, lon


def build_eta_table(route_df, cities_order, coords_list, start_time=None):
    """Таблица ETA для каждой точки маршрута."""
    if start_time is None:
        start_time = datetime.now()
    etas = []
    cumul_hrs = 0
    for i, row in route_df.iterrows():
        cumul_hrs += row['Время (ч)']
        eta = start_time + timedelta(hours=cumul_hrs)
        city = row['К'] if i < len(route_df) - 1 else cities_order[-1]
        etas.append({
            'Точка': row['К'],
            'Через (ч)': round(cumul_hrs, 1),
            'Прибытие': eta.strftime('%H:%M'),
        })
    return etas


def get_stop_status(route_df, cities_order, progress_0_to_1, eta_table):
    """
    Определяет текущую и следующую остановку на основе прогресса.
    route_df: таблица шагов маршрута (по строкам legs), где есть колонка 'Время (ч)'.
    cities_order: список городов маршрута (start + destinations).
    eta_table: результат build_eta_table (длина = len(route_df)).
    """
    if route_df is None or route_df.empty or not cities_order:
        return None

    # elapsed_hours зависит только от прогресса, так что статус стабильный при каждом ререндере
    total_hours = float(route_df['Время (ч)'].sum())
    elapsed = total_hours * float(progress_0_to_1)

    cumul = 0.0
    reached_legs = 0  # сколько legs успели завершить
    for leg_idx, leg_time in enumerate(route_df['Время (ч)'].tolist()):
        cumul += float(leg_time)
        if cumul <= elapsed + 1e-9:
            reached_legs = leg_idx + 1
        else:
            break

    # cities_order: [start, city1, city2, ...]
    current_city = cities_order[min(reached_legs, len(cities_order) - 1)]
    next_city = cities_order[reached_legs + 1] if reached_legs + 1 < len(cities_order) else None

    next_eta = None
    if next_city is not None and eta_table and reached_legs < len(eta_table):
        next_eta = eta_table[reached_legs].get('Прибытие')

    return {
        'current_city': current_city,
        'next_city': next_city,
        'next_eta': next_eta,
        'reached_legs': reached_legs,
        'elapsed_hours': round(elapsed, 2),
        'total_hours': round(total_hours, 2),
    }
