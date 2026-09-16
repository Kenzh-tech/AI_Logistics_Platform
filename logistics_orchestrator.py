"""
Оркестратор логистических процессов: объединяет маршрутизацию и оптимизацию.
Тема диплома: управление логистическими процессами на основе ML.
"""
import pandas as pd
import numpy as np
import routing_engine as re
from route_optimizer import calculate_route as ortools_route

try:
    import company_config as cfg
except ImportError:
    cfg = None


class LogisticsOrchestrator:
    """
    Интеллектуальная система планирования маршрутов доставки.
    Использует OR-Tools для оптимизации (задача коммивояжёра).
    """
    
    @staticmethod
    def build_distance_matrix(cities_df):
        """
        cities_df: DataFrame с колонками city, lat, lon (индекс = порядок города)
        Возвращает матрицу расстояний в км.
        """
        n = len(cities_df)
        matrix = np.zeros((n, n))
        coords = list(zip(cities_df['lat'], cities_df['lon']))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = int(re.RoutingEngine.calculate_distance(
                        coords[i][0], coords[i][1], coords[j][0], coords[j][1]
                    ))
                else:
                    matrix[i][j] = 0
        return matrix
    
    @staticmethod
    def get_optimal_delivery_route(df, start_city, max_cities=10):
        """
        Оптимальный маршрут доставки от склада (start_city) по городам из БД.
        Использует алгоритм TSP (OR-Tools) для минимизации пробега.
        start_city может быть складом компании (Алматы, Астана, Шымкент) или городом из БД.
        
        Возвращает: (route_df, total_km, cities_order)
        """
        all_cities = LogisticsOrchestrator.get_cities_for_routing(df, limit=max_cities + 10)
        if start_city not in all_cities['city'].values and start_city not in df['city'].values:
            return None, 0, []
        
        # Стартовый город (склад или город из БД) + города назначения из БД (исключая склады без заказов)
        db_cities = df.drop_duplicates('city')[['city', 'lat', 'lon']].dropna(subset=['lat', 'lon'])
        start_match = all_cities[all_cities['city'] == start_city]
        if start_match.empty:
            start_match = db_cities[db_cities['city'] == start_city]
        if start_match.empty:
            return None, 0, []
        start_row = start_match.iloc[0][['city', 'lat', 'lon']]
        others = db_cities[db_cities['city'] != start_city].head(max_cities - 1)
        cities_subset = pd.concat([pd.DataFrame([start_row]), others.reset_index(drop=True)], ignore_index=True)
        
        if len(cities_subset) < 2:
            return re.RoutingEngine.get_optimal_route(df, start_city), 0, [start_city]
        
        dist_matrix = LogisticsOrchestrator.build_distance_matrix(cities_subset)
        
        try:
            plan = ortools_route(dist_matrix.tolist())
        except Exception:
            plan = list(range(len(cities_subset)))
        
        if plan is None:
            plan = list(range(len(cities_subset)))
        
        # Собираем маршрут по порядку
        route_cities = []
        total_km = 0
        prev_idx = plan[0]
        
        for idx in plan[1:]:
            from_city = cities_subset.iloc[prev_idx]
            to_city = cities_subset.iloc[idx]
            dist = re.RoutingEngine.calculate_distance(
                from_city['lat'], from_city['lon'],
                to_city['lat'], to_city['lon']
            )
            route_cities.append({
                'Этап': len(route_cities) + 1,
                'От': from_city['city'],
                'К': to_city['city'],
                'Дистанция (км)': round(dist, 1),
                'Время (ч)': round(dist / 70 * 1.1, 1),
            })
            total_km += dist
            prev_idx = idx
        
        cities_order = [cities_subset.iloc[i]['city'] for i in plan]
        route_df = pd.DataFrame(route_cities)
        
        return route_df, round(total_km, 1), cities_order
    
    @staticmethod
    def get_cities_for_routing(df, limit=15, include_warehouses=True):
        """Список городов для выбора: склады компании + топ по объёму продаж."""
        rows = []
        if include_warehouses and cfg and hasattr(cfg, 'WAREHOUSES'):
            for w in cfg.WAREHOUSES['locations']:
                rows.append({'city': w['city'], 'lat': w['lat'], 'lon': w['lon'], 'sales': 0})
        city_sales = df.groupby('city').agg({
            'sales': 'sum',
            'lat': 'first',
            'lon': 'first'
        }).reset_index()
        city_sales = city_sales.dropna(subset=['lat', 'lon'])
        city_sales = city_sales[~city_sales['city'].isin([r['city'] for r in rows])]
        combined = pd.concat([pd.DataFrame(rows), city_sales.nlargest(limit - len(rows), 'sales')], ignore_index=True)
        return combined

    @staticmethod
    def get_route_from_accepted_orders(accepted_orders, start_city, df, return_to_start: bool = True):
        """
        Маршрут из принятых заказов (список dict с city, lat, lon).
        start_city — склад. Возвращает (route_df, total_km, cities_order, coords).
        """
        if not accepted_orders:
            return None, 0, [], []
        start_coords = None
        if cfg and hasattr(cfg, 'WAREHOUSES'):
            for w in cfg.WAREHOUSES['locations']:
                if w['city'] == start_city:
                    start_coords = {'city': w['city'], 'lat': w['lat'], 'lon': w['lon']}
                    break
        if start_coords is None:
            match = df[df['city'] == start_city][['city', 'lat', 'lon']].drop_duplicates()
            if not match.empty:
                start_coords = match.iloc[0].to_dict()
        if start_coords is None:
            start_coords = {'city': accepted_orders[0]['city'], 'lat': accepted_orders[0]['lat'], 'lon': accepted_orders[0]['lon']}
        
        others = [{"city": o["city"], "lat": o["lat"], "lon": o["lon"]} for o in accepted_orders if o["city"] != start_coords["city"]]
        cities_df = pd.DataFrame([start_coords] + others).drop_duplicates("city")
        if len(cities_df) < 2:
            return None, 0, cities_df["city"].tolist(), cities_df[["lat", "lon"]].to_dict("records")
        
        dist_matrix = LogisticsOrchestrator.build_distance_matrix(cities_df)
        try:
            plan = ortools_route(dist_matrix.tolist())
        except Exception:
            plan = list(range(len(cities_df)))
        if plan is None:
            plan = list(range(len(cities_df)))
        
        route_cities = []
        total_km = 0
        prev_idx = plan[0]
        for idx in plan[1:]:
            from_c = cities_df.iloc[prev_idx]
            to_c = cities_df.iloc[idx]
            dist = re.RoutingEngine.calculate_distance(from_c['lat'], from_c['lon'], to_c['lat'], to_c['lon'])
            route_cities.append({"Этап": len(route_cities)+1, "От": from_c['city'], "К": to_c['city'], "Дистанция (км)": round(dist,1), "Время (ч)": round(dist/70*1.1, 1)})
            total_km += dist
            prev_idx = idx
        
        cities_order = cities_df.iloc[plan]["city"].tolist()
        coords = [{"lat": cities_df.iloc[i]["lat"], "lon": cities_df.iloc[i]["lon"]} for i in plan]

        # Возврат на склад после доставки: добавляем финальный leg обратно в start_city
        if return_to_start and cities_order and cities_order[-1] != start_coords["city"]:
            last_coords = coords[-1]
            dist = re.RoutingEngine.calculate_distance(
                float(last_coords["lat"]), float(last_coords["lon"]),
                float(start_coords["lat"]), float(start_coords["lon"])
            )
            route_cities.append({
                "Этап": len(route_cities) + 1,
                "От": cities_order[-1],
                "К": start_coords["city"],
                "Дистанция (км)": round(dist, 1),
                "Время (ч)": round(dist / 70 * 1.1, 1),
            })
            total_km += dist
            cities_order.append(start_coords["city"])
            coords.append({"lat": start_coords["lat"], "lon": start_coords["lon"]})

        return pd.DataFrame(route_cities), round(total_km, 1), cities_order, coords
