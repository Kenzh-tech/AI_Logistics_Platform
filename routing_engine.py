import pandas as pd
import numpy as np

class RoutingEngine:
    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        # Формула гаверсинуса для точного расчета км по координатам из БД
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    @staticmethod
    def get_optimal_route(df, start_city):
        if start_city not in df['city'].values:
            return None
        
        start_point = df[df['city'] == start_city].iloc[0]
        
        # Берем уникальные города из ВАШЕЙ базы данных
        all_cities = df[df['city'] != start_city].drop_duplicates('city').copy()
        
        routes = []
        for _, row in all_cities.iterrows():
            dist = RoutingEngine.calculate_distance(
                start_point['lat'], start_point['lon'],
                row['lat'], row['lon']
            )
            
            # Фильтр: берем только те страны/города из БД, которые в радиусе 4000 км
            # Это исключит межконтинентальные маршруты, сократив время
            if 10 < dist < 4000:
                # Скорость 70 км/ч + 10% задержек
                travel_time = (dist / 70) * np.random.uniform(1.05, 1.15)
                
                routes.append({
                    'Назначение': row['city'],
                    'Регион': row['region'],
                    'Дистанция (км)': round(dist, 1),
                    'Время (ч)': round(travel_time, 1),
                    'Расход (л)': round(dist * 0.31, 1),
                    'Приоритет': 'Оптимально' if dist < 1500 else 'Дальний рейс'
                })
        
        # Сортируем по близости и берем топ-8 для красивого графика
        return sorted(routes, key=lambda x: x['Дистанция (км)'])[:8]