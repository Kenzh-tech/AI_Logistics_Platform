"""
Аналитика перевозок и доходов.
Ответ на вопрос 6: Как проводится анализ по перевозкам и доходам?
"""
import pandas as pd
import numpy as np


class TransportAnalytics:
    """
    Анализ перевозок и финансовых показателей:
    - доходы по регионам/городам/маршрутам
    - расходы на доставку (оценка)
    - рентабельность
    - эффективность перевозок
    """
    
    FUEL_PRICE_PER_KM = 70     # тенге/км (топливо + амортизация)
    BASE_COST_PER_DELIVERY = 22500  # тенге на заказ (разгрузка, документы)
    
    @staticmethod
    def get_income_by_region(df):
        """Доходы по регионам."""
        if 'region' not in df.columns or 'sales' not in df.columns:
            return pd.DataFrame()
        agg = df.groupby('region').agg(
            revenue=('sales', 'sum'),
            orders=('sales', 'count'),
            profit=('profit', 'sum')
        ).reset_index()
        agg.columns = ['Регион', 'Выручка', 'Заказов', 'Прибыль']
        return agg.sort_values('Выручка', ascending=False)
    
    @staticmethod
    def get_income_by_city(df, top_n=15):
        """Доходы по городам доставки."""
        if 'city' not in df.columns:
            return pd.DataFrame()
        agg = df.groupby('city').agg(
            revenue=('sales', 'sum'),
            orders=('sales', 'count'),
        ).reset_index()
        agg.columns = ['Город', 'Выручка', 'Заказов']
        return agg.nlargest(top_n, 'Выручка')
    
    @staticmethod
    def get_transport_costs_estimate(df):
        """
        Оценка расходов на перевозки.
        Упрощённая модель: км × тариф + базовая стоимость заказа.
        """
        if 'days_real' not in df.columns and 'days_sched' not in df.columns:
            return 0, 0
        df = df.copy()
        days = df['days_real'] if 'days_real' in df.columns else df['days_sched']
        df['est_km'] = (days.astype(float) * 500).clip(100, 5000)
        total_km = df['est_km'].sum()
        total_cost = total_km * TransportAnalytics.FUEL_PRICE_PER_KM
        total_cost += len(df) * TransportAnalytics.BASE_COST_PER_DELIVERY
        return total_cost, total_km
    
    @staticmethod
    def get_profitability_summary(df):
        """Сводка: выручка, расходы, прибыль."""
        revenue = df['sales'].sum()
        costs, total_km = TransportAnalytics.get_transport_costs_estimate(df)
        profit = df['profit'].sum() if 'profit' in df.columns else revenue * 0.1
        return {
            'Выручка': revenue,
            'Расходы на перевозки (оценка)': costs,
            'Прибыль (из данных)': profit,
            'Пробег (оценка, км)': total_km,
            'Заказов': len(df),
        }
    
    @staticmethod
    def get_delivery_efficiency(df):
        """Эффективность доставки: своевременность, ошибки."""
        if 'delivery_error' not in df.columns:
            return {}
        on_time = (df['delivery_error'] <= 0).sum()
        late = (df['delivery_error'] > 0).sum()
        return {
            'Своевременных': int(on_time),
            'С опозданием': int(late),
            'Доля вовремя (%)': round(on_time / len(df) * 100, 1) if len(df) > 0 else 0,
            'Средняя ошибка (дней)': round(df['delivery_error'].mean(), 2),
        }
    
    @staticmethod
    def get_route_income_analysis(df, cities_in_route):
        """Анализ доходности по выбранному маршруту (города)."""
        if not cities_in_route:
            return {}
        subset = df[df['city'].isin(cities_in_route)]
        if subset.empty:
            return {}
        rev = subset['sales'].sum()
        ord_cnt = len(subset)
        return {
            'Выручка по маршруту': rev,
            'Заказов': ord_cnt,
            'Средний чек': round(rev / ord_cnt, 2) if ord_cnt > 0 else 0,
        }
