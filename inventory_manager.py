"""
Модуль управления складскими запасами с учётом прогноза спроса.
Связывает прогнозирование (ML) с логистикой пополнения.
"""
import pandas as pd


class InventoryManager:
    """
    Интеллектуальное управление запасами: остатки + рекомендации пополнения
    на основе прогноза спроса (demand forecasting).
    """
    
    # Параметры для расчёта точки заказа
    LEAD_DAYS = 5      # Средний срок поставки (дней)
    SAFETY_FACTOR = 1.5  # Коэффициент страхового запаса
    
    @staticmethod
    def get_inventory_status(df):
        """
        Текущие остатки по товарам + статус (OK / Low / Deficit).
        Использует данные из БД (имитация склада из sales count).
        """
        stock = df.groupby('product')['sales'].count().reset_index()
        stock.columns = ['Тауар (Product)', 'Саны (Qty)']
        
        def check_status(qty):
            if qty > 50: return "🟢 OK"
            elif qty > 20: return "🟡 Low"
            else: return "🔴 Deficit"
        
        stock['Статус'] = stock['Саны (Qty)'].apply(check_status)
        return stock
    
    @staticmethod
    def get_reorder_recommendations(df, forecast_by_product_fn, inventory_df=None):
        """
        Рекомендации по пополнению на основе прогноза спроса.
        
        Логика: Точка заказа = (дневной спрос × срок поставки) + страховой запас
        Если текущий остаток < точка заказа → рекомендовать заказ.
        
        forecast_by_product_fn: функция(product) -> прогноз спроса на 7 дней
        """
        if inventory_df is None:
            inventory_df = InventoryManager.get_inventory_status(df)
        
        products = df['product'].unique()[:50]  # Топ-50 товаров
        recommendations = []
        
        for product in products:
            current = inventory_df[inventory_df['Тауар (Product)'] == product]['Саны (Qty)']
            current_qty = int(current.values[0]) if len(current) > 0 else 0
            
            # Прогноз спроса на 7 дней (число)
            predicted_demand = forecast_by_product_fn(df, product, 7)
            daily_demand = max(1, float(predicted_demand) / 7)
            
            # Точка заказа (reorder point)
            reorder_point = (daily_demand * InventoryManager.LEAD_DAYS) + \
                            (InventoryManager.SAFETY_FACTOR * (daily_demand ** 0.5))
            reorder_point = int(max(10, reorder_point))
            
            # Рекомендуемый объём заказа (EOQ-упрощённо: на 2 недели)
            order_qty = max(0, int(daily_demand * 14) - current_qty)
            
            if current_qty < reorder_point and order_qty > 0:
                recommendations.append({
                    'Товар': product[:40] + ('...' if len(str(product)) > 40 else ''),
                    'ProductKey': product,
                    'Текущий остаток': current_qty,
                    'Точка заказа': reorder_point,
                    'Рекомендуемый заказ': order_qty,
                    'Прогноз спроса (7 дн)': round(predicted_demand, 0)
                })
        
        return pd.DataFrame(recommendations).head(15)

    @staticmethod
    def get_inventory_by_warehouse(df, warehouses, min_stock_per_product=5):
        """
        Упрощённая модель остатков по товарам в разрезе складов.

        Входные данные из demo-датасета не содержат "склад → товар", поэтому
        мы имитируем распределение остатков по складам:
        считаем, что на складе в городе X лежит запас товара, пропорциональный
        числу записей (заказов) этого товара, где город назначения == X.
        """
        if df is None or df.empty:
            return {}
        if not warehouses:
            return {}

        warehouse_cities = [w["city"] for w in warehouses if "city" in w]
        if not warehouse_cities:
            return {}

        # Глобальный "спрос/партии" по товарам в датасете
        global_product_counts = df.groupby("product").size().to_dict()

        # Локальные "партии" по складам (город назначения совпадает с городом склада)
        local_city_product_counts = (
            df.groupby(["city", "product"]).size().reset_index(name="qty")
        )

        # Удобная структура для быстрого доступа
        local_map = {}
        for _, r in local_city_product_counts.iterrows():
            local_map.setdefault(r["city"], {})[r["product"]] = int(r["qty"])

        inv = {city: {} for city in warehouse_cities}
        num_wh = len(warehouse_cities)

        for product, total_qty in global_product_counts.items():
            # Сколько "партии" товара приходится на города складов
            total_on_wh = sum(local_map.get(city, {}).get(product, 0) for city in warehouse_cities)

            for city in warehouse_cities:
                local_qty = local_map.get(city, {}).get(product, 0)
                if total_on_wh > 0:
                    share = local_qty / total_on_wh
                    qty = int(round(total_qty * share))
                else:
                    qty = int(round(total_qty / num_wh))

                # Чтобы пользователь мог реально "принять" хотя бы немного
                if qty <= 0:
                    qty = int(min_stock_per_product)

                inv[city][product] = int(qty)

        return inv
