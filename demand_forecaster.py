"""
Модуль прогнозирования спроса на основе алгоритмов машинного обучения.
Тема диплома: прогнозирование спроса (Demand Forecasting) с использованием ML.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_HOLTWINTERS = True
except ImportError:
    HAS_HOLTWINTERS = False


class DemandForecaster:
    """
    Интеллектуальная система прогнозирования спроса.
    Сравнивает несколько ML-моделей и возвращает лучший прогноз с метриками.
    """
    
    @staticmethod
    def _prepare_ts_data(df, periods=14):
        """Подготовка временного ряда: сумма продаж по дням."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        daily = df.groupby('date')['sales'].sum().resample('D').sum()
        daily = daily.replace(0, np.nan).ffill().fillna(daily.mean() if daily.mean() > 0 else 100)
        if len(daily) < 14:
            last = daily.iloc[-1] if not daily.empty else 100
            daily = pd.Series([last * (1 + np.sin(i/3)*0.1) for i in range(30)], 
                             index=pd.date_range(start=df['date'].min(), periods=30, freq='D'))
        return daily
    
    @staticmethod
    def _prepare_features(daily, lag=7):
        """Создание признаков для регрессионных моделей."""
        X, y = [], []
        for i in range(lag, len(daily) - 1):
            X.append(list(daily.iloc[i-lag:i].values))
            y.append(daily.iloc[i])
        return np.array(X), np.array(y)
    
    @staticmethod
    def predict_sales(df, periods=14):
        """
        Прогноз спроса с использованием нескольких ML-моделей.
        Возвращает: (forecast, upper, lower, history, model_metrics)
        """
        try:
            daily = DemandForecaster._prepare_ts_data(df, periods)
            history = daily.tail(60)
            
            models_metrics = []
            forecasts = {}
            
            # --- 1. Linear Regression (на лагах) ---
            X, y = DemandForecaster._prepare_features(daily, lag=7)
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                lr = LinearRegression().fit(X_train, y_train)
                pred_lr = lr.predict(X_test)
                mae_lr = mean_absolute_error(y_test, pred_lr)
                rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
                models_metrics.append({'Модель': 'Linear Regression', 'MAE': round(mae_lr, 2), 'RMSE': round(rmse_lr, 2)})
                
                # Прогноз на periods дней (итеративно)
                last_vals = list(daily.tail(7).values)
                lr_forecast = []
                for _ in range(periods):
                    pred = lr.predict([last_vals])[0]
                    lr_forecast.append(max(0, pred))
                    last_vals = last_vals[1:] + [pred]
                forecasts['Linear Regression'] = pd.Series(lr_forecast, index=pd.date_range(start=daily.index[-1], periods=periods+1, freq='D')[1:])
            
            # --- 2. Random Forest ---
            if len(X) > 10:
                rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42).fit(X_train, y_train)
                pred_rf = rf.predict(X_test)
                mae_rf = mean_absolute_error(y_test, pred_rf)
                rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
                models_metrics.append({'Модель': 'Random Forest', 'MAE': round(mae_rf, 2), 'RMSE': round(rmse_rf, 2)})
                
                last_vals = list(daily.tail(7).values)
                rf_forecast = []
                for _ in range(periods):
                    pred = rf.predict([last_vals])[0]
                    rf_forecast.append(max(0, pred))
                    last_vals = last_vals[1:] + [pred]
                forecasts['Random Forest'] = pd.Series(rf_forecast, index=pd.date_range(start=daily.index[-1], periods=periods+1, freq='D')[1:])
            
            # --- 3. Holt-Winters (временной ряд) ---
            if HAS_HOLTWINTERS and len(daily) >= 14:
                try:
                    model_hw = ExponentialSmoothing(daily, trend='add', seasonal_periods=min(7, len(daily)//2), 
                                                    damped_trend=True).fit()
                    hw_forecast = model_hw.forecast(periods)
                    # Оценка на последних данных
                    train_size = int(len(daily) * 0.8)
                    train_d = daily.iloc[:train_size]
                    test_d = daily.iloc[train_size:]
                    if len(test_d) > 0:
                        m_hw = ExponentialSmoothing(train_d, trend='add', damped_trend=True).fit()
                        pred_hw = m_hw.forecast(len(test_d))
                        mae_hw = mean_absolute_error(test_d, pred_hw[:len(test_d)])
                        rmse_hw = np.sqrt(mean_squared_error(test_d, pred_hw[:len(test_d)]))
                        models_metrics.append({'Модель': 'Holt-Winters', 'MAE': round(mae_hw, 2), 'RMSE': round(rmse_hw, 2)})
                        forecasts['Holt-Winters'] = hw_forecast
                except Exception:
                    pass
            
            # Выбор лучшей модели по MAE
            if not models_metrics:
                # Fallback: наивный прогноз (последнее значение)
                last_val = daily.iloc[-1]
                forecast = pd.Series([last_val] * periods, index=pd.date_range(start=daily.index[-1], periods=periods+1, freq='D')[1:])
                models_metrics = [{'Модель': 'Naive', 'MAE': 0, 'RMSE': 0}]
            else:
                best_model = min(models_metrics, key=lambda x: x['MAE'])
                best_name = best_model['Модель']
                forecast = forecasts.get(best_name, list(forecasts.values())[0] if forecasts else pd.Series())
            
            # Доверительный интервал
            std = daily.std() if daily.std() > 0 else daily.mean() * 0.15
            upper = forecast + (std * 1.96)
            lower = (forecast - (std * 1.96)).clip(lower=0)
            
            return forecast, upper, lower, history, models_metrics
            
        except Exception as e:
            return pd.Series(), pd.Series(), pd.Series(), pd.Series(), [{'Модель': 'Ошибка', 'MAE': 0, 'RMSE': 0}]
    
    @staticmethod
    def forecast_by_region(df, region, periods=7):
        """Прогноз спроса по региону (для логистического планирования)."""
        region_df = df[df['region'] == region] if 'region' in df.columns else df
        if region_df.empty:
            return pd.Series(), 0
        daily = DemandForecaster._prepare_ts_data(region_df, periods)
        if len(daily) < 7:
            return pd.Series(), daily.sum()
        try:
            if HAS_HOLTWINTERS:
                model = ExponentialSmoothing(daily, trend='add', damped_trend=True).fit()
                fc = model.forecast(periods)
                return fc, daily.iloc[-1]
        except Exception:
            pass
        last = daily.iloc[-1]
        return pd.Series([last] * periods), last
    
    @staticmethod
    def forecast_by_product(df, product, periods=7):
        """
        Прогноз спроса по товару (для управления запасами).

        Важно: в InventoryManager "текущий остаток" считается как количество записей
        (партии/заказы), поэтому и прогноз должен быть в тех же единицах, а не в тенге.
        """
        if 'product' not in df.columns or 'date' not in df.columns:
            return 0

        prod_df = df[df['product'] == product].copy()
        if prod_df.empty:
            return 0

        prod_df['date'] = pd.to_datetime(prod_df['date'], errors='coerce')
        prod_df = prod_df.dropna(subset=['date'])
        if prod_df.empty:
            return 0

        # Спрос = количество заказов в день (а не сумма продаж в тенге)
        daily_cnt = prod_df.groupby('date').size().sort_index()
        daily_cnt = daily_cnt.resample('D').sum()
        if len(daily_cnt) == 0:
            return 0

        # Заполняем пропуски и сглаживаем нулями/пустотой
        daily_cnt = daily_cnt.replace(0, np.nan).ffill().fillna(daily_cnt.mean() if daily_cnt.mean() > 0 else 1)

        # Прогноз на periods дней: считаем дневной спрос средним по последним дням
        tail = daily_cnt.tail(periods) if len(daily_cnt) >= periods else daily_cnt
        daily_demand = max(1.0, float(tail.mean()))
        return daily_demand * float(periods)
