"""
Модуль прогнозирования риска опоздания доставки на основе ML.
Часть интеллектуальной системы управления логистическими процессами.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DeliveryRiskPredictor:
    """
    ML-модель для предсказания Late_delivery_risk.
    Использует: регион, режим доставки, категорию, запланированные дни и др.
    """
    
    _model = None
    _encoders = {}
    _metrics = {}
    
    @staticmethod
    def _prepare_features(df):
        """Подготовка признаков для классификации."""
        df = df.copy()
        
        # Числовые признаки
        numeric = ['days_sched', 'days_real', 'profit', 'sales']
        for c in numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # Категориальные -> кодирование
        cat_cols = ['region', 'mode', 'category', 'status']
        X = pd.DataFrame()
        
        for col in cat_cols:
            if col in df.columns and df[col].notna().any():
                vals = df[col].astype(str).fillna('unknown')
                if col in DeliveryRiskPredictor._encoders:
                    le = DeliveryRiskPredictor._encoders[col]
                    try:
                        X[col] = [le.transform([v])[0] if v in le.classes_ else 0 for v in vals]
                    except Exception:
                        X[col] = 0
                else:
                    le = LabelEncoder()
                    try:
                        X[col] = le.fit_transform(vals)
                        DeliveryRiskPredictor._encoders[col] = le
                    except Exception:
                        X[col] = 0
        
        for c in numeric:
            if c in df.columns:
                X[c] = df[c].values
        
        if 'risk' in df.columns:
            y = df['risk'].astype(int).values
        else:
            y = None
        
        return X, y
    
    @staticmethod
    def train(df):
        """Обучение модели на исторических данных."""
        X, y = DeliveryRiskPredictor._prepare_features(df)
        if y is None or len(np.unique(y)) < 2:
            DeliveryRiskPredictor._metrics = {'accuracy': 0, 'precision': 0, 'recall': 0}
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        DeliveryRiskPredictor._model = model
        DeliveryRiskPredictor._metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 1),
            'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
            'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
            'f1': round(f1_score(y_test, y_pred, zero_division=0) * 100, 1)
        }
        return True
    
    @staticmethod
    def get_metrics():
        """Возвращает метрики обученной модели."""
        return DeliveryRiskPredictor._metrics
    
    @staticmethod
    def predict_risk(df_subset):
        """Предсказание риска для набора заказов. Возвращает средний риск и детали."""
        if DeliveryRiskPredictor._model is None:
            return 0.0, []
        
        X, _ = DeliveryRiskPredictor._prepare_features(df_subset)
        if X.empty:
            return 0.0, []
        
        # Выравниваем колонки под обученную модель
        for c in DeliveryRiskPredictor._model.feature_names_in_:
            if c not in X.columns:
                X[c] = 0
        X = X[[c for c in DeliveryRiskPredictor._model.feature_names_in_ if c in X.columns]]
        
        preds = DeliveryRiskPredictor._model.predict_proba(X)
        risk_proba = preds[:, 1] if preds.shape[1] > 1 else preds[:, 0]
        
        return float(risk_proba.mean()), risk_proba.tolist()
    
    @staticmethod
    def predict_route_risk(route_cities, df):
        """Оценка риска опоздания для маршрута по городам."""
        if route_cities is None or len(route_cities) == 0:
            return 0.0
        subset = df[df['city'].isin(route_cities)]
        if subset.empty:
            return 0.0
        return DeliveryRiskPredictor.predict_risk(subset)[0]
