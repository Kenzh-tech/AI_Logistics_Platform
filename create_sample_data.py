"""Генерация компактного датасета для QazLogistics (Казахстан)."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

CITIES = [
    ("Алматы", 43.238949, 76.945465, "Южный КЗ"),
    ("Астана", 51.169392, 71.449074, "Северный КЗ"),
    ("Шымкент", 42.341686, 69.590101, "Южный КЗ"),
    ("Актобе", 50.2839, 57.1670, "Западный КЗ"),
    ("Караганда", 49.8047, 73.0856, "Центральный КЗ"),
    ("Павлодар", 52.2740, 76.9483, "Северный КЗ"),
    ("Семей", 50.4111, 80.2275, "Восточный КЗ"),
    ("Уральск", 51.2278, 51.3865, "Западный КЗ"),
    ("Тараз", 42.8983, 71.3660, "Южный КЗ"),
    ("Атырау", 47.1167, 51.9167, "Западный КЗ"),
    ("Костанай", 53.2144, 63.6246, "Северный КЗ"),
    ("Усть-Каменогорск", 49.9485, 82.6113, "Восточный КЗ"),
]

# Товар -> примерный диапазон цен в тенге (min, max) для заказа
# (оставляем только позиции, относящиеся к строительству)
PRODUCT_PRICES_KZT = {
    "Цемент (мешок 50кг)": (45000, 180000),      # партия 15-60 мешков
    "Арматура 12м": (80000, 350000),
    "Плитка керамическая": (60000, 400000),
    "Кабель электрический": (35000, 150000),
    "Перчатки х/б": (8000, 45000),
    "Рабочая одежда": (35000, 150000),
    "Обувь защитная": (25000, 120000),
    "Каска": (5000, 35000),
    "Дрель электрическая": (28000, 120000),
    "Болгарка": (35000, 180000),
    "Электроды сварочные": (20000, 95000),
}

PRODUCTS = list(PRODUCT_PRICES_KZT.keys())
MODES = ["Standard", "Express", "Same Day"]
STATUSES = ["Advance shipping", "Late delivery", "Shipping on time", "Shipping on time", "Shipping on time"]
# Т.к. нам нужна "строительная" специфика, категория фиксируется.
CATEGORIES = ["Стройматериалы"]

n = 700
start_date = datetime(2024, 1, 1)
rows = []
for i in range(n):
    city_info = CITIES[np.random.randint(0, len(CITIES))]
    city, lat, lon, region = city_info
    product = PRODUCTS[np.random.randint(0, len(PRODUCTS))]
    min_p, max_p = PRODUCT_PRICES_KZT[product]
    days_sched = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.15, 0.05])
    days_real = days_sched + np.random.choice([-1, 0, 0, 1, 2], p=[0.1, 0.5, 0.2, 0.15, 0.05])
    days_real = max(0, days_real)
    risk = 1 if days_real > days_sched else 0
    sales = round(np.random.uniform(min_p, max_p), 0)  # в тенге
    profit = round(sales * np.random.uniform(0.05, 0.25), 0)  # прибыль 5-25%
    cat = CATEGORIES[0]
    date = (start_date + timedelta(days=np.random.randint(0, 300))).strftime("%Y-%m-%d")
    mode = MODES[np.random.randint(0, len(MODES))]
    status = STATUSES[np.random.randint(0, len(STATUSES))]
    
    rows.append({
        "date": date, "sales": sales, "profit": profit,
        "days_real": days_real, "days_sched": days_sched, "risk": risk,
        "region": region, "category": cat, "city": city, "product": product,
        "lat": lat, "lon": lon, "status": status, "mode": mode
    })

df = pd.DataFrame(rows)
df.to_csv("data.csv", index=False, encoding="cp1251")
print(f"Создано {len(df)} записей. Уникальных городов: {df['city'].nunique()}, товаров: {df['product'].nunique()}")
