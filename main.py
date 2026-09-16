"""
Интеллектуальная информационная система управления логистическими процессами
и прогнозирования спроса на основе алгоритмов машинного обучения.

Тема диплома: «Разработка интеллектуальной информационной системы управления
логистическими процессами и прогнозирования спроса на основе алгоритмов машинного обучения»
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import os
from datetime import datetime

import database_handler as db
import demand_forecaster as forecaster
import inventory_manager as inv_mgr
import risk_predictor as risk_mdl
import logistics_orchestrator as log_orch
import company_config as cfg
import data_info as dbinfo
import transport_analytics as ta
import order_dispatcher as disp
import osrm_routing
import json

# --- 1. АВТОРИЗАЦИЯ ---
def ensure_login():
    if "user" in st.session_state and st.session_state["user"]:
        return True

    st.markdown("<h1 style='text-align: center; color: white;'>AI Logistics Login</h1>", unsafe_allow_html=True)
    st.caption("Войдите в личный кабинет по логину/паролю (роль задается в базе).")

    col_l, col_m, col_r = st.columns([1, 1.5, 1])
    with col_m:
        st.text_input("Логин", key="login_username")
        st.text_input("Құпия сөз", type="password", key="login_password")
        if st.button("Кіру / Вход", use_container_width=True):
            u = db.authenticate_user(st.session_state.get("login_username", ""), st.session_state.get("login_password", ""))
            if u:
                st.session_state["user"] = u
                # очистка полей формы
                del st.session_state["login_password"]
                del st.session_state["login_username"]
                st.rerun()
            else:
                st.error("Неверный логин или пароль.")
    return False


if not ensure_login():
    st.stop()

# --- 2. ТЕХНИЧЕСКИЕ ФУНКЦИИ ---
def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"CSS қатесі: {e}")

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))) * 1.15

st.set_page_config(layout="wide", page_title="AI Logistics Platform", page_icon="AI")
local_css("style.css")

# --- 3. ЗАГРУЗКА ДАННЫХ ---
df = db.get_data("SELECT * FROM main_table")
if df.empty:
    st.info("Таблица `main_table` пуста. Выполняю синхронизацию...")
    db.init_db()
    df = db.get_data("SELECT * FROM main_table")
    if df.empty:
        st.error("Деректер базасы бос! Синхронизация не помогла.")
        st.stop()

# --- ФИЛЬТР ТОВАРОВ: ТОЛЬКО СТРОИТЕЛЬСТВО ---
# На случай, если БД уже заполнена старыми данными и пользователь ещё не сделал Sync.
if "product" in df.columns and hasattr(cfg, "CONSTRUCTION_PRODUCTS"):
    df = df[df["product"].isin(cfg.CONSTRUCTION_PRODUCTS)].copy()

# Инициализация ML-модели риска (один раз за сессию)
if "risk_trained" not in st.session_state:
    risk_mdl.DeliveryRiskPredictor.train(df)
    st.session_state["risk_trained"] = True

# --- 4. HEADER ---
c_title, c_sync = st.columns([4, 1])
with c_title:
    st.markdown(f"<h1>{cfg.COMPANY['name']} — Интеллектуальная система управления логистикой</h1>", unsafe_allow_html=True)
with c_sync:
    if st.button("СИНХРОНИЗАЦИЯ", use_container_width=True):
        db.init_db()
        st.rerun()
    if st.button("Logout / Шығу"):
        st.session_state.pop("user", None)
        st.rerun()

# --- SIDEBAR: Профиль компании (ответы на вопросы 1–5) ---
with st.sidebar:
    st.markdown("### Профиль компании")
    st.markdown(f"**{cfg.COMPANY['full_name']}**")
    st.caption(cfg.COMPANY['description'])
    st.divider()
    st.markdown("**5. Парк и склады**")
    st.metric("Грузовиков", cfg.FLEET["trucks_total"])
    st.metric("Складов", cfg.WAREHOUSES["count"])
    for w in cfg.WAREHOUSES["locations"]:
        st.caption(f"• {w['city']} ({w['type']})")
    st.divider()
    st.markdown("**3–4. Регион и валюта**")
    st.caption(f"Регион: {cfg.REGION['description']}")
    use_kzt = st.checkbox("Показывать в тенге", value=True, key="use_kzt")
    if use_kzt:
        cfg.CURRENCY["display"] = "KZT"
    else:
        cfg.CURRENCY["display"] = "USD"
    st.divider()
    st.markdown("**1. База данных**")
    st.caption(f"{dbinfo.DATABASE_INFO['type']}")
    st.caption(f"Источник: {dbinfo.DATABASE_INFO['source']}")
    st.caption(f"Записей: ~{dbinfo.DATABASE_INFO['records_count']}")
    st.markdown("**2. Номенклатура**")
    st.caption(dbinfo.PRODUCT_INFO['description'])

# KPI: реальные метрики из ML
risk_metrics = risk_mdl.DeliveryRiskPredictor.get_metrics()
model_acc = risk_metrics.get("accuracy", 0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Жалпы айналым", cfg.format_money(df['sales'].sum()))
k2.metric("Тәуекел деңгейі (факт)", f"{(df['risk'].mean() * 100):.1f}%")
k3.metric("ML: дәлдік (риск)", f"{model_acc}%")
k4.metric("Активті бағыттар", len(df['city'].unique()))

st.divider()

current_user = st.session_state.get("user", {}) or {}
role = current_user.get("role", "dispatcher")
user_id = int(current_user.get("user_id", 0)) if current_user.get("user_id") is not None else 0

# --- КАБИНЕТЫ НЕ ДИСПЕТЧЕРА ---
if role == "customer":
    st.markdown("## Личный кабинет заказчика")
    st.caption(f"Пользователь: {current_user.get('full_name', current_user.get('username',''))}")

    warehouse_cities = [w["city"] for w in cfg.WAREHOUSES["locations"]]
    warehouse_city = st.selectbox("Склад отправления", warehouse_cities, index=0 if warehouse_cities else 0)
    wh = next(w for w in cfg.WAREHOUSES["locations"] if w["city"] == warehouse_city)
    wh_lat, wh_lon = float(wh["lat"]), float(wh["lon"])

    destination_city = st.selectbox("Место доставки (город)", sorted(df["city"].dropna().unique().tolist()))
    dest_row = df[df["city"] == destination_city].iloc[0]
    dest_lat, dest_lon = float(dest_row["lat"]), float(dest_row["lon"])

    construction_products = sorted(list(cfg.CONSTRUCTION_PRODUCTS)) if hasattr(cfg, "CONSTRUCTION_PRODUCTS") else sorted(df["product"].dropna().unique().tolist())
    product = st.selectbox("Товар", construction_products, index=0 if construction_products else 0)

    qty = st.number_input("Количество (партий/единиц)", min_value=1, value=1, step=1)

    # Предполагаем риск и предлагаемую цену по той же формуле, что в диспетчере
    dist_km = calculate_distance(wh_lat, wh_lon, dest_lat, dest_lon)
    risk_subset = df[(df["city"] == destination_city) & (df["product"] == product)]
    risk_rate = float(risk_subset["risk"].mean()) if not risk_subset.empty and "risk" in risk_subset.columns else 0.0
    base = cfg.COST_MODEL["base_cost_kzt"] + dist_km * cfg.COST_MODEL["cost_per_km_kzt"]
    suggested = int(base * cfg.COST_MODEL["markup_factor"] * (1.0 + cfg.COST_MODEL["risk_surcharge_factor"] * risk_rate))

    proposed_price = st.number_input(
        "Предложить цену (тенге)",
        min_value=0.0,
        value=float(suggested),
        step=1000.0,
        format="%.0f",
    )

    st.info(f"Расчётная рекомендованная цена: {cfg.format_money(suggested)} (риск: {risk_rate:.2f})")

    if st.button("Отправить предложение диспетчеру"):
        offer_id = db.create_customer_offer(
            customer_user_id=user_id,
            warehouse_city=warehouse_city,
            destination_city=destination_city,
            destination_lat=dest_lat,
            destination_lon=dest_lon,
            product=product,
            qty=int(qty),
            proposed_price=float(proposed_price),
            suggested_price=float(suggested),
            risk_rate=float(risk_rate),
        )
        if offer_id:
            st.success(f"Предложение отправлено (id={offer_id}).")
            st.rerun()
        else:
            st.error("Не удалось сохранить предложение.")

    st.divider()
    st.markdown("### Ваши предложения")
    my_offers = db.get_offers_for_customer(user_id)
    if my_offers.empty:
        st.caption("Пока предложений нет.")
    else:
        st.dataframe(
            my_offers[
                [
                    "offer_id",
                    "warehouse_city",
                    "destination_city",
                    "product",
                    "qty",
                    "proposed_price",
                    "risk_rate",
                    "status",
                    "created_at",
                    "delivery_id",
                    "completed_at",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.stop()

if role == "driver":
    st.markdown("## Личный кабинет водителя")
    st.caption(f"Пользователь: {current_user.get('full_name', current_user.get('username',''))}")

    deliveries = db.get_deliveries_by_driver(user_id)
    if deliveries.empty:
        st.info("Активных рейсов пока нет.")
        st.stop()

    delivery_ids = deliveries["delivery_id"].astype(int).tolist()
    delivery_choice = st.selectbox("Выберите активный рейс", delivery_ids, index=0)
    drow = deliveries[deliveries["delivery_id"] == delivery_choice].iloc[0]

    route_cities = []
    coords_list = []
    try:
        route_cities = json.loads(drow["route_cities_json"]) if drow["route_cities_json"] else []
    except Exception:
        route_cities = []
    try:
        coords_list = json.loads(drow["coords_json"]) if drow["coords_json"] else []
    except Exception:
        coords_list = []

    total_duration_h = float(drow.get("total_duration_h") or 0.0)
    start_time_txt = drow.get("start_time")
    try:
        start_time_dt = datetime.fromisoformat(str(start_time_txt))
    except Exception:
        start_time_dt = datetime.now()

    elapsed_h = (datetime.now() - start_time_dt).total_seconds() / 3600.0
    prog = min(1.0, max(0.0, elapsed_h / total_duration_h)) if total_duration_h > 0 else 0.0

    st.metric("Прогресс рейса", f"{prog*100:.0f}%")
    st.caption(f"Маршрут городов: {', '.join(route_cities)}")

    if coords_list and len(coords_list) >= 2:
        # Позиция грузовика на полилинии
        lat_t, lon_t = disp.get_truck_position(coords_list, prog)
        poly_df = pd.DataFrame(coords_list)

        fig = go.Figure()
        fig.add_trace(
            go.Scattermapbox(
                lat=poly_df["lat"],
                lon=poly_df["lon"],
                mode="lines",
                name="Маршрут",
                line=dict(width=4, color="#00d4ff"),
            )
        )
        if lat_t is not None and lon_t is not None:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[lat_t],
                    lon=[lon_t],
                    mode="markers",
                    name="Грузовик",
                    marker=dict(size=22, color="red"),
                )
            )
        center = coords_list[0] if coords_list else {"lat": 43.0, "lon": 76.0}
        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=float(center["lat"]), lon=float(center["lon"])),
                zoom=4,
            ),
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.stop()

# --- 5. ОСНОВНЫЕ ВКЛАДКИ (диспетчер) ---
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Диспетчер (заказы, карта, грузовик)",
    "Данные",
    "Прогноз спроса (ML)",
    "Маршруты",
    "Склад и пополнение",
    "Аналитика перевозок и доходов",
])

# Инициализация состояния диспетчера
if "accepted_orders" not in st.session_state:
    st.session_state["accepted_orders"] = []  # [{city, product, qty, agreed_price, lat, lon, sales}, ...]
if "active_deliveries" not in st.session_state:
    st.session_state["active_deliveries"] = []  # list of active deliveries (one per truck)
if "sim_time_hours" not in st.session_state:
    # Глобальная симуляция времени; прогресс каждого рейса считается относительно своего start_sim_time
    st.session_state["sim_time_hours"] = 0.0
if "truck_progress" not in st.session_state:
    # Legacy key (оставляем, чтобы не ломать код/сохранённые сессии), дальше можно убрать
    st.session_state["truck_progress"] = 0.0

if "inventory_by_warehouse" not in st.session_state:
    # Сначала пытаемся прочитать реальные остатки из БД (warehouse_inventory),
    # если таблицы ещё нет — используем демо-модель.
    try:
        inv_df = db.get_warehouse_inventory()
        if (
            inv_df is not None
            and not inv_df.empty
            and {"warehouse_city", "product", "qty"}.issubset(inv_df.columns)
        ):
            inv_map = {}
            allowed_products = getattr(cfg, "CONSTRUCTION_PRODUCTS", None)
            for _, r in inv_df.iterrows():
                product = r["product"]
                if allowed_products is not None and product not in allowed_products:
                    continue
                inv_map.setdefault(r["warehouse_city"], {})[product] = int(r["qty"])
            st.session_state["inventory_by_warehouse"] = inv_map
        else:
            raise ValueError("warehouse_inventory пустой/неподходящий формат")
    except Exception:
        st.session_state["inventory_by_warehouse"] = inv_mgr.InventoryManager.get_inventory_by_warehouse(
            df,
            cfg.WAREHOUSES.get("locations", []),
        )

# --- TAB 1: ДИСПЕТЧЕР (приём заказов, карта, грузовик, согласование стоимости) ---
with t1:
    st.markdown("Диспетчер: приём заказов и отслеживание грузовика")
    st.caption("Процесс: заявки → согласование цены → построение маршрута → движение и ETA.")

    # Зимний период: предупреждение диспетчеру (для реалистичности рисков/ETA)
    now = datetime.now()
    if now.month in (11, 12, 1, 2, 3):
        st.warning("Зимний период: учитывайте гололёд и возможные задержки. Рекомендуем закладывать запас по времени.")

    proposed_offers_df = db.get_offers("proposed")
    accepted_offers_df = db.get_offers("accepted")
    # Структура, которую использует LogisticsOrchestrator.get_route_from_accepted_orders()
    ac = []
    for _, r in accepted_offers_df.iterrows():
        ac.append(
            {
                "offer_id": int(r["offer_id"]),
                "city": r["destination_city"],
                "product": r["product"],
                "qty": int(r["qty"]),
                "risk_rate": float(r.get("risk_rate", 0.0) or 0.0),
                "agreed_price": float(r["proposed_price"]),
                "suggested_price": float(r.get("suggested_price", 0.0) or 0.0),
                "warehouse_city": r["warehouse_city"],
                "lat": float(r["destination_lat"]),
                "lon": float(r["destination_lon"]),
                "sales": 0.0,
            }
        )

    active_deliveries = st.session_state.get("active_deliveries", [])
    active = active_deliveries[0] if active_deliveries else None

    # === Секция 1: Предложения клиентов (ввод цены уже сделал клиент) ===
    st.markdown("Предложения клиентов на доставку")

    warehouse_cities = [w["city"] for w in cfg.WAREHOUSES["locations"]]
    warehouse_city = st.selectbox("Склад отправления", warehouse_cities, index=0)
    wh = next(w for w in cfg.WAREHOUSES["locations"] if w["city"] == warehouse_city)
    wh_lat, wh_lon = float(wh["lat"]), float(wh["lon"])
    ac_wh = [a for a in ac if a.get("warehouse_city") == warehouse_city]

    for _, row in proposed_offers_df.iterrows():
        if row.get("warehouse_city") != warehouse_city:
            continue

        offer_id = int(row["offer_id"])
        product = row["product"]
        qty_needed = int(row["qty"])

        # Если уже принят — не показываем повторно
        if any(a.get("offer_id") == offer_id for a in ac_wh):
            continue

        stock_qty = int(st.session_state.get("inventory_by_warehouse", {}).get(warehouse_city, {}).get(product, 0))
        reserved_qty = sum(int(a.get("qty", 0)) for a in ac_wh if a.get("product") == product)
        remaining_qty = max(0, stock_qty - reserved_qty)

        col_btn, col_info = st.columns([1, 4])
        with col_info:
            dist_km = calculate_distance(wh_lat, wh_lon, float(row["destination_lat"]), float(row["destination_lon"]))
            risk_pct = float(row.get("risk_rate", 0.0) or 0.0) * 100.0
            st.write(
                f"Клиент: {row.get('customer_full_name', '')} ({row.get('customer_username','')})\n"
                f"Город: {row['destination_city']}\n"
                f"Товар: {product}\n"
                f"Количество: {qty_needed}\n"
                f"Остаток на складе: {stock_qty} | Доступно: {remaining_qty}\n"
                f"Предложение: {cfg.format_money(float(row['proposed_price']))}\n"
                f"Расстояние: {dist_km:,.0f} км\n"
                f"Риск задержки (факт): {risk_pct:.1f}%"
            )

        with col_btn:
            if remaining_qty < qty_needed:
                st.caption(f"Нет остатка: нужно {qty_needed}, доступно {remaining_qty}")
            else:
                if st.button("Принять", key=f"accept_offer_{offer_id}"):
                    ok = db.accept_offer(offer_id=offer_id, dispatcher_user_id=user_id)
                    if ok:
                        st.rerun()
                    else:
                        st.error("Не удалось принять предложение.")

    # === Секция 2: Принятые заказы + запуск рейса ===
    st.markdown("Принятые заказы")
    if ac_wh:
        for i, o in enumerate(ac_wh):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(
                    f"{o['city']} | {o.get('product', '')}\n"
                    f"Партии: {o.get('qty', 0)} | Сумма: {cfg.format_money(o['agreed_price'])}"
                )
            with c2:
                if st.button("Отклонить", key=f"reject_{o['offer_id']}"):
                    if db.reject_offer(int(o["offer_id"])):
                        st.rerun()
        total_agreed = sum(o["agreed_price"] for o in ac_wh)
        st.metric("Сумма согласованных заказов", cfg.format_money(total_agreed))

        if len(ac_wh) >= 1 and st.button("Запустить рейсы (построить маршруты)"):
            # Проверяем, хватает ли запаса на все принятые товары
            inv_wh = st.session_state.get("inventory_by_warehouse", {}).get(warehouse_city, {})
            need_per_product = {}
            for o in ac_wh:
                need_per_product[o["product"]] = int(
                    need_per_product.get(o["product"], 0) + int(o.get("qty", 0))
                )

            if any(int(inv_wh.get(p, 0)) < int(q) for p, q in need_per_product.items()):
                st.error("Недостаточно остатка на складе для запуска рейса.")
            else:
                # Авто-распределение по грузовикам:
                # "Простые" отправляем 1 машиной, "сложные" (много заявок/риск) - несколькими.
                truck_counts = db.get_truck_counts()
                free_trucks = int(truck_counts.get("free", 0))
                if free_trucks <= 0:
                    st.error("Нет свободных грузовиков для запуска рейсов.")
                else:
                    avg_risk = float(np.mean([float(o.get("risk_rate", 0.0)) for o in ac_wh])) if ac_wh else 0.0
                    complexity = float(len(ac_wh)) + avg_risk * 2.0
                    trucks_to_use = int(math.ceil(complexity / 3.0))
                    trucks_to_use = max(1, min(trucks_to_use, free_trucks, len(ac_wh)))

                    groups = [[] for _ in range(trucks_to_use)]
                    for i, o in enumerate(ac_wh):
                        groups[i % trucks_to_use].append(o)

                    dispatched_any = False
                    start_sim_time = float(st.session_state.get("sim_time_hours", 0.0))

                    for group in groups:
                        if not group:
                            continue

                        # Считаем нужды по продуктам в группе
                        need_per_group = {}
                        for o in group:
                            need_per_group[o["product"]] = int(need_per_group.get(o["product"], 0)) + int(o.get("qty", 0))

                        truck_id, driver_id = db.reserve_truck(current_city=warehouse_city)
                        if truck_id is None:
                            continue

                        # Списываем товар со склада под этот грузовик
                        inv_map = st.session_state.setdefault("inventory_by_warehouse", {})
                        inv_for_wh = inv_map.setdefault(warehouse_city, {})
                        for product, qty in need_per_group.items():
                            inv_for_wh[product] = int(inv_for_wh.get(product, 0)) - int(qty)
                            db.adjust_warehouse_inventory(warehouse_city, product, -int(qty))

                        route_df, total_km, cities_order, coords_list = log_orch.LogisticsOrchestrator.get_route_from_accepted_orders(
                            group,
                            warehouse_city,
                            df,
                            return_to_start=True,
                        )

                        if route_df is None or route_df.empty:
                            # На всякий случай откатываем резерв и списание, если рейс не построился.
                            try:
                                db.release_truck(int(truck_id))
                            except Exception:
                                pass
                            for product, qty in need_per_group.items():
                                inv_for_wh[product] = int(inv_for_wh.get(product, 0)) + int(qty)
                                db.adjust_warehouse_inventory(warehouse_city, product, int(qty))
                            continue

                        if route_df is not None and not route_df.empty:
                            # Суммируем выручку по городам (для отображения)
                            prices = {}
                            for o in group:
                                prices[o["city"]] = float(prices.get(o["city"], 0.0)) + float(o["agreed_price"])

                            # Делаем роутинг по реальным дорогам (OSRM)
                            total_duration_h = float(route_df["Время (ч)"].sum()) if "Время (ч)" in route_df.columns else 0.0
                            route_polyline = None
                            route_cumdist_km = None
                            try:
                                (
                                    route_polyline_candidate,
                                    total_dist_osrm,
                                    total_duration_osrm,
                                    leg_dist_km,
                                    leg_duration_h,
                                ) = osrm_routing.build_osrm_polyline_for_waypoints(coords_list)

                                if route_polyline_candidate and len(leg_duration_h) == len(route_df):
                                    route_df = route_df.copy()
                                    route_df["Время (ч)"] = [round(float(x), 1) for x in leg_duration_h]
                                    route_df["Дистанция (км)"] = [round(float(x), 1) for x in leg_dist_km]
                                    total_duration_h = float(sum(leg_duration_h))
                                    total_km = float(sum(leg_dist_km))
                                    route_polyline = route_polyline_candidate
                                    route_cumdist_km = osrm_routing.build_cumulative_distances_km(route_polyline_candidate)
                                else:
                                    route_polyline = route_polyline_candidate
                            except Exception:
                                # Если OSRM недоступен/лимит, оставляем "квази-роутинг" из route_df
                                pass

                            delivery_start_time = datetime.now()
                            delivery_id = db.create_delivery(
                                dispatcher_user_id=user_id,
                                truck_id=int(truck_id),
                                driver_id=int(driver_id) if driver_id is not None else None,
                                warehouse_city=warehouse_city,
                                route_cities=cities_order,
                                coords_list=coords_list,
                                total_km=float(total_km),
                                total_duration_h=float(total_duration_h),
                                start_time=delivery_start_time,
                            )
                            if delivery_id is None:
                                try:
                                    db.release_truck(int(truck_id))
                                except Exception:
                                    pass
                                continue

                            offer_ids = [int(o["offer_id"]) for o in group if o.get("offer_id") is not None]
                            db.mark_offers_dispatched(offer_ids, int(delivery_id))

                            st.session_state["active_deliveries"].append(
                                {
                                    "route_df": route_df,
                                    "cities_order": cities_order,
                                    "coords": coords_list,
                                    "total_km": total_km,
                                    "prices": prices,
                                    "warehouse_city": warehouse_city,
                                    "start_time": delivery_start_time,
                                    "truck_id": truck_id,
                                    "driver_id": driver_id,
                                    "delivery_id": delivery_id,
                                    "offer_ids": offer_ids,
                                    "replenished": False,
                                    "goods_loaded": True,
                                    "total_duration_h": total_duration_h,
                                    "route_polyline": route_polyline,
                                    "route_cumdist_km": route_cumdist_km,
                                    "start_sim_time": start_sim_time,
                                }
                            )
                            dispatched_any = True

                    if not dispatched_any:
                        st.error("Не удалось сформировать ни одного рейса.")
                    else:
                        st.rerun()
    else:
        route_res = st.session_state.get("route_result", None)
        if route_res and route_res[0] is not None and not route_res[0].empty:
            if st.button("Использовать рассчитанный маршрут из вкладки «Маршруты»"):
                route_df, total_km, cities_order = route_res
                all_cities = log_orch.LogisticsOrchestrator.get_cities_for_routing(df, 50)
                coords_list = []
                for c in cities_order:
                    r = all_cities[all_cities["city"] == c]
                    if not r.empty:
                        coords_list.append({"lat": r.iloc[0]["lat"], "lon": r.iloc[0]["lon"]})
                    else:
                        r2 = df[df["city"] == c][["lat", "lon"]].drop_duplicates()
                        if not r2.empty:
                            coords_list.append({"lat": r2.iloc[0]["lat"], "lon": r2.iloc[0]["lon"]})
                if len(coords_list) >= 2:
                    total_duration_h = float(route_df["Время (ч)"].sum()) if "Время (ч)" in route_df.columns else 0.0
                    st.session_state["active_deliveries"] = [
                        {
                            "route_df": route_df,
                            "cities_order": cities_order,
                            "coords": coords_list,
                            "total_km": total_km,
                            "prices": {},
                            "warehouse_city": warehouse_city,
                            "start_time": datetime.now(),
                            "truck_id": None,
                            "replenished": True,
                            "goods_loaded": False,
                            "total_duration_h": total_duration_h,
                            "route_polyline": None,
                            "route_cumdist_km": None,
                            "start_sim_time": float(st.session_state.get("sim_time_hours", 0.0)),
                        }
                    ]
                    st.rerun()
        if not active_deliveries:
            st.info("Нет активных рейсов. Примите заявки выше или используйте маршрут во вкладке «Маршруты».")

    # === Секция 3: Активные рейсы — карта с грузовиками, ETA ===
    if active_deliveries:
        st.markdown("Грузовики в пути")

        # Диапазон слайдера времени: берём максимум среди активных рейсов
        max_duration_h = 0.1
        for d in active_deliveries:
            td = d.get("total_duration_h")
            if td is None:
                rf = d.get("route_df")
                if rf is not None and not rf.empty and "Время (ч)" in rf.columns:
                    td = float(rf["Время (ч)"].sum())
            if td is not None:
                max_duration_h = max(max_duration_h, float(td))

        st.metric("Активных рейсов", len(active_deliveries))

        # Сколько грузовиков свободно/занято
        try:
            truck_counts = db.get_truck_counts()
            col_tc1, col_tc2 = st.columns(2)
            col_tc1.metric("Свободно", f"{truck_counts.get('free', 0)}")
            col_tc2.metric("В рейсе/занято", f"{truck_counts.get('busy', 0)}")
        except Exception:
            pass

        # Управление общей симуляцией
        sim_time_hours = st.slider(
            "Симуляция времени (ч)",
            0.0,
            max_duration_h,
            float(st.session_state.get("sim_time_hours", 0.0)),
            step=0.05,
        )
        st.session_state["sim_time_hours"] = sim_time_hours

        col_speed1, col_speed2, col_speed3 = st.columns(3)
        with col_speed1:
            if st.button("Ускорить (+15 мин)"):
                st.session_state["sim_time_hours"] = min(max_duration_h, sim_time_hours + 0.25)
                st.rerun()
        with col_speed2:
            if st.button("Ускорить (+30 мин)"):
                st.session_state["sim_time_hours"] = min(max_duration_h, sim_time_hours + 0.5)
                st.rerun()
        with col_speed3:
            if st.button("Назад ( -15 мин)"):
                st.session_state["sim_time_hours"] = max(0.0, sim_time_hours - 0.25)
                st.rerun()

        # Общая карта: один Figure с маркерами на каждый грузовик
        fig = go.Figure()
        marker_positions = []

        for idx, d in enumerate(active_deliveries):
            route_df = d.get("route_df")
            cities_order = d.get("cities_order", [])
            waypoints = d.get("coords", [])
            total_km = float(d.get("total_km", 0.0))

            total_duration_h = d.get("total_duration_h")
            if total_duration_h is None:
                total_duration_h = float(route_df["Время (ч)"].sum()) if route_df is not None and not route_df.empty and "Время (ч)" in route_df.columns else 0.0

            start_sim_time = float(d.get("start_sim_time", 0.0))
            elapsed = sim_time_hours - start_sim_time
            prog = max(0.0, elapsed / float(total_duration_h)) if float(total_duration_h) > 0 else 0.0
            prog = min(1.0, prog)

            route_polyline = d.get("route_polyline")
            route_cumdist_km = d.get("route_cumdist_km")

            # Позиция грузовика в текущий момент
            lat_t, lon_t = None, None
            if route_polyline and route_cumdist_km:
                target_km = total_km * prog
                pos = osrm_routing.point_on_polyline_by_distance_km(route_polyline, route_cumdist_km, target_km)
                lat_t, lon_t = pos["lat"], pos["lon"]
                poly_df = pd.DataFrame(route_polyline)
                fig.add_trace(
                    go.Scattermapbox(
                        lat=poly_df["lat"],
                        lon=poly_df["lon"],
                        mode="lines",
                        name=f"Маршрут грузовика {d.get('truck_id', idx+1)}",
                        line=dict(width=3, color="#00d4ff"),
                        opacity=0.6,
                    )
                )
            else:
                lat_t, lon_t = disp.get_truck_position(waypoints, prog)
                if lat_t is not None and waypoints:
                    map_df = pd.DataFrame(waypoints)
                    fig.add_trace(
                        go.Scattermapbox(
                            lat=map_df["lat"],
                            lon=map_df["lon"],
                            mode="lines",
                            name=f"Маршрут грузовика {d.get('truck_id', idx+1)}",
                            line=dict(width=3, color="#00d4ff"),
                            opacity=0.6,
                        )
                    )

            if lat_t is not None and lon_t is not None:
                marker_positions.append((lat_t, lon_t))

                fig.add_trace(
                    go.Scattermapbox(
                        lat=[lat_t],
                        lon=[lon_t],
                        mode="markers",
                        name=f"Грузовик {d.get('truck_id', idx+1)} ({prog*100:.0f}%)",
                        marker=dict(size=20 + idx * 2, color="red"),
                    )
                )

            # Быстрый ETA/остановка под рейс
            if route_df is not None and waypoints and cities_order:
                eta_table = disp.build_eta_table(
                    route_df,
                    cities_order,
                    waypoints,
                    start_time=d.get("start_time", datetime.now()),
                )
                status = disp.get_stop_status(route_df, cities_order, prog, eta_table)
                if status:
                    st.write(
                        f"Грузовик {d.get('truck_id', idx+1)}: текущая остановка {status.get('current_city')} "
                        f"(следующая: {status.get('next_city')})"
                    )

        # Настройка отображения карты
        center = marker_positions[0] if marker_positions else (43.0, 76.0)
        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=center[0], lon=center[1]),
                zoom=4,
            ),
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Завершение рейсов и освобождение грузовиков
        to_remove = []
        for idx, d in enumerate(active_deliveries):
            total_duration_h = float(d.get("total_duration_h") or 0.0)
            if total_duration_h <= 0:
                continue
            start_sim_time = float(d.get("start_sim_time", 0.0))
            elapsed = sim_time_hours - start_sim_time
            prog = max(0.0, min(1.0, elapsed / total_duration_h))

            # Пользовательское завершение
            truck_id = d.get("truck_id")
            finish_key = f"finish_{truck_id}_{idx}"
            if st.button("Завершить рейс", key=finish_key):
                delivery_id = d.get("delivery_id")
                if delivery_id is not None:
                    try:
                        db.mark_delivery_completed(int(delivery_id))
                    except Exception:
                        pass
                if truck_id is not None:
                    try:
                        db.release_truck(int(truck_id))
                    except Exception:
                        pass
                to_remove.append(idx)
                continue

            if prog >= 0.999 and d.get("goods_loaded", False) and not d.get("replenished", False):
                delivery_id = d.get("delivery_id")
                warehouse_city = d.get("warehouse_city")
                inv_map = st.session_state.get("inventory_by_warehouse", {})
                inv_for_wh = inv_map.get(warehouse_city, {})

                # Пополнение (по текущей логике автопополнения из старого кода)
                inventory_df = pd.DataFrame(
                    [{"Тауар (Product)": p, "Саны (Qty)": int(q)} for p, q in inv_for_wh.items()]
                )
                reorder_df = inv_mgr.InventoryManager.get_reorder_recommendations(
                    df,
                    forecast_by_product_fn=lambda d2, p, n: forecaster.DemandForecaster.forecast_by_product(d2, p, n),
                    inventory_df=inventory_df,
                )
                added_count = 0
                if not reorder_df.empty and "Рекомендуемый заказ" in reorder_df.columns:
                    for _, r in reorder_df.iterrows():
                        pk = r.get("ProductKey", None)
                        qty_add = int(r.get("Рекомендуемый заказ", 0))
                        if pk and qty_add > 0:
                            inv_for_wh[pk] = int(inv_for_wh.get(pk, 0)) + qty_add
                            db.adjust_warehouse_inventory(warehouse_city, pk, qty_add)
                            added_count += 1

                if truck_id is not None:
                    try:
                        db.release_truck(int(truck_id))
                    except Exception:
                        pass

                if delivery_id is not None:
                    try:
                        db.mark_delivery_completed(int(delivery_id))
                    except Exception:
                        pass

                st.success(f"Рейс грузовика {truck_id} завершён. Пополнение позиций: {added_count}.")
                to_remove.append(idx)

        # Удаляем завершённые рейсы
        if to_remove:
            for ridx in sorted(set(to_remove), reverse=True):
                active_deliveries.pop(ridx)
            st.session_state["active_deliveries"] = active_deliveries
            st.rerun()

# --- TAB 2: ДАННЫЕ ---
with t2:
    st.markdown("Данные заказов (демо)")
    st.dataframe(df.head(250), use_container_width=True, height=420)
    st.divider()
    st.markdown("### Выполненные заказы (из базы)")
    try:
        completed = db.get_data(
            """
            SELECT
                co.*,
                u.username AS customer_username,
                u.full_name AS customer_full_name
            FROM completed_orders co
            JOIN users u ON u.user_id = co.customer_user_id
            ORDER BY co.completed_at DESC
            """
        )
    except Exception:
        completed = pd.DataFrame()
    if completed.empty:
        st.caption("Пока выполненных заказов нет.")
    else:
        st.dataframe(completed, use_container_width=True, hide_index=True)

# --- TAB 3: ПРОГНОЗ СПРОСА (ML) ---
with t3:
    st.markdown("Прогнозирование спроса на основе машинного обучения")
    result = forecaster.DemandForecaster.predict_sales(df)
    if len(result) == 5:
        f_val, upper, lower, h_data, model_metrics = result
    else:
        f_val, upper, lower, h_data = result[0], result[1], result[2], result[3]
        model_metrics = [{'Модель': 'Holt-Winters', 'MAE': 0, 'RMSE': 0}]

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        fig_f = go.Figure()
        if not h_data.empty:
            fig_f.add_trace(go.Scatter(x=h_data.index, y=h_data.values, name="История", line=dict(color='#00d4ff')))
        if not f_val.empty:
            fig_f.add_trace(go.Scatter(x=f_val.index, y=f_val.values, name="Прогноз", line=dict(dash='dot', color='#ff3b30')))
            if not upper.empty and not lower.empty:
                fig_f.add_trace(go.Scatter(x=f_val.index, y=upper.values, name="Верхняя граница", line=dict(dash='dash', color='rgba(255,59,48,0.4)')))
                fig_f.add_trace(go.Scatter(x=f_val.index, y=lower.values, name="Нижняя граница", line=dict(dash='dash', color='rgba(255,59,48,0.4)'), fill='tonexty'))
        fig_f.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=400, title="Прогноз общего спроса (выбрана лучшая ML-модель по MAE)")
        st.plotly_chart(fig_f, use_container_width=True)

    with col_f2:
        st.markdown("**Сравнение ML-моделей (реальные метрики)**")
        if model_metrics:
            ml_df = pd.DataFrame(model_metrics)
            # Относительный MAE: чем меньше — тем лучше (нормализуем для графика)
            if 'MAE' in ml_df.columns and ml_df['MAE'].max() > 0:
                min_mae = ml_df['MAE'].min()
                ml_df['Оценка (1/MAE норм.)'] = (min_mae / ml_df['MAE'] * 100).clip(0, 100).astype(int)
                st.plotly_chart(px.bar(ml_df, x='Модель', y='Оценка (1/MAE норм.)', color='Модель', 
                    title="Сравнение моделей (выше = лучше)", template="plotly_dark"), use_container_width=True)
            st.dataframe(ml_df[['Модель', 'MAE', 'RMSE']], use_container_width=True, hide_index=True)
        st.caption("Выбирается модель с минимальным MAE. Linear Regression, Random Forest, Holt-Winters.")

# --- TAB 4: ЛОГИСТИКА ---
with t4:
    st.markdown("Оптимизация маршрутов доставки (OR-Tools TSP)")

    cities_df = log_orch.LogisticsOrchestrator.get_cities_for_routing(df, limit=20)
    city_list = cities_df['city'].tolist() if not cities_df.empty else []

    col_sel, col_opt = st.columns([1, 2])
    with col_sel:
        start_city = st.selectbox("Склад / стартовый город (Алматы, Астана, Шымкент или город из БД)", city_list, index=0 if city_list else 0)
        max_cities = st.slider("Макс. городов в маршруте", 3, 12, 8)
        if st.button("🔀 Рассчитать оптимальный маршрут"):
            route_df, total_km, cities_order = log_orch.LogisticsOrchestrator.get_optimal_delivery_route(df, start_city, max_cities)
            st.session_state["route_result"] = (route_df, total_km, cities_order)

    route_result = st.session_state.get("route_result", None)
    if route_result:
        route_df, total_km, cities_order = route_result
        if route_df is not None and not route_df.empty:
            st.metric("Общая дистанция маршрута", f"{total_km:,.0f} км")
            st.dataframe(route_df, use_container_width=True)

            # Оценка риска опоздания для маршрута (ML)
            risk_val = risk_mdl.DeliveryRiskPredictor.predict_route_risk(cities_order, df)
            st.metric("ML: прогноз риска опоздания по маршруту", f"{risk_val * 100:.1f}%")

            # Карта маршрута (OSRM по дорогам)
            city_coords = df[df["city"].isin(cities_order)].drop_duplicates("city")[
                ["city", "lat", "lon"]
            ]
            waypoints = []
            for c in cities_order:
                row = city_coords[city_coords["city"] == c]
                if not row.empty:
                    waypoints.append({"lat": float(row.iloc[0]["lat"]), "lon": float(row.iloc[0]["lon"])})

            if len(waypoints) >= 2:
                try:
                    (
                        route_polyline,
                        total_dist_osrm,
                        total_duration_osrm,
                        leg_dist_km,
                        leg_duration_h,
                    ) = osrm_routing.build_osrm_polyline_for_waypoints(waypoints)

                    # Пытаемся обновить route_df по времени/дистанции, если размер совпадает
                    if route_polyline and len(leg_duration_h) == len(route_df):
                        route_df = route_df.copy()
                        route_df["Время (ч)"] = [round(float(x), 1) for x in leg_duration_h]
                        route_df["Дистанция (км)"] = [round(float(x), 1) for x in leg_dist_km]
                        st.session_state["route_result"] = (route_df, float(sum(leg_dist_km)), cities_order)

                    poly_df = pd.DataFrame(route_polyline)
                    fig_map = go.Figure()
                    fig_map.add_trace(
                        go.Scattermapbox(
                            lat=poly_df["lat"],
                            lon=poly_df["lon"],
                            mode="lines",
                            name="Маршрут (OSRM)",
                            line=dict(width=4, color="#00d4ff"),
                        )
                    )
                    # отметки по ключевым точкам (склад/города)
                    key_df = pd.DataFrame(waypoints)
                    fig_map.add_trace(
                        go.Scattermapbox(
                            lat=key_df["lat"],
                            lon=key_df["lon"],
                            mode="markers",
                            name="Точки",
                            marker=dict(size=10, color="#ffffff"),
                        )
                    )
                    fig_map.update_layout(
                        mapbox=dict(style="carto-darkmatter", zoom=2, center=key_df.iloc[0][["lat", "lon"]].to_dict()),
                        height=350,
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception:
                    # fallback: старая линия по точкам
                    map_rows = waypoints
                    map_df = pd.DataFrame(map_rows) if map_rows else None
                    if map_df is not None and len(map_df) >= 2:
                        fig_map = px.line_mapbox(map_df, lat="lat", lon="lon", zoom=2, mapbox_style="carto-darkmatter")
                        fig_map.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Маршрут не построен. Проверьте наличие городов с координатами.")

    # Гео-калькулятор (дополнительно)
    with st.expander("Калькулятор расстояния между точками"):
        ca, cb, cp = st.columns(3)
        with ca:
            la = st.number_input("Lat A:", value=51.16, key="la")
            oa = st.number_input("Lon A:", value=71.42, key="oa")
        with cb:
            lb = st.number_input("Lat B:", value=43.22, key="lb")
            ob = st.number_input("Lon B:", value=76.85, key="ob")
        with cp:
            fuel = st.slider("Тариф (тенге/км):", 50, 200, 100, key="fuel")
            dist = calculate_distance(la, oa, lb, ob)
            cost_estimate = int(dist * fuel + 20000)  # тенге
            st.metric("Расстояние:", f"{dist:,.0f} км")
            st.metric("Оценка стоимости:", cfg.format_money(cost_estimate))

# --- TAB 5: СКЛАД ---
with t5:
    st.markdown("Управление запасами и рекомендации пополнения")

    # Пытаемся взять реальные qty по складам из БД (warehouse_inventory).
    # Если таблицы ещё нет — используем демо-оценку.
    try:
        inv_wh = db.get_warehouse_inventory()
        if inv_wh is not None and not inv_wh.empty and {"product", "qty"}.issubset(inv_wh.columns):
            stock_df = (
                inv_wh.groupby("product")["qty"]
                .sum()
                .reset_index()
                .rename(columns={"product": "Тауар (Product)", "qty": "Саны (Qty)"})
            )
        else:
            raise ValueError("warehouse_inventory пустой/неподходящий формат")
    except Exception:
        stock_df = inv_mgr.InventoryManager.get_inventory_status(df)
    try:
        fig_stock = px.bar(stock_df.head(15), x='Тауар (Product)', y='Саны (Qty)',
                           color='Саны (Qty)', title="Текущие остатки на складе", template="plotly_dark")
        st.plotly_chart(fig_stock, use_container_width=True)
    except Exception:
        st.warning("График: неверный формат данных.")

    st.markdown("#### Рекомендации по пополнению (на основе прогноза спроса ML)")
    reorder_df = inv_mgr.InventoryManager.get_reorder_recommendations(
        df,
        forecast_by_product_fn=lambda d, p, n: forecaster.DemandForecaster.forecast_by_product(d, p, n),
        inventory_df=stock_df
    )
    if not reorder_df.empty:
        st.dataframe(reorder_df, use_container_width=True)
        st.caption(
            "Товары с остатком ниже точки заказа. Рекомендуемый объём рассчитан по прогнозу спроса "
            "в тех же единицах, что и остаток (партии/заказы)."
        )
    else:
        st.success("Все товары в норме. Критичных рекомендаций по пополнению нет.")

    st.dataframe(stock_df, use_container_width=True)

# --- TAB 6: АНАЛИТИКА ПЕРЕВОЗОК И ДОХОДОВ ---
with t6:
    st.markdown("Анализ перевозок и доходов")
    st.caption("Как проводится анализ: выручка по регионам/городам, оценка расходов на перевозки, эффективность доставки.")

    summary = ta.TransportAnalytics.get_profitability_summary(df)
    eff = ta.TransportAnalytics.get_delivery_efficiency(df)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Выручка", cfg.format_money(summary['Выручка']))
        st.metric("Прибыль (из данных)", cfg.format_money(summary['Прибыль (из данных)']))
    with col_s2:
        st.metric("Расходы на перевозки (оценка)", cfg.format_money(summary['Расходы на перевозки (оценка)']))
        st.metric("Заказов", f"{summary['Заказов']:,}")
    with col_s3:
        st.metric("Пробег (оценка)", f"{summary['Пробег (оценка, км)']:,.0f} км")
        if eff:
            st.metric("Доставок вовремя", f"{eff.get('Доля вовремя (%)', 0)}%")

    st.markdown("#### Доходы по регионам")
    inc_region = ta.TransportAnalytics.get_income_by_region(df)
    if not inc_region.empty:
        inc_region_display = inc_region.copy()
        inc_region_display['Выручка'] = inc_region_display['Выручка'].apply(lambda x: cfg.format_money(x))
        inc_region_display['Прибыль'] = inc_region_display['Прибыль'].apply(lambda x: cfg.format_money(x))
        st.dataframe(inc_region_display, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(inc_region.head(10), x='Регион', y='Выручка', title="Выручка по регионам", template="plotly_dark"), use_container_width=True)

    st.markdown("#### Доходы по городам (топ-15)")
    inc_city = ta.TransportAnalytics.get_income_by_city(df, 15)
    if not inc_city.empty:
        inc_city['Выручка_число'] = inc_city['Выручка']
        st.plotly_chart(px.bar(inc_city, x='Город', y='Выручка_число', title="Выручка по городам доставки", template="plotly_dark"), use_container_width=True)

    # Анализ по выбранному маршруту (если есть)
    route_res = st.session_state.get("route_result", None)
    if route_res and route_res[2]:
        st.markdown("#### Доходность по последнему рассчитанному маршруту")
        route_inc = ta.TransportAnalytics.get_route_income_analysis(df, route_res[2])
        if route_inc:
            st.metric("Выручка по маршруту", cfg.format_money(route_inc['Выручка по маршруту']))
            st.metric("Заказов в маршруте", route_inc['Заказов'])
            st.metric("Средний чек", cfg.format_money(route_inc['Средний чек']))

# --- FOOTER ---
st.divider()
st.markdown("Интеллектуальная ИС: прогноз спроса (ML), оптимизация маршрутов (OR-Tools), прогноз риска доставки (Random Forest).")
