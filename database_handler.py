"""
Обработчик базы данных.
Загружает data.csv в SQLite. Поддерживает готовый формат (date, sales, city, ...).
"""
import sqlite3
import pandas as pd
import numpy as np
import os
import hashlib
import json
from datetime import datetime

import company_config as cfg
import inventory_manager as inv_mgr

REQUIRED_COLUMNS = ['date', 'sales', 'profit', 'days_real', 'days_sched', 'risk', 
                    'region', 'category', 'city', 'product', 'lat', 'lon', 'status', 'mode']


def _hash_password(password: str) -> str:
    # Простое хеширование для диплома (без соли), чтобы не усложнять.
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def authenticate_user(username: str, password: str) -> dict | None:
    """
    Возвращает dict пользователя: {user_id, username, role, full_name}, либо None.
    """
    conn = sqlite3.connect("logistics_data.db")
    try:
        row = conn.execute(
            "SELECT user_id, username, password_hash, role, full_name FROM users WHERE username=?",
            (username,),
        ).fetchone()
        if not row:
            return None
        user_id, uname, pw_hash, role, full_name = row
        if pw_hash != _hash_password(password):
            return None
        return {"user_id": int(user_id), "username": uname, "role": role, "full_name": full_name}
    finally:
        conn.close()


def _seed_default_users_if_empty(conn):
    cnt = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    cnt = int(cnt[0]) if cnt else 0
    if cnt > 0:
        return

    # Примеры аккаунтов (можно заменить пароли по желанию)
    # Важно: для роли `driver` делаем user_id = driver_id,
    # чтобы кабинет водителя мог читать deliveries по driver_id.
    default_users = [
        # Диспетчер
        (5, "dispatcher", "qazaqstan2026", "dispatcher", "Диспетчер"),
        # Заказчики
        (3, "customer1", "customer2026", "customer", "Заказчик 1"),
        (4, "customer2", "customer2026", "customer", "Заказчик 2"),
        # Водители
        (1, "driver1", "driver2026", "driver", "Водитель 1"),
        (2, "driver2", "driver2026", "driver", "Водитель 2"),
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO users (user_id, username, password_hash, role, full_name)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (int(uid), uname, _hash_password(pw), role, full_name)
            for uid, uname, pw, role, full_name in default_users
        ],
    )
    conn.commit()

def init_db():
    conn = sqlite3.connect('logistics_data.db')
    base_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_path, 'data.csv')
    
    if not os.path.exists(path):
        print("Файл data.csv не найден!")
        conn.close()
        return

    try:
        # Читаем CSV с кодировкой, которую использует генератор данных
        try:
            df = pd.read_csv(path, encoding='cp1251', low_memory=False)
        except Exception:
            # Фолбэк: иногда файл могли сохранить в UTF-8
            df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]

        # Готовый формат — используем как есть
        if all(c in df.columns for c in REQUIRED_COLUMNS):
            main_df = df[REQUIRED_COLUMNS].copy()
        else:
            # Старый формат DataCo — маппинг
            mapping = {
                'order date (DateOrders)': 'date', 'Sales': 'sales',
                'Order Item Profit Ratio': 'profit', 'Days for shipping (real)': 'days_real',
                'Days for shipment (scheduled)': 'days_sched', 'Late_delivery_risk': 'risk',
                'Order Region': 'region', 'Category Name': 'category',
                'Order City': 'city', 'Product Name': 'product',
                'Latitude': 'lat', 'Longitude': 'lon', 'Delivery Status': 'status', 'Shipping Mode': 'mode'
            }
            actual = {k: v for k, v in mapping.items() if k in df.columns}
            main_df = df[list(actual.keys())].copy()
            main_df.rename(columns=actual, inplace=True)

        main_df['date'] = pd.to_datetime(main_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        main_df['delivery_error'] = main_df['days_real'].astype(float) - main_df['days_sched'].astype(float)

        # --- ФИЛЬТР ТОВАРОВ: ТОЛЬКО СТРОИТЕЛЬСТВО ---
        # Даже если исходный `data.csv` содержит "нестроительные" позиции,
        # в БД должны попадать только стройматериалы.
        if "product" in main_df.columns and hasattr(cfg, "CONSTRUCTION_PRODUCTS"):
            before = len(main_df)
            main_df = main_df[main_df["product"].isin(cfg.CONSTRUCTION_PRODUCTS)].copy()
            filtered = len(main_df)
            print(f"Filtered construction products: {before} -> {filtered} rows")

        if 'product' in main_df.columns:
            inv = main_df.groupby('product')['sales'].count().reset_index()
            inv.columns = ['product', 'sales_count']
            np.random.seed(42)
            inv['stock_level'] = (inv['sales_count'] * np.random.uniform(0.5, 1.8)).astype(int)
            inv.to_sql('inventory', conn, if_exists='replace', index=False)

        main_df.to_sql('main_table', conn, if_exists='replace', index=False)

        # ---- Доп. таблицы для реального учета склада/парка ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS warehouse_inventory (
                warehouse_city TEXT NOT NULL,
                product TEXT NOT NULL,
                qty INTEGER NOT NULL,
                PRIMARY KEY (warehouse_city, product)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS truck_fleet (
                truck_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                current_city TEXT,
                updated_at TEXT
            )
            """
        )

        # ---- БАЗЫ: ВОДИТЕЛИ И ИСТОРИЯ СОСТОЯНИЙ ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drivers (
                driver_id INTEGER PRIMARY KEY,
                full_name TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS driver_fleet (
                driver_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                current_city TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS truck_state_history (
                hist_id INTEGER PRIMARY KEY AUTOINCREMENT,
                truck_id INTEGER NOT NULL,
                driver_id INTEGER,
                status TEXT NOT NULL,
                location_city TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

        # ---- АВТОРАЗГРАНИЧЕНИЕ (ЛИЧНЫЕ КАБИНЕТЫ) ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_offers (
                offer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_user_id INTEGER NOT NULL,
                warehouse_city TEXT NOT NULL,
                destination_city TEXT NOT NULL,
                destination_lat REAL NOT NULL,
                destination_lon REAL NOT NULL,
                product TEXT NOT NULL,
                qty INTEGER NOT NULL,
                proposed_price REAL NOT NULL,
                suggested_price REAL,
                risk_rate REAL,
                status TEXT NOT NULL,
                dispatcher_user_id INTEGER,
                accepted_at TEXT,
                delivery_id INTEGER,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deliveries (
                delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dispatcher_user_id INTEGER NOT NULL,
                truck_id INTEGER NOT NULL,
                driver_id INTEGER,
                warehouse_city TEXT NOT NULL,
                route_cities_json TEXT NOT NULL,
                coords_json TEXT NOT NULL,
                total_km REAL,
                total_duration_h REAL,
                start_time TEXT NOT NULL,
                status TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS completed_orders (
                completed_id INTEGER PRIMARY KEY AUTOINCREMENT,
                offer_id INTEGER NOT NULL UNIQUE,
                delivery_id INTEGER,
                customer_user_id INTEGER,
                destination_city TEXT NOT NULL,
                product TEXT NOT NULL,
                qty INTEGER NOT NULL,
                final_price REAL NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )

        _seed_default_users_if_empty(conn)

        # Сидим warehouse_inventory (демо-модель распределения по складам)
        inv_by_wh = inv_mgr.InventoryManager.get_inventory_by_warehouse(
            main_df,
            cfg.WAREHOUSES.get("locations", []),
        )
        wh_rows = []
        for wh_city, prod_map in inv_by_wh.items():
            for product, qty in prod_map.items():
                wh_rows.append({"warehouse_city": wh_city, "product": product, "qty": int(qty)})
        wh_df = pd.DataFrame(wh_rows)
        if not wh_df.empty:
            wh_df.to_sql('warehouse_inventory', conn, if_exists='replace', index=False)

        # Сидим парк грузовиков
        trucks_total = int(cfg.FLEET.get("trucks_total", 0))
        trucks_available = int(cfg.FLEET.get("trucks_available", 0))
        fleet_rows = []
        for tid in range(1, trucks_total + 1):
            status = "free" if tid <= trucks_available else "busy"
            fleet_rows.append({
                "truck_id": tid,
                "status": status,
                "current_city": None,
                "updated_at": None,
            })
        if fleet_rows:
            pd.DataFrame(fleet_rows).to_sql('truck_fleet', conn, if_exists='replace', index=False)

        # Сидим пул водителей (демо)
        existing_drivers = conn.execute("SELECT COUNT(*) FROM drivers").fetchone()
        existing_drivers = int(existing_drivers[0]) if existing_drivers else 0
        if existing_drivers == 0:
            driver_names = [
                "Айбек Абдрахманов",
                "Данияр Куанышев",
                "Ерлан Сапаргалиев",
                "Марат Абишев",
                "Нурлан Баймурзаев",
                "Санат Садыков",
                "Темирлан Махметов",
                "Фарид Ахметов",
                "Айдос Касенов",
                "Альмира Смагулова",
                "Гульназ Исмагулова",
                "Жанар Турсынбекова",
                "Карлыгаш Сулейменова",
                "Мадина Орынбасар",
                "София Жакупова",
            ]
            drivers_total = max(trucks_total, trucks_available, len(driver_names))
            driver_rows = []
            for did in range(1, drivers_total + 1):
                driver_rows.append({"driver_id": did, "full_name": driver_names[(did - 1) % len(driver_names)]})
            pd.DataFrame(driver_rows).to_sql("drivers", conn, if_exists="replace", index=False)

        # Сидим статусы водителей (free), если таблица пустая
        cnt_driver_fleet = conn.execute("SELECT COUNT(*) FROM driver_fleet").fetchone()
        cnt_driver_fleet = int(cnt_driver_fleet[0]) if cnt_driver_fleet else 0
        if cnt_driver_fleet == 0:
            driver_rows = conn.execute("SELECT driver_id FROM drivers").fetchall()
            if driver_rows:
                pd.DataFrame(
                    [{"driver_id": int(r[0]), "status": "free", "current_city": None, "updated_at": None} for r in driver_rows]
                ).to_sql("driver_fleet", conn, if_exists="replace", index=False)

        print("База данных обновлена! Записей:", len(main_df))
        
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        conn.commit()
        conn.close()


def get_data(query):
    conn = sqlite3.connect('logistics_data.db')
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        return pd.DataFrame()
    finally:
        conn.close()


def get_warehouse_inventory():
    """Текущие остатки по складам (warehouse_city, product, qty)."""
    return get_data("SELECT warehouse_city, product, qty FROM warehouse_inventory")


def adjust_warehouse_inventory(warehouse_city: str, product: str, delta_qty: int):
    """
    Изменяет остаток qty на delta_qty (может быть отрицательным).
    qty не опускается ниже 0.
    """
    conn = sqlite3.connect('logistics_data.db')
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS warehouse_inventory (
                warehouse_city TEXT NOT NULL,
                product TEXT NOT NULL,
                qty INTEGER NOT NULL,
                PRIMARY KEY (warehouse_city, product)
            )
            """
        )
        cur = conn.execute(
            "SELECT qty FROM warehouse_inventory WHERE warehouse_city=? AND product=?",
            (warehouse_city, product),
        )
        row = cur.fetchone()
        if row is None:
            new_qty = max(0, int(delta_qty))
            conn.execute(
                "INSERT INTO warehouse_inventory (warehouse_city, product, qty) VALUES (?, ?, ?)",
                (warehouse_city, product, int(new_qty)),
            )
        else:
            old_qty = int(row[0])
            new_qty = max(0, old_qty + int(delta_qty))
            conn.execute(
                "UPDATE warehouse_inventory SET qty=? WHERE warehouse_city=? AND product=?",
                (int(new_qty), warehouse_city, product),
            )
        conn.commit()
    finally:
        conn.close()


def get_truck_counts():
    # На случай, если таблицы ещё нет (пользователь не синхронизировал)
    conn = sqlite3.connect('logistics_data.db')
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS truck_fleet (
                truck_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                current_city TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()

        # Если таблица пустая (Sync не делали) — сидим базовый парк
        cnt = conn.execute("SELECT COUNT(*) FROM truck_fleet").fetchone()
        if cnt and int(cnt[0]) == 0:
            trucks_total = int(cfg.FLEET.get("trucks_total", 0))
            trucks_available = int(cfg.FLEET.get("trucks_available", 0))
            fleet_rows = []
            for tid in range(1, trucks_total + 1):
                status = "free" if tid <= trucks_available else "busy"
                fleet_rows.append((tid, status, None, None))
            if fleet_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO truck_fleet (truck_id, status, current_city, updated_at) VALUES (?, ?, ?, ?)",
                    fleet_rows,
                )
                conn.commit()
    finally:
        conn.close()

    df = get_data("SELECT status, COUNT(*) as cnt FROM truck_fleet GROUP BY status")
    free_cnt = int(df[df["status"] == "free"]["cnt"].iloc[0]) if not df.empty and (df["status"] == "free").any() else 0
    busy_cnt = int(df[df["status"] == "busy"]["cnt"].iloc[0]) if not df.empty and (df["status"] == "busy").any() else 0
    return {"free": free_cnt, "busy": busy_cnt, "total": free_cnt + busy_cnt}


def reserve_truck(current_city: str | None = None) -> tuple[int | None, int | None]:
    """
    Резервирует первый свободный грузовик и назначает свободного водителя.
    Возвращает (truck_id, driver_id) или (None, None).
    """
    conn = sqlite3.connect('logistics_data.db')
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS truck_fleet (
                truck_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                current_city TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS driver_fleet (
                driver_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                current_city TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drivers (
                driver_id INTEGER PRIMARY KEY,
                full_name TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS truck_state_history (
                hist_id INTEGER PRIMARY KEY AUTOINCREMENT,
                truck_id INTEGER NOT NULL,
                driver_id INTEGER,
                status TEXT NOT NULL,
                location_city TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

        # Если Sync ещё не делали — сидим минимальный пул водителей.
        cnt_drivers = conn.execute("SELECT COUNT(*) FROM drivers").fetchone()
        if cnt_drivers and int(cnt_drivers[0]) == 0:
            driver_names = [
                "Айбек Абдрахманов",
                "Данияр Куанышев",
                "Ерлан Сапаргалиев",
                "Марат Абишев",
                "Нурлан Баймурзаев",
                "Санат Садыков",
                "Темирлан Махметов",
                "Фарид Ахметов",
            ]
            driver_rows = [{"driver_id": did, "full_name": driver_names[(did - 1) % len(driver_names)]} for did in range(1, len(driver_names) + 1)]
            pd.DataFrame(driver_rows).to_sql("drivers", conn, if_exists="replace", index=False)

        # Если таблица пустая (не было Sync), создаём базовый парк.
        cnt = conn.execute("SELECT COUNT(*) FROM truck_fleet").fetchone()
        if cnt and int(cnt[0]) == 0:
            trucks_total = int(cfg.FLEET.get("trucks_total", 0))
            trucks_available = int(cfg.FLEET.get("trucks_available", 0))
            fleet_rows = []
            for tid in range(1, trucks_total + 1):
                status = "free" if tid <= trucks_available else "busy"
                fleet_rows.append((tid, status, None, None))
            conn.executemany(
                "INSERT OR REPLACE INTO truck_fleet (truck_id, status, current_city, updated_at) VALUES (?, ?, ?, ?)",
                fleet_rows,
            )
            conn.commit()

        cur = conn.execute(
            "SELECT truck_id FROM truck_fleet WHERE status='free' ORDER BY truck_id LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None, None
        truck_id = int(row[0])

        driver_id = None
        try:
            drow = conn.execute(
                "SELECT driver_id FROM driver_fleet WHERE status='free' ORDER BY driver_id LIMIT 1"
            ).fetchone()
            driver_id = int(drow[0]) if drow else None
        except Exception:
            driver_id = None

        if driver_id is not None:
            try:
                conn.execute(
                    """
                    UPDATE driver_fleet
                    SET status='in_trip', current_city=?, updated_at=datetime('now')
                    WHERE driver_id=?
                    """,
                    (current_city, driver_id),
                )
            except Exception:
                pass

        # Запись о начале поездки
        conn.execute(
            """
            INSERT INTO truck_state_history (truck_id, driver_id, status, location_city, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (truck_id, driver_id, "in_trip", current_city),
        )

        conn.execute(
            "UPDATE truck_fleet SET status='busy', current_city=?, updated_at=datetime('now') WHERE truck_id=?",
            (current_city, truck_id),
        )
        conn.commit()
        return truck_id, driver_id
    finally:
        conn.close()


def release_truck(truck_id: int):
    """Освобождает грузовик."""
    conn = sqlite3.connect('logistics_data.db')
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS truck_fleet (
                truck_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                current_city TEXT,
                updated_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS truck_state_history (
                hist_id INTEGER PRIMARY KEY AUTOINCREMENT,
                truck_id INTEGER NOT NULL,
                driver_id INTEGER,
                status TEXT NOT NULL,
                location_city TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS driver_fleet (
                driver_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                current_city TEXT,
                updated_at TEXT
            )
            """
        )

        driver_id = None
        try:
            drow = conn.execute(
                """
                SELECT driver_id
                FROM truck_state_history
                WHERE truck_id=? AND status='in_trip'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (int(truck_id),),
            ).fetchone()
            driver_id = int(drow[0]) if drow and drow[0] is not None else None
        except Exception:
            driver_id = None

        if driver_id is not None:
            try:
                conn.execute(
                    """
                    UPDATE driver_fleet
                    SET status='free', current_city=NULL, updated_at=datetime('now')
                    WHERE driver_id=?
                    """,
                    (driver_id,),
                )
            except Exception:
                pass

        # Запись о завершении поездки
        conn.execute(
            """
            INSERT INTO truck_state_history (truck_id, driver_id, status, location_city, updated_at)
            VALUES (?, ?, ?, NULL, datetime('now'))
            """,
            (int(truck_id), driver_id, "free"),
        )
        conn.execute(
            "UPDATE truck_fleet SET status='free', current_city=NULL, updated_at=datetime('now') WHERE truck_id=?",
            (int(truck_id),),
        )
        conn.commit()
    finally:
        conn.close()


def create_customer_offer(
    customer_user_id: int,
    warehouse_city: str,
    destination_city: str,
    destination_lat: float,
    destination_lon: float,
    product: str,
    qty: int,
    proposed_price: float,
    suggested_price: float | None,
    risk_rate: float | None,
) -> int | None:
    """Создаёт предложение клиента диспетчеру."""
    conn = sqlite3.connect("logistics_data.db")
    try:
        cur = conn.execute(
            """
            INSERT INTO customer_offers (
                customer_user_id,
                warehouse_city,
                destination_city,
                destination_lat,
                destination_lon,
                product,
                qty,
                proposed_price,
                suggested_price,
                risk_rate,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', datetime('now'))
            """,
            (
                int(customer_user_id),
                warehouse_city,
                destination_city,
                float(destination_lat),
                float(destination_lon),
                product,
                int(qty),
                float(proposed_price),
                float(suggested_price) if suggested_price is not None else None,
                float(risk_rate) if risk_rate is not None else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid) if cur.lastrowid is not None else None
    except Exception:
        return None
    finally:
        conn.close()


def get_customer_offers_by_status(status: str):
    """Список предложений по статусу (dataframe)."""
    return get_offers(status)


def _get_offers_df(where_sql: str, params: tuple):
    conn = sqlite3.connect("logistics_data.db")
    try:
        query = f"""
            SELECT
                o.*,
                u.username AS customer_username,
                u.full_name AS customer_full_name
            FROM customer_offers o
            JOIN users u ON u.user_id = o.customer_user_id
            {where_sql}
            ORDER BY o.created_at DESC
        """
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_offers(status: str):
    return _get_offers_df("WHERE o.status = ?", (status,))


def get_offers_for_customer(customer_user_id: int):
    return _get_offers_df("WHERE o.customer_user_id = ?", (int(customer_user_id),))


def accept_offer(offer_id: int, dispatcher_user_id: int) -> bool:
    conn = sqlite3.connect("logistics_data.db")
    try:
        conn.execute(
            """
            UPDATE customer_offers
            SET status='accepted', dispatcher_user_id=?, accepted_at=datetime('now')
            WHERE offer_id=? AND status='proposed'
            """,
            (int(dispatcher_user_id), int(offer_id)),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def reject_offer(offer_id: int) -> bool:
    conn = sqlite3.connect("logistics_data.db")
    try:
        conn.execute(
            """
            UPDATE customer_offers
            SET status='rejected', dispatcher_user_id=NULL, accepted_at=NULL, delivery_id=NULL
            WHERE offer_id=? AND status IN ('proposed','accepted')
            """,
            (int(offer_id),),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def create_delivery(
    dispatcher_user_id: int,
    truck_id: int,
    driver_id: int | None,
    warehouse_city: str,
    route_cities: list[str],
    coords_list: list[dict],
    total_km: float,
    total_duration_h: float,
    start_time: datetime | None = None,
) -> int | None:
    start_time = start_time or datetime.now()
    conn = sqlite3.connect("logistics_data.db")
    try:
        route_cities_json = json.dumps(route_cities, ensure_ascii=False)
        coords_json = json.dumps(coords_list, ensure_ascii=False)
        cur = conn.execute(
            """
            INSERT INTO deliveries (
                dispatcher_user_id,
                truck_id,
                driver_id,
                warehouse_city,
                route_cities_json,
                coords_json,
                total_km,
                total_duration_h,
                start_time,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'in_progress')
            """,
            (
                int(dispatcher_user_id),
                int(truck_id),
                int(driver_id) if driver_id is not None else None,
                warehouse_city,
                route_cities_json,
                coords_json,
                float(total_km) if total_km is not None else None,
                float(total_duration_h) if total_duration_h is not None else None,
                str(start_time),
            ),
        )
        conn.commit()
        return int(cur.lastrowid) if cur.lastrowid is not None else None
    except Exception:
        return None
    finally:
        conn.close()


def mark_offers_dispatched(offer_ids: list[int], delivery_id: int) -> bool:
    conn = sqlite3.connect("logistics_data.db")
    try:
        if not offer_ids:
            return False
        placeholders = ",".join(["?"] * len(offer_ids))
        params = [int(delivery_id)] + [int(x) for x in offer_ids]
        conn.execute(
            f"""
            UPDATE customer_offers
            SET status='dispatched', delivery_id=?
            WHERE offer_id IN ({placeholders})
            """,
            tuple(params),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_deliveries_by_driver(driver_id: int):
    conn = sqlite3.connect("logistics_data.db")
    try:
        df_out = pd.read_sql_query(
            """
            SELECT *
            FROM deliveries
            WHERE driver_id=? AND status='in_progress'
            ORDER BY start_time DESC
            """,
            conn,
            params=(int(driver_id),),
        )
        return df_out
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def mark_delivery_completed(delivery_id: int):
    """
    Помечает рейс завершённым:
    - deliveries.status -> completed
    - customer_offers.status -> completed
    - вставляет строки в completed_orders
    """
    delivery_id = int(delivery_id)
    conn = sqlite3.connect("logistics_data.db")
    try:
        conn.execute(
            """
            UPDATE deliveries
            SET status='completed', completed_at=datetime('now')
            WHERE delivery_id=?
            """,
            (delivery_id,),
        )

        # Обновляем статусы предложений
        conn.execute(
            """
            UPDATE customer_offers
            SET status='completed', completed_at=datetime('now')
            WHERE delivery_id=?
            """,
            (delivery_id,),
        )

        # Вставляем в completed_orders (по каждому offer)
        conn.execute(
            """
            INSERT OR IGNORE INTO completed_orders (
                offer_id,
                delivery_id,
                customer_user_id,
                destination_city,
                product,
                qty,
                final_price,
                created_at,
                completed_at
            )
            SELECT
                o.offer_id,
                o.delivery_id,
                o.customer_user_id,
                o.destination_city,
                o.product,
                o.qty,
                o.proposed_price AS final_price,
                datetime('now') AS created_at,
                datetime('now') AS completed_at
            FROM customer_offers o
            WHERE o.delivery_id=?
            """,
            (delivery_id,),
        )

        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()
