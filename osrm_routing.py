"""
Маршрутизация по дорогам через публичный OSRM.

Используется для:
- получения polyline маршрута (для движения грузовика по карте)
- получения duration/distance (для ETA и тайм-слайдера)
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _fetch_osrm_leg(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> Dict:
    # OSRM: расстояние в метрах, duration в секундах.
    base_url = "http://router.project-osrm.org/route/v1/driving"
    coords = f"{from_lon},{from_lat};{to_lon},{to_lat}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
        "alternatives": "false",
    }
    url = f"{base_url}/{coords}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def get_osrm_leg_polyline(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
) -> Tuple[List[Dict[str, float]], float, float]:
    """
    Returns:
      polyline: список точек [{'lat': ..., 'lon': ...}, ...]
      distance_km: суммарная дистанция
      duration_h: суммарное время
    """
    data = _fetch_osrm_leg(from_lat, from_lon, to_lat, to_lon)
    routes = data.get("routes") or []
    if not routes:
        return [], 0.0, 0.0

    r0 = routes[0]
    distance_m = float(r0.get("distance", 0.0))
    duration_s = float(r0.get("duration", 0.0))
    distance_km = distance_m / 1000.0
    duration_h = duration_s / 3600.0

    geometry = r0.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    # coordinates: [[lon, lat], ...]
    polyline = [{"lat": float(lat), "lon": float(lon)} for lon, lat in coords]
    return polyline, distance_km, duration_h


def build_osrm_polyline_for_waypoints(
    waypoints: List[Dict[str, float]],
) -> Tuple[List[Dict[str, float]], float, float, List[float], List[float]]:
    """
    waypoints: [{'lat':..., 'lon':...}, ...] в требуемом порядке
    Returns:
      polyline_all: склеенная polyline по всем legs
      total_distance_km
      total_duration_h
      leg_distance_km: per-leg
      leg_duration_h: per-leg
    """
    if not waypoints or len(waypoints) < 2:
        return [], 0.0, 0.0, [], []

    polyline_all: List[Dict[str, float]] = []
    total_distance_km = 0.0
    total_duration_h = 0.0
    leg_distance_km: List[float] = []
    leg_duration_h: List[float] = []

    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]

        leg_polyline, d_km, t_h = get_osrm_leg_polyline(
            float(a["lat"]), float(a["lon"]),
            float(b["lat"]), float(b["lon"]),
        )

        # Если OSRM не вернул геометрию — создаём прямой сегмент из двух точек
        if not leg_polyline:
            leg_polyline = [
                {"lat": float(a["lat"]), "lon": float(a["lon"])},
                {"lat": float(b["lat"]), "lon": float(b["lon"])},
            ]
            d_km = _haversine_km(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
            # ~70 км/ч, чтобы не было деления на ноль
            t_h = d_km / 70.0 * 1.1

        # Сцепляем: не дублируем первую точку следующего leg
        if polyline_all and leg_polyline:
            polyline_all.extend(leg_polyline[1:])
        else:
            polyline_all.extend(leg_polyline)

        total_distance_km += float(d_km)
        total_duration_h += float(t_h)
        leg_distance_km.append(float(d_km))
        leg_duration_h.append(float(t_h))

    return polyline_all, total_distance_km, total_duration_h, leg_distance_km, leg_duration_h


def build_cumulative_distances_km(polyline: List[Dict[str, float]]) -> List[float]:
    """
    cumulative[i] = расстояние от начала до polyline[i] (в км)
    cumulative[0] == 0
    """
    if not polyline:
        return []

    cumulative = [0.0]
    total = 0.0
    for i in range(1, len(polyline)):
        a = polyline[i - 1]
        b = polyline[i]
        total += _haversine_km(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
        cumulative.append(total)
    return cumulative


def point_on_polyline_by_distance_km(
    polyline: List[Dict[str, float]],
    cumulative_km: List[float],
    target_km: float,
) -> Dict[str, float]:
    """
    Ищет точку на polyline по target_km.
    Если target_km выходит за диапазон — clamp к концам.
    """
    if not polyline or len(polyline) == 1:
        return polyline[0] if polyline else {"lat": 0.0, "lon": 0.0}

    if target_km <= 0:
        return polyline[0]
    if target_km >= cumulative_km[-1]:
        return polyline[-1]

    # Ищем сегмент: cumulative[i-1] <= target < cumulative[i]
    # Линейный поиск ок для небольших polyline, для больших — можно бинарный.
    for i in range(1, len(cumulative_km)):
        if cumulative_km[i] >= target_km:
            prev_d = cumulative_km[i - 1]
            next_d = cumulative_km[i]
            if next_d - prev_d <= 1e-9:
                return polyline[i]
            t = (target_km - prev_d) / (next_d - prev_d)
            a = polyline[i - 1]
            b = polyline[i]
            return {
                "lat": float(a["lat"] + (b["lat"] - a["lat"]) * t),
                "lon": float(a["lon"] + (b["lon"] - a["lon"]) * t),
            }

    return polyline[-1]

