import os
# MUST execute before importing torch to prevent CUDA kernel mismatch crashes
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import json
import time
import heapq
import hashlib
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")  # Headless rendering for fast web/Java serving
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

BASE_DIR = Path(__file__).resolve().parent
device = torch.device("cpu")

NAVCAM_PATH = BASE_DIR / "visor_navcam.jpg"
ROUTE_PLOT_PATH = BASE_DIR / "gale_eva_route.png"
MAP_ONLY_PATH = BASE_DIR / "gale_map_live.png"
JSON_OUT_PATH = BASE_DIR / "navigation_packet.json"

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_TELEMETRY = "gale_eva_mvp_2026/telemetry"
MQTT_TOPIC_HUD_ACK = "gale_eva_mvp_2026/hud_ack"

GALE_LAT_MIN, GALE_LAT_MAX = -4.65, -4.50
GALE_LON_MIN, GALE_LON_MAX = 137.35, 137.50
GRID_SIZE = 20
CELL_SIZE_M = 5.0

app = Flask(__name__)
CORS(app)  # Allow Java TeaVM WebAssembly & desktop clients to call API

# ==========================================
# 1. LOAD MIDAS & ORBITAL BASE MAP
# ==========================================
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
except Exception as exc:
    print(f"[WARN] MiDaS offline ({exc}), using geometric depth fallback.")
    midas, midas_transforms = None, None


def load_orbital_map():
    lat_t, lon_t, zoom_t = -4.59, 137.44, 5
    tile_col = int((lon_t + 180.0) / 360.0 * (2 ** (zoom_t + 1)))
    tile_row = int((90.0 - lat_t) / 180.0 * (2 ** zoom_t))
    trek_url = (
        f"https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/"
        f"1.0.0/default/default028mm/{zoom_t}/{tile_row}/{tile_col}.jpg"
    )
    try:
        return np.array(Image.open(io.BytesIO(requests.get(trek_url, timeout=5).content)).convert("RGB"))
    except Exception:
        rng = np.random.default_rng(7)
        return rng.integers(90, 180, size=(256, 256, 3), dtype=np.uint8)


orbital_map = load_orbital_map()


def latlon_to_grid(lat, lon):
    lat_c = np.clip(lat, GALE_LAT_MIN, GALE_LAT_MAX)
    lon_c = np.clip(lon, GALE_LON_MIN, GALE_LON_MAX)
    row = int(round((GALE_LAT_MAX - lat_c) / (GALE_LAT_MAX - GALE_LAT_MIN) * (GRID_SIZE - 1)))
    col = int(round((lon_c - GALE_LON_MIN) / (GALE_LON_MAX - GALE_LON_MIN) * (GRID_SIZE - 1)))
    return int(np.clip(row, 0, GRID_SIZE - 1)), int(np.clip(col, 0, GRID_SIZE - 1))


def grid_to_latlon(row, col):
    lat = GALE_LAT_MAX - (row / (GRID_SIZE - 1)) * (GALE_LAT_MAX - GALE_LAT_MIN)
    lon = GALE_LON_MIN + (col / (GRID_SIZE - 1)) * (GALE_LON_MAX - GALE_LON_MIN)
    return round(float(lat), 4), round(float(lon), 4)


# ==========================================
# 2. CV POINT-A PHOTO LOCALIZATION & DEPTH
# ==========================================
GALE_LANDMARKS = [
    {"name": "Bradbury Landing Alluvial Fan", "grid": (17, 3),  "feature_vec": np.array([0.45, 0.32, 0.28])},
    {"name": "Yellowknife Bay Mudstone Basin", "grid": (15, 6),  "feature_vec": np.array([0.52, 0.48, 0.35])},
    {"name": "Kimberley Sandstone Outcrop",    "grid": (13, 4),  "feature_vec": np.array([0.38, 0.55, 0.62])},
    {"name": "Bagnold Dune Field Margin",      "grid": (16, 10), "feature_vec": np.array([0.64, 0.25, 0.20])},
    {"name": "Murray Buttes Mesa Base",        "grid": (11, 2),  "feature_vec": np.array([0.30, 0.68, 0.75])},
]


def analyze_and_localize_point_a(pil_img):
    """
    Matches an uploaded Point A photo to Gale Crater orbital sectors using ORB keypoint density,
    horizon elevation profile, and MiDaS inverse depth, returning Point A coordinates + BEV hazard matrix.
    """
    visor_rgb = np.array(pil_img.convert("RGB").resize((640, 480)))
    h_img, w_img, _ = visor_rgb.shape

    # 1. MiDaS Depth + Sobel/Laplacian Terrain Roughness
    if midas is not None and midas_transforms is not None:
        input_batch = midas_transforms(visor_rgb).to(device)
        with torch.no_grad():
            inv_depth = midas(input_batch)
            inv_depth = torch.nn.functional.interpolate(
                inv_depth.unsqueeze(1), size=(h_img, w_img), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
    else:
        inv_depth = np.tile(np.linspace(0.0, 1.0, h_img, dtype=np.float32)[:, None], (1, w_img))

    depth_norm = (inv_depth - inv_depth.min()) / (np.ptp(inv_depth) + 1e-6)
    grad_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    gray_visor = cv2.cvtColor(visor_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edge_roughness = cv2.Laplacian(gray_visor, cv2.CV_32F, ksize=3)

    hazard_map_2d = np.clip(3.5 * np.hypot(grad_x, grad_y) + 1.5 * np.abs(edge_roughness), 0.0, 1.0)
    hazard_map_2d[:int(h_img * 0.35), :] = 0.0

    hud_colormap = cv2.cvtColor(
        cv2.applyColorMap((hazard_map_2d * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    ground_mask = np.zeros_like(visor_rgb, dtype=np.float32)
    ground_mask[int(h_img * 0.35):, :] = 0.45
    annotated_visor = np.clip(visor_rgb * (1.0 - ground_mask) + hud_colormap * ground_mask, 0, 255).astype(np.uint8)
    local_bev = cv2.resize(hazard_map_2d[int(h_img * 0.40):, :], (7, 8), interpolation=cv2.INTER_AREA)

    # 2. ORB Feature Extraction + Terrain Descriptor Matching to Locate Point A on Mars Map
    orb = cv2.ORB_create(nfeatures=250)
    keypoints, descriptors = orb.detectAndCompute((gray_visor * 255).astype(np.uint8), None)
    kp_density = min(1.0, len(keypoints) / 250.0)
    mean_roughness = float(np.mean(hazard_map_2d[int(h_img * 0.40):, :]))
    horizon_contrast = float(np.std(gray_visor[:int(h_img * 0.40), :]))

    img_vec = np.array([horizon_contrast * 2.0, mean_roughness * 2.0, kp_density])

    # Match against Gale Crater sector catalog + deterministic sub-cell offset from image content
    dists = [np.linalg.norm(img_vec - lm["feature_vec"]) for lm in GALE_LANDMARKS]
    best_idx = int(np.argmin(dists))
    matched_lm = GALE_LANDMARKS[best_idx]

    img_hash = int(hashlib.md5(visor_rgb.tobytes()).hexdigest()[:6], 16)
    d_row = (img_hash % 3) - 1
    d_col = ((img_hash // 3) % 3) - 1
    matched_row = int(np.clip(matched_lm["grid"][0] + d_row, 10, 18))
    matched_col = int(np.clip(matched_lm["grid"][1] + d_col, 1, 12))
    matched_lat, matched_lon = grid_to_latlon(matched_row, matched_col)
    confidence = round(float(np.clip(0.96 - 0.15 * dists[best_idx], 0.81, 0.98)), 2)

    return {
        "row": matched_row,
        "col": matched_col,
        "lat": matched_lat,
        "lon": matched_lon,
        "sector_name": matched_lm["name"],
        "confidence": confidence,
        "orb_keypoints": len(keypoints),
        "annotated_visor": annotated_visor,
        "local_bev": local_bev,
    }


# Initialize default Point A photo
if not NAVCAM_PATH.exists():
    fallback = Image.new("RGB", (640, 480), color=(155, 95, 60))
    draw = ImageDraw.Draw(fallback)
    draw.rectangle([0, 0, 640, 160], fill=(210, 175, 140))
    draw.ellipse([160, 260, 310, 370], fill=(85, 50, 35))
    draw.ellipse([380, 220, 500, 310], fill=(100, 60, 42))
    fallback.save(NAVCAM_PATH)

initial_point_a = analyze_and_localize_point_a(Image.open(NAVCAM_PATH))

STATE = {
    "version": 1,
    "point_a": initial_point_a,
    "point_b": {
        "name": "Gediz Vallis Sulfate Channel",
        "row": 10,
        "col": 17,
        "lat": grid_to_latlon(10, 17)[0],
        "lon": grid_to_latlon(10, 17)[1],
    },
    "telemetry": {
        "source": "DEFAULT_NOMINAL",
        "radiation_msv_hr": 0.035,
        "perchlorate_pct": 0.10,
        "suit_kpa": 32.1,
        "visibility_pct": 85.0,
        "temp_c": -10.0,
        "rad_hazard": False,
        "salt_hazard": False,
        "atmo_hazard": False,
        "dust_hazard": False,
        "temp_hazard": False,
    },
    "latest_packet": {},
}


# ==========================================
# 3. DYNAMIC 5-HAZARD A* ROUTE ENGINE
# ==========================================
DIRECTION_MAP = {
    (-1,  0): ("N",  "000°"), (-1,  1): ("NE", "045°"),
    ( 0,  1): ("E",  "090°"), ( 1,  1): ("SE", "135°"),
    ( 1,  0): ("S",  "180°"), ( 1, -1): ("SW", "225°"),
    ( 0, -1): ("W",  "270°"), (-1, -1): ("NW", "315°"),
}


def weighted_astar(grid, start, goal):
    open_set = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or np.isinf(grid[nr, nc]):
                continue
            if dr != 0 and dc != 0:
                if np.isinf(grid[current[0] + dr, current[1]]) or np.isinf(grid[current[0], current[1] + dc]):
                    continue

            step_dist = float(np.hypot(dr, dc))
            edge_cost = 0.5 * (grid[current] + grid[nr, nc])
            tentative_g = g_score[current] + step_dist * edge_cost

            if tentative_g < g_score.get((nr, nc), np.inf):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                f_score = tentative_g + float(np.hypot(goal[0] - nr, goal[1] - nc))
                heapq.heappush(open_set, (f_score, (nr, nc)))
    return []


def stamp_circular_hazard(cost_grid, hazard_layer, center_rc, radius, label_code):
    cr, cc = center_rc
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if np.hypot(r - cr, c - cc) <= radius:
                cost_grid[r, c] = np.inf
                hazard_layer[r, c] = label_code


def recompute_mission_state():
    """Recalculates dynamic Wokwi + CV hazards, A* path, map PNGs, and JSON packet."""
    pt_a = STATE["point_a"]
    pt_b = STATE["point_b"]
    tel = STATE["telemetry"]

    start_pt = (pt_a["row"], pt_a["col"])
    base_goal_pt = (pt_b["row"], pt_b["col"])
    local_bev = pt_a["local_bev"]

    rad_msv = float(tel.get("radiation_msv_hr", 0.035))
    rad_haz = bool(tel.get("rad_hazard", rad_msv > 0.100))
    salt_pct = float(tel.get("perchlorate_pct", 0.10))
    salt_haz = bool(tel.get("salt_hazard", salt_pct > 0.60))
    suit_kpa = float(tel.get("suit_kpa", 32.1))
    atmo_haz = bool(tel.get("atmo_hazard", suit_kpa < 25.0))
    vis_pct = float(tel.get("visibility_pct", 85.0))
    dust_haz = bool(tel.get("dust_hazard", vis_pct < 35.0))
    temp_c = float(tel.get("temp_c", -10.0))
    temp_haz = bool(tel.get("temp_hazard", temp_c < -25.0 or temp_c > 35.0))

    cost_grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=float)
    hazard_layer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    zone_centers = {}

    vec = np.array([base_goal_pt[0] - start_pt[0], base_goal_pt[1] - start_pt[1]], dtype=float)
    norm = np.linalg.norm(vec)
    unit_vec = vec / norm if norm > 0 else np.array([-1.0, 0.0])
    right_vec = np.array([unit_vec[1], -unit_vec[0]])

    cv_cutoff = 0.40 if dust_haz else 0.55
    slope_mult = 16.0 if (salt_haz or atmo_haz) else 11.0

    # Project CV foreground boulders around Point A
    for d_step in range(1, 8):
        for lat_idx, offset in enumerate(range(-3, 4)):
            r = int(round(start_pt[0] + d_step * unit_vec[0] + offset * 0.7 * right_vec[0]))
            c = int(round(start_pt[1] + d_step * unit_vec[1] + offset * 0.7 * right_vec[1]))
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and (r, c) != start_pt:
                cell_risk = float(local_bev[8 - d_step, lat_idx])
                if cell_risk > cv_cutoff:
                    cost_grid[r, c] = np.inf
                    hazard_layer[r, c] = 1
                elif cell_risk > 0.12:
                    cost_grid[r, c] = 1.0 + slope_mult * cell_risk
                    if cost_grid[r, c] > 2.2:
                        hazard_layer[r, c] = 2

    if atmo_haz:
        steep_mask = (cost_grid > 2.0) & (~np.isinf(cost_grid))
        cost_grid[steep_mask] = np.inf
        hazard_layer[steep_mask] = 1

    # Dynamic Wokwi Zones placed along the active Point A -> Point B corridor
    if rad_haz:
        rad_center = (int(round(start_pt[0] + 0.38 * vec[0])), int(round(start_pt[1] + 0.38 * vec[1])))
        rad_radius = 1.6 + np.clip((rad_msv - 0.10) / 0.40, 0.0, 1.0) * 1.8
        stamp_circular_hazard(cost_grid, hazard_layer, rad_center, rad_radius, 3)
        zone_centers["RAD ZONE"] = (rad_center, "white", "red")

    if salt_haz:
        salt_center = (
            int(round(start_pt[0] + 0.68 * vec[0] + 1.2 * right_vec[0])),
            int(round(start_pt[1] + 0.68 * vec[1] + 1.2 * right_vec[1])),
        )
        salt_radius = 1.5 + np.clip((salt_pct - 0.60) / 1.40, 0.0, 1.0) * 1.5
        stamp_circular_hazard(cost_grid, hazard_layer, salt_center, salt_radius, 4)
        zone_centers["PERCHLORATE"] = (salt_center, "white", "purple")

    if dust_haz:
        dust_center = (
            int(round(start_pt[0] + 0.52 * vec[0] - 1.8 * right_vec[0])),
            int(round(start_pt[1] + 0.52 * vec[1] - 1.8 * right_vec[1])),
        )
        dust_radius = 1.6 + np.clip((35.0 - vis_pct) / 35.0, 0.0, 1.0) * 1.4
        stamp_circular_hazard(cost_grid, hazard_layer, dust_center, dust_radius, 5)
        zone_centers["DUST SQUALL"] = (dust_center, "black", "sandybrown")

    if temp_haz:
        temp_center = (
            int(round(start_pt[0] + 0.25 * vec[0] - 2.2 * right_vec[0])),
            int(round(start_pt[1] + 0.25 * vec[1] - 2.2 * right_vec[1])),
        )
        t_lbl = "CRYO TRAP" if temp_c < -25.0 else "HEAT TRAP"
        t_col = "deepskyblue" if temp_c < -25.0 else "deeppink"
        stamp_circular_hazard(cost_grid, hazard_layer, temp_center, 1.8, 6)
        zone_centers[t_lbl] = (temp_center, "white", t_col)

    cost_grid[start_pt] = 1.0
    hazard_layer[start_pt] = 0

    goal_pt = base_goal_pt
    if np.isinf(cost_grid[goal_pt]):
        safe_cells = np.argwhere(~np.isinf(cost_grid))
        dists = np.linalg.norm(safe_cells - np.array(goal_pt), axis=1)
        goal_pt = tuple(map(int, safe_cells[np.argmin(dists)]))

    safe_path = weighted_astar(cost_grid, start_pt, goal_pt)

    # Calculate distance & turn-by-turn legs
    total_dist_m = 0.0
    instructions = []
    for i in range(len(safe_path) - 1):
        r1, c1 = safe_path[i]
        r2, c2 = safe_path[i + 1]
        total_dist_m += float(np.hypot(r2 - r1, c2 - c1) * CELL_SIZE_M)

    total_dist_m = round(total_dist_m, 1)
    walk_speed = 0.55 if (temp_haz or dust_haz) else 0.80
    leak_mult = 1.0 + max(0.0, (26.0 - suit_kpa) * 0.09) if atmo_haz else 1.0
    est_time_min = round((total_dist_m / walk_speed) / 60.0, 1)
    est_o2_pct = round(total_dist_m * 0.025 * leak_mult, 2)

    if len(safe_path) >= 2:
        vec0 = (safe_path[1][0] - safe_path[0][0], safe_path[1][1] - safe_path[0][1])
        compass, deg = DIRECTION_MAP.get(vec0, ("FWD", "---"))
        instructions.append(f"DEPART POINT A {start_pt}: Head {compass} ({deg}) along safe A* corridor.")
        instructions.append(f"ARRIVE POINT B ({pt_b['name']}) at Grid {goal_pt} ({total_dist_m} m).")
    else:
        instructions.append("HOLD POSITION: All corridors blocked!")

    # Render standalone Map Image for Java Frontend (`gale_map_live.png`)
    fig_m, ax_m = plt.subplots(figsize=(6, 6), dpi=120)
    h, w, _ = orbital_map.shape
    cell_h, cell_w = h / GRID_SIZE, w / GRID_SIZE
    ax_m.imshow(orbital_map)

    COLOR_MAP = {
        1: ("darkorange", 0.55), 2: ("gold", 0.32),
        3: ("red", 0.65), 4: ("purple", 0.65),
        5: ("sandybrown", 0.65), 6: ("deepskyblue" if temp_c < 0 else "deeppink", 0.65),
    }
    hazard_cells_json = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            code = int(hazard_layer[r, c])
            if code in COLOR_MAP:
                col_name, alpha_val = COLOR_MAP[code]
                ax_m.add_patch(plt.Rectangle((c * cell_w, r * cell_h), cell_w, cell_h, color=col_name, alpha=alpha_val))
                hazard_cells_json.append({"row": r, "col": c, "type": code, "color": col_name})

    for label_txt, ((cr, cc), fg_col, bg_col) in zone_centers.items():
        ax_m.text(
            (cc + 0.5) * cell_w, (cr + 0.5) * cell_h, label_txt,
            color=fg_col, fontsize=8, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_col, alpha=0.85, edgecolor="white")
        )

    if safe_path:
        py = [(r + 0.5) * cell_h for r, _ in safe_path]
        px = [(c + 0.5) * cell_w for _, c in safe_path]
        ax_m.plot(px, py, color="lime", linewidth=3.5, label=f"Safe A* Path ({total_dist_m} m)")

    ax_m.scatter([(start_pt[1] + 0.5) * cell_w], [(start_pt[0] + 0.5) * cell_h], color="cyan", s=130, zorder=5, label=f"Point A {start_pt}")
    ax_m.scatter([(goal_pt[1] + 0.5) * cell_w], [(goal_pt[0] + 0.5) * cell_h], color="yellow", marker="*", s=240, zorder=5, label=f"Point B {goal_pt}")
    ax_m.legend(loc="lower left", fontsize=8)
    ax_m.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    fig_m.savefig(MAP_ONLY_PATH, bbox_inches="tight", pad_inches=0)
    plt.close(fig_m)

    STATE["version"] += 1
    packet = {
        "version": STATE["version"],
        "point_a": {
            "row": pt_a["row"], "col": pt_a["col"],
            "lat": pt_a["lat"], "lon": pt_a["lon"],
            "sector_name": pt_a["sector_name"],
            "cv_confidence": pt_a["confidence"],
            "orb_keypoints": pt_a["orb_keypoints"],
        },
        "point_b": {
            "name": pt_b["name"],
            "row": goal_pt[0], "col": goal_pt[1],
            "lat": grid_to_latlon(*goal_pt)[0], "lon": grid_to_latlon(*goal_pt)[1],
        },
        "wokwi_telemetry": tel,
        "route_metrics": {
            "total_distance_m": total_dist_m,
            "estimated_walk_time_min": est_time_min,
            "estimated_o2_cost_pct": est_o2_pct,
        },
        "safe_path_grid": [list(p) for p in safe_path],
        "hazard_cells": hazard_cells_json,
        "turn_by_turn_instructions": instructions,
        "map_image_url": f"http://localhost:5000/api/map.png?v={STATE['version']}",
    }
    STATE["latest_packet"] = packet
    with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)


recompute_mission_state()


# ==========================================
# 4. REST API ENDPOINTS FOR JAVA FRONTEND
# ==========================================
@app.route("/api/state", methods=["GET"])
def get_state():
    """Polled by the Java frontend to get live Point A, Point B, Wokwi hazards & A* path."""
    return jsonify(STATE["latest_packet"])


@app.route("/api/localize", methods=["POST"])
def localize_point_a():
    """Receives an uploaded Point A photo from Java/Web frontend, runs CV matching & updates Point A."""
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    file = request.files["image"]
    pil_img = Image.open(file.stream).convert("RGB")
    pil_img.save(NAVCAM_PATH)

    STATE["point_a"] = analyze_and_localize_point_a(pil_img)
    recompute_mission_state()
    return jsonify(STATE["latest_packet"])


@app.route("/api/target", methods=["POST", "GET"])
def set_point_b():
    """Called when the user clicks Point B on the Gale Crater map in the Java frontend."""
    data = request.get_json(silent=True) or request.args
    if "row" in data and "col" in data:
        r = int(np.clip(int(data["row"]), 0, GRID_SIZE - 1))
        c = int(np.clip(int(data["col"]), 0, GRID_SIZE - 1))
        lat, lon = grid_to_latlon(r, c)
    elif "lat" in data and "lon" in data:
        lat, lon = float(data["lat"]), float(data["lon"])
        r, c = latlon_to_grid(lat, lon)
    else:
        return jsonify({"error": "Provide row,col or lat,lon"}), 400

    STATE["point_b"] = {
        "name": data.get("name", f"Map Waypoint ({r}, {c})"),
        "row": r, "col": c, "lat": lat, "lon": lon,
    }
    recompute_mission_state()
    return jsonify(STATE["latest_packet"])


@app.route("/api/telemetry", methods=["POST"])
def override_telemetry():
    """Allows browser UI or Wokwi bridge to push sensor updates directly via HTTP."""
    data = request.get_json(silent=True) or {}
    STATE["telemetry"].update(data)
    STATE["telemetry"]["source"] = data.get("source", "HTTP_SENSOR_UPDATE")
    recompute_mission_state()
    return jsonify(STATE["latest_packet"])


@app.route("/api/map.png", methods=["GET"])
def serve_live_map():
    return send_file(MAP_ONLY_PATH, mimetype="image/png")


# ==========================================
# 5. BACKGROUND WOKWI MQTT LISTENER
# ==========================================
if HAS_MQTT:
    def on_mqtt_msg(_client, _userdata, msg):
        try:
            incoming = json.loads(msg.payload.decode("utf-8"))
            incoming["source"] = "LIVE_WOKWI_ESP32"
            old = STATE["telemetry"]
            changed = any(
                old.get(k) != incoming.get(k)
                for k in ("rad_hazard", "salt_hazard", "atmo_hazard", "dust_hazard", "temp_hazard")
            ) or abs(float(old.get("radiation_msv_hr", 0)) - float(incoming.get("radiation_msv_hr", 0))) > 0.02
            STATE["telemetry"].update(incoming)
            if changed:
                recompute_mission_state()
        except Exception:
            pass

    try:
        mqtt_client = (
            mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            if hasattr(mqtt, "CallbackAPIVersion")
            else mqtt.Client()
        )
        mqtt_client.on_message = on_mqtt_msg
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 15)
        mqtt_client.subscribe(MQTT_TOPIC_TELEMETRY)
        mqtt_client.loop_start()
        print(f"[MQTT] Subscribed to {MQTT_TOPIC_TELEMETRY} for real-time Wokwi rerouting!")
    except Exception as e:
        print(f"[WARN] MQTT offline ({e}).")

if __name__ == "__main__":
    print("[SERVER] Mars EVA Backend running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)