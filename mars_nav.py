import os
# MUST execute before importing torch to prevent CUDA kernel mismatch crashes
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import json
import heapq
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
import torch

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

# ==========================================
# 0. PATHS, DEVICE & MODEL INITIALIZATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
device = torch.device("cpu")

NAVCAM_PATH = BASE_DIR / "visor_navcam.jpg"
SCIENCE_PATH = BASE_DIR / "science_macro.jpg"
ROUTE_PLOT_PATH = BASE_DIR / "gale_eva_route.png"
JSON_OUT_PATH = BASE_DIR / "navigation_packet.json"

HEADERS = {"User-Agent": "MarsEVACoPilotDemo/1.0"}
START_SOL = 1000

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_TELEMETRY = "gale_eva_mvp_2026/telemetry"
MQTT_TOPIC_HUD_ACK = "gale_eva_mvp_2026/hud_ack"


def load_depth_models(dev):
    """Loads MiDaS_small and its transform safely with an offline geometric fallback."""
    try:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(dev).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        return model, transforms.small_transform
    except Exception as exc:
        print(f"[WARN] MiDaS hub load failed ({exc}). Using geometric horizon depth fallback.")
        return None, None


midas, midas_transforms = load_depth_models(device)


# ==========================================
# 1. NASA API FETCH & OFFLINE FALLBACKS
# ==========================================
def fetch_mars2020_bundle(sol_num):
    try:
        api_url = (
            f"https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020"
            f"&feedtype=json&sol={sol_num}&num=100&page=0"
        )
        resp = requests.get(api_url, headers=HEADERS, timeout=4)
        resp.raise_for_status()
        images = resp.json().get("images", [])

        n_urls, s_urls = [], []
        for img in images:
            files = img.get("image_files", {})
            url = files.get("large") or files.get("full_res") or files.get("medium")
            cam = img.get("camera", {}).get("instrument", "").upper()
            img_id = img.get("imageid", "unknown")

            if url:
                if "NAVCAM" in cam:
                    n_urls.append((img_id, url))
                elif any(k in cam for k in ("SHERLOC", "WATSON", "PIXL")):
                    s_urls.append((img_id, url))
        return n_urls, s_urls
    except Exception:
        return [], []


def make_fallback_nav_image():
    img = Image.new("RGB", (640, 480), color=(155, 95, 60))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 640, 160], fill=(210, 175, 140))
    draw.ellipse([160, 260, 310, 370], fill=(85, 50, 35))
    draw.ellipse([380, 220, 500, 310], fill=(100, 60, 42))
    draw.polygon([(80, 420), (140, 350), (210, 430)], fill=(70, 42, 30))
    return img


def make_fallback_sci_image():
    rng = np.random.default_rng(42)
    base = rng.normal(loc=130, scale=22, size=(480, 640, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.line([(60, 120), (280, 240), (550, 190)], fill=(235, 220, 195), width=6)
    draw.line([(120, 380), (340, 290), (590, 340)], fill=(220, 205, 180), width=4)
    for x, y, r in [(180, 180, 28), (420, 260, 35), (290, 340, 22), (480, 130, 19)]:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(95, 65, 50), outline=(60, 40, 30))
    return img


active_sol = START_SOL
if NAVCAM_PATH.exists() and SCIENCE_PATH.exists():
    nav_img = Image.open(NAVCAM_PATH).convert("RGB")
    sci_img = Image.open(SCIENCE_PATH).convert("RGB")
    nav_id, sci_id = "CACHED_NAV", "CACHED_SCI"
else:
    navcam_urls, science_urls = [], []
    for current_sol in range(START_SOL, START_SOL - 6, -1):
        n_urls, s_urls = fetch_mars2020_bundle(current_sol)
        if n_urls and s_urls:
            active_sol = current_sol
            navcam_urls, science_urls = n_urls, s_urls
            break
    if navcam_urls and science_urls:
        nav_id, nav_url = navcam_urls[0]
        sci_id, sci_url = science_urls[0]
        try:
            nav_img = Image.open(BytesIO(requests.get(nav_url, headers=HEADERS, timeout=6).content)).convert("RGB")
            sci_img = Image.open(BytesIO(requests.get(sci_url, headers=HEADERS, timeout=6).content)).convert("RGB")
        except Exception:
            nav_img, sci_img = make_fallback_nav_image(), make_fallback_sci_image()
            nav_id, sci_id = "FALLBACK_NAV", "FALLBACK_SCI"
    else:
        nav_img, sci_img = make_fallback_nav_image(), make_fallback_sci_image()
        nav_id, sci_id = "FALLBACK_NAV", "FALLBACK_SCI"
    nav_img.save(NAVCAM_PATH)
    sci_img.save(SCIENCE_PATH)

# Fetch Orbital Map Once
lat_t, lon_t, zoom_t = -4.59, 137.44, 5
tile_col = int((lon_t + 180.0) / 360.0 * (2 ** (zoom_t + 1)))
tile_row = int((90.0 - lat_t) / 180.0 * (2 ** zoom_t))
trek_url = (
    f"https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/"
    f"1.0.0/default/default028mm/{zoom_t}/{tile_row}/{tile_col}.jpg"
)
try:
    orbital_map = np.array(Image.open(BytesIO(requests.get(trek_url, timeout=5).content)).convert("RGB"))
except Exception:
    orbital_map = np.full((256, 256, 3), 165, dtype=np.uint8)


# ==========================================
# 2. PRECOMPUTE CV DEPTH & SCIENCE CONTOURS
# ==========================================
visor_rgb = np.array(nav_img)
h_img, w_img, _ = visor_rgb.shape

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
base_annotated_visor = np.clip(visor_rgb * (1.0 - ground_mask) + hud_colormap * ground_mask, 0, 255).astype(np.uint8)
local_bev = cv2.resize(hazard_map_2d[int(h_img * 0.40):, :], (7, 8), interpolation=cv2.INTER_AREA)

# Science macro contours
sci_rgb = np.array(sci_img)
sci_edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(sci_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0), 45, 130)
contours, _ = cv2.findContours(sci_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
annotated_science = sci_rgb.copy()
cv2.drawContours(annotated_science, contours, -1, (0, 255, 180), 1)
mineral_feature_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 12.0])


# ==========================================
# 3. COORDINATE GRID & TARGET DEFINITIONS
# ==========================================
GALE_LAT_MIN, GALE_LAT_MAX = -4.65, -4.50
GALE_LON_MIN, GALE_LON_MAX = 137.35, 137.50
GRID_SIZE = 20
CELL_SIZE_M = 5.0


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


start_pt = (17, 3)
start_lat, start_lon = grid_to_latlon(*start_pt)

PRESET_TARGETS = {
    "1": ("Mount Sharp Smectite Clay Unit", (-4.515, 137.470), "Hydrated phyllosilicates (biosignature preservation)"),
    "2": ("Vera Rubin Hematite Ridge",      (-4.525, 137.390), "Iron-oxide concretions (ancient redox boundary)"),
    "3": ("Gediz Vallis Sulfate Channel",   (-4.580, 137.485), "Late-stage evaporative sulfates & boulder deposit"),
}

print(f"\n=== GALE CRATER EVA NAVIGATOR (Point A Visor Pos: {start_lat}, {start_lon} | Grid {start_pt}) ===")
for k, (name, coords, _) in PRESET_TARGETS.items():
    print(f"  [{k}] {name} -> Lat/Lon: {coords} (Grid {latlon_to_grid(*coords)})")

if len(sys.argv) > 1:
    user_choice = sys.argv[1].strip()
else:
    try:
        user_choice = input("Choose Target [1/2/3] (or press Enter for [3] Gediz Vallis): ").strip() or "3"
    except EOFError:
        user_choice = "3"

if user_choice not in PRESET_TARGETS:
    user_choice = "3"


# ==========================================
# 4. DYNAMIC 5-HAZARD SPATIAL MAP & A* ENGINE
# ==========================================
STATE = {
    "target_key": user_choice,
    "telemetry": {
        "source": "DEFAULT_NOMINAL (Press R/S/D/T/A or adjust Wokwi)",
        "radiation_msv_hr": 0.035,
        "perchlorate_pct": 0.10,
        "suit_kpa": 32.1,
        "visibility_pct": 85.0,
        "temp_c": -10.0,
        "rad_hazard": False,     # OFF by default -> NO Red Zone!
        "salt_hazard": False,    # OFF by default -> NO Purple Zone!
        "atmo_hazard": False,
        "dust_hazard": False,
        "temp_hazard": False,
    },
    "needs_redraw": True,
}

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


def stamp_circular_hazard(cost_grid, hazard_layer, center_rc, radius, label_code, cost_val=np.inf):
    """Stamps a circular spatial hazard onto the grid so A* must curve around it."""
    cr, cc = center_rc
    stamped = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if np.hypot(r - cr, c - cc) <= radius:
                cost_grid[r, c] = cost_val
                hazard_layer[r, c] = label_code
                stamped.append((r, c))
    return stamped


def build_dynamic_environment():
    """Constructs the 20x20 cost grid and spatial hazard zones from current Wokwi telemetry."""
    tel = STATE["telemetry"]
    t_name, (g_lat, g_lon), sci_rat = PRESET_TARGETS[STATE["target_key"]]
    base_goal_pt = latlon_to_grid(g_lat, g_lon)

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
    # hazard_layer codes: 0=None, 1=CV_Boulder, 2=CV_Slope, 3=RAD, 4=SALT, 5=DUST, 6=TEMP
    hazard_layer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    zone_centers = {}

    # 1. Project foreground CV hazards along heading vector
    vec = np.array([base_goal_pt[0] - start_pt[0], base_goal_pt[1] - start_pt[1]], dtype=float)
    norm = np.linalg.norm(vec)
    unit_vec = vec / norm if norm > 0 else np.array([-1.0, 0.0])
    right_vec = np.array([unit_vec[1], -unit_vec[0]])

    cv_cutoff = 0.40 if dust_haz else 0.55
    slope_mult = 16.0 if (salt_haz or atmo_haz) else 11.0

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

    # Suit Pressure Leak (atmo_hazard): Astronaut cannot climb rough slopes!
    if atmo_haz:
        steep_mask = (cost_grid > 2.0) & (~np.isinf(cost_grid))
        cost_grid[steep_mask] = np.inf
        hazard_layer[steep_mask] = 1

    # 2. Wokwi Hazard 1: RADIATION ZONE (ONLY active when rad_hazard is True!)
    if rad_haz:
        # Placed 38% along the direct path from Point A to Point B so A* MUST reroute around it!
        rad_center = (
            int(round(start_pt[0] + 0.38 * vec[0])),
            int(round(start_pt[1] + 0.38 * vec[1])),
        )
        # Radius grows dynamically from 1.8 cells (at 0.11 mSv/h) to 3.4 cells (at 0.50 mSv/h)
        rad_radius = 1.6 + np.clip((rad_msv - 0.10) / 0.40, 0.0, 1.0) * 1.8
        stamp_circular_hazard(cost_grid, hazard_layer, rad_center, rad_radius, label_code=3, cost_val=np.inf)
        zone_centers["RAD ZONE"] = (rad_center, "white", "red")

    # 3. Wokwi Hazard 2: PERCHLORATE SALT BASIN (ONLY active when salt_hazard is True!)
    if salt_haz:
        # Placed 68% along the direct path + slightly offset so it creates a second S-bend
        salt_center = (
            int(round(start_pt[0] + 0.68 * vec[0] + 1.2 * right_vec[0])),
            int(round(start_pt[1] + 0.68 * vec[1] + 1.2 * right_vec[1])),
        )
        salt_radius = 1.5 + np.clip((salt_pct - 0.60) / 1.40, 0.0, 1.0) * 1.5
        stamp_circular_hazard(cost_grid, hazard_layer, salt_center, salt_radius, label_code=4, cost_val=np.inf)
        zone_centers["PERCHLORATE"] = (salt_center, "white", "purple")

    # 4. Wokwi Hazard 4: DUST STORM SQUALL WALL (ONLY active when dust_hazard is True!)
    if dust_haz:
        dust_center = (
            int(round(start_pt[0] + 0.52 * vec[0] - 1.8 * right_vec[0])),
            int(round(start_pt[1] + 0.52 * vec[1] - 1.8 * right_vec[1])),
        )
        dust_radius = 1.6 + np.clip((35.0 - vis_pct) / 35.0, 0.0, 1.0) * 1.4
        stamp_circular_hazard(cost_grid, hazard_layer, dust_center, dust_radius, label_code=5, cost_val=np.inf)
        zone_centers["DUST SQUALL"] = (dust_center, "black", "sandybrown")

    # 5. Wokwi Hazard 5: EXTREME TEMP ZONE (ONLY active when temp_hazard is True!)
    if temp_haz:
        temp_center = (
            int(round(start_pt[0] + 0.25 * vec[0] - 2.2 * right_vec[0])),
            int(round(start_pt[1] + 0.25 * vec[1] - 2.2 * right_vec[1])),
        )
        temp_label = "CRYO TRAP" if temp_c < -25.0 else "HEAT TRAP"
        temp_bg = "deepskyblue" if temp_c < -25.0 else "deeppink"
        stamp_circular_hazard(cost_grid, hazard_layer, temp_center, 1.8, label_code=6, cost_val=np.inf)
        zone_centers[temp_label] = (temp_center, "white", temp_bg)

    # Guarantee Start & Goal are walkable
    cost_grid[start_pt] = 1.0
    hazard_layer[start_pt] = 0

    goal_pt = base_goal_pt
    if np.isinf(cost_grid[goal_pt]):
        safe_cells = np.argwhere(~np.isinf(cost_grid))
        dists = np.linalg.norm(safe_cells - np.array(goal_pt), axis=1)
        goal_pt = tuple(map(int, safe_cells[np.argmin(dists)]))

    safe_path = weighted_astar(cost_grid, start_pt, goal_pt)
    return cost_grid, hazard_layer, zone_centers, goal_pt, safe_path


def generate_eva_instructions(path, cost_grid, hazard_layer, goal_pt):
    t_name, _, _ = PRESET_TARGETS[STATE["target_key"]]
    goal_lat, goal_lon = grid_to_latlon(*goal_pt)
    if len(path) < 2:
        return ["HOLD POSITION: All corridors blocked by extreme hazards!"], 0.0

    LABEL_NAMES = {
        1: "Visor Steep Boulder",
        3: "Radiation Plume",
        4: "Toxic Perchlorate Basin",
        5: "Dust Storm Squall",
        6: "Extreme Thermal Zone",
    }

    segments = []
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        vec = (r2 - r1, c2 - c1)
        step_m = float(np.hypot(vec[0], vec[1]) * CELL_SIZE_M)

        nearby = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r2 + dr, c2 + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    code = int(hazard_layer[nr, nc])
                    if code in LABEL_NAMES:
                        nearby.add(LABEL_NAMES[code])

        if segments and segments[-1]["vec"] == vec:
            segments[-1]["end_cell"] = (r2, c2)
            segments[-1]["dist_m"] += step_m
            segments[-1]["steps"] += 1
            segments[-1]["hazards"].update(nearby)
        else:
            segments.append({
                "vec": vec, "end_cell": (r2, c2), "dist_m": step_m, "steps": 1, "hazards": set(nearby)
            })

    instructions = []
    total_dist_m = 0.0
    for idx, seg in enumerate(segments, 1):
        compass, deg = DIRECTION_MAP.get(seg["vec"], ("FWD", "---"))
        dist = round(seg["dist_m"], 1)
        total_dist_m += seg["dist_m"]
        end_lat, end_lon = grid_to_latlon(*seg["end_cell"])
        action = "Start EVA: Walk" if idx == 1 else "Turn & walk"
        haz_str = f" | AVOIDING: {', '.join(sorted(seg['hazards']))}" if seg["hazards"] else ""
        instructions.append(
            f"LEG {idx}: {action} {compass} ({deg}) for {dist} m to {seg['end_cell']} [{end_lat}, {end_lon}]{haz_str}"
        )

    instructions.append(f"ARRIVAL: Point B ({t_name}) at Grid {goal_pt} [{goal_lat}, {goal_lon}].")
    return instructions, round(total_dist_m, 1)


# ==========================================
# 5. LIVE MATPLOTLIB DASHBOARD & KEYBOARD UI
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6.6))


def update_dashboard():
    tel = STATE["telemetry"]
    t_name, _, sci_rat = PRESET_TARGETS[STATE["target_key"]]
    cost_grid, hazard_layer, zone_centers, goal_pt, safe_path = build_dynamic_environment()
    eva_instructions, total_distance_m = generate_eva_instructions(safe_path, cost_grid, hazard_layer, goal_pt)

    rad_msv = float(tel.get("radiation_msv_hr", 0.035))
    salt_pct = float(tel.get("perchlorate_pct", 0.10))
    suit_kpa = float(tel.get("suit_kpa", 32.1))
    vis_pct = float(tel.get("visibility_pct", 85.0))
    temp_c = float(tel.get("temp_c", -10.0))

    dust_haz = bool(tel.get("dust_hazard", vis_pct < 35.0))
    temp_haz = bool(tel.get("temp_hazard", temp_c < -25.0 or temp_c > 35.0))
    atmo_haz = bool(tel.get("atmo_hazard", suit_kpa < 25.0))

    walk_speed = 0.55 if (temp_haz or dust_haz) else 0.80
    leak_mult = 1.0 + max(0.0, (26.0 - suit_kpa) * 0.09) if atmo_haz else 1.0
    est_time_min = round((total_distance_m / walk_speed) / 60.0, 1)
    est_o2_pct = round(total_distance_m * 0.025 * leak_mult, 2)

    active_flags = [
        name for name, flag in [
            ("RADIATION", bool(tel.get("rad_hazard", rad_msv > 0.100))),
            ("PERCHLORATE", bool(tel.get("salt_hazard", salt_pct > 0.60))),
            ("SUIT LEAK", atmo_haz),
            ("DUST STORM", dust_haz),
            ("EXTREME TEMP", temp_haz),
        ] if flag
    ]
    status_str = ", ".join(active_flags) if active_flags else "ALL NOMINAL (CLEAR CORRIDOR)"

    # Console output
    print("\n" + "=" * 92)
    print(f"LIVE EVA ROUTER [{tel.get('source')}] | Active Hazards: {status_str}")
    print(f"Target: {t_name} {goal_pt} | Dist: {total_distance_m} m | Time: {est_time_min} min | O2: {est_o2_pct}%")
    for line in eva_instructions:
        print("  • " + line)
    print("=" * 92)

    # Clear axes and redraw cleanly without overlapping titles
    for ax in axes:
        ax.clear()

    fig.suptitle(
        f"Wokwi Suit Telemetry [{tel.get('source')}]  |  Active Alerts: {status_str}\n"
        f"RAD: {rad_msv:.3f} mSv/h  •  Salt: {salt_pct:.2f}%  •  Suit: {suit_kpa:.1f} kPa  •  "
        f"Vis: {vis_pct:.0f}%  •  Temp: {temp_c:.1f}°C",
        fontsize=11, fontweight="bold", y=0.97
    )

    # Panel 1: Visor HUD (Tints amber during a Dust Storm)
    visor_img = base_annotated_visor.copy()
    if dust_haz:
        amber = np.full_like(visor_img, (215, 115, 35), dtype=np.uint8)
        visor_img = cv2.addWeighted(visor_img, 0.65, amber, 0.35, 0)
    num_cv_cells = int(np.sum((hazard_layer == 1) | (hazard_layer == 2)))
    axes[0].imshow(visor_img)
    axes[0].set_title(
        f"1. Point A Visor HUD ({nav_id})\n{num_cv_cells} CV Rock/Slope Cells | Vis: {vis_pct:.0f}%",
        fontsize=10, pad=8
    )
    axes[0].axis("off")

    # Panel 2: Dynamic Gale Crater Map
    h, w, _ = orbital_map.shape
    cell_h, cell_w = h / GRID_SIZE, w / GRID_SIZE
    axes[1].imshow(orbital_map)

    COLOR_MAP = {
        1: ("darkorange", 0.55),  # CV Impassable Boulder
        2: ("gold",       0.32),  # CV Rough Slope
        3: ("red",        0.65),  # Wokwi Radiation Plume
        4: ("purple",     0.65),  # Wokwi Perchlorate Salt Basin
        5: ("sandybrown", 0.65),  # Wokwi Dust Storm Squall
        6: ("deepskyblue" if temp_c < 0 else "deeppink", 0.65),  # Wokwi Extreme Temp
    }

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            code = int(hazard_layer[r, c])
            if code in COLOR_MAP:
                col_name, alpha_val = COLOR_MAP[code]
                axes[1].add_patch(
                    plt.Rectangle((c * cell_w, r * cell_h), cell_w, cell_h, color=col_name, alpha=alpha_val)
                )

    for label_txt, ((cr, cc), fg_col, bg_col) in zone_centers.items():
        axes[1].text(
            (cc + 0.5) * cell_w, (cr + 0.5) * cell_h, label_txt,
            color=fg_col, fontsize=7.5, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_col, alpha=0.85, edgecolor="white")
        )

    if safe_path:
        py = [(r + 0.5) * cell_h for r, _ in safe_path]
        px = [(c + 0.5) * cell_w for _, c in safe_path]
        axes[1].plot(px, py, color="lime", linewidth=3.2, label=f"Safe A* Path ({total_distance_m} m)")

    axes[1].scatter([(start_pt[1] + 0.5) * cell_w], [(start_pt[0] + 0.5) * cell_h], color="cyan", s=110, zorder=5, label=f"Point A {start_pt}")
    axes[1].scatter([(goal_pt[1] + 0.5) * cell_w], [(goal_pt[0] + 0.5) * cell_h], color="yellow", marker="*", s=220, zorder=5, label=f"Point B {goal_pt}")
    axes[1].set_title(
        f"2. Gale Crater Route: {t_name}\nDist: {total_distance_m} m | Time: {est_time_min} min | Est. O2: {est_o2_pct}%",
        fontsize=10, pad=8
    )
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].axis("off")

    # Panel 3: Micro-Science Inspector
    axes[2].imshow(annotated_science)
    axes[2].set_title(
        f"3. Point B Science Macro ({sci_id})\n{mineral_feature_count} Grain Boundaries | {sci_rat[:34]}...",
        fontsize=10, pad=8
    )
    axes[2].axis("off")

    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.02, right=0.98, wspace=0.08)
    plt.savefig(ROUTE_PLOT_PATH, dpi=150)
    fig.canvas.draw_idle()

    # Export updated JSON for Java UI
    with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "wokwi_suit_telemetry": tel,
            "active_hazards": active_flags,
            "point_a_visor_start": {"grid": list(start_pt)},
            "point_b_target": {"name": t_name, "grid": list(goal_pt)},
            "route_metrics": {
                "total_distance_m": total_distance_m,
                "estimated_walk_time_min": est_time_min,
                "estimated_o2_cost_pct": est_o2_pct,
            },
            "turn_by_turn_instructions": eva_instructions,
        }, f, indent=2)


def on_key_press(event):
    """Interactive keyboard controls so you can toggle any hazard live in the Figure window."""
    if not event.key:
        return
    k = event.key.lower()
    tel = STATE["telemetry"]

    if k in ("1", "2", "3"):
        STATE["target_key"] = k
        tel["source"] = f"KEYBOARD_TARGET_{k}"
    elif k == "r":
        # Cycle Radiation: OFF -> MODERATE (0.18) -> SEVERE (0.45) -> OFF
        if not tel["rad_hazard"]:
            tel["radiation_msv_hr"], tel["rad_hazard"] = 0.180, True
        elif tel["radiation_msv_hr"] < 0.35:
            tel["radiation_msv_hr"], tel["rad_hazard"] = 0.450, True
        else:
            tel["radiation_msv_hr"], tel["rad_hazard"] = 0.030, False
        tel["source"] = "KEYBOARD_OVERRIDE (RAD)"
    elif k == "s":
        tel["salt_hazard"] = not tel["salt_hazard"]
        tel["perchlorate_pct"] = 1.45 if tel["salt_hazard"] else 0.10
        tel["source"] = "KEYBOARD_OVERRIDE (SALT)"
    elif k == "d":
        tel["dust_hazard"] = not tel["dust_hazard"]
        tel["visibility_pct"] = 18.0 if tel["dust_hazard"] else 85.0
        tel["source"] = "KEYBOARD_OVERRIDE (DUST)"
    elif k == "t":
        if not tel["temp_hazard"]:
            tel["temp_c"], tel["temp_hazard"] = -38.0, True
        elif tel["temp_c"] < 0:
            tel["temp_c"], tel["temp_hazard"] = 46.0, True
        else:
            tel["temp_c"], tel["temp_hazard"] = -10.0, False
        tel["source"] = "KEYBOARD_OVERRIDE (TEMP)"
    elif k == "a":
        tel["atmo_hazard"] = not tel["atmo_hazard"]
        tel["suit_kpa"] = 19.5 if tel["atmo_hazard"] else 32.1
        tel["source"] = "KEYBOARD_OVERRIDE (SUIT_KPA)"
    elif k == "c":
        tel.update({
            "source": "ALL_NOMINAL_RESET",
            "radiation_msv_hr": 0.030, "rad_hazard": False,
            "perchlorate_pct": 0.10,   "salt_hazard": False,
            "suit_kpa": 32.1,          "atmo_hazard": False,
            "visibility_pct": 85.0,    "dust_hazard": False,
            "temp_c": -10.0,           "temp_hazard": False,
        })
    else:
        return

    update_dashboard()


fig.text(
    0.5, 0.015,
    "LIVE CONTROLS — Wokwi MQTT Auto-Sync Active  |  Keyboard Shortcuts:  "
    "[R] Radiation (Off/Med/High)   [S] Perchlorate Salt   [D] Dust Storm   "
    "[T] Temp (Cryo/Hot)   [A] Suit Leak   [1/2/3] Switch Target   [C] Clear All Hazards",
    ha="center", va="bottom", fontsize=9.5, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0f4f8", edgecolor="#8899a6")
)
fig.canvas.mpl_connect("key_press_event", on_key_press)


# ==========================================
# 6. BACKGROUND LIVE WOKWI MQTT LISTENER
# ==========================================
mqtt_client = None
if HAS_MQTT:
    def on_mqtt_msg(_client, _userdata, msg):
        try:
            incoming = json.loads(msg.payload.decode("utf-8"))
            incoming["source"] = "LIVE_WOKWI_ESP32"
            # Only trigger redraw if a hazard state or value meaningfully changed
            old = STATE["telemetry"]
            changed = any(
                old.get(k) != incoming.get(k)
                for k in ("rad_hazard", "salt_hazard", "atmo_hazard", "dust_hazard", "temp_hazard")
            ) or abs(float(old.get("radiation_msv_hr", 0)) - float(incoming.get("radiation_msv_hr", 0))) > 0.02
            STATE["telemetry"].update(incoming)
            if changed or old.get("source") != "LIVE_WOKWI_ESP32":
                STATE["needs_redraw"] = True
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
        print(f"[MQTT] Listening continuously to {MQTT_TOPIC_TELEMETRY} for live Wokwi knob changes...")
    except Exception as e:
        print(f"[WARN] Live MQTT listener offline ({e}). Use keyboard keys [R/S/D/T/A] in Figure window.")


def check_mqtt_timer():
    if STATE["needs_redraw"]:
        STATE["needs_redraw"] = False
        update_dashboard()


timer = fig.canvas.new_timer(interval=400)
timer.add_callback(check_mqtt_timer)
timer.start()

# Initial draw & show
update_dashboard()
plt.show()

if mqtt_client is not None:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
