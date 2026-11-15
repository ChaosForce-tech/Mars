import os
# MUST execute before importing torch to prevent CUDA kernel mismatch crashes
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import time
import json
import heapq
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
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
# 1. LIVE WOKWI ESP32 TELEMETRY INGESTION
# ==========================================
DEFAULT_WOKWI_TELEMETRY = {
    "source": "DEFAULT_FALLBACK",
    "radiation_msv_hr": 0.037,
    "perchlorate_pct": 0.10,
    "suit_kpa": 32.1,
    "visibility_pct": 85.0,
    "temp_c": -10.0,
    "rad_hazard": False,
    "salt_hazard": False,
    "atmo_hazard": False,
    "dust_hazard": False,
    "temp_hazard": False,
}


def fetch_wokwi_telemetry(timeout_sec=3.0):
    """Subscribes to HiveMQ to pull live 5-hazard telemetry from the Wokwi ESP32 simulator."""
    if not HAS_MQTT:
        print("[WARN] paho-mqtt not installed. Using nominal suit telemetry defaults.")
        return DEFAULT_WOKWI_TELEMETRY.copy()

    received = {}

    def on_message(_client, _userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            data["source"] = "LIVE_WOKWI_ESP32"
            received.update(data)
        except Exception:
            pass

    try:
        client = (
            mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            if hasattr(mqtt, "CallbackAPIVersion")
            else mqtt.Client()
        )
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 15)
        client.subscribe(MQTT_TOPIC_TELEMETRY)
        client.loop_start()

        deadline = time.time() + timeout_sec
        while time.time() < deadline and not received:
            time.sleep(0.05)

        client.loop_stop()
        client.disconnect()
    except Exception as exc:
        print(f"[WARN] MQTT broker unreachable ({exc}). Using fallback suit telemetry.")

    if received:
        print(f"[WOKWI LIVE] Telemetry locked from {MQTT_TOPIC_TELEMETRY}: {json.dumps(received)}")
        return received

    print("[INFO] No live Wokwi packet detected within 3s window (is simulation running?). Using nominal telemetry.")
    return DEFAULT_WOKWI_TELEMETRY.copy()


def publish_hud_to_wokwi(summary_dict):
    """Pushes the calculated A* route summary back to the Wokwi ESP32 Serial Monitor."""
    if not HAS_MQTT:
        return
    try:
        client = (
            mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            if hasattr(mqtt, "CallbackAPIVersion")
            else mqtt.Client()
        )
        client.connect(MQTT_BROKER, MQTT_PORT, 10)
        client.publish(MQTT_TOPIC_HUD_ACK, json.dumps(summary_dict))
        client.disconnect()
    except Exception:
        pass


wokwi_data = fetch_wokwi_telemetry(timeout_sec=3.0)


# ==========================================
# 2. NASA API FETCH & OFFLINE FALLBACKS
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


navcam_urls, science_urls = [], []
active_sol = START_SOL

for current_sol in range(START_SOL, START_SOL - 8, -1):
    n_urls, s_urls = fetch_mars2020_bundle(current_sol)
    if n_urls and s_urls:
        active_sol = current_sol
        navcam_urls, science_urls = n_urls, s_urls
        print(f"[INFO] NASA API: Found paired Navcam + Macro bundle on Sol {active_sol}.")
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
elif NAVCAM_PATH.exists() and SCIENCE_PATH.exists():
    nav_img = Image.open(NAVCAM_PATH).convert("RGB")
    sci_img = Image.open(SCIENCE_PATH).convert("RGB")
    nav_id, sci_id = "CACHED_NAV", "CACHED_SCI"
else:
    nav_img, sci_img = make_fallback_nav_image(), make_fallback_sci_image()
    nav_id, sci_id = "FALLBACK_NAV", "FALLBACK_SCI"

nav_img.save(NAVCAM_PATH)
sci_img.save(SCIENCE_PATH)


# ==========================================
# 3. GALE CRATER COORDINATE & TARGET INPUT
# ==========================================
GALE_LAT_MIN, GALE_LAT_MAX = -4.65, -4.50
GALE_LON_MIN, GALE_LON_MAX = 137.35, 137.50
GRID_SIZE = 20
CELL_SIZE_M = 5.0


def latlon_to_grid(lat, lon):
    lat_clamped = np.clip(lat, GALE_LAT_MIN, GALE_LAT_MAX)
    lon_clamped = np.clip(lon, GALE_LON_MIN, GALE_LON_MAX)
    row = int(round((GALE_LAT_MAX - lat_clamped) / (GALE_LAT_MAX - GALE_LAT_MIN) * (GRID_SIZE - 1)))
    col = int(round((lon_clamped - GALE_LON_MIN) / (GALE_LON_MAX - GALE_LON_MIN) * (GRID_SIZE - 1)))
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
print("  [4] Enter custom Gale Crater Grid (row,col) or Lat,Lon")

if len(sys.argv) > 1:
    user_choice = sys.argv[1].strip()
else:
    try:
        user_choice = input("Choose [1/2/3/4] (or press Enter for default [1]): ").strip() or "1"
    except EOFError:
        user_choice = "1"

if user_choice in PRESET_TARGETS:
    target_name, (goal_lat, goal_lon), science_rationale = PRESET_TARGETS[user_choice]
    goal_pt = latlon_to_grid(goal_lat, goal_lon)
else:
    raw_in = user_choice if "," in user_choice else ""
    if not raw_in:
        try:
            raw_in = input("Enter 'row,col' (0-19) OR 'lat,lon' (e.g. -4.52,137.46): ").strip()
        except EOFError:
            raw_in = ""
    try:
        v1, v2 = [float(x.strip()) for x in raw_in.split(",")]
        if 0 <= v1 < GRID_SIZE and 0 <= v2 < GRID_SIZE:
            goal_pt = (int(v1), int(v2))
            goal_lat, goal_lon = grid_to_latlon(*goal_pt)
            target_name = f"Custom Grid Target {goal_pt}"
        else:
            goal_pt = latlon_to_grid(v1, v2)
            goal_lat, goal_lon = grid_to_latlon(*goal_pt)
            target_name = f"Custom Gale Waypoint ({goal_lat}, {goal_lon})"
        science_rationale = "User-designated geological sampling waypoint"
    except Exception:
        target_name, (goal_lat, goal_lon), science_rationale = PRESET_TARGETS["1"]
        goal_pt = latlon_to_grid(goal_lat, goal_lon)


# ==========================================
# 4. CV DEPTH + WOKWI 5-HAZARD FUSION
# ==========================================
visor_rgb = np.array(Image.open(NAVCAM_PATH).convert("RGB"))
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
annotated_visor = np.clip(visor_rgb * (1.0 - ground_mask) + hud_colormap * ground_mask, 0, 255).astype(np.uint8)

# Wokwi Hazard 4: Dust Storm Visibility lowers CV obstacle cutoff & tints visor
vis_pct = float(wokwi_data.get("visibility_pct", 85.0))
dust_hazard_active = bool(wokwi_data.get("dust_hazard", False))
cv_risk_cutoff = 0.42 if dust_hazard_active else 0.55

if dust_hazard_active:
    dust_tint = np.full_like(annotated_visor, (215, 120, 45), dtype=np.uint8)
    annotated_visor = cv2.addWeighted(annotated_visor, 0.72, dust_tint, 0.28, 0)

cost_grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=float)
ground_crop = hazard_map_2d[int(h_img * 0.40):, :]
local_bev = cv2.resize(ground_crop, (7, 8), interpolation=cv2.INTER_AREA)

heading_vec = np.array([goal_pt[0] - start_pt[0], goal_pt[1] - start_pt[1]], dtype=float)
heading_norm = np.linalg.norm(heading_vec)
heading_vec = heading_vec / heading_norm if heading_norm > 0 else np.array([-1.0, 0.0])
right_vec = np.array([heading_vec[1], -heading_vec[0]])

# Wokwi Hazard 2: Perchlorate salts increase dust-kickup slope cost
salt_pct = float(wokwi_data.get("perchlorate_pct", 0.10))
salt_hazard_active = bool(wokwi_data.get("salt_hazard", False))
slope_penalty_mult = 18.0 if salt_hazard_active else 12.0

for d_step in range(1, 9):
    for lat_idx, offset in enumerate(range(-3, 4)):
        r = int(round(start_pt[0] + d_step * heading_vec[0] + offset * 0.7 * right_vec[0]))
        c = int(round(start_pt[1] + d_step * heading_vec[1] + offset * 0.7 * right_vec[1]))
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and (r, c) != start_pt:
            cell_risk = float(local_bev[8 - d_step, lat_idx])
            if cell_risk > cv_risk_cutoff:
                cost_grid[r, c] = np.inf
            else:
                cost_grid[r, c] = max(cost_grid[r, c], 1.0 + slope_penalty_mult * cell_risk)

num_cv_hazards = int(np.sum(cost_grid > 2.2))

# Wokwi Hazard 1: Ionizing Radiation dynamically expands RAD Zone
rad_msv = float(wokwi_data.get("radiation_msv_hr", 0.037))
rad_hazard_active = bool(wokwi_data.get("rad_hazard", False))
if rad_msv > 0.300:
    RAD_ROWS, RAD_COLS = (3, 10), (9, 16)   # Severe radiation flare (7x7 cells)
elif rad_hazard_active:
    RAD_ROWS, RAD_COLS = (4, 9), (10, 15)   # Elevated radiation zone (5x5 cells)
else:
    RAD_ROWS, RAD_COLS = (5, 8), (11, 14)   # Baseline RAD pocket (3x3 cells)

cost_grid[RAD_ROWS[0]:RAD_ROWS[1], RAD_COLS[0]:RAD_COLS[1]] = np.inf

# Wokwi Hazard 2 (cont.): High perchlorate concentration marks toxic salt basin
SALT_ROWS, SALT_COLS = (11, 14), (6, 9)
if salt_hazard_active:
    cost_grid[SALT_ROWS[0]:SALT_ROWS[1], SALT_COLS[0]:SALT_COLS[1]] = np.inf

cost_grid[start_pt] = 1.0

if np.isinf(cost_grid[goal_pt]):
    safe_cells = np.argwhere(~np.isinf(cost_grid))
    dists = np.linalg.norm(safe_cells - np.array(goal_pt), axis=1)
    nearest_safe = tuple(map(int, safe_cells[np.argmin(dists)]))
    print(f"[ALERT] Point B {goal_pt} is inside an active hazard zone! Auto-snapped to {nearest_safe}.")
    goal_pt = nearest_safe
    goal_lat, goal_lon = grid_to_latlon(*goal_pt)

# Micro-Science CV Grain/Vein Inspector
sci_rgb = np.array(Image.open(SCIENCE_PATH).convert("RGB"))
sci_gray = cv2.cvtColor(sci_rgb, cv2.COLOR_RGB2GRAY)
sci_edges = cv2.Canny(cv2.GaussianBlur(sci_gray, (5, 5), 0), 45, 130)
contours, _ = cv2.findContours(sci_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
annotated_science = sci_rgb.copy()
cv2.drawContours(annotated_science, contours, -1, (0, 255, 180), 1)
mineral_feature_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 12.0])


# ==========================================
# 5. WEIGHTED A* + TURN-BY-TURN GENERATOR
# ==========================================
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


DIRECTION_MAP = {
    (-1,  0): ("N",  "000°"), (-1,  1): ("NE", "045°"),
    ( 0,  1): ("E",  "090°"), ( 1,  1): ("SE", "135°"),
    ( 1,  0): ("S",  "180°"), ( 1, -1): ("SW", "225°"),
    ( 0, -1): ("W",  "270°"), (-1, -1): ("NW", "315°"),
}


def check_adjacent_hazards(r, c, grid):
    alerts = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if np.isinf(grid[nr, nc]):
                    if RAD_ROWS[0] <= nr < RAD_ROWS[1] and RAD_COLS[0] <= nc < RAD_COLS[1]:
                        alerts.add(f"Radiation Zone ({rad_msv:.3f} mSv/h)")
                    elif salt_hazard_active and SALT_ROWS[0] <= nr < SALT_ROWS[1] and SALT_COLS[0] <= nc < SALT_COLS[1]:
                        alerts.add(f"Toxic Perchlorate Basin ({salt_pct:.2f}%)")
                    else:
                        alerts.add("Visor-Detected Steep Boulder")
                elif grid[nr, nc] > 2.2:
                    alerts.add("Rough Regolith Slope")
    return sorted(alerts)


def generate_eva_instructions(path, grid, cell_m=5.0):
    if len(path) < 2:
        return ["HOLD POSITION: Target unreachable or already at destination."], 0.0

    segments = []
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        vec = (r2 - r1, c2 - c1)
        step_m = float(np.hypot(vec[0], vec[1]) * cell_m)
        hazards = check_adjacent_hazards(r2, c2, grid)
        terrain_cost = float(grid[r2, c2])

        if segments and segments[-1]["vec"] == vec:
            segments[-1]["end_cell"] = (r2, c2)
            segments[-1]["dist_m"] += step_m
            segments[-1]["steps"] += 1
            segments[-1]["max_cost"] = max(segments[-1]["max_cost"], terrain_cost)
            segments[-1]["hazards"].update(hazards)
        else:
            segments.append({
                "vec": vec, "start_cell": (r1, c1), "end_cell": (r2, c2),
                "dist_m": step_m, "steps": 1, "max_cost": terrain_cost, "hazards": set(hazards),
            })

    instructions = []
    total_dist_m = 0.0
    for idx, seg in enumerate(segments, 1):
        compass, deg = DIRECTION_MAP.get(seg["vec"], ("FWD", "---"))
        dist = round(seg["dist_m"], 1)
        total_dist_m += seg["dist_m"]
        end_lat, end_lon = grid_to_latlon(*seg["end_cell"])

        action = "Start EVA: Walk" if idx == 1 else "Turn & walk"
        surface_note = "Rough slope (slow pace)" if seg["max_cost"] > 2.2 else "Flat regolith (nominal)"
        hazard_note = f" | CAUTION: Bypassing {', '.join(sorted(seg['hazards']))}" if seg["hazards"] else ""

        instructions.append(
            f"LEG {idx}: {action} {compass} ({deg}) for {dist} m ({seg['steps']} steps) "
            f"to {seg['end_cell']} [{end_lat}, {end_lon}] — {surface_note}{hazard_note}"
        )

    # Append dynamic Wokwi suit advisories
    suit_alerts = []
    if bool(wokwi_data.get("atmo_hazard", False)):
        suit_alerts.append(f"SUIT PRESSURE DROP ({wokwi_data.get('suit_kpa', 0):.1f} kPa - check CO2 seal)")
    if salt_hazard_active:
        suit_alerts.append(f"PERCHLORATE DUST ({salt_pct:.2f}% - airlock decon required)")
    if dust_hazard_active:
        suit_alerts.append(f"LOW VISIBILITY ({vis_pct:.1f}% - HUD contrast boost active)")
    if bool(wokwi_data.get("temp_hazard", False)):
        suit_alerts.append(f"EXTREME TEMP ({wokwi_data.get('temp_c', -10.0):.1f} C - PLSS thermal regulation max)")

    if suit_alerts:
        instructions.append("WOKWI SUIT ALERTS: " + " | ".join(suit_alerts))

    instructions.append(
        f"ARRIVAL: Reached Point B ({target_name}) at Grid {path[-1]} [{goal_lat}, {goal_lon}]. "
        f"({mineral_feature_count} micro-structures flagged)."
    )
    return instructions, round(total_dist_m, 1)


safe_path = weighted_astar(cost_grid, start_pt, goal_pt)
eva_instructions, total_distance_m = generate_eva_instructions(safe_path, cost_grid, CELL_SIZE_M)

# Wokwi Hazard 3 & 5: Adjust Walk Speed (Temp/Dust) & O2 Burn (Suit Pressure Leak)
temp_hazard_active = bool(wokwi_data.get("temp_hazard", False))
suit_kpa = float(wokwi_data.get("suit_kpa", 32.1))
atmo_hazard_active = bool(wokwi_data.get("atmo_hazard", False))

walk_speed_mps = 0.55 if (temp_hazard_active or dust_hazard_active) else 0.80
leak_multiplier = 1.0 + max(0.0, (28.0 - suit_kpa) * 0.08) if atmo_hazard_active else 1.0

est_walk_time_min = round((total_distance_m / walk_speed_mps) / 60.0, 1)
est_o2_used_pct = round(total_distance_m * 0.025 * leak_multiplier, 2)


# ==========================================
# 6. PRINT HUD, PUBLISH TO WOKWI & PLOT
# ==========================================
print("\n" + "=" * 90)
print(f"OFFLINE VISOR HUD + WOKWI TELEMETRY ({wokwi_data.get('source')}): {start_pt} -> {goal_pt}")
print(
    f"Target: {target_name} | Dist: {total_distance_m} m | "
    f"Time: {est_walk_time_min} min | Est. O2: {est_o2_used_pct}%"
)
print(
    f"Wokwi Sensors -> RAD: {rad_msv:.3f} mSv/h | Perchlorate: {salt_pct:.2f}% | "
    f"Suit: {suit_kpa:.1f} kPa | Vis: {vis_pct:.1f}% | Temp: {float(wokwi_data.get('temp_c', -10.0)):.1f} C"
)
print("=" * 90)
for step_text in eva_instructions:
    print("  • " + step_text)
print("=" * 90 + "\n")

# Send calculated route back to Wokwi ESP32 Serial Monitor
publish_hud_to_wokwi({
    "target": target_name[:24],
    "dist_m": total_distance_m,
    "time_min": est_walk_time_min,
    "o2_pct": est_o2_used_pct,
    "first_leg": eva_instructions[0][:60] if eva_instructions else "HOLD",
})

lat, lon, zoom = -4.59, 137.44, 5
tile_col = int((lon + 180.0) / 360.0 * (2 ** (zoom + 1)))
tile_row = int((90.0 - lat) / 180.0 * (2 ** zoom))
trek_url = (
    f"https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/"
    f"1.0.0/default/default028mm/{zoom}/{tile_row}/{tile_col}.jpg"
)
try:
    orbital_map = np.array(Image.open(BytesIO(requests.get(trek_url, timeout=6).content)).convert("RGB"))
except Exception:
    orbital_map = np.full((256, 256, 3), 165, dtype=np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
fig.suptitle(
    f"Wokwi Suit Telemetry [{wokwi_data.get('source')}] — "
    f"RAD: {rad_msv:.3f} mSv/h | Salt: {salt_pct:.2f}% | Suit: {suit_kpa:.1f} kPa | "
    f"Vis: {vis_pct:.1f}% | Temp: {float(wokwi_data.get('temp_c', -10.0)):.1f}°C",
    fontsize=11, fontweight="bold"
)

axes[0].imshow(annotated_visor)
axes[0].set_title(f"1. Point A Visor HUD ({nav_id})\n{num_cv_hazards} CV Hazards | Vis: {vis_pct:.0f}%")
axes[0].axis("off")

h, w, _ = orbital_map.shape
cell_h, cell_w = h / GRID_SIZE, w / GRID_SIZE
axes[1].imshow(orbital_map)

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if np.isinf(cost_grid[r, c]):
            is_rad = (RAD_ROWS[0] <= r < RAD_ROWS[1] and RAD_COLS[0] <= c < RAD_COLS[1])
            is_salt = (salt_hazard_active and SALT_ROWS[0] <= r < SALT_ROWS[1] and SALT_COLS[0] <= c < SALT_COLS[1])
            color = "red" if is_rad else ("purple" if is_salt else "darkorange")
            axes[1].add_patch(plt.Rectangle((c * cell_w, r * cell_h), cell_w, cell_h, color=color, alpha=0.58))
        elif cost_grid[r, c] > 2.2:
            axes[1].add_patch(plt.Rectangle((c * cell_w, r * cell_h), cell_w, cell_h, color="gold", alpha=0.35))

if safe_path:
    py = [(r + 0.5) * cell_h for r, _ in safe_path]
    px = [(c + 0.5) * cell_w for _, c in safe_path]
    axes[1].plot(px, py, color="lime", linewidth=3, label=f"Safe A* Path ({total_distance_m} m)")

axes[1].scatter([(start_pt[1] + 0.5) * cell_w], [(start_pt[0] + 0.5) * cell_h], color="cyan", s=110, zorder=5, label=f"Point A {start_pt}")
axes[1].scatter([(goal_pt[1] + 0.5) * cell_w], [(goal_pt[0] + 0.5) * cell_h], color="yellow", marker="*", s=220, zorder=5, label=f"Point B {goal_pt}")
axes[1].set_title(f"2. Gale Crater Route: {target_name}\n(Red=RAD, Purple=Perchlorate, Orange=Boulder)")
axes[1].legend(loc="lower left", fontsize=8)
axes[1].axis("off")

axes[2].imshow(annotated_science)
axes[2].set_title(f"3. Point B Science Macro ({sci_id})\n{mineral_feature_count} Grain Boundaries | {science_rationale[:34]}...")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(ROUTE_PLOT_PATH, dpi=150)
plt.show()

navigation_packet = {
    "sector": "Gale Crater, Mars",
    "sol_sampled": active_sol,
    "wokwi_suit_telemetry": wokwi_data,
    "point_a_visor_start": {"grid": list(start_pt), "lat": start_lat, "lon": start_lon, "visor_image": str(NAVCAM_PATH.name)},
    "point_b_target": {
        "name": target_name, "grid": list(goal_pt), "lat": goal_lat, "lon": goal_lon,
        "science_rationale": science_rationale, "macro_image": str(SCIENCE_PATH.name),
        "cv_detected_micro_features": mineral_feature_count,
    },
    "route_metrics": {
        "total_distance_m": total_distance_m,
        "estimated_walk_time_min": est_walk_time_min,
        "estimated_o2_cost_pct": est_o2_used_pct,
        "cv_hazard_cells_avoided": num_cv_hazards,
    },
    "turn_by_turn_instructions": eva_instructions,
}

with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(navigation_packet, f, indent=2)