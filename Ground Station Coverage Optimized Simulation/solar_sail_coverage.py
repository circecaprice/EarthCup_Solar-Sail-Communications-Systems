"""
Solar Sail TinyGS Coverage Analysis Tool
=========================================
Compatible with Python 3.13+. No cartopy required.

Dependencies (all pip-installable on Python 3.13):
    pip install numpy matplotlib requests scipy skyfield geopandas

Usage:
    python solar_sail_coverage.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import requests
import math
import os
import tempfile
import warnings
import geopandas as gpd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ALTITUDE_KM       = 500.0
MIN_ELEVATION_DEG = 10.0
SIM_DAYS          = 50
TIME_STEP_SEC     = 30
INCLINATIONS      = [0, 28.5, 45, 53, 63, 70, 80, 90, 97.4, 110, 130]
EARTH_RADIUS_KM   = 6371.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Natural Earth shapefile base URL
NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/110m_physical/ne_110m_"
NE_CULT = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/110m_cultural/ne_110m_"

# ─────────────────────────────────────────────
# WORLD MAP DATA
# ─────────────────────────────────────────────

_tmpdir = tempfile.mkdtemp()
_geo_cache = {}

def fetch_shapefile(name, cultural=False):
    """Download a Natural Earth 110m shapefile set and load with geopandas."""
    if name in _geo_cache:
        return _geo_cache[name]
    base = NE_CULT if cultural else NE_BASE
    try:
        for ext in ['shp', 'dbf', 'shx']:
            url = f"{base}{name}.{ext}"
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(os.path.join(_tmpdir, f"ne_{name}.{ext}"), 'wb') as f:
                f.write(r.content)
        gdf = gpd.read_file(os.path.join(_tmpdir, f"ne_{name}.shp"))
        _geo_cache[name] = gdf
        return gdf
    except Exception as e:
        print(f"  ⚠  Could not fetch shapefile '{name}': {e}")
        return None


def get_world_geometries():
    print("  Downloading world map data...")
    land  = fetch_shapefile("land")
    coast = fetch_shapefile("coastline")
    return land, coast


# ─────────────────────────────────────────────
# TINYGS STATIONS
# ─────────────────────────────────────────────

FALLBACK_STATIONS = [
    (40.42, -3.70, "Madrid-ES"),       (51.51, -0.13, "London-UK"),
    (48.85,  2.35, "Paris-FR"),        (52.52, 13.40, "Berlin-DE"),
    (55.75, 37.62, "Moscow-RU"),       (59.33, 18.07, "Stockholm-SE"),
    (60.17, 24.94, "Helsinki-FI"),     (50.08, 14.44, "Prague-CZ"),
    (47.50, 19.04, "Budapest-HU"),     (41.01, 28.97, "Istanbul-TR"),
    (35.69,139.69, "Tokyo-JP"),        (37.57,126.98, "Seoul-KR"),
    (39.91,116.39, "Beijing-CN"),      (31.23,121.47, "Shanghai-CN"),
    (22.54,114.06, "Shenzhen-CN"),     (28.61, 77.21, "Delhi-IN"),
    (19.08, 72.88, "Mumbai-IN"),       (12.97, 77.59, "Bangalore-IN"),
    (-33.87,151.21,"Sydney-AU"),       (-37.81,144.96,"Melbourne-AU"),
    (-27.47,153.03,"Brisbane-AU"),     (43.65,-79.38, "Toronto-CA"),
    (45.42,-75.70, "Ottawa-CA"),       (49.25,-123.12,"Vancouver-CA"),
    (37.77,-122.42,"SanFrancisco-US"), (34.05,-118.24,"LosAngeles-US"),
    (47.61,-122.33,"Seattle-US"),      (39.74,-104.98,"Denver-US"),
    (41.88,-87.63, "Chicago-US"),      (40.71,-74.01, "NewYork-US"),
    (29.76,-95.37, "Houston-US"),      (33.75,-84.39, "Atlanta-US"),
    (25.77,-80.19, "Miami-US"),        (19.43,-99.13, "MexicoCity-MX"),
    (4.71, -74.07, "Bogota-CO"),       (-12.05,-77.04,"Lima-PE"),
    (-23.55,-46.63,"SaoPaulo-BR"),     (-22.91,-43.17,"RioDeJaneiro-BR"),
    (-34.61,-58.38,"BuenosAires-AR"),  (-33.46,-70.65,"Santiago-CL"),
    (6.37,   2.39, "Cotonou-BJ"),      (-1.29, 36.82, "Nairobi-KE"),
    (-25.75, 28.19,"Pretoria-ZA"),     (30.04, 31.24, "Cairo-EG"),
    (36.74,  3.06, "Algiers-DZ"),      (14.69,-17.44, "Dakar-SN"),
    (5.35,  -4.01, "Abidjan-CI"),      (9.06,   7.50, "Abuja-NG"),
    (24.69, 46.72, "Riyadh-SA"),       (25.20, 55.27, "Dubai-AE"),
    (35.69, 51.39, "Tehran-IR"),       (33.34, 44.40, "Baghdad-IQ"),
    (31.77, 35.22, "Jerusalem-IL"),    (1.35, 103.82, "Singapore-SG"),
    (3.15, 101.69, "KualaLumpur-MY"),  (13.75,100.52, "Bangkok-TH"),
    (21.03,105.85, "Hanoi-VN"),        (-6.21,106.85, "Jakarta-ID"),
    (64.13,-21.94, "Reykjavik-IS"),    (63.43, 10.39, "Trondheim-NO"),
    (65.01, 25.47, "Oulu-FI"),         (-54.80,-68.30,"Ushuaia-AR"),
    (78.22, 15.65, "Svalbard-NO"),     (53.35, -6.26, "Dublin-IE"),
    (45.46,  9.19, "Milan-IT"),        (37.98, 23.73, "Athens-GR"),
    (50.45, 30.52, "Kyiv-UA"),         (44.80, 20.46, "Belgrade-RS"),
]


def fetch_tinygs_stations():
    for url in ["https://api.tinygs.com/v1/stations"]:
        try:
            r = requests.get(url, timeout=8,
                             headers={"User-Agent": "SolarSailCoverageAnalyzer/1.0"})
            if r.status_code == 200:
                stations = []
                for s in r.json():
                    lat = s.get("lat") or s.get("latitude")
                    lon = s.get("lon") or s.get("longitude")
                    name = s.get("name", "Unknown")
                    if lat is not None and lon is not None:
                        stations.append((float(lat), float(lon), str(name)))
                if len(stations) > 10:
                    print(f"  ✓ Fetched {len(stations)} live TinyGS stations")
                    return stations, True
        except Exception:
            pass
    print(f"  ℹ  Using curated fallback of {len(FALLBACK_STATIONS)} TinyGS stations")
    return list(FALLBACK_STATIONS), False


# ─────────────────────────────────────────────
# ORBITAL MECHANICS
# ─────────────────────────────────────────────

def orbital_period_sec(alt_km):
    mu = 398600.4418
    r  = EARTH_RADIUS_KM + alt_km
    return 2 * math.pi * math.sqrt(r**3 / mu)


def footprint_radius_deg(alt_km, min_elev_deg):
    rho = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + alt_km)
    eta = math.asin(rho * math.cos(math.radians(min_elev_deg)))
    lam = math.pi / 2 - math.radians(min_elev_deg) - eta
    return math.degrees(lam)


def ground_track(inc_deg, alt_km, duration_days, dt_sec=30):
    mu      = 398600.4418
    r       = EARTH_RADIUS_KM + alt_km
    n       = math.sqrt(mu / r**3)
    omega_E = 7.2921150e-5
    inc     = math.radians(inc_deg)
    t_arr   = np.arange(0, duration_days * 86400, dt_sec)
    lats, lons = [], []
    for t in t_arr:
        nu    = (n * t) % (2 * math.pi)
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)
        x_eci = math.cos(0)*x_orb - math.cos(inc)*math.sin(0)*y_orb
        y_eci = math.sin(0)*x_orb + math.cos(inc)*math.cos(0)*y_orb
        z_eci = math.sin(inc)*y_orb
        th    = omega_E * t
        x_ecef =  x_eci*math.cos(th) + y_eci*math.sin(th)
        y_ecef = -x_eci*math.sin(th) + y_eci*math.cos(th)
        z_ecef =  z_eci
        lats.append(math.degrees(math.asin(z_ecef / r)))
        lons.append(math.degrees(math.atan2(y_ecef, x_ecef)))
    return np.array(lats), np.array(lons), t_arr


def compute_coverage(stations, inc_deg, alt_km, duration_days, dt_sec=30):
    fp_rad      = math.radians(footprint_radius_deg(alt_km, MIN_ELEVATION_DEG))
    lats, lons, _ = ground_track(inc_deg, alt_km, duration_days, dt_sec)
    sat_lat_rad = np.radians(lats)
    sat_lon_rad = np.radians(lons)
    total_steps = 0
    total_passes = 0
    for (gs_lat, gs_lon, _) in stations:
        gl = math.radians(gs_lat)
        gn = math.radians(gs_lon)
        dlat = sat_lat_rad - gl
        dlon = sat_lon_rad - gn
        a   = np.sin(dlat/2)**2 + math.cos(gl)*np.cos(sat_lat_rad)*np.sin(dlon/2)**2
        sep = 2*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        iv  = sep <= fp_rad
        total_steps  += int(np.sum(iv))
        total_passes += int(np.sum(np.diff(iv.astype(int)) == 1))
    return total_steps * dt_sec / 3600.0, total_passes


# ─────────────────────────────────────────────
# MAP HELPERS
# ─────────────────────────────────────────────

def draw_world(ax, land, coast, facecolor='#0d2137', landcolor='#1c2b1e',
               coastcolor='#3a6b47', coastlw=0.5):
    ax.set_facecolor(facecolor)
    if land is not None:
        land.plot(ax=ax, color=landcolor, edgecolor='none', zorder=1)
    if coast is not None:
        coast.plot(ax=ax, color=coastcolor, linewidth=coastlw, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    # Grid lines
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color='#1a2e3a', lw=0.3, zorder=0)
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color='#1a2e3a', lw=0.3, zorder=0)
    ax.tick_params(colors='#8b949e', labelsize=7)
    ax.set_xlabel('Longitude (°)', color='#8b949e', fontsize=8)
    ax.set_ylabel('Latitude (°)',  color='#8b949e', fontsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')


def plot_ground_track(ax, lats, lons, color='#4fc3f7', lw=0.7, alpha=0.7):
    """Plot ground track with dateline wrapping handled."""
    split_idx = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
    for slat, slon in zip(np.split(lats, split_idx), np.split(lons, split_idx)):
        ax.plot(slon, slat, color=color, lw=lw, alpha=alpha, zorder=3)


def lat_band_line(ax, lat, **kwargs):
    ax.axhline(lat, **kwargs)


# ─────────────────────────────────────────────
# SAIL DIAGRAM
# ─────────────────────────────────────────────

def draw_sail_diagram(ax):
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('#0d1117')

    # Earth
    earth = plt.Circle((0, -2.0), 0.55, color='#1a6faf', zorder=5)
    ax.add_patch(earth)
    ax.text(0, -2.0, '🌍', ha='center', va='center', fontsize=14, zorder=7)

    # Orbit arc
    theta = np.linspace(np.pi*0.15, np.pi*0.85, 80)
    ox = 1.55*np.cos(theta); oy = 1.55*np.sin(theta) - 2.0
    ax.plot(ox, oy, '--', color='#4fc3f7', lw=1.2, alpha=0.7, zorder=4)

    # Satellite body
    ax.add_patch(plt.Rectangle((-0.08, -0.57), 0.16, 0.24, color='#aaa', zorder=8))

    # Sail (edge-on = thin line)
    ax.plot([-0.45, 0.45], [-0.45, -0.45],
            color='#f0c040', lw=5, solid_capstyle='round', zorder=9)
    for sx in [-0.45, 0.45]:
        ax.plot(sx, -0.45, 'o', color='#f0c040', ms=7, zorder=10)

    # Normal vector (nadir)
    ax.annotate('', xy=(0, -1.15), xytext=(0, -0.7),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.0), zorder=11)
    ax.text(0.12, -0.93, 'Normal\n(Nadir)', color='#e74c3c', fontsize=7, va='center')

    # Sunlight
    for yi in [0.4, 0.0, -0.4]:
        ax.annotate('', xy=(-1.55, yi), xytext=(-2.5, yi),
                    arrowprops=dict(arrowstyle='->', color='#ffd54f', lw=1.5), zorder=6)
    ax.text(-2.75, 0.0, '☀\nSRP', color='#ffd54f', fontsize=8, va='center', ha='center')

    # RF link
    ax.plot([0, 0.35], [-0.7, -1.55], ':', color='#00e5ff', lw=1.5, zorder=7)
    ax.text(0.45, -1.15, 'RF\nLink', color='#00e5ff', fontsize=7, va='center')

    # Ground station
    ax.plot(0.42, -1.88, '^', color='#ff6b6b', ms=9, zorder=8)
    ax.text(0.6, -1.88, 'TinyGS', color='#ff6b6b', fontsize=7, va='center')

    ax.set_title('Edge-On Solar Sail\n(Normal → Nadir)',
                 color='white', fontsize=9, pad=4)


# ─────────────────────────────────────────────
# FIGURE 1 — COVERAGE SUMMARY
# ─────────────────────────────────────────────

def plot_coverage_summary(stations, results, live_data, land, coast):
    inclinations = [r[0] for r in results]
    contact_hrs  = [r[1] for r in results]
    passes       = [r[2] for r in results]
    best_idx     = int(np.argmax(contact_hrs))
    best_inc     = inclinations[best_idx]

    cmap   = plt.cm.RdYlGn
    norm_c = plt.Normalize(min(contact_hrs), max(contact_hrs))
    colors = [cmap(norm_c(v)) for v in contact_hrs]

    fig = plt.figure(figsize=(20, 13), facecolor='#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.30,
                            left=0.06, right=0.97, top=0.93, bottom=0.06)

    # ── Bar: contact time
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.set_facecolor('#161b22')
    ax_bar.bar(inclinations, contact_hrs, color=colors, edgecolor='#30363d', width=5)
    ax_bar.axvline(best_inc, color='#f0c040', lw=1.5, ls='--', alpha=0.9)
    ax_bar.text(best_inc + 1.5, max(contact_hrs)*0.96,
                f'Best: {best_inc}°', color='#f0c040', fontsize=8)
    ax_bar.set_xlabel('Inclination (°)', color='#8b949e')
    ax_bar.set_ylabel('Total Contact Time (hrs)', color='#8b949e')
    ax_bar.set_title(f'TinyGS Coverage vs Inclination\n'
                     f'(500 km | {SIM_DAYS}-day sim | elev ≥ {MIN_ELEVATION_DEG}°)',
                     color='white', fontsize=10)
    ax_bar.tick_params(colors='#8b949e')
    for sp in ax_bar.spines.values(): sp.set_color('#30363d')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_bar, pad=0.02, fraction=0.04)
    cb.set_label('Contact hrs', color='#8b949e', fontsize=8)
    cb.ax.tick_params(colors='#8b949e')

    # ── Bar: passes
    ax_pass = fig.add_subplot(gs[1, 0])
    ax_pass.set_facecolor('#161b22')
    ax_pass.bar(inclinations, passes, color=colors, edgecolor='#30363d', width=5)
    ax_pass.axvline(best_inc, color='#f0c040', lw=1.5, ls='--', alpha=0.9)
    ax_pass.set_xlabel('Inclination (°)', color='#8b949e')
    ax_pass.set_ylabel('Total Passes', color='#8b949e')
    ax_pass.set_title('Number of Passes vs Inclination', color='white', fontsize=10)
    ax_pass.tick_params(colors='#8b949e')
    for sp in ax_pass.spines.values(): sp.set_color('#30363d')

    # ── Ground track map (best inclination)
    ax_map = fig.add_subplot(gs[:, 1:])
    draw_world(ax_map, land, coast)

    lats, lons, _ = ground_track(best_inc, ALTITUDE_KM, min(2, SIM_DAYS), dt_sec=30)
    plot_ground_track(ax_map, lats, lons, color='#4fc3f7', lw=0.7, alpha=0.65)

    # Footprint circles at sample positions
    fp_deg = footprint_radius_deg(ALTITUDE_KM, MIN_ELEVATION_DEG)
    theta_fp = np.linspace(0, 2*np.pi, 60)
    for i in range(0, len(lats), max(1, len(lats)//20)):
        clat, clon = lats[i], lons[i]
        fp_lats = clat + fp_deg*np.cos(theta_fp)
        fp_lons = clon + (fp_deg/max(0.01, math.cos(math.radians(clat))))*np.sin(theta_fp)
        fp_lons = ((fp_lons+180) % 360) - 180
        ax_map.plot(fp_lons, fp_lats, color='#4fc3f7', lw=0.25, alpha=0.15, zorder=3)

    # Max latitude band
    max_lat = best_inc if best_inc <= 90 else 180 - best_inc
    for lat_l in [max_lat, -max_lat]:
        ax_map.axhline(lat_l, color='#f0c040', lw=1.0, ls='--', alpha=0.55, zorder=4)

    # Stations
    gs_lats = [s[0] for s in stations]
    gs_lons = [s[1] for s in stations]
    gs_col  = ['#ff6b6b' if abs(s[0])>60 else
               '#ffd54f' if abs(s[0])>30 else '#69f0ae' for s in stations]
    ax_map.scatter(gs_lons, gs_lats, c=gs_col, s=22, zorder=5,
                   marker='^', edgecolors='white', linewidths=0.3)

    ax_map.set_title(
        f'Ground Track — Best Inclination: {best_inc}°  |  '
        f'{len(stations)} TinyGS Stations  |  Footprint ≥{MIN_ELEVATION_DEG}° elev  |  2-day track',
        color='white', fontsize=10, pad=8)

    legend_els = [
        mpatches.Patch(color='#69f0ae', label='Equatorial (|lat|≤30°)'),
        mpatches.Patch(color='#ffd54f', label='Mid-latitude (30°–60°)'),
        mpatches.Patch(color='#ff6b6b', label='Polar (|lat|>60°)'),
        plt.Line2D([0],[0], color='#4fc3f7', lw=1.5, label=f'Ground track (i={best_inc}°)'),
        plt.Line2D([0],[0], color='#f0c040', lw=1.2, ls='--', label='Max coverage latitude'),
    ]
    ax_map.legend(handles=legend_els, loc='lower left',
                  facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#c9d1d9', fontsize=8)

    # Sail diagram inset
    ax_sail = fig.add_axes([0.035, 0.53, 0.175, 0.36], facecolor='#0d1117')
    draw_sail_diagram(ax_sail)

    # Title bar
    period_min = orbital_period_sec(ALTITUDE_KM) / 60
    src = "live API" if live_data else "curated stations"
    fig.text(0.05, 0.975,
             f"Solar Sail TinyGS Coverage Optimizer  |  Alt: {ALTITUDE_KM} km  |  "
             f"Period: {period_min:.1f} min  |  Stations: {len(stations)} ({src})  |  "
             f"Best: i={best_inc}°  →  {contact_hrs[best_idx]:.1f} hrs / {passes[best_idx]} passes ({SIM_DAYS} days)",
             color='#c9d1d9', fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22', edgecolor='#30363d'))

    path = os.path.join(OUTPUT_DIR, "solar_sail_coverage.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ─────────────────────────────────────────────
# FIGURE 2 — MULTI-INCLINATION GROUND TRACKS
# ─────────────────────────────────────────────

def plot_multi_groundtrack(stations, land, coast):
    key_incs = [28.5, 53.0, 97.4, 130.0]
    labels   = ["28.5° — ISS-like", "53° — Starlink-like",
                "97.4° — Sun-synchronous", "130° — Retrograde"]
    track_colors = ['#4fc3f7', '#69f0ae', '#ff9800', '#e040fb']

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor='#0d1117')
    fig.subplots_adjust(hspace=0.30, wspace=0.10,
                        top=0.92, bottom=0.05, left=0.04, right=0.98)

    gs_lats = [s[0] for s in stations]
    gs_lons = [s[1] for s in stations]

    for ax, inc, label, tcol in zip(axes.flat, key_incs, labels, track_colors):
        draw_world(ax, land, coast)
        lats, lons, _ = ground_track(inc, ALTITUDE_KM, 1.0, dt_sec=30)
        plot_ground_track(ax, lats, lons, color=tcol, lw=0.9, alpha=0.8)

        ml = inc if inc <= 90 else 180 - inc
        ax.axhline( ml, color='#f0c040', lw=0.9, ls='--', alpha=0.6, zorder=4)
        ax.axhline(-ml, color='#f0c040', lw=0.9, ls='--', alpha=0.6, zorder=4)

        ax.scatter(gs_lons, gs_lats, c='#ff6b6b', s=12, zorder=5,
                   marker='^', edgecolors='white', linewidths=0.2)

        ax.set_title(label, color='white', fontsize=11, pad=5)

    fig.text(0.5, 0.97,
             "Solar Sail Ground Tracks — 1-Day Comparison  |  "
             "▲ TinyGS Stations  |  ─ ─ Max Latitude Band",
             color='#c9d1d9', fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#161b22', edgecolor='#30363d'))

    path = os.path.join(OUTPUT_DIR, "solar_sail_groundtracks.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ─────────────────────────────────────────────
# FIGURE 3 — PER-STATION PASS COUNT HEATMAP
# ─────────────────────────────────────────────

def plot_station_heatmap(stations, results, land, coast):
    best_inc = results[int(np.argmax([r[1] for r in results]))][0]
    fp_rad   = math.radians(footprint_radius_deg(ALTITUDE_KM, MIN_ELEVATION_DEG))
    lats, lons, _ = ground_track(best_inc, ALTITUDE_KM, SIM_DAYS, dt_sec=30)
    slr = np.radians(lats); snr = np.radians(lons)

    station_passes = []
    for (gs_lat, gs_lon, _) in stations:
        gl = math.radians(gs_lat); gn = math.radians(gs_lon)
        dlat = slr - gl; dlon = snr - gn
        a   = np.sin(dlat/2)**2 + math.cos(gl)*np.cos(slr)*np.sin(dlon/2)**2
        sep = 2*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        iv  = sep <= fp_rad
        station_passes.append(int(np.sum(np.diff(iv.astype(int)) == 1)))

    fig, ax = plt.subplots(figsize=(16, 8), facecolor='#0d1117')
    fig.subplots_adjust(left=0.04, right=0.95, top=0.91, bottom=0.07)
    draw_world(ax, land, coast)

    # Draw 1-day ground track faintly
    lt, ln, _ = ground_track(best_inc, ALTITUDE_KM, 1.0, dt_sec=30)
    plot_ground_track(ax, lt, ln, color='#4fc3f7', lw=0.45, alpha=0.3)

    gs_lats = [s[0] for s in stations]
    gs_lons = [s[1] for s in stations]
    vmax = max(station_passes) if station_passes else 1

    sc = ax.scatter(gs_lons, gs_lats,
                    c=station_passes, cmap='YlOrRd',
                    s=80, zorder=5, marker='^',
                    edgecolors='white', linewidths=0.4,
                    vmin=0, vmax=vmax)

    cb = plt.colorbar(sc, ax=ax, fraction=0.022, pad=0.01)
    cb.set_label(f'Passes over {SIM_DAYS} days', color='#c9d1d9', fontsize=9)
    cb.ax.tick_params(colors='#8b949e')

    # Annotate top-5 stations
    top5 = sorted(range(len(station_passes)), key=lambda i: station_passes[i], reverse=True)[:5]
    for i in top5:
        ax.annotate(f"{stations[i][2].split('-')[0]}\n({station_passes[i]}p)",
                    xy=(gs_lons[i], gs_lats[i]),
                    xytext=(gs_lons[i]+3, gs_lats[i]+4),
                    color='white', fontsize=6.5, zorder=8,
                    arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.7))

    ax.set_title(
        f"Per-Station Pass Count  |  Best Inclination: {best_inc}°  |  "
        f"{SIM_DAYS}-day simulation  |  {len(stations)} TinyGS Stations",
        color='white', fontsize=11, pad=8)

    path = os.path.join(OUTPUT_DIR, "solar_sail_station_coverage.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n═══════════════════════════════════════════════════════")
    print("  Solar Sail TinyGS Coverage Optimizer")
    print(f"  Altitude : {ALTITUDE_KM} km circular")
    print(f"  Sail     : circular, edge-on (normal = nadir)")
    print(f"  Min elev : {MIN_ELEVATION_DEG}°  |  Sim window: {SIM_DAYS} days")
    print("═══════════════════════════════════════════════════════\n")

    print("[ 1/4 ] Fetching TinyGS station locations...")
    stations, live = fetch_tinygs_stations()

    print("\n[ 2/4 ] Computing coverage sweep across inclinations...")
    results = []
    for inc in INCLINATIONS:
        hrs, n_passes = compute_coverage(stations, inc, ALTITUDE_KM, SIM_DAYS, TIME_STEP_SEC)
        results.append((inc, hrs, n_passes))
        marker = " ◀ BEST" if inc == INCLINATIONS[int(np.argmax([r[1] for r in results]))] else ""
        print(f"  i={inc:6.1f}°  →  {hrs:7.2f} hrs  |  {n_passes:5d} passes{marker}")

    best = max(results, key=lambda r: r[1])
    print(f"\n  ★  Best inclination: {best[0]}°  →  {best[1]:.2f} hrs  |  {best[2]} passes\n")

    print("[ 3/4 ] Loading world map data...")
    land, coast = get_world_geometries()

    print("\n[ 4/4 ] Generating visualizations...")
    p1 = plot_coverage_summary(stations, results, live, land, coast)
    p2 = plot_multi_groundtrack(stations, land, coast)
    p3 = plot_station_heatmap(stations, results, land, coast)

    period = orbital_period_sec(ALTITUDE_KM) / 60
    fp     = footprint_radius_deg(ALTITUDE_KM, MIN_ELEVATION_DEG)
    print(f"""
─────────────────────────────────────────────────
  RESULTS SUMMARY
  Orbital period    : {period:.1f} min
  RF footprint      : ±{fp:.1f}° great-circle radius
  Best inclination  : {best[0]}°
  Contact time      : {best[1]:.1f} hrs over {SIM_DAYS} days
  Total passes      : {best[2]}
  Avg pass duration : {best[1]*60/max(1,best[2]):.1f} min

  Output images saved to: {OUTPUT_DIR}
─────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
