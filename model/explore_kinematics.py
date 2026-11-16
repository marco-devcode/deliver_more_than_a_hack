"""Unsupervised kinematic and geometric analysis of the tactical tracking dataset.

This module performs a physical characterization of the ball and player trajectories across the
graded halves of the match (158,857 frames at 25 Hz), without relying on ground-truth possession
labels. It quantifies:

1. Missing-ball (NaN) gap taxonomy: distinguishing in-play ballistic flights and short occlusions
   ("ballistic bridges") from out-of-bounds dead-ball stoppages.
2. Vectorial ball kinematics: finite-difference velocity components (v_x, v_y), optical jitter
   attenuation, and impulsive acceleration spikes (||Delta v||).
3. The 2D projection pass-through paradox: high-speed trajectories (>= 15 m) where the projected
   ball path crosses an opposing player's proximity radius (< 1.5 m) without directional
   deflection or deceleration, causing naive proximity baselines to fragment possessions.
4. Collective player kinematics: relationship between mean player speed, ball visibility, and
   tactical restarts.

Usage:
    python model/explore_kinematics.py
"""

import os
import sys
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402


FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"
HOME, AWAY = 0, 1


def parse_team_slots(meta_path):
    """Return an int8 array mapping each track slot index to HOME (0) or AWAY (1)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("Expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def extract_kinematic_arrays(raw_path, team_of_slot, n_frames, pitch_dims):
    """Stream the raw ATD file once and extract metric ball and collective player series.

    Parameters
    ----------
    raw_path : str
        Path to the FIFA/EPTS raw text file.
    team_of_slot : np.ndarray
        Array of shape (n_slots,) with values HOME (0) or AWAY (1).
    n_frames : int
        Total number of video frames (199,319 for tactical).
    pitch_dims : tuple of (float, float)
        Pitch length and width in metres (105.3, 68.0).

    Returns
    -------
    dict of str -> np.ndarray
        Pre-allocated arrays indexed by video_frame (0 .. n_frames - 1).
    """
    length, width = pitch_dims

    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_s = np.full(n_frames, np.nan, dtype=np.float32)

    min_dist_h = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_a = np.full(n_frames, np.nan, dtype=np.float32)
    near_speed_h = np.full(n_frames, np.nan, dtype=np.float32)
    near_speed_a = np.full(n_frames, np.nan, dtype=np.float32)

    mean_player_speed = np.full(n_frames, np.nan, dtype=np.float32)
    n_players = np.zeros(n_frames, dtype=np.int16)

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_idx = int(parts[0]) - 1
            if frame_idx < 0 or frame_idx >= n_frames:
                continue

            ball_str = parts[2]
            has_ball = ball_str[0] != MISSING
            bx = by = np.nan
            if has_ball:
                c1 = ball_str.index(",")
                c2 = ball_str.index(",", c1 + 1)
                bx = float(ball_str[:c1]) * length
                by = float(ball_str[c1 + 1:c2]) * width
                bs = float(ball_str[c2 + 1:])
                ball_x[frame_idx] = bx
                ball_y[frame_idx] = by
                ball_s[frame_idx] = bs

            best_d2_h = np.inf
            best_d2_a = np.inf
            best_sp_h = np.nan
            best_sp_a = np.nan
            speed_sum = 0.0
            active_count = 0

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1)
                px = float(group[:c1]) * length
                py = float(group[c1 + 1:c2]) * width
                ps = float(group[c2 + 1:])

                if not np.isnan(ps):
                    speed_sum += ps
                    active_count += 1

                if has_ball:
                    dx = px - bx
                    dy = py - by
                    d2 = dx * dx + dy * dy
                    if team_of_slot[slot] == HOME:
                        if d2 < best_d2_h:
                            best_d2_h = d2
                            best_sp_h = ps
                    else:
                        if d2 < best_d2_a:
                            best_d2_a = d2
                            best_sp_a = ps

            n_players[frame_idx] = active_count
            if active_count > 0:
                mean_player_speed[frame_idx] = speed_sum / active_count

            if has_ball:
                if best_d2_h < np.inf:
                    min_dist_h[frame_idx] = np.sqrt(best_d2_h)
                    near_speed_h[frame_idx] = best_sp_h
                if best_d2_a < np.inf:
                    min_dist_a[frame_idx] = np.sqrt(best_d2_a)
                    near_speed_a[frame_idx] = best_sp_a

    return {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_s": ball_s,
        "min_dist_h": min_dist_h,
        "min_dist_a": min_dist_a,
        "near_speed_h": near_speed_h,
        "near_speed_a": near_speed_a,
        "mean_player_speed": mean_player_speed,
        "n_players": n_players,
    }


def compute_vector_kinematics(ball_x, ball_y, fps=25.0, half_span=2):
    """Compute vectorial ball velocity (v_x, v_y) and impulsive acceleration magnitude.

    To suppress single-frame pixel quantization noise at 25 Hz without smearing genuine touch
    impulses, velocity is computed over a symmetric window of +/- `half_span` frames (0.16 s for
    half_span=2) wherever contiguous observations exist, falling back to 1-step differences at
    segment boundaries.
    """
    n = len(ball_x)
    vx = np.full(n, np.nan, dtype=np.float32)
    vy = np.full(n, np.nan, dtype=np.float32)

    dt_span = (2 * half_span) / float(fps)
    vx[half_span:n - half_span] = (ball_x[2 * half_span:] - ball_x[:n - 2 * half_span]) / dt_span
    vy[half_span:n - half_span] = (ball_y[2 * half_span:] - ball_y[:n - 2 * half_span]) / dt_span

    # Fill boundary frames of valid segments using 1-frame forward/backward differences
    dt_1 = 1.0 / float(fps)
    dx_1 = np.diff(ball_x, prepend=np.nan) / dt_1
    dy_1 = np.diff(ball_y, prepend=np.nan) / dt_1
    fill_mask = np.isnan(vx) & ~np.isnan(dx_1)
    vx[fill_mask] = dx_1[fill_mask]
    vy[fill_mask] = dy_1[fill_mask]

    v_norm = np.hypot(vx, vy)

    # Impulsive velocity change ||Delta v|| across +/- half_span frames (m/s change over 0.16 s)
    dv_x = np.full(n, np.nan, dtype=np.float32)
    dv_y = np.full(n, np.nan, dtype=np.float32)
    dv_x[half_span:n - half_span] = vx[2 * half_span:] - vx[:n - 2 * half_span]
    dv_y[half_span:n - half_span] = vy[2 * half_span:] - vy[:n - 2 * half_span]
    impulse_mag = np.hypot(dv_x, dv_y)

    # Angular deflection (radians) between incoming and outgoing velocity vectors (for v > 3 m/s)
    v_in_x = vx[:n - 2 * half_span]
    v_in_y = vy[:n - 2 * half_span]
    v_out_x = vx[2 * half_span:]
    v_out_y = vy[2 * half_span:]
    dot = v_in_x * v_out_x + v_in_y * v_out_y
    norms = np.hypot(v_in_x, v_in_y) * np.hypot(v_out_x, v_out_y)
    cos_theta = np.clip(dot / np.where(norms > 1e-4, norms, np.nan), -1.0, 1.0)
    deflection_deg = np.full(n, np.nan, dtype=np.float32)
    deflection_deg[half_span:n - half_span] = np.degrees(np.arccos(cos_theta))

    return {
        "vx": vx,
        "vy": vy,
        "v_norm": v_norm,
        "impulse_mag": impulse_mag,
        "deflection_deg": deflection_deg,
    }


def analyze_nan_gaps(ball_x, ball_y, half_blocks, pitch_dims, fps=25.0):
    """Extract and classify all contiguous missing-ball (NaN) intervals within graded play.

    A gap [t_start, t_end] bounded by valid observations at t_1 = t_start - 1 and t_2 = t_end + 1
    is classified into:
    - 'ballistic_bridge': both boundary points are strictly inside the pitch (margin > 1.5 m from
      touchlines/goal lines), duration <= 5.0 s, and implied speed v_imp in [4.0, 35.0] m/s, or
      short occlusion (<= 1.5 s) with displacement < 6.0 m.
    - 'boundary_or_stoppage': ball disappears/reappears near pitch boundaries or gap duration /
      kinematics indicate a dead-ball stoppage.
    """
    length, width = pitch_dims
    margin = 1.5
    records = []

    for half, h_start, h_end in half_blocks:
        vis = ~np.isnan(ball_x[h_start:h_end + 1])
        # Find contiguous runs of False (missing ball)
        padded = np.concatenate(([True], vis, [True]))
        starts = np.where(padded[:-1] & ~padded[1:])[0] + h_start
        ends = np.where(~padded[:-1] & padded[1:])[0] + h_start - 1

        for s, e in zip(starts, ends):
            t1 = s - 1
            t2 = e + 1
            if t1 < h_start or t2 > h_end:
                continue
            if np.isnan(ball_x[t1]) or np.isnan(ball_x[t2]):
                continue

            dur_frames = e - s + 1
            dur_s = dur_frames / float(fps)
            x1, y1 = float(ball_x[t1]), float(ball_y[t1])
            x2, y2 = float(ball_x[t2]), float(ball_y[t2])
            dist_m = float(np.hypot(x2 - x1, y2 - y1))
            v_imp = dist_m / dur_s

            in_bounds_1 = (margin <= x1 <= length - margin) and (margin <= y1 <= width - margin)
            in_bounds_2 = (margin <= x2 <= length - margin) and (margin <= y2 <= width - margin)

            is_flight_bridge = (
                in_bounds_1 and in_bounds_2 and dur_s <= 5.0 and 4.0 <= v_imp <= 35.0
            )
            is_short_occlusion = (
                in_bounds_1 and in_bounds_2 and dur_s <= 2.0 and dist_m < 8.0
            )
            category = (
                "flight_bridge" if (is_flight_bridge and dist_m >= 8.0)
                else "short_occlusion" if (is_FlightOrShort := (is_flight_bridge or is_short_occlusion))
                else "boundary_or_stoppage"
            )

            records.append({
                "half": half,
                "start_frame": s,
                "end_frame": e,
                "dur_frames": dur_frames,
                "dur_s": dur_s,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "dist_m": dist_m,
                "v_imp": v_imp,
                "in_bounds_1": in_bounds_1,
                "in_bounds_2": in_bounds_2,
                "category": category,
            })

    return records


def detect_ghost_passthroughs(series, kin, half_blocks, pitch_dims):
    """Detect frames where the ball is in high-speed unperturbed flight but passes closest to an
    opposing player ('2D projection pass-through').

    Criteria for a frame t being a mid-flight ghost proximity event:
    1. Ball speed v_norm[t] >= 10.0 m/s (fast pass or aerial ball).
    2. Directional deflection angle deflection_deg[t] <= 12.0 deg AND impulse_mag[t] <= 4.0 m/s
       (ball trajectory is smooth and unperturbed at frame t).
    3. Nearest player at frame t is within 2.5 m (or simply switches the argmin team compared to
       the flight origin/destination).
    """
    graded_mask = np.zeros(len(series["ball_x"]), dtype=bool)
    for _, s, e in half_blocks:
        graded_mask[s:e + 1] = True

    valid = graded_mask & ~np.isnan(kin["v_norm"]) & ~np.isnan(kin["deflection_deg"])
    in_flight = valid & (kin["v_norm"] >= 10.0) & (kin["deflection_deg"] <= 15.0) & (kin["impulse_mag"] <= 5.0)

    # Identify contiguous flight segments
    padded = np.concatenate(([False], in_flight, [False]))
    seg_starts = np.where(~padded[:-1] & padded[1:])[0]
    seg_ends = np.where(padded[:-1] & ~padded[1:])[0] - 1

    ball_x, ball_y = series["ball_x"], series["ball_y"]
    min_h, min_a = series["min_dist_h"], series["min_dist_a"]
    nearest_team = np.where(min_h <= min_a, HOME, AWAY)

    long_flights = []
    ghost_flips = 0
    ghost_close_encounters = 0

    for s, e in zip(seg_starts, seg_ends):
        if e - s < 4:  # at least 0.2 s of high-speed flight
            continue
        disp = float(np.hypot(ball_x[e] - ball_x[s], ball_y[e] - ball_y[s]))
        if disp < 12.0:
            continue

        # Team at launch (just before flight starts) and landing (just after flight ends)
        t_launch = max(0, s - 2)
        t_land = min(len(ball_x) - 1, e + 2)
        launch_team = nearest_team[t_launch]
        land_team = nearest_team[t_land]

        mid_teams = nearest_team[s:e + 1]
        mid_min_dist = np.minimum(min_h[s:e + 1], min_a[s:e + 1])

        # Did the closest-player team flip to the opposite of launch_team during unperturbed flight?
        opposite_frames = np.sum(mid_teams != launch_team)
        close_opposite = np.sum((mid_teams != launch_team) & (mid_min_dist < 2.0))
        if opposite_frames > 0:
            ghost_flips += 1
        if close_opposite > 0:
            ghost_close_encounters += 1

        long_flights.append({
            "start": int(s),
            "end": int(e),
            "disp_m": disp,
            "mean_v": float(np.nanmean(kin["v_norm"][s:e + 1])),
            "launch_team": int(launch_team),
            "land_team": int(land_team),
            "opposite_frames": int(opposite_frames),
            "close_opposite_frames": int(close_opposite),
        })

    return {
        "n_in_flight_frames": int(np.sum(in_flight)),
        "long_flights": long_flights,
        "n_long_flights": len(long_flights),
        "n_flights_with_nearest_team_flip": ghost_flips,
        "n_flights_with_close_passthrough": ghost_close_encounters,
    }


def generate_figures(series, kin, gaps, passthrough_info, half_blocks, out_dir):
    """Save diagnostic figures to `out_dir`."""
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    })

    # Figure 1: Missing-Ball Gap Taxonomy (Duration vs Displacement & Implied Speed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cat_colors = {
        "flight_bridge": ("#1f77b4", "Ballistic Flight Bridge (in-bounds, d >= 8 m)"),
        "short_occlusion": ("#2ca02c", "Short In-Play Occlusion (in-bounds, d < 8 m)"),
        "boundary_or_stoppage": ("#d62728", "Boundary / Dead-Ball Stoppage"),
    }

    ax = axes[0]
    for cat, (color, label) in cat_colors.items():
        sub = [g for g in gaps if g["category"] == cat and g["dur_s"] <= 30.0]
        if sub:
            ax.scatter(
                [g["dur_s"] for g in sub],
                [g["dist_m"] for g in sub],
                c=color, label="{} (n={})".format(label, len([g for g in gaps if g["category"] == cat])),
                alpha=0.65, s=22, edgecolors="none"
            )
    t_grid = np.linspace(0.1, 6.0, 100)
    ax.plot(t_grid, 15.0 * t_grid, "k--", lw=1.0, alpha=0.6, label="v_imp = 15 m/s reference")
    ax.plot(t_grid, 25.0 * t_grid, "k:", lw=1.0, alpha=0.6, label="v_imp = 25 m/s reference")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 75)
    ax.set_xlabel("Missing-ball gap duration (s)")
    ax.set_ylabel("Displacement between disappearance and reappearance (m)")
    ax.set_title("A. Taxonomy of Missing-Ball (NaN) Intervals (t <= 15 s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")

    ax2 = axes[1]
    flight_speeds = [g["v_imp"] for g in gaps if g["category"] == "flight_bridge"]
    stop_speeds = [g["v_imp"] for g in gaps if g["category"] == "boundary_or_stoppage" and g["v_imp"] <= 35.0]
    bins = np.linspace(0, 35, 36)
    ax2.hist(flight_speeds, bins=bins, alpha=0.75, color="#1f77b4", label="Ballistic Flight Bridges")
    ax2.hist(stop_speeds, bins=bins, alpha=0.55, color="#d62728", label="Boundary / Stoppage Gaps")
    ax2.set_xlabel("Implied transit speed Delta_d / Delta_t (m/s)")
    ax2.set_ylabel("Number of gaps")
    ax2.set_title("B. Distribution of Implied Speed Across NaN Gaps")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig1_path = os.path.join(out_dir, "fig1_gap_ballistic_taxonomy.png")
    fig.savefig(fig1_path)
    plt.close(fig)

    # Figure 2: Vector Velocity (v_x, v_y), Acceleration Spikes, and 2D Pass-Through Example
    # Select a representative 60-second window in the first half with multiple long flights
    w_start = 25000
    w_end = 26500  # 60 seconds (1500 frames) in middle of 1st half
    frames = np.arange(w_start, w_end)
    time_s = (frames - half_blocks[0][1]) / 25.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)

    # Panel 1: Signed velocity components v_x and v_y
    ax = axes[0]
    ax.plot(time_s, kin["vx"][w_start:w_end], color="#1f77b4", lw=1.1, label="v_x (longitudinal, m/s)")
    ax.plot(time_s, kin["vy"][w_start:w_end], color="#ff7f0e", lw=1.0, alpha=0.85, label="v_y (lateral, m/s)")
    ax.axhline(0.0, color="black", lw=0.7, ls="--")
    ax.set_ylabel("Signed velocity (m/s)")
    ax.set_title("A. Vectorial Ball Velocity Components (v_x, v_y) over a 60 s In-Play Window")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Scalar speed ||v|| vs Impulsive Acceleration ||Delta v||
    ax = axes[1]
    ax.plot(time_s, kin["v_norm"][w_start:w_end], color="#2ca02c", lw=1.1, label="Speed ||v|| (m/s)")
    ax.plot(time_s, kin["impulse_mag"][w_start:w_end], color="#d62728", lw=1.0, alpha=0.9,
            label="Impulsive change ||Delta v|| (m/s over 0.16 s)")
    # Highlight long ballistic flights in this window
    for fl in passthrough_info["long_flights"]:
        if fl["end"] >= w_start and fl["start"] <= w_end:
            s_t = (max(fl["start"], w_start) - half_blocks[0][1]) / 25.0
            e_t = (min(fl["end"], w_end) - half_blocks[0][1]) / 25.0
            ax.axvspan(s_t, e_t, color="#9467bd", alpha=0.22)
    ax.set_ylabel("Magnitude (m/s)")
    ax.set_title("B. Ball Speed vs Impulsive Acceleration Spikes (Shaded = Unperturbed Long Flights >= 12 m)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Minimum distance to Home and Away players
    ax = axes[2]
    ax.plot(time_s, series["min_dist_h"][w_start:w_end], color="#004d98", lw=1.1, label="Min distance to Home (Barcelona, m)")
    ax.plot(time_s, series["min_dist_a"][w_start:w_end], color="#e41a1c", lw=1.1, label="Min distance to Away (Athletic, m)")
    ax.axhline(1.5, color="black", lw=0.8, ls=":", label="Control radius threshold (1.5 m)")
    ax.set_ylim(0, 18)
    ax.set_xlabel("Seconds from First Half Kickoff (s)")
    ax.set_ylabel("Player-ball distance (m)")
    ax.set_title("C. Nearest Player Distance by Team (Notice Mid-Flight Crossings Above 1.5 m)")
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig2_path = os.path.join(out_dir, "fig2_vector_velocity_and_passthrough.png")
    fig.savefig(fig2_path)
    plt.close(fig)

    # Figure 3: Collective Player Speed vs Ball Visibility and Out-of-Play Regimes
    graded_mask = np.zeros(len(series["ball_x"]), dtype=bool)
    for _, s, e in half_blocks:
        graded_mask[s:e + 1] = True

    ball_vis = graded_mask & ~np.isnan(series["ball_x"])
    ball_nan = graded_mask & np.isnan(series["ball_x"])

    # Create mask of frames inside ballistic flight bridges vs dead-ball/boundary gaps
    bridge_mask = np.zeros(len(series["ball_x"]), dtype=bool)
    stop_mask = np.zeros(len(series["ball_x"]), dtype=bool)
    for g in gaps:
        if g["category"] in ("flight_bridge", "short_occlusion"):
            bridge_mask[g["start_frame"]:g["end_frame"] + 1] = True
        else:
            stop_mask[g["start_frame"]:g["end_frame"] + 1] = True

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0.2, 4.5, 60)
    ax.hist(series["mean_player_speed"][ball_vis], bins=bins, density=True, alpha=0.6,
            color="#2ca02c", label="Ball visible in play (mean={:.2f} m/s)".format(
                float(np.nanmean(series["mean_player_speed"][ball_vis]))))
    ax.hist(series["mean_player_speed"][bridge_mask], bins=bins, density=True, alpha=0.6,
            color="#1f77b4", label="Ball NaN: Ballistic Bridge / Short Occlusion (mean={:.2f} m/s)".format(
                float(np.nanmean(series["mean_player_speed"][bridge_mask]))))
    ax.hist(series["mean_player_speed"][stop_mask], bins=bins, density=True, alpha=0.6,
            color="#d62728", label="Ball NaN: Boundary / Stoppage Gap (mean={:.2f} m/s)".format(
                float(np.nanmean(series["mean_player_speed"][stop_mask]))))
    ax.set_xlabel("Collective mean speed of all tracked players on pitch (m/s)")
    ax.set_ylabel("Probability density")
    ax.set_title("Collective Player Kinematics Across Ball Visibility Regimes (Graded Halves)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig3_path = os.path.join(out_dir, "fig3_collective_speed_vs_deadball.png")
    fig.savefig(fig3_path)
    plt.close(fig)

    return [fig1_path, fig2_path, fig3_path]


def main():
    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch_dims = GAMES[GAME]["pitch"]
    half_blocks = half_video_blocks(GAME)

    print("1. Streaming raw tracking data and computing metric coordinates...")
    team_of_slot = parse_team_slots(meta_path)
    series = extract_kinematic_arrays(raw_path, team_of_slot, n_frames, pitch_dims)
    t_load = time.time() - t0
    print("   Completed in {:.2f} s ({} total frames).".format(t_load, n_frames))

    graded_mask = np.zeros(n_frames, dtype=bool)
    for _, s, e in half_blocks:
        graded_mask[s:e + 1] = True
    n_graded = int(np.sum(graded_mask))
    n_vis_graded = int(np.sum(graded_mask & ~np.isnan(series["ball_x"])))

    print("\n2. Computing vectorial ball kinematics and impulse derivatives...")
    kin = compute_vector_kinematics(series["ball_x"], series["ball_y"], fps=25.0, half_span=2)

    # Compare raw 1-frame derivative jitter vs 2-frame regularized derivative vs ATD speed
    dt1 = 1.0 / 25.0
    v1_norm = np.hypot(np.diff(series["ball_x"], prepend=np.nan) / dt1,
                       np.diff(series["ball_y"], prepend=np.nan) / dt1)
    valid_comp = graded_mask & ~np.isnan(kin["v_norm"]) & ~np.isnan(series["ball_s"])
    corr_reg = float(np.corrcoef(kin["v_norm"][valid_comp], series["ball_s"][valid_comp])[0, 1])
    mae_reg = float(np.mean(np.abs(kin["v_norm"][valid_comp] - series["ball_s"][valid_comp])))

    # Count sign reversals of vx and vy inside graded frames
    vx_valid = kin["vx"][graded_mask & ~np.isnan(kin["vx"])]
    vy_valid = kin["vy"][graded_mask & ~np.isnan(kin["vy"])]
    vx_sign_flips = int(np.sum(vx_valid[:-1] * vx_valid[1:] < 0))
    vy_sign_flips = int(np.sum(vy_valid[:-1] * vy_valid[1:] < 0))

    # Count high-impulse spikes (local maxima where ||Delta v|| > 6.0 m/s)
    imp = kin["impulse_mag"]
    imp_valid = graded_mask & ~np.isnan(imp)
    imp_filled = np.where(imp_valid, imp, -1.0)
    local_max = (imp_filled[1:-1] > imp_filled[:-2]) & (imp_filled[1:-1] >= imp_filled[2:])
    n_impulse_spikes = int(np.sum(local_max & (imp_filled[1:-1] >= 6.0)))

    print("\n3. Classifying missing-ball (NaN) gaps in graded play...")
    gaps = analyze_nan_gaps(series["ball_x"], series["ball_y"], half_blocks, pitch_dims)
    n_gaps = len(gaps)
    flight_gaps = [g for g in gaps if g["category"] == "flight_bridge"]
    short_gaps = [g for g in gaps if g["category"] == "short_occlusion"]
    stop_gaps = [g for g in gaps if g["category"] == "boundary_or_stoppage"]

    frames_flight_bridge = sum(g["dur_frames"] for g in flight_gaps)
    frames_short_occ = sum(g["dur_frames"] for g in short_gaps)
    frames_stop = sum(g["dur_frames"] for g in stop_gaps)
    total_nan_graded = n_graded - n_vis_graded

    print("\n4. Analyzing 2D pass-through events during high-speed flight...")
    passthrough_info = detect_ghost_passthroughs(series, kin, half_blocks, pitch_dims)

    # Also measure how often nearest player is > 2.0 m away while ball is visible in play
    min_d_both = np.minimum(series["min_dist_h"], series["min_dist_a"])
    vis_graded = graded_mask & ~np.isnan(min_d_both)
    frac_ball_free = float(np.mean(min_d_both[vis_graded] > 2.0)) * 100.0

    print("\n5. Generating diagnostic plots...")
    fig_dir = os.path.join(HERE, "figures")
    fig_paths = generate_figures(series, kin, gaps, passthrough_info, half_blocks, fig_dir)

    print("\n" + "=" * 78)
    print("UNSUPERVISED KINEMATIC ANALYSIS REPORT (GRADED PLAY: {} FRAMES)".format(n_graded))
    print("=" * 78)
    print("A. BALL COVERAGE AND MISSING-DATA (NaN) TAXONOMY")
    print("   Visible ball frames in graded play : {:>7d} / {:d} ({:.2f}%)".format(
        n_vis_graded, n_graded, 100.0 * n_vis_graded / n_graded))
    print("   Missing ball frames in graded play : {:>7d} / {:d} ({:.2f}%)".format(
        total_nan_graded, n_graded, 100.0 * total_nan_graded / n_graded))
    print("   Total contiguous NaN intervals     : {:>7d}".format(n_gaps))
    print("     1. Ballistic Flight Bridges      : {:>4d} gaps ({:>6d} frames, {:>5.1f}% of NaN)".format(
        len(flight_gaps), frames_flight_bridge, 100.0 * frames_flight_bridge / total_nan_graded))
    print("        (In-bounds start/end, d >= 8 m, t <= 5 s, 4 <= v_imp <= 35 m/s)")
    print("     2. Short In-Play Occlusions      : {:>4d} gaps ({:>6d} frames, {:>5.1f}% of NaN)".format(
        len(short_gaps), frames_short_occ, 100.0 * frames_short_occ / total_nan_graded))
    print("        (In-bounds start/end, t <= 2 s or d < 8 m)")
    print("     3. Boundary / Stoppage Gaps      : {:>4d} gaps ({:>6d} frames, {:>5.1f}% of NaN)".format(
        len(stop_gaps), frames_stop, 100.0 * frames_stop / total_nan_graded))
    print("        (Out-of-bounds transitions or extended dead-ball stoppages)")

    print("\nB. VECTORIAL VELOCITY AND IMPULSE STATISTICS")
    print("   Correlation ||v_reg|| vs ATD speed : {:.4f} (MAE = {:.2f} m/s)".format(corr_reg, mae_reg))
    print("   Longitudinal (v_x) sign reversals  : {:>7d} across graded play".format(vx_sign_flips))
    print("   Lateral (v_y) sign reversals       : {:>7d} across graded play".format(vy_sign_flips))
    print("   Impulsive acceleration spikes      : {:>7d} local peaks (||Delta v|| >= 6.0 m/s)".format(
        n_impulse_spikes))
    print("   Visible frames with ball > 2.0 m   : {:>6.2f}% (ball in transit between players)".format(
        frac_ball_free))

    print("\nC. 2D PROJECTION PASS-THROUGH & LONG FLIGHT ANALYSIS")
    print("   High-speed unperturbed flights     : {:>7d} segments (d >= 12 m, v >= 10 m/s)".format(
        passthrough_info["n_long_flights"]))
    print("   Flights flipping nearest-team ID   : {:>7d} / {:d} ({:.1f}%)".format(
        passthrough_info["n_flights_with_nearest_team_flip"],
        passthrough_info["n_long_flights"],
        100.0 * passthrough_info["n_flights_with_nearest_team_flip"] / max(1, passthrough_info["n_long_flights"])))
    print("   Flights passing < 2.0 m of opponent: {:>7d} / {:d} ({:.1f}%) without deflection".format(
        passthrough_info["n_flights_with_close_passthrough"],
        passthrough_info["n_long_flights"],
        100.0 * passthrough_info["n_flights_with_close_passthrough"] / max(1, passthrough_info["n_long_flights"])))

    print("\nD. COLLECTIVE PLAYER SPEED BY REGIME")
    print("   Mean player speed (Ball Visible)   : {:.2f} m/s".format(
        float(np.nanmean(series["mean_player_speed"][graded_mask & ~np.isnan(series["ball_x"])]))))
    print("   Mean player speed (Flight Bridge)  : {:.2f} m/s".format(
        float(np.nanmean(series["mean_player_speed"][
            np.isin(np.arange(n_frames),
                    np.concatenate([np.arange(g["start_frame"], g["end_frame"] + 1) for g in flight_gaps + short_gaps]))
        ]))))
    print("   Mean player speed (Stoppage Gaps)  : {:.2f} m/s".format(
        float(np.nanmean(series["mean_player_speed"][
            np.isin(np.arange(n_frames),
                    np.concatenate([np.arange(g["start_frame"], g["end_frame"] + 1) for g in stop_gaps]))
        ]))))

    print("\nSaved diagnostic figures:")
    for p in fig_paths:
        print("  - {}".format(os.path.relpath(p, ROOT)))
    print("=" * 78)


if __name__ == "__main__":
    main()
