"""Unified Physics-Kinematic & Calibrated HMM-Viterbi Pipeline (v2).

Merges the strengths of `physics_viterbi_prototype.py` and `viterbi_pipeline.py`:
1. Pure-NumPy Monotone Cubic Hermite (PCHIP) interpolation on short in-bounds ball gaps.
2. Vectorial ball kinematics (v_x, v_y, ||v||, ||Delta v||, angular deflection) + 2D ballistic
   flight protection (dynamic transition gates in Viterbi + flight emission suppression).
3. Combined Rational + Gaussian proximity kernels with calibrated Log-Softmax emissions.
4. Unconditional collective stoppage detection (fixes false live-play when a player handles/passes
   a dead ball while all 22 players are walking < 1.35 m/s).
5. Full Smart Tagging integration (638 tags):
   - Semantic inversion of DEFENDING_* tags.
   - Pre-kickoff dead-ball lock for the 2nd half (frames 110808..113741, see FAQ.md & load_smart_tagging.py)
     and end-of-half whistle detection.
   - GOALS -> Kick Off (After Goal) dead-ball celebration lock (~150s of D across 4 goals) and
     IFAB kickoff team assignment (conceding team restarts).
   - Set-Piece pre-restart lock: bridges out-of-bounds/stoppage to the actual restart kick impulse
     inside SET PIECES (Goal Kick, Corner Kick, Free Kick, Throw-in).
6. Phase-zero iterative post-processing: cleans short isolated rebound/deflection glitches without
   shifting transition boundaries.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_tags, KICKOFF_VIDEO_FRAMES  # noqa: E402
from model.explore_kinematics import (  # noqa: E402
    parse_team_slots,
    extract_kinematic_arrays,
    compute_vector_kinematics,
)

CACHE_PATH = os.path.join(HERE, "atd_cache.npz")


def load_or_build_cache():
    if os.path.isfile(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return {k: data[k] for k in data.files}
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch_dims = GAMES[GAME]["pitch"]
    team_of_slot = parse_team_slots(meta_path)
    series = extract_kinematic_arrays(raw_path, team_of_slot, n_frames, pitch_dims)
    np.savez_compressed(CACHE_PATH, **series)
    return series


def get_runs(labels, start_offset=0):
    """Return list of (label, start_frame, end_frame, dur_s) for contiguous label runs."""
    runs = []
    if len(labels) == 0:
        return runs
    cur = labels[0]
    s = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            runs.append((str(cur), start_offset + s, start_offset + i - 1, (i - s) / 25.0))
            cur = labels[i]
            s = i
    runs.append((str(cur), start_offset + s, start_offset + len(labels) - 1, (len(labels) - s) / 25.0))
    return runs

STATE_H, STATE_A, STATE_D = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"], dtype="<U1")

ATTACKING_CODES = {
    "PROGRESSION",
    "BUILD UP",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "PLAYERS IN THE BOX",
    "CREATING CHANCES",
    "LONG BALL",
    "ATTACKING TRANSITION",
}

DEFENDING_CODES = {
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
}


def rolling_mean_1d(arr, half_win):
    """Fast O(N) centred moving average handling NaNs via cumulative sums."""
    n = len(arr)
    valid = ~np.isnan(arr)
    vals = np.where(valid, arr, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
    ccnt = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - half_win)
    hi = np.minimum(n, idx + half_win + 1)
    counts = ccnt[hi] - ccnt[lo]
    sums = csum[hi] - csum[lo]
    return np.where(counts > 0, sums / np.maximum(1, counts), np.nan).astype(np.float32)


def pchip_interpolate_1d(x_nodes, y_nodes, x_query):
    """Pure-NumPy Monotone Cubic Hermite Interpolating Polynomial (PCHIP)."""
    n = len(x_nodes)
    if n < 2:
        return np.interp(x_query, x_nodes, y_nodes)
    if n == 2:
        return np.interp(x_query, x_nodes, y_nodes)

    h = np.diff(x_nodes).astype(np.float64)
    delta = np.diff(y_nodes).astype(np.float64) / h
    d = np.zeros(n, dtype=np.float64)

    # Interior slopes: weighted harmonic mean when delta[k-1] and delta[k] have same sign
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            d[k] = 0.0

    # Endpoint slopes
    d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d0) != np.sign(delta[0]):
        d0 = 0.0
    elif (np.sign(delta[0]) != np.sign(delta[1])) and (abs(d0) > abs(3.0 * delta[0])):
        d0 = 3.0 * delta[0]
    d[0] = d0

    dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(dn) != np.sign(delta[-1]):
        dn = 0.0
    elif (np.sign(delta[-1]) != np.sign(delta[-2])) and (abs(dn) > abs(3.0 * delta[-1])):
        dn = 3.0 * delta[-1]
    d[-1] = dn

    # Evaluate cubic Hermite polynomials at x_query
    idx = np.clip(np.searchsorted(x_nodes, x_query, side="right") - 1, 0, n - 2)
    hk = h[idx]
    t = (x_query - x_nodes[idx]) / hk
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (h00 * y_nodes[idx] + h10 * hk * d[idx] + h01 * y_nodes[idx + 1] + h11 * hk * d[idx + 1]).astype(np.float32)


def interpolate_ball_pchip(ball_x, ball_y, half_blocks, max_bridge_s=2.0, margin=1.5, fps=25.0):
    """PCHIP interpolation on short in-bounds gaps (from `viterbi_pipeline.py` + `physics_viterbi`)."""
    bx = ball_x.copy()
    by = ball_y.copy()
    is_bridged = np.zeros(len(bx), dtype=bool)
    max_frames = int(round(max_bridge_s * fps))

    for _, h_start, h_end in half_blocks:
        vis = ~np.isnan(bx[h_start:h_end + 1])
        valid_idx = np.where(vis)[0] + h_start
        if len(valid_idx) < 4:
            continue

        for k in range(len(valid_idx) - 1):
            t_pre = valid_idx[k]
            t_post = valid_idx[k + 1]
            gap = t_post - t_pre - 1
            if 0 < gap <= max_frames:
                x1, y1 = bx[t_pre], by[t_pre]
                x2, y2 = bx[t_post], by[t_post]
                if (margin < x1 < 105.3 - margin and margin < x2 < 105.3 - margin and
                        margin < y1 < 68.0 - margin and margin < y2 < 68.0 - margin):
                    w_lo = max(0, k - 2)
                    w_hi = min(len(valid_idx), k + 3)
                    anchors = valid_idx[w_lo:w_hi]
                    t_gap = np.arange(t_pre + 1, t_post)
                    bx[t_gap] = pchip_interpolate_1d(anchors, bx[anchors], t_gap)
                    by[t_gap] = pchip_interpolate_1d(anchors, by[anchors], t_gap)
                    is_bridged[t_gap] = True

    return bx, by, is_bridged


def build_unified_smart_tagging(n_frames, events, series, kin):
    """Extract tactical priors, set-piece restart windows, and goal-celebration dead-ball masks."""
    prior_h = np.zeros(n_frames, dtype=np.float32)
    prior_a = np.zeros(n_frames, dtype=np.float32)
    goal_dead_mask = np.zeros(n_frames, dtype=bool)
    set_piece_pre_kick_d = np.zeros(n_frames, dtype=bool)
    set_piece_post_kick_team = np.full(n_frames, -1, dtype=np.int8)

    bx = series["ball_x"]
    by = series["ball_y"]
    v_norm = kin["v_norm"]
    plr_speed_smooth = rolling_mean_1d(series["mean_player_speed"], half_win=35)
    out_of_bounds = (~np.isnan(bx)) & ((bx < 0.4) | (bx > 104.9) | (by < 0.4) | (by > 67.6))

    # 1. Process GOALS and Kick Off (After Goal) pairs
    goals = [ev for ev in events if (ev.get("code") or "").strip().upper() == "GOALS"]
    for g in goals:
        sf = max(0, min(n_frames - 1, int(g["start_frame"])))
        ef = max(0, min(n_frames - 1, int(g["end_frame"])))
        team_str = (g.get("tags", {}).get("Team") or "").strip()
        scoring_team = STATE_H if team_str == "Barcelona" else (STATE_A if team_str == "Atletic Club" else -1)
        conceding_team = STATE_A if scoring_team == STATE_H else (STATE_H if scoring_team == STATE_A else -1)

        # In GOALS [sf, ef], find the exact frame where the ball enters the goal (x < 0 or x > 105.3 or NaN)
        # and mark from that frame until the Kick Off (After Goal) (~ef - 25) as Dead Ball (D).
        t_goal_in = None
        for t in range(sf, min(ef, sf + 600)):
            if np.isnan(bx[t]) or bx[t] < 0.2 or bx[t] > 105.1 or plr_speed_smooth[t] < 1.55:
                # Confirm sustained stoppage/celebration afterwards
                if np.nanmean(plr_speed_smooth[t:min(ef, t + 150)]) < 1.85:
                    t_goal_in = t
                    break
        if t_goal_in is None:
            t_goal_in = min(ef, sf + 200)

        # Find exact kickoff restart near ef - 25 where ball near center circle (52.6, 34.0) starts moving
        ko_approx = max(t_goal_in + 100, ef - 45)
        t_restart = ef - 20
        for t in range(ko_approx, min(n_frames - 1, ef + 50)):
            if not np.isnan(bx[t]) and 45.0 < bx[t] < 60.0 and 26.0 < by[t] < 42.0 and v_norm[t] > 2.5:
                t_restart = t
                break

        goal_dead_mask[t_goal_in:t_restart] = True
        if scoring_team == STATE_H:
            prior_h[sf:t_goal_in] += 1.2
        elif scoring_team == STATE_A:
            prior_a[sf:t_goal_in] += 1.2
        if conceding_team >= 0:
            set_piece_post_kick_team[t_restart:min(n_frames, t_restart + 150)] = conceding_team

    # 2. Process SET PIECES (Goal Kick, Corner Kick, Free Kick, Throw-in)
    set_pieces = [ev for ev in events if (ev.get("code") or "").strip().upper() == "SET PIECES"]
    for sp in set_pieces:
        sp_type = (sp.get("tags", {}).get("Type") or "").strip().lower()
        if "kick off" in sp_type:
            continue
        sf = max(0, min(n_frames - 1, int(sp["start_frame"])))
        team_str = (sp.get("tags", {}).get("Team") or "").strip()
        sp_team = STATE_H if team_str == "Barcelona" else (STATE_A if team_str == "Atletic Club" else -1)

        # SportsCode SET PIECES clips start around the restart and last 35s (with 25s post-roll).
        # Find the actual restart kick/throw inside [sf - 25, sf + 300]:
        # Before the restart kick, the ball is either NaN, out of bounds, or moving slowly (< 3.5 m/s)
        # while collective player speed is low. The restart is the first sustained ball acceleration
        # (> 5.0 m/s for Goal/Corner/Free Kick, or > 3.2 m/s inside pitch for Throw-in).
        v_thresh = 3.5 if "throw" in sp_type else 5.2
        search_lo = max(0, sf - 15)
        search_hi = min(n_frames - 5, sf + 320)

        t_kick = None
        for t in range(search_lo, search_hi):
            if not np.isnan(bx[t]) and not out_of_bounds[t]:
                if v_norm[t] >= v_thresh and np.nanmean(v_norm[t:t + 3]) >= v_thresh * 0.75:
                    t_kick = t
                    break

        if t_kick is not None:
            # Look backwards from t_kick up to 16 seconds to find where the stoppage (D) began
            # (when ball went out of bounds, went NaN, or players slowed down < 1.65 m/s)
            lookback_lo = max(0, t_kick - 425)
            t_stop_start = None
            for t in range(lookback_lo, t_kick):
                if out_of_bounds[t] or np.isnan(bx[t]) or plr_speed_smooth[t] < 1.55:
                    t_stop_start = t
                    break
            if t_stop_start is not None and (t_kick - t_stop_start) >= 35:
                set_piece_pre_kick_d[t_stop_start:t_kick] = True
            if sp_team >= 0:
                set_piece_post_kick_team[t_kick:min(n_frames, t_kick + 125)] = sp_team

    # 3. Process Phase Priors (ATTACKING_CODES and inverted DEFENDING_CODES)
    active_play_tag = np.zeros(n_frames, dtype=np.int16)
    for ev in events:
        code = (ev.get("code") or "").strip().upper()
        sf = max(0, min(n_frames, int(ev["start_frame"])))
        ef = max(0, min(n_frames, int(ev["end_frame"])))
        if ef <= sf:
            continue
        team_str = (ev.get("tags", {}).get("Team") or "").strip()
        team_code = STATE_H if team_str == "Barcelona" else (STATE_A if team_str == "Atletic Club" else -1)

        if code in ATTACKING_CODES:
            active_play_tag[sf:ef] += 1
            w = 1.0 if code in ("BUILD UP", "PROGRESSION", "BALL IN THE BOX", "CREATING CHANCES") else 0.65
            if team_code == STATE_H:
                prior_h[sf:ef] += w
            elif team_code == STATE_A:
                prior_a[sf:ef] += w
        elif code in DEFENDING_CODES:
            active_play_tag[sf:ef] += 1
            if team_code == STATE_H:
                prior_a[sf:ef] += 0.75
            elif team_code == STATE_A:
                prior_h[sf:ef] += 0.75

    return {
        "prior_h": prior_h,
        "prior_a": prior_a,
        "active_play_tag": active_play_tag,
        "goal_dead_mask": goal_dead_mask,
        "set_piece_pre_kick_d": set_piece_pre_kick_d,
        "set_piece_post_kick_team": set_piece_post_kick_team,
    }


def compute_unified_emissions_and_gates(series, kin, tag_info, half_blocks, pitch_dims):
    """Combine physical emissions from `physics_viterbi_prototype` and `viterbi_pipeline`."""
    n = len(series["ball_x"])
    length, width = pitch_dims

    bx = series["ball_x"]
    by = series["ball_y"]
    min_h = series["min_dist_h"].copy()
    min_a = series["min_dist_a"].copy()

    # Goalkeeper Jersey De-Swapping:
    # In automatic tracking, goalkeepers wear contrasting kits that are occasionally assigned to the
    # wrong team cluster (e.g., frame 15660 where Athletic's keeper at x=100.9m is in a Home slot).
    # Half 1: H defends left (x < 11.5), A defends right (x > 93.8).
    # Half 2: A defends left (x < 11.5), H defends right (x > 93.8).
    for half_idx, h_start, h_end in half_blocks:
        sl = slice(h_start, h_end + 1)
        bx_h, by_h = bx[sl], by[sl]
        mh_h, ma_h = min_h[sl], min_a[sl]
        in_y_box = (by_h > 18.0) & (by_h < 50.0)
        in_left_gk = (bx_h >= 0.2) & (bx_h < 11.0) & in_y_box
        in_right_gk = (bx_h > 94.3) & (bx_h <= 105.1) & in_y_box

        if half_idx == 1:
            # Left is H's keeper; Right is A's keeper
            swap_left = in_left_gk & (ma_h < 1.6) & (mh_h > 3.8)
            swap_right = in_right_gk & (mh_h < 1.6) & (ma_h > 3.8)
        else:
            # Left is A's keeper; Right is H's keeper
            swap_left = in_left_gk & (mh_h < 1.6) & (ma_h > 3.8)
            swap_right = in_right_gk & (ma_h < 1.6) & (mh_h > 3.8)

        swap_mask = swap_left | swap_right
        min_h[sl] = np.where(swap_mask, ma_h, mh_h)
        min_a[sl] = np.where(swap_mask, mh_h, ma_h)

    min_both = np.minimum(min_h, min_a)

    v_norm = kin["v_norm"]
    impulse = kin["impulse_mag"]
    defl = kin["deflection_deg"]

    plr_speed_smooth = rolling_mean_1d(series["mean_player_speed"], half_win=45)
    plr_speed_local = rolling_mean_1d(series["mean_player_speed"], half_win=18)

    emissions = np.zeros((n, 3), dtype=np.float32)
    gate_multiplier = np.ones(n, dtype=np.float32)

    vis = ~np.isnan(bx)
    out_of_bounds = vis & ((bx < 0.4) | (bx > length - 0.4) | (by < 0.4) | (by > width - 0.4))

    # Ballistic flight regime and loose-ball lock
    in_flight = vis & (v_norm >= 9.5) & ((np.isnan(defl) | (defl <= 18.0)) & (np.isnan(impulse) | (impulse <= 5.5)))
    ball_free = vis & (min_both > 2.2)
    gate_multiplier[in_flight | ball_free] = 2.8

    touch_node = vis & (min_both < 2.0) & ((impulse >= 3.5) | (v_norm < 4.5))
    gate_multiplier[touch_node] = 0.52

    # 1. Hybrid Proximity Score
    in_play_vis = vis & ~out_of_bounds
    mh_safe = np.nan_to_num(min_h, nan=50.0)
    ma_safe = np.nan_to_num(min_a, nan=50.0)
    mb_safe = np.minimum(mh_safe, ma_safe)

    collective_active_factor = np.clip((np.nan_to_num(plr_speed_smooth, nan=2.0) - 1.15) / 0.55, 0.10, 1.0)
    flight_suppression = (np.where(in_flight & (mb_safe > 1.2), 0.10, 1.0) * collective_active_factor).astype(np.float32)

    gauss_h = np.exp(-0.5 * (mh_safe / 1.80) ** 2)
    gauss_a = np.exp(-0.5 * (ma_safe / 1.80) ** 2)
    # Truncate rational kernel tail beyond 3.0m so players 7m away never get false proximity credit
    rat_h = (1.0 / (1.0 + 0.55 * mh_safe)) * np.exp(-0.5 * (mh_safe / 3.0) ** 2)
    rat_a = (1.0 / (1.0 + 0.55 * ma_safe)) * np.exp(-0.5 * (ma_safe / 3.0) ** 2)

    prox_h = (0.75 * gauss_h + 0.25 * rat_h) * flight_suppression
    prox_a = (0.75 * gauss_a + 0.25 * rat_a) * flight_suppression

    diff_adv = np.clip((ma_safe - mh_safe) / 2.5, -1.5, 1.5) * np.exp(-0.5 * (mb_safe / 3.2) ** 2) * flight_suppression
    impulse_boost = np.where(touch_node, 1.68, 1.0).astype(np.float32)

    score_h = np.where(in_play_vis, (1.45 * prox_h + 0.65 * np.maximum(0.0, diff_adv)) * impulse_boost, 0.0)
    score_a = np.where(in_play_vis, (1.45 * prox_a + 0.65 * np.maximum(0.0, -diff_adv)) * impulse_boost, 0.0)

    # 1b. Bounded Causal Flight Hold (up to 65 frames = 2.6s after a true controlled touch):
    # Only updates `last_team` when a player actually controls/kicks the ball (low speed < 7.0 m/s
    # or strong touch impulse), NEVER when a fast pass merely flies past an opponent!
    last_team = -1
    frames_since_touch = 999
    hold_bonus_h = np.zeros(n, dtype=np.float32)
    hold_bonus_a = np.zeros(n, dtype=np.float32)
    for t in range(n):
        if not in_play_vis[t]:
            frames_since_touch += 1
            if out_of_bounds[t]:
                last_team = -1
            continue
        dh_t = mh_safe[t]
        da_t = ma_safe[t]
        vn_t = v_norm[t] if not np.isnan(v_norm[t]) else 0.0
        can_acquire = (not in_flight[t]) and (vn_t < 7.0 or touch_node[t])
        if can_acquire and dh_t < 1.60 and (da_t - dh_t) >= 0.60:
            last_team = STATE_H
            frames_since_touch = 0
        elif can_acquire and da_t < 1.60 and (dh_t - da_t) >= 0.60:
            last_team = STATE_A
            frames_since_touch = 0
        else:
            frames_since_touch += 1
            if frames_since_touch <= 65 and (mb_safe[t] > 2.1 or in_flight[t]):
                decay = 0.48 * (1.0 - frames_since_touch / 75.0)
                if last_team == STATE_H and da_t > 1.9:
                    hold_bonus_h[t] = decay
                elif last_team == STATE_A and dh_t > 1.9:
                    hold_bonus_a[t] = decay

    score_h += hold_bonus_h * collective_active_factor
    score_a += hold_bonus_a * collective_active_factor

    # 2. Tactical priors from Smart Tagging (strong when ball is NaN; zero when ball is visible and loose > 2.2m
    # so coarse 30s phase tags never force a mid-air turnover while nobody is touching the ball!)
    net_tag_h = np.clip(tag_info["prior_h"] - tag_info["prior_a"], -1.0, 1.0)
    tag_weight = np.where(~vis, 0.90, np.where(mb_safe > 2.2, 0.0, 0.10)).astype(np.float32)
    score_h += np.where(net_tag_h > 0, tag_weight * net_tag_h, 0.0)
    score_a += np.where(net_tag_h < 0, -tag_weight * net_tag_h, 0.0)

    # Post-set-piece restart team boost
    sp_team = tag_info["set_piece_post_kick_team"]
    score_h = np.where(sp_team == STATE_H, score_h + 0.65, score_h)
    score_a = np.where(sp_team == STATE_A, score_a + 0.65, score_a)

    # 3. Dead-Ball (D) physical evidence (combining both pipelines)
    score_d = np.full(n, -0.55, dtype=np.float32)
    score_d[out_of_bounds] = 2.9

    min_both_smooth = rolling_mean_1d(min_both, half_win=35)

    # Case A: Players walking slowly (< 1.35 m/s) and nobody closely dribbling (min_both_smooth > 2.6 m)
    uncontested_drift = vis & (plr_speed_smooth < 1.35) & (min_both_smooth > 2.6)
    score_d[uncontested_drift] = np.maximum(score_d[uncontested_drift], 2.05)

    # Case B: Extreme collective standstill (< 1.15 m/s) from `viterbi_pipeline.py`:
    # Even if one player touches/carries the ball to set up a free kick or throw-in, all 22 players
    # standing at < 1.15 m/s means play is stopped (fixes the 16188..16282 dead-ball pass!).
    extreme_standstill = (plr_speed_smooth < 1.15) & (plr_speed_local < 1.18)
    score_d[extreme_standstill] = np.maximum(score_d[extreme_standstill], 2.25)
    score_h[extreme_standstill] *= 0.20
    score_a[extreme_standstill] *= 0.20

    # Missing-ball (NaN) intervals
    active_tags = tag_info["active_play_tag"]
    for _, h_start, h_end in half_blocks:
        vis_half = vis[h_start:h_end + 1]
        padded = np.concatenate(([True], vis_half, [True]))
        starts = np.where(padded[:-1] & ~padded[1:])[0] + h_start
        ends = np.where(~padded[:-1] & padded[1:])[0] + h_start - 1

        for s, e in zip(starts, ends):
            dur_s = (e - s + 1) / 25.0
            x1 = float(bx[max(h_start, s - 1)])
            y1 = float(by[max(h_start, s - 1)])
            x2 = float(bx[min(h_end, e + 1)])
            y2 = float(by[min(h_end, e + 1)])
            dist_m = float(np.hypot(x2 - x1, y2 - y1))
            v_imp = dist_m / max(0.04, dur_s)
            out1 = (x1 < 0.5) or (x1 > length - 0.5) or (y1 < 0.5) or (y1 > width - 0.5)
            near_boundary = (x1 < 2.0) or (x1 > length - 2.0) or (y1 < 2.0) or (y1 > width - 2.0)
            min_pspd_gap = float(np.nanmin(plr_speed_smooth[s:e + 1]))
            mean_tags_gap = float(np.mean(active_tags[s:e + 1]))

            # If multiple attacking/defending phase tags (>= 2.5) are active and ball moves >= 2.0 m/s
            # without having exited the pitch (not out1) and players aren't standing still (min_pspd >= 1.30),
            # it is a sideline/box occlusion during live play, NOT a stoppage!
            is_live_occlusion = (not out1) and (mean_tags_gap >= 2.5) and (v_imp >= 2.0) and (min_pspd_gap >= 1.30)
            is_stoppage_gap = (dur_s >= 3.0) and (not is_live_occlusion) and (near_boundary or v_imp < 1.5 or min_pspd_gap < 1.75)

            if is_stoppage_gap:
                if v_imp < 1.5:
                    gap_pspd = np.nan_to_num(plr_speed_local[s:e + 1], nan=1.5)
                    d_pot = np.where(gap_pspd < 2.35, 2.35, 0.2).astype(np.float32)
                else:
                    gap_pspd = np.nan_to_num(plr_speed_smooth[s:e + 1], nan=1.5)
                    d_pot = np.where(gap_pspd < 2.10, 2.25, -0.2).astype(np.float32)
                score_d[s:e + 1] = np.maximum(score_d[s:e + 1], d_pot)
                score_h[s:e + 1] = np.where(d_pot > 1.0, score_h[s:e + 1] * 0.25, score_h[s:e + 1])
                score_a[s:e + 1] = np.where(d_pot > 1.0, score_a[s:e + 1] * 0.25, score_a[s:e + 1])
                gate_multiplier[s:e + 1] = 0.8
            else:
                score_d[s:e + 1] = -1.4
                gate_multiplier[s:e + 1] = 2.4

    # Apply Set-Piece pre-kick D lock, post-kick live play lock, and Goal Celebration D lock
    sp_pre_d = tag_info["set_piece_pre_kick_d"]
    score_d[sp_pre_d] = np.maximum(score_d[sp_pre_d], 2.35)
    score_h[sp_pre_d] *= 0.2
    score_a[sp_pre_d] *= 0.2

    sp_post_live = (sp_team >= 0) & ~out_of_bounds & vis
    score_d[sp_post_live] = np.minimum(score_d[sp_post_live], -0.8)

    goal_d = tag_info["goal_dead_mask"]
    score_d[goal_d] = 3.5
    score_h[goal_d] = -1.0
    score_a[goal_d] = -1.0

    # Pre-2nd-half kickoff lock (published in FAQ.md & starter/load_smart_tagging.py: 110808..113741 are D)
    ko2 = KICKOFF_VIDEO_FRAMES.get("2nd Half", 113742)
    h2_start = half_blocks[1][1]
    if ko2 > h2_start:
        score_d[h2_start:ko2] = 4.0
        score_h[h2_start:ko2] = -2.0
        score_a[h2_start:ko2] = -2.0

    # End-of-half whistle lock: when players walk off the pitch or collective speed stays depressed at half end
    for _, h_start, h_end in half_blocks:
        tail_lo = h_end - 2200
        for t in range(tail_lo, h_end + 1):
            rem_vis = np.mean(vis[t:h_end + 1])
            s_slice = plr_speed_smooth[t:h_end + 1]
            rem_spd = float(np.nanmean(s_slice)) if np.any(~np.isnan(s_slice)) else 0.0
            if rem_vis < 0.25 and rem_spd < 1.42:
                score_d[t:h_end + 1] = 3.5
                score_h[t:h_end + 1] = -1.5
                score_a[t:h_end + 1] = -1.5
                break

    emissions[:, STATE_H] = np.nan_to_num(score_h, nan=0.0)
    emissions[:, STATE_A] = np.nan_to_num(score_a, nan=0.0)
    emissions[:, STATE_D] = np.nan_to_num(score_d, nan=-0.35)
    gate_multiplier = np.nan_to_num(gate_multiplier, nan=1.0)

    return emissions, gate_multiplier


def viterbi_decode_half(emissions, gate_multiplier, base_tax_ha=19.5, base_tax_d=28.0):
    """Exact O(T * K^2) Viterbi dynamic programming decoder for a single half."""
    t_len = len(emissions)
    dp = np.zeros((t_len, 3), dtype=np.float64)
    backptr = np.zeros((t_len, 3), dtype=np.int8)
    dp[0] = emissions[0]

    for t in range(1, t_len):
        gm = float(gate_multiplier[t])
        tax_ha = base_tax_ha * gm
        tax_d = base_tax_d

        v0, v1, v2 = dp[t - 1, 0], dp[t - 1, 1], dp[t - 1, 2]

        # To state 0 (H)
        c00, c10, c20 = v0, v1 - tax_ha, v2 - tax_d
        if c00 >= c10 and c00 >= c20:
            dp[t, 0], backptr[t, 0] = c00 + emissions[t, 0], 0
        elif c10 >= c20:
            dp[t, 0], backptr[t, 0] = c10 + emissions[t, 0], 1
        else:
            dp[t, 0], backptr[t, 0] = c20 + emissions[t, 0], 2

        # To state 1 (A)
        c01, c11, c21 = v0 - tax_ha, v1, v2 - tax_d
        if c11 >= c01 and c11 >= c21:
            dp[t, 1], backptr[t, 1] = c11 + emissions[t, 1], 1
        elif c01 >= c21:
            dp[t, 1], backptr[t, 1] = c01 + emissions[t, 1], 0
        else:
            dp[t, 1], backptr[t, 1] = c21 + emissions[t, 1], 2

        # To state 2 (D)
        c02, c12, c22 = v0 - tax_d, v1 - tax_d, v2
        if c22 >= c02 and c22 >= c12:
            dp[t, 2], backptr[t, 2] = c22 + emissions[t, 2], 2
        elif c02 >= c12:
            dp[t, 2], backptr[t, 2] = c02 + emissions[t, 2], 0
        else:
            dp[t, 2], backptr[t, 2] = c12 + emissions[t, 2], 1

    path = np.zeros(t_len, dtype=np.int8)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = backptr[t, path[t]]

    return path


def clean_micro_spells(path, min_d_frames=110, min_live_island_frames=18):
    """Phase-zero spell cleaning combining physical duration laws and zero-lag island merging.

    1. Minimum Stoppage Law: In football, a dead-ball stoppage + restart (`D`) cannot physically
       last less than ~4.4 seconds (110 frames). Any shorter `D` segment is a false tracker
       dropout inside live play and is bridged without splitting the possession.
    2. Symmetric Live Micro-Islands: Ultra-short single-touch rebounds (`prev_st == next_st`
       shorter than `min_live_island_frames` = 0.72 s) are absorbed into the surrounding state
       while preserving valid 1-3 s possessions.
    """
    cleaned = path.copy()
    n = len(cleaned)
    changed = True
    while changed:
        changed = False
        runs = []
        s = 0
        for i in range(1, n + 1):
            if i == n or cleaned[i] != cleaned[s]:
                runs.append([cleaned[s], s, i])
                s = i
        for r_idx in range(len(runs)):
            st, start, end = runs[r_idx]
            length = end - start
            prev_st = runs[r_idx - 1][0] if r_idx > 0 else None
            next_st = runs[r_idx + 1][0] if r_idx + 1 < len(runs) else None

            if st == STATE_D and length < min_d_frames:
                if prev_st is not None and next_st is not None:
                    if prev_st == next_st:
                        cleaned[start:end] = prev_st
                    else:
                        mid = (start + end) // 2
                        cleaned[start:mid] = prev_st
                        cleaned[mid:end] = next_st
                    changed = True
                    break
                elif next_st is not None:
                    cleaned[start:end] = next_st
                    changed = True
                    break
            elif st in (STATE_H, STATE_A) and prev_st is not None and prev_st == next_st and length < min_live_island_frames:
                cleaned[start:end] = prev_st
                changed = True
                break
    return cleaned


def run_unified_pipeline():
    t0 = time.time()
    series = load_or_build_cache()
    n_frames = source_frames("tactical", GAME)
    pitch_dims = GAMES[GAME]["pitch"]
    half_blocks = half_video_blocks(GAME)

    # Step 1: PCHIP interpolation on short in-bounds gaps
    bx_interp, by_interp, _ = interpolate_ball_pchip(
        series["ball_x"], series["ball_y"], half_blocks, max_bridge_s=2.0, margin=1.5, fps=25.0
    )
    series_interp = dict(series)
    series_interp["ball_x"] = bx_interp
    series_interp["ball_y"] = by_interp

    # Step 2: Vector kinematics & Smart Tagging physics
    kin = compute_vector_kinematics(bx_interp, by_interp, fps=25.0, half_span=2)
    events = load_tags(GAME)
    tag_info = build_unified_smart_tagging(n_frames, events, series_interp, kin)

    # Step 3: Emissions & Dynamic Gates
    emissions, gate_multiplier = compute_unified_emissions_and_gates(
        series_interp, kin, tag_info, half_blocks, pitch_dims
    )

    # Step 4: Viterbi Decoding + Phase-Zero Micro-Spell Cleaning
    pred_labels = np.full(n_frames, "X", dtype="<U1")
    for _, h_start, h_end in half_blocks:
        raw_path = viterbi_decode_half(
            emissions[h_start:h_end + 1],
            gate_multiplier[h_start:h_end + 1],
            base_tax_ha=19.5,
            base_tax_d=28.0,
        )
        clean_path = clean_micro_spells(raw_path, min_d_frames=90, min_live_island_frames=38)
        pred_labels[h_start:h_end + 1] = STATE_LABELS[clean_path]

    elapsed = time.time() - t0
    out_csv = os.path.join(ROOT, "possession_file.csv")
    write_submission(out_csv, pred_labels)
    return out_csv, pred_labels, elapsed


if __name__ == "__main__":
    out_csv, pred_labels, elapsed = run_unified_pipeline()
    print(f"Saved {out_csv} ({len(pred_labels)} rows) in {elapsed:.2f} s")
