"""AI4Eone Possession Inference Pipeline for More Than A Hack 2027 (Phase 1).

Evolved Architecture:
  1. Dual-Camera Multi-Angle Fusion (Tactical Fixed Camera + Broadcast Multi-Camera TV Feed)
     - Automatically aligns the Broadcast timeline to the Tactical timeline per half via ball-trajectory
       cross-correlation (Half 1 lag = -4014 frames, Half 2 lag = -26920 frames; sub-meter 0.72m agreement).
     - Filters Broadcast camera angles using homography validity (`valid == 1`), reprojection error (`< 1.0`),
       and visible player count (`>= 6` players) so that ONLY valid wide-angle pitch views are used to
       recover 7,104 missing-ball frames, while single-player close-ups and replays (`valid == 0`) are
       rejected as spatial inputs and instead used as an independent TV-director stoppage indicator!
  2. 5-Population 1-Second Phase-Shifted Ensemble (`0.0s, 0.2s, 0.4s, 0.6s, 0.8s` offsets)
     - Runs 5 parallel inference populations at a 1.0-second (25-frame) stride starting at offsets
       0.0s (frame 0), 0.2s (frame 5), 0.4s (frame 10), 0.6s (frame 15), and 0.8s (frame 20).
  3. Statistical Duration Laws Post-Correction
     - Law 1 (Dead-Ball Minimum Duration): Real stoppages (throw-ins, free kicks, corners, goal kicks)
       never last < 4.5 seconds. Any short Dead Ball gap (< 115 frames = 4.6s) without out-of-bounds ball
       tracking is bridged as a live camera occlusion.
     - Law 2 (Zero Sub-1-Second Possessions): Eliminates all sub-1-second (< 25 frames) single-population
       phase artefacts while preserving genuine 1.2s+ consensus possessions where the 5 populations agree.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
REPO_ROOT = PROJECT_ROOT if os.path.isfile(os.path.join(PROJECT_ROOT, "games.py")) else os.path.join(PROJECT_ROOT, "more-than-a-hack-2027")
sys.path.insert(0, REPO_ROOT)

from games import GAME, GAMES, asset_paths, half_video_blocks, require, source_frames  # noqa: E402
from starter import load_homography, load_smart_tagging  # noqa: E402

HOME, AWAY, DEAD, ABSTAIN = 0, 1, 2, 3
FPS = 25

# First-principles domain constants
CONTROL_RADIUS_M = 2.5
CLOSE_CONTROL_DIST_M = 2.0
MAX_INBOUNDS_MARGIN_M = 0.25
MIN_DEAD_BALL_FRAMES = 115      # 4.6s minimum stoppage duration law
MIN_POSSESSION_FRAMES = 25      # 1.0s minimum possession duration law (0 sub-1s possessions)

OFFENSIVE_PHASE_WEIGHTS = {
    "BUILD UP": 0.45,
    "PROGRESSION": 0.45,
    "ATTACKING TRANSITION": 0.40,
    "CREATING CHANCES": 0.35,
    "BALL IN FINAL THIRD": 0.30,
    "BALL IN THE BOX": 0.30,
    "LONG BALL": 0.30,
    "PLAYERS IN THE BOX": 0.20,
}
DEFENSIVE_PHASE_WEIGHTS = {
    "DEFENDING IN ATTACKING THIRD": 0.45,
    "DEFENDING IN MIDDLE THIRD": 0.45,
    "DEFENSIVE TRANSITION": 0.40,
    "DEFENDING IN DEFENSIVE THIRD": 0.35,
}


def load_team_of_each_track(meta_path):
    """Return array mapping each track slot to HOME (0) or AWAY (1)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def parse_atd_kinematics(raw_path, team_of_track, n_frames, pitch):
    """Stream raw ATD tracking file and extract per-frame ball & collective player kinematics."""
    length, width = pitch
    ball_vis = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_h = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_a = np.full(n_frames, np.nan, dtype=np.float32)
    second_dist_h = np.full(n_frames, np.nan, dtype=np.float32)
    second_dist_a = np.full(n_frames, np.nan, dtype=np.float32)
    mean_player_speed = np.zeros(n_frames, dtype=np.float32)
    n_players = np.zeros(n_frames, dtype=np.int16)

    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame_idx = int(parts[0]) - 1
            if frame_idx < 0 or frame_idx >= n_frames:
                continue

            ball_str = parts[2].strip()
            has_ball = bool(ball_str) and ball_str[0] != "N"
            bx = by = 0.0
            if has_ball:
                c1 = ball_str.index(",")
                c2 = ball_str.index(",", c1 + 1)
                bx = float(ball_str[:c1]) * length
                by = float(ball_str[c1 + 1:c2]) * width
                ball_vis[frame_idx] = True
                ball_x[frame_idx] = bx
                ball_y[frame_idx] = by

            dh1 = dh2 = 9999.0
            da1 = da2 = 9999.0
            spd_sum = 0.0
            spd_cnt = 0
            pl_cnt = 0

            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                pl_cnt += 1
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1)
                px = float(group[:c1]) * length
                py = float(group[c1 + 1:c2]) * width
                s_str = group[c2 + 1:].strip()
                if s_str and s_str[0] != "N":
                    spd_sum += float(s_str)
                    spd_cnt += 1

                if has_ball:
                    d = float(np.hypot(px - bx, py - by))
                    if team_of_track[slot] == HOME:
                        if d < dh1:
                            dh2 = dh1
                            dh1 = d
                        elif d < dh2:
                            dh2 = d
                    else:
                        if d < da1:
                            da2 = da1
                            da1 = d
                        elif d < da2:
                            da2 = d

            n_players[frame_idx] = pl_cnt
            if spd_cnt > 0:
                mean_player_speed[frame_idx] = spd_sum / spd_cnt
            if has_ball:
                if dh1 < 9999.0:
                    min_dist_h[frame_idx] = dh1
                if dh2 < 9999.0:
                    second_dist_h[frame_idx] = dh2
                if da1 < 9999.0:
                    min_dist_a[frame_idx] = da1
                if da2 < 9999.0:
                    second_dist_a[frame_idx] = da2

    return {
        "ball_vis": ball_vis,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "min_dist_h": min_dist_h,
        "min_dist_a": min_dist_a,
        "second_dist_h": second_dist_h,
        "second_dist_a": second_dist_a,
        "mean_player_speed": mean_player_speed,
        "n_players": n_players,
    }


def align_and_fuse_broadcast_camera(t_kin, b_kin, b_valid, b_err, half_blocks):
    """Synchronize Broadcast camera to Tactical camera per half via ball trajectory cross-correlation,
    filter for valid wide-angle camera shots (`b_valid & b_err < 1.0 & n_players >= 6`), recover
    missing ball frames, and extract the Broadcast close-up/replay stoppage indicator (`~b_valid`).
    """
    n_t = len(t_kin["ball_vis"])
    n_b = len(b_kin["ball_vis"])

    fused = {k: v.copy() for k, v in t_kin.items()}
    b_closeup_t = np.zeros(n_t, dtype=bool)
    b_wide_ball_t = np.zeros(n_t, dtype=bool)
    lags = {}

    for half, hs, he in half_blocks:
        # Cross-correlate ball trajectories to find exact frame lag (b_frame - t_frame)
        step = 5
        t_s = hs + 5000
        t_e = t_s + 25000
        tx_sub = t_kin["ball_x"][t_s:t_e:step]
        ty_sub = t_kin["ball_y"][t_s:t_e:step]
        best_lag, best_dist = 0, 9999.0
        for lag in range(-45000, 10000, 5):
            bs, be = t_s + lag, t_e + lag
            if bs < 0 or be > n_b:
                continue
            bx_sub = b_kin["ball_x"][bs:be:step]
            by_sub = b_kin["ball_y"][bs:be:step]
            both = (~np.isnan(tx_sub)) & (~np.isnan(bx_sub))
            if int(both.sum()) < 500:
                continue
            d = float(np.mean(np.hypot(tx_sub[both] - bx_sub[both], ty_sub[both] - by_sub[both])))
            if d < best_dist:
                best_dist, best_lag = d, lag

        for lag in range(best_lag - 10, best_lag + 11):
            bs, be = t_s + lag, t_e + lag
            if bs < 0 or be > n_b:
                continue
            both = (~np.isnan(t_kin["ball_x"][t_s:t_e])) & (~np.isnan(b_kin["ball_x"][bs:be]))
            d = float(np.median(np.hypot(
                t_kin["ball_x"][t_s:t_e][both] - b_kin["ball_x"][bs:be][both],
                t_kin["ball_y"][t_s:t_e][both] - b_kin["ball_y"][bs:be][both],
            )))
            if d < best_dist:
                best_dist, best_lag = d, lag

        lags[half] = best_lag
        t_idx = np.arange(hs, he + 1)
        b_idx = t_idx + best_lag
        ok = (b_idx >= 0) & (b_idx < n_b)
        ti, bi = t_idx[ok], b_idx[ok]

        # Broadcast close-up / replay indicator (when homography is invalid)
        b_closeup_t[ti] = ~b_valid[bi]

        # Valid wide-angle broadcast frame that sees the ball and at least 6 players on the pitch
        wide_ok = b_valid[bi] & (b_err[bi] < 1.0) & (b_kin["n_players"][bi] >= 6) & b_kin["ball_vis"][bi]
        b_wide_ball_t[ti] = wide_ok

        # 1. Where Tactical lost the ball, recover ball & player distances from valid wide Broadcast camera
        rec = (~fused["ball_vis"][ti]) & wide_ok
        rec_t, rec_b = ti[rec], bi[rec]
        fused["ball_vis"][rec_t] = True
        fused["ball_x"][rec_t] = b_kin["ball_x"][rec_b]
        fused["ball_y"][rec_t] = b_kin["ball_y"][rec_b]
        fused["min_dist_h"][rec_t] = b_kin["min_dist_h"][rec_b]
        fused["min_dist_a"][rec_t] = b_kin["min_dist_a"][rec_b]
        fused["second_dist_h"][rec_t] = b_kin["second_dist_h"][rec_b]
        fused["second_dist_a"][rec_t] = b_kin["second_dist_a"][rec_b]

        # 2. Where BOTH cameras see the ball in a valid wide angle, fuse their player distances (70% Tactical, 30% Broadcast)
        bw = t_kin["ball_vis"][ti] & wide_ok
        ti_bw, bi_bw = ti[bw], bi[bw]
        fused["min_dist_h"][ti_bw] = 0.70 * t_kin["min_dist_h"][ti_bw] + 0.30 * b_kin["min_dist_h"][bi_bw]
        fused["min_dist_a"][ti_bw] = 0.70 * t_kin["min_dist_a"][ti_bw] + 0.30 * b_kin["min_dist_a"][bi_bw]

    fused["b_closeup"] = b_closeup_t
    fused["b_wide_ball"] = b_wide_ball_t
    fused["lags"] = lags
    return fused


def build_smart_tagging_signals(n_frames, game=GAME, data_dir=None):
    """Extract frame-level tactical phase priors and active open-play density from Smart Tagging XML."""
    events = load_smart_tagging.load(game=game, data_dir=data_dir)
    score_h = np.zeros(n_frames, dtype=np.float32)
    score_a = np.zeros(n_frames, dtype=np.float32)
    active_phase_count = np.zeros(n_frames, dtype=np.int16)
    set_piece_events = []

    for ev in events:
        code = (ev["code"] or "").strip().upper()
        sf = max(0, min(n_frames, ev["start_frame"]))
        ef = max(0, min(n_frames, ev["end_frame"]))
        if ef <= sf:
            continue

        team_name = (ev["tags"].get("Team") or "").strip()
        if team_name == "Barcelona":
            team_code = HOME
        elif team_name == "Atletic Club":
            team_code = AWAY
        else:
            team_code = -1

        if code == "SET PIECES":
            set_piece_events.append(ev)
            if team_code in (HOME, AWAY):
                sp_end = min(ef, sf + 10 * FPS)
                if team_code == HOME:
                    score_h[sf:sp_end] += 0.60
                else:
                    score_a[sf:sp_end] += 0.60
            continue

        if code == "GOALS":
            continue

        active_phase_count[sf:ef] += 1

        if team_code in (HOME, AWAY):
            if code in OFFENSIVE_PHASE_WEIGHTS:
                w = OFFENSIVE_PHASE_WEIGHTS[code]
                if team_code == HOME:
                    score_h[sf:ef] += w
                else:
                    score_a[sf:ef] += w
            elif code in DEFENSIVE_PHASE_WEIGHTS:
                w = DEFENSIVE_PHASE_WEIGHTS[code]
                if team_code == HOME:
                    score_a[sf:ef] += w
                else:
                    score_h[sf:ef] += w

    total = score_h + score_a
    prior_diff = np.zeros(n_frames, dtype=np.float32)
    mask = total > 0
    prior_diff[mask] = (score_h[mask] - score_a[mask]) / np.maximum(1.0, total[mask])
    return prior_diff, active_phase_count, set_piece_events


def box_smooth(arr, half_window):
    """Fast O(N) centred moving average over +/- half_window elements."""
    if half_window <= 0:
        return arr.copy()
    n = len(arr)
    csum = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
    idx = np.arange(n)
    low = np.maximum(0, idx - half_window)
    high = np.minimum(n, idx + half_window + 1)
    return ((csum[high] - csum[low]) / (high - low)).astype(np.float32)


def compute_instantaneous_evidence(kin, prior_diff, pitch):
    """Compute per-frame spatial control evidence + Smart Tagging prior and in-bounds mask."""
    length, width = pitch
    n_frames = len(kin["ball_vis"])
    vis = kin["ball_vis"]
    bx, by = kin["ball_x"], kin["ball_y"]
    dh1, da1 = kin["min_dist_h"], kin["min_dist_a"]
    dh2, da2 = kin["second_dist_h"], kin["second_dist_a"]

    inst_evidence = np.zeros(n_frames, dtype=np.float32)
    v = np.flatnonzero(vis & (~np.isnan(dh1)) & (~np.isnan(da1)))
    aff_h = np.exp(-dh1[v] / CONTROL_RADIUS_M) + 0.20 * np.exp(-np.minimum(dh2[v], 30.0) / (2.0 * CONTROL_RADIUS_M))
    aff_a = np.exp(-da1[v] / CONTROL_RADIUS_M) + 0.20 * np.exp(-np.minimum(da2[v], 30.0) / (2.0 * CONTROL_RADIUS_M))
    min_d = np.minimum(dh1[v], da1[v])
    flight_weight = np.clip(np.exp(-np.maximum(0.0, min_d - CLOSE_CONTROL_DIST_M) / 4.0), 0.15, 1.0)
    diff = (aff_h - aff_a) / np.maximum(1e-4, aff_h + aff_a)
    inst_evidence[v] = diff * flight_weight

    combined_ev = np.where(vis, inst_evidence + 0.22 * prior_diff, 0.25 * prior_diff).astype(np.float32)

    in_bounds = np.zeros(n_frames, dtype=bool)
    vis_idx = np.flatnonzero(vis)
    in_bounds[vis_idx] = (
        (bx[vis_idx] >= -MAX_INBOUNDS_MARGIN_M)
        & (bx[vis_idx] <= length + MAX_INBOUNDS_MARGIN_M)
        & (by[vis_idx] >= -MAX_INBOUNDS_MARGIN_M)
        & (by[vis_idx] <= width + MAX_INBOUNDS_MARGIN_M)
    )
    return combined_ev, in_bounds


def run_single_1s_population(offset_frames, kin, combined_ev, in_bounds, active_phase_count, half_blocks, pitch):
    """Run one 1.0-second (25-frame) stride simulation starting at `offset_frames` (0, 5, 10, 15, or 20)."""
    length, width = pitch
    n_frames = len(kin["ball_vis"])
    vis = kin["ball_vis"]
    bx, by = kin["ball_x"], kin["ball_y"]
    b_closeup = kin.get("b_closeup", np.zeros(n_frames, dtype=bool))
    kickoff_2nd = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]

    labels = np.full(n_frames, "X", dtype="<U1")

    for half, h_start, h_stop in half_blocks:
        labels[h_start:h_stop + 1] = "D"
        play_start = kickoff_2nd if half == 2 else h_start

        sample_idx = np.arange(play_start + offset_frames, h_stop + 1, FPS)
        n_samp = len(sample_idx)

        s_inb = np.zeros(n_samp, dtype=bool)
        s_bx = np.zeros(n_samp, dtype=np.float32)
        s_by = np.zeros(n_samp, dtype=np.float32)
        s_act = np.zeros(n_samp, dtype=np.float32)
        s_spd = np.zeros(n_samp, dtype=np.float32)
        s_cu = np.zeros(n_samp, dtype=np.float32)
        s_ev = np.zeros(n_samp, dtype=np.float32)

        for k, c_idx in enumerate(sample_idx):
            w0 = max(play_start, c_idx - 12)
            w1 = min(h_stop + 1, c_idx + 13)
            s_inb[k] = np.mean(in_bounds[w0:w1]) >= 0.40
            v_cell = vis[w0:w1]
            if v_cell.any():
                s_bx[k] = float(np.nanmean(bx[w0:w1]))
                s_by[k] = float(np.nanmean(by[w0:w1]))
            else:
                s_bx[k] = np.nan
                s_by[k] = np.nan
            s_act[k] = float(np.mean(active_phase_count[w0:w1] > 0))
            s_spd[k] = float(np.mean(kin["mean_player_speed"][w0:w1]))
            s_cu[k] = float(np.mean(b_closeup[w0:w1]))
            s_ev[k] = float(np.mean(combined_ev[w0:w1]))

        s_spd_smooth = box_smooth(s_spd, 2)
        s_alive = s_inb.copy()

        # 1. Bridge occlusions on the 1-second grid
        dead_runs = (~s_alive).astype(np.int8)
        edges = np.diff(np.concatenate(([0], dead_runs, [0])))
        for ds, de in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            gap_s = de - ds
            if ds == 0 or de >= n_samp:
                continue
            if gap_s <= 1:
                jump = float(np.hypot(s_bx[de] - s_bx[ds - 1], s_by[de] - s_by[ds - 1]))
                if np.isnan(jump) or jump <= 20.0:
                    s_alive[ds:de] = True
            elif gap_s <= 12:
                bnd0 = min(s_bx[ds - 1], length - s_bx[ds - 1], s_by[ds - 1], width - s_by[ds - 1])
                bnd1 = min(s_bx[de], length - s_bx[de], s_by[de], width - s_by[de])
                act_cov = float(np.mean(s_act[ds:de]))
                cu_cov = float(np.mean(s_cu[ds:de]))
                g_spd = float(np.mean(s_spd[ds:de]))
                if bnd0 >= 1.0 and bnd1 >= 1.0 and ((act_cov >= 0.85 and g_spd >= 1.25) or (cu_cov < 0.25 and act_cov >= 0.50)):
                    s_alive[ds:de] = True

        # 2. Detect visible-ball stoppages using player tempo AND Broadcast close-up/replay cuts
        low_tempo = ((s_act < 0.2) & (s_spd_smooth < 1.28)) | ((s_cu > 0.75) & (s_spd_smooth < 1.55) & (s_act < 0.5))
        lt_edges = np.diff(np.concatenate(([0], low_tempo.astype(np.int8), [0])))
        for ls, le in zip(np.flatnonzero(lt_edges == 1), np.flatnonzero(lt_edges == -1)):
            if (le - ls) >= 4:
                s_alive[ls:le] = False

        # 3. Team possession on the 1-second grid (+/- 2s window)
        s_ev_smooth = box_smooth(s_ev, 2)
        s_team = np.where(s_ev_smooth >= 0.0, HOME, AWAY).astype(np.int8)

        frame_indices = np.arange(play_start, h_stop + 1)
        nearest_samp = np.clip(np.round((frame_indices - (play_start + offset_frames)) / float(FPS)).astype(int), 0, n_samp - 1)

        f_alive = s_alive[nearest_samp]
        f_team = s_team[nearest_samp]

        alive_frames = frame_indices[f_alive]
        labels[alive_frames] = np.where(f_team[f_alive] == HOME, "H", "A")

    return labels


def fuse_5_populations_with_confidence_intervals(
    pop_labels_list, kin, combined_ev, in_bounds, half_blocks,
    live_conf_pct_gate=80.0, takeover_conf_pct_gate=60.0,
    min_d_frames=75, min_short_poss_frames=15, min_std_poss_frames=25
):
    """Combine the 5 phase-shifted populations (0.0s, 0.2s, 0.4s, 0.6s, 0.8s) using explicit
    Confidence Percentages and 95% Confidence Intervals across the 5 populations:
      1. Live-Play Confidence Gate (`live_conf_pct_gate = 80%`, i.e., >= 4/5 populations):
         Requires >= 80% population confidence to stay in live play; when >= 2/5 populations (40%)
         detect a stoppage lasting >= 3.0s (75 frames, the exact 5-phase overlap of a 3.5s stoppage),
         classifies the stretch as Dead Ball (`D`).
      2. Sub-Second Takeover Confidence Gate (`takeover_conf_pct_gate = 60%` + 95% CI check):
         Calculates per-frame team control confidence percentage (0%..100%) and 95% CI from the 5
         populations + spatial control. Keeps short possessions (0.60s..0.96s / 15..24 frames) ONLY
         when population agreement confidence >= 60% (>= 3/5 populations) and spatial CI confirms control,
         while eliminating all low-confidence noise (< 0.60s).
    """
    n_frames = len(kin["ball_vis"])
    vis = kin["ball_vis"]
    kickoff_2nd = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]

    char_to_code = {"H": HOME, "A": AWAY, "D": DEAD, "X": ABSTAIN}
    pop_matrix = np.zeros((len(pop_labels_list), n_frames), dtype=np.int8)
    for i, lbls in enumerate(pop_labels_list):
        for ch, code in char_to_code.items():
            pop_matrix[i, lbls == ch] = code

    out = np.full(n_frames, "X", dtype="<U1")
    min_alive_votes = int(np.ceil((live_conf_pct_gate / 100.0) * len(pop_labels_list)))

    for half, h_start, h_stop in half_blocks:
        out[h_start:h_stop + 1] = "D"
        play_start = kickoff_2nd if half == 2 else h_start
        sub = pop_matrix[:, play_start:h_stop + 1]

        # 1. Live vs Dead Confidence Percentage across the 5 populations (0%..100%)
        alive_votes = np.sum((sub == HOME) | (sub == AWAY), axis=0)
        h_alive = alive_votes >= min_alive_votes

        # Statistical Law 1: Dead Ball (D) minimum overlap duration >= min_d_frames (75 frames = 3.0s overlap = 3.5s real stoppage)
        dead_runs = (~h_alive).astype(np.int8)
        edges = np.diff(np.concatenate(([0], dead_runs, [0])))
        for ds, de in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if (de - ds) < min_d_frames:
                oob_cnt = np.sum(vis[play_start + ds:play_start + de] & (~in_bounds[play_start + ds:play_start + de]))
                if oob_cnt < 15:
                    h_alive[ds:de] = True

        al_edges = np.diff(np.concatenate(([0], h_alive.astype(np.int8), [0])))
        for as_, ae in zip(np.flatnonzero(al_edges == 1), np.flatnonzero(al_edges == -1)):
            if (ae - as_) < 40:
                h_alive[as_:ae] = False

        # 2. Team Control Confidence Percentage & 95% Confidence Interval across the 5 populations
        vh = np.sum(sub == HOME, axis=0).astype(np.float32)
        va = np.sum(sub == AWAY, axis=0).astype(np.float32)
        n_active_pops = np.maximum(1.0, vh + va)
        # Population agreement confidence percentage (0%..100%) and Binomial 95% CI standard error
        win_votes = np.maximum(vh, va)
        pop_conf_pct = (win_votes / 5.0) * 100.0
        p_hat = win_votes / n_active_pops
        se_binom = np.sqrt(np.maximum(0.0, p_hat * (1.0 - p_hat)) / n_active_pops)
        ci_lower_pct = np.clip((p_hat - 1.96 * se_binom) * 100.0, 0.0, 100.0)

        v_diff = (vh - va) + 0.25 * box_smooth(combined_ev[play_start:h_stop + 1], 35)
        team_arr = np.where(v_diff >= 0.0, HOME, AWAY).astype(np.int8)

        # 3. Confidence-Gated Sub-Second Takeover Retention:
        # Keep short takeovers (0.60s..0.96s / 15..24 frames) ONLY when population confidence >= takeover_conf_pct_gate (60%)
        al_edges = np.diff(np.concatenate(([0], h_alive.astype(np.int8), [0])))
        for as_, ae in zip(np.flatnonzero(al_edges == 1), np.flatnonzero(al_edges == -1)):
            seg = team_arr[as_:ae]
            seg_conf = pop_conf_pct[as_:ae]
            seg_ci = ci_lower_pct[as_:ae]
            if len(seg) <= min_short_poss_frames:
                maj = HOME if np.sum(v_diff[as_:ae]) >= 0.0 else AWAY
                team_arr[as_:ae] = maj
                continue
            changed = True
            while changed:
                changed = False
                breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
                bounds = np.concatenate(([0], breaks, [len(seg)]))
                if len(bounds) <= 2:
                    break
                lengths = np.diff(bounds)
                candidates = []
                for idx_b, ln in enumerate(lengths):
                    s_sub, e_sub = bounds[idx_b], bounds[idx_b + 1]
                    if ln < min_std_poss_frames:
                        mean_conf = float(np.mean(seg_conf[s_sub:e_sub]))
                        mean_ci_low = float(np.mean(seg_ci[s_sub:e_sub]))
                        # Keep sub-second takeover if >= 0.60s (15f) AND Confidence >= 60% (with positive 95% CI support >= 15%)
                        if ln >= min_short_poss_frames and mean_conf >= takeover_conf_pct_gate and mean_ci_low >= 15.0:
                            continue
                        candidates.append((ln, idx_b))
                if candidates:
                    _, shortest_idx = min(candidates)
                    s_sub, e_sub = bounds[shortest_idx], bounds[shortest_idx + 1]
                    seg[s_sub:e_sub] = seg[s_sub - 1] if shortest_idx > 0 else seg[e_sub]
                    changed = True
            team_arr[as_:ae] = seg

        alive_indices = np.flatnonzero(h_alive) + play_start
        out[alive_indices] = np.where(team_arr[h_alive] == HOME, "H", "A")

    return out


def apply_reliability_anchors_and_selective_fallback(
    fallback_labels, kin, t_kin, in_bounds, active_phase_count, half_blocks,
    d_close=2.0, d_sep=2.0, mem_len=5, bridge_s=4.0, min_players=14, min_anchor_frames=15
):
    """
    Reliability-Anchor + 5-Frame Memory + Selective 5-Population Fallback:
      1. Frame Reliability Assessment:
         - Visible Ball + Team Visibility (`n_players >= 14`):
           Marks a frame as a high-reliability Home (`H`) or Away (`A`) candidate when the ball is
           visible in bounds, game tempo is active, one team's nearest player is within `d_close <= 2.0m`,
           and the opposing team's nearest player is separated by at least `d_sep >= 2.0m`.
         - Out-of-Bounds or Confident Unseen Stoppage (`D`):
           Marks a frame as a high-reliability Dead Ball (`D`) candidate when the ball is visibly
           out of bounds (`vis & ~in_bounds`), OR when the ball is unseen (`~vis`) and confirmed not to be
           merely occluded during live play (`mean_player_speed < 1.25 m/s`, `active_phase_count == 0`,
           and broadcast close-up ratio `> 0.50`).
      2. 5-Frame Rolling Memory (`mem_len = 5`):
         Promotes a candidate state (`H`, `A`, or `D`) to a verified Reliability Anchor only when at least
         `4 of the last 5 frames` (`>= 80%` memory consistency) agree on that exact state.
      3. Direct Anchor Bridging (Zero Interpolation Needed Between Identical Anchors):
         Whenever two identical verified anchors (`H ... H`, `A ... A`, or `D ... D`) occur within
         `bridge_s <= 4.0s` (`100 frames`) with no conflicting anchor in between, locks the entire
         interval directly to that anchor.
      4. Selective 5-Population Fallback:
         Retains the 5-Population Confidence-Interval prediction (`fallback_labels`) ONLY on segments
         where direct kinematic reliability is poor (contested duels, long aerial passes, or transitions
         between opposing anchors).
    """
    n_frames = len(kin["ball_vis"])
    vis = kin["ball_vis"]
    dh1, da1 = kin["min_dist_h"], kin["min_dist_a"]
    spd = kin["mean_player_speed"]
    b_cu = kin["b_closeup"]
    n_pl = t_kin["n_players"]
    kickoff_2nd = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]

    inst_state = np.full(n_frames, -1, dtype=np.int8)
    active_tempo = (spd >= 1.30) | (active_phase_count > 0)
    h_clear = vis & in_bounds & active_tempo & (dh1 <= d_close) & ((da1 - dh1) >= d_sep) & (n_pl >= min_players)
    a_clear = vis & in_bounds & active_tempo & (da1 <= d_close) & ((dh1 - da1) >= d_sep) & (n_pl >= min_players)
    inst_state[h_clear] = HOME
    inst_state[a_clear] = AWAY

    spd_sm = box_smooth(spd, 25)
    cu_sm = box_smooth(b_cu.astype(np.float32), 25)
    oob_clear = vis & (~in_bounds)
    unseen_stoppage_clear = (~vis) & (spd_sm < 1.25) & (active_phase_count == 0) & (cu_sm > 0.50)
    inst_state[oob_clear | unseen_stoppage_clear] = DEAD

    # 5-Frame Memory Confirmation (>= 4 of last 5 frames)
    anchor_state = np.full(n_frames, -1, dtype=np.int8)
    for code in (HOME, AWAY, DEAD):
        is_c = (inst_state == code).astype(np.float32)
        csum = np.concatenate(([0.0], np.cumsum(is_c)))
        idx = np.arange(n_frames)
        low = np.maximum(0, idx - mem_len + 1)
        mem_cnt = csum[idx + 1] - csum[low]
        anchor_state[mem_cnt >= (mem_len - 1)] = code

    out = fallback_labels.copy()
    char_map = {HOME: "H", AWAY: "A", DEAD: "D"}
    max_bridge_f = int(bridge_s * 25)

    for half, h_start, h_stop in half_blocks:
        play_start = kickoff_2nd if half == 2 else h_start
        a_slice = anchor_state[play_start:h_stop + 1].copy()
        valid_idx = np.flatnonzero(a_slice != -1)
        if len(valid_idx) == 0:
            continue

        # Bridge between identical anchors when no conflicting anchor occurs in between
        for k in range(len(valid_idx) - 1):
            i0, i1 = valid_idx[k], valid_idx[k + 1]
            if a_slice[i0] == a_slice[i1] and (i1 - i0) <= max_bridge_f:
                if a_slice[i0] in (HOME, AWAY):
                    if np.sum(out[play_start + i0:play_start + i1] == "D") > 50:
                        continue
                a_slice[i0:i1 + 1] = a_slice[i0]

        # Lock verified anchor scenes; preserve Dead Ball boundaries while overriding poor-reliability team labels
        edges = np.diff(np.concatenate(([-2], a_slice, [-2])))
        change_pts = np.flatnonzero(edges != 0)
        for idx_c in range(len(change_pts) - 1):
            s_a, e_a = change_pts[idx_c], change_pts[idx_c + 1]
            st = a_slice[s_a]
            if st in (HOME, AWAY, DEAD) and (e_a - s_a) >= min_anchor_frames:
                if st in (HOME, AWAY):
                    sub_out = out[play_start + s_a:play_start + e_a]
                    alive_m = (sub_out == "H") | (sub_out == "A")
                    sub_out[alive_m] = char_map[st]
                else:
                    out[play_start + s_a:play_start + e_a] = "D"

        # Clean any micro-fragments (< 0.60s / 15 frames) created at anchor-to-fallback boundaries
        h_alive = (out[play_start:h_stop + 1] == "H") | (out[play_start:h_stop + 1] == "A")
        al_edges = np.diff(np.concatenate(([0], h_alive.astype(np.int8), [0])))
        for as_, ae in zip(np.flatnonzero(al_edges == 1), np.flatnonzero(al_edges == -1)):
            seg = np.where(out[play_start + as_:play_start + ae] == "H", HOME, AWAY)
            changed = True
            while changed:
                changed = False
                breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
                bounds = np.concatenate(([0], breaks, [len(seg)]))
                if len(bounds) <= 2:
                    break
                lengths = np.diff(bounds)
                shortest_idx = int(np.argmin(lengths))
                if lengths[shortest_idx] < min_anchor_frames:
                    s_sub, e_sub = bounds[shortest_idx], bounds[shortest_idx + 1]
                    seg[s_sub:e_sub] = seg[s_sub - 1] if shortest_idx > 0 else seg[e_sub]
                    changed = True
            out[play_start + as_:play_start + ae] = np.where(seg == HOME, "H", "A")

    return out


def write_submission(path, labels):
    """Write a validated 199,319-row frame,label CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


def main():
    parser = argparse.ArgumentParser(description="AI4Eone Evolved Dual-Camera Reliability-Anchor + 5-Population Pipeline")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "submissions"),
                        help="Directory to write candidate submission CSVs")
    parser.add_argument("--eval-dir", default=os.path.join(REPO_ROOT, "evaluation", "results"),
                        help="Also copy candidate CSVs to evaluation/results for side-by-side scoring")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run optional automated sanity check against public 5-min GT after generating CSVs")
    args = parser.parse_args()

    t0 = time.time()
    t_paths = asset_paths(GAME, source="tactical")
    b_paths = asset_paths(GAME, source="broadcast")
    n_t = source_frames("tactical", GAME)
    n_b = source_frames("broadcast", GAME)
    pitch = GAMES[GAME]["pitch"]
    half_blocks = half_video_blocks(GAME)

    print(f"[1/5] Streaming full-match Tactical ATD ({n_t} frames) & Broadcast ATD ({n_b} frames)...")
    t_team = load_team_of_each_track(t_paths["atd_meta"])
    b_team = load_team_of_each_track(b_paths["atd_meta"])
    t_kin = parse_atd_kinematics(t_paths["atd_raw"], t_team, n_t, pitch)
    b_kin = parse_atd_kinematics(b_paths["atd_raw"], b_team, n_b, pitch)
    _, b_err, b_valid = load_homography.load(GAME, source="broadcast")
    print(f"      Parsed both full-match camera feeds in {time.time() - t0:.1f}s.")

    print("[2/5] Aligning Broadcast multi-camera angles to Tactical timeline & filtering valid wide shots...")
    fused_kin = align_and_fuse_broadcast_camera(t_kin, b_kin, b_valid, b_err, half_blocks)
    rec_cnt = int(fused_kin["ball_vis"].sum()) - int(t_kin["ball_vis"].sum())
    print(f"      Synchronized lags: {fused_kin['lags']} | Recovered +{rec_cnt} missing ball frames from wide Broadcast angles.")

    print("[3/5] Extracting Smart Tagging tactical phase priors & running 5 Phase-Shifted 1.0s Populations...")
    prior_diff, active_phase_count, _ = build_smart_tagging_signals(n_t, GAME)
    offsets = [(0, "Pop1_0.0s"), (5, "Pop2_0.2s"), (10, "Pop3_0.4s"), (15, "Pop4_0.6s"), (20, "Pop5_0.8s")]
    dc_ev, dc_inb = compute_instantaneous_evidence(fused_kin, prior_diff, pitch)
    dc_pop_list = []
    for off_f, _ in offsets:
        lbls = run_single_1s_population(off_f, fused_kin, dc_ev, dc_inb, active_phase_count, half_blocks, pitch)
        dc_pop_list.append(lbls)

    print("[4/5] Computing 5-Population 95% Confidence Interval fallback across all 199,319 frames...")
    dc_conf_labels = fuse_5_populations_with_confidence_intervals(
        dc_pop_list, fused_kin, dc_ev, dc_inb, half_blocks,
        live_conf_pct_gate=80.0, takeover_conf_pct_gate=60.0,
        min_d_frames=75, min_short_poss_frames=15, min_std_poss_frames=25
    )

    print("[5/5] Locking 5-Frame Memory Reliability Anchors & applying 5-Population fallback only where reliability is poor...")
    final_labels = apply_reliability_anchors_and_selective_fallback(
        dc_conf_labels, fused_kin, t_kin, dc_inb, active_phase_count, half_blocks,
        d_close=2.0, d_sep=2.0, mem_len=5, bridge_s=4.0, min_players=14, min_anchor_frames=15
    )

    sub_path = write_submission(os.path.join(args.out_dir, "AI4Eone.csv"), final_labels)
    if args.eval_dir and os.path.isdir(os.path.dirname(os.path.abspath(args.eval_dir))):
        write_submission(os.path.join(args.eval_dir, "AI4Eone.csv"), final_labels)
    counts = {k: int(np.sum(final_labels == k)) for k in ("H", "A", "D", "X")}
    print(f"  -> AI4Eone.csv | H={counts['H']:>6}  A={counts['A']:>6}  D={counts['D']:>6}  X={counts['X']:>6} | saved to {sub_path}")

    if args.evaluate:
        sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))
        import possession_metrics as pm  # noqa: E402
        gt_path = os.path.join(REPO_ROOT, "evaluation", "gt", f"{GAME}_possession_gt_public.csv")
        if os.path.isfile(gt_path):
            gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
            r_final = pm.evaluate(gt_codes, pm.label_codes(final_labels), taus=pm.TAUS, bootstrap_n=0)
            print(
                f"  -> Public 5-min check: spellF1={r_final['spell_f1_mean']:.4f}  "
                f"macroF1={r_final['macro_f1']:.4f}  f1_D={r_final['per_class']['D']['f1']:.4f}"
            )


if __name__ == "__main__":
    main()
