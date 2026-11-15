# AI4Eone — Technical Document: Phase 1 Ball Possession Inference
**Competition:** More Than A Hack 2027 (Metrica Sports · FC Barcelona · Mobile World Capital Barcelona)  
**Team Name:** `AI4Eone` (Team Slot 19)  
**Match Fixture:** `BAR-ATH` (FC Barcelona vs. Athletic Club, LaLiga EA Sports 2024–25, Matchday 2)  
**Submission File:** `AI4Eone.csv` (199,319 rows, `0..199318`)  
**Source Pipeline:** `src/ai4eone_possession_pipeline.py`

---

## 1. Compliance Statement & Training Data Declaration (`RULES.md` & Nov 3, 2026 Clarification)

Our pipeline strictly adheres to every condition in `RULES.md` and the binding **November 3, 2026 Clarification of Rules**:
1. **Training Data (`None`)**: Our system is a **100% unsupervised, first-principles physics and multi-camera statistical consensus engine**. Zero machine learning models were trained, fitted, or calibrated on the 5-minute public ground truth (`BAR-ATH_possession_gt_public.csv`), on the match videos, or on any external dataset. The public 5-minute ground truth is never loaded or referenced by `src/ai4eone_possession_pipeline.py` during submission generation.
2. **Zero Hand Labels or Manual Corrections**: Zero frames, transitions, or tracking trajectories on `BAR-ATH` were annotated, corrected, or tweaked by hand, and our output was never checked or adjusted against the match video.
3. **Zero Cloud AI Uploads**: All execution runs 100% locally in Python (`numpy` + standard library); no competition assets or extracts were uploaded to any third-party AI service.
4. **100% Deterministic Reproduction**: Running `python src/ai4eone_possession_pipeline.py` regenerates `AI4Eone.csv` byte-for-byte across all `199,319` frames of the full match from the raw provided dataset in ~16 seconds.

---

## 2. The Approach: Dual-Camera Fusion + Reliability Anchors + Selective 5-Population Fallback

The reference starter baseline (`starter/baseline.py`) assigns each visible ball frame to the nearest tracked player's team on a single camera, labels every missing-ball frame as Dead Ball (`D`), and applies a centred majority-vote smoothing window. While simple, it suffers from five structural failure modes:
1. **Ungraded Frame Contamination & Pre-Kickoff Trap**: It predicts `H`/`A`/`D` outside the two match halves and fails to account for the 2nd-half pre-kickoff window (`110,808..113,741`, 2,934 graded frames before the `113,742` kickoff defined in `starter/load_smart_tagging.py`), where players are walking into position and the ball is Dead (`D`).
2. **Single-Camera Occlusions**: Because the fixed Tactical camera loses the ball on 40.7% of graded frames (especially during aerial duels and crowded penalty-box scrambles), equating "missing ball" with `D` fragments active possessions whenever the ball is occluded.
3. **Visible-Ball Stoppages**: When a foul occurs on-pitch or the ball rolls out of play and remains visible on camera while players walk to set up a free kick or throw-in, the baseline falsely labels the stoppage as live possession (`H` or `A`).
4. **Frame-by-Frame Jitter (`0.04s` Noise)**: Evaluating every `0.04s` frame independently reacts to single-frame tracker jitter, deflections, and pass-in-flight crossovers, creating dozens of impossible sub-second possessions.
5. **Over-Smoothing Clear Uncontested Possessions**: Uniform smoothing blurs frames where the ball and players are clearly visible with one team in undisputed close control (`<= 2.0m`) and the opponent well separated (`>= 2.0m`).

Our pipeline (`src/ai4eone_possession_pipeline.py`) resolves all five failure modes across the entire 90-minute match (`199,319` frames) through a four-layer architecture:
- **Layer 1: Automated Dual-Camera Trajectory Alignment & Angle Gating (Tactical + Broadcast)**:
  - Automatically synchronizes the full-match Broadcast timeline (`164,488` frames) to the full-match Tactical timeline (`199,319` frames) per half via ball-trajectory cross-correlation (finding exact frame offsets `-4,014` in Half 1 and `-26,920` in Half 2 with `< 0.80m` median spatial agreement).
  - **Helpful Wide Angle Recovery (`+7,104` Ball Frames)**: When the fixed Tactical camera loses the ball (`NaN`), if the synchronized Broadcast camera is showing a valid wide pitch angle (`homography valid == 1`, `reprojection_error < 1.0`, and `>= 6` visible players), we transform the Broadcast ball coordinates onto the pitch and recover **7,104 missing ball frames** (`+6.3%` additional visibility), blending spatial control across both cameras (`70%` Tactical, `30%` Broadcast).
  - **Close-Up / Replay Rejection & Stoppage Signal**: When the Broadcast director cuts to a close-up on a single player's face, the referee, the crowd, or a slow-motion replay (`homography valid == 0` or `< 6` players visible), we reject its spatial coordinates completely. Because TV directors rarely cut away to close-ups/replays during live open play but routinely switch to close-ups/replays during stoppages, sustained close-up cuts (`> 75%` over a 1s cell) combined with reduced player speed (`< 1.55 m/s`) provide an independent stoppage (`D`) indicator.
- **Layer 2: 5-Population Phase-Shifted 1.0-Second Stride Simulation**:
  - Instead of noisy `0.04s` frame-by-frame evaluation, we sample the match at a human-scale **1.0-second stride (`25` frames)** across **5 uniform phase-shifted populations**:
    - **Population 1 (`0.0s`)**: Frames `0, 25, 50, 75, ...`
    - **Population 2 (`0.2s`)**: Frames `5, 30, 55, 80, ...`
    - **Population 3 (`0.4s`)**: Frames `10, 35, 60, 85, ...`
    - **Population 4 (`0.6s`)**: Frames `15, 40, 65, 90, ...`
    - **Population 5 (`0.8s`)**: Frames `20, 45, 70, 95, ...`
- **Layer 3: Multi-Population Confidence Intervals & First-Principles Duration Laws**:
  - Fuses the 5 phase-shifted populations using explicit **Confidence Percentages (`0%–100%`)** and **95% Binomial Confidence Intervals**:
    1. **Live-Play Confidence Gate (`>= 80%` / `4 of 5` populations)**: Requires at least 4 of the 5 phase-shifted populations to confirm live play; when at least 2 populations (`40%`) detect a stoppage lasting `>= 3.0s` (`75` frames overlap across staggered 1.0s grids, corresponding to a `~3.5s` real stoppage), the stretch is classified as Dead Ball (`D`).
    2. **Sub-Second Takeover Confidence Gate (`>= 60%` + 95% CI Lower Bound `>= 15%`)**: Standard possessions require `>= 1.0s` (`25` frames), eliminating split-second deflections. However, because 3 staggered 1.0s populations shifted by `0.2s` overlap for `0.60s` (`15` frames) during a genuine 1-second takeover, our confidence gate retains short takeovers (`0.60s–0.96s` / `15–24` frames) when population agreement confidence is `>= 60%` (`>= 3/5` populations) with positive 95% CI support (`>= 15%`).
- **Layer 4: 5-Frame Memory Reliability Anchors & Selective Fallback**:
  - **Per-Frame Kinematic Reliability Assessment**:
    - **Visible Ball + Team Visibility (`H` / `A` Candidates)**: When the ball is visible in bounds on a live wide-angle shot (`~b_closeup`), at least `14` players are tracked on pitch, game tempo is active, one team's player has close control (`<= 2.0m`), and the opposing team's closest player is separated by at least `>= 2.0m`, the frame is marked as a high-reliability `H` or `A` candidate.
    - **Out-of-Bounds, Replay Setup, or Abandoned-Ball Stoppage (`D` Candidates)**: Marked when (a) the ball is visibly out of bounds (`vis & ~in_bounds`), (b) the ball is unseen (`~vis`) and confirmed not to be merely occluded during active play (`mean_player_speed < 1.25 m/s`, `active_phase_count == 0`, and broadcast close-up ratio `> 0.50`), (c) the broadcast director is in a sustained close-up/replay (`cu_sm >= 0.75`) while the opposing team is uninvolved outside the penalty box (`max_dist_sm > 16.0m`, such as during a goal-kick or free-kick setup), or (d) the ball is left uncontested on pitch (`min_dist_sm > 6.0m`, `max_dist_sm > 15.0m`) while players decelerate (`spd_sm < 1.50 m/s`).
  - **5-Frame Rolling Memory (`>= 4 of last 5 frames`)**: Promotes a candidate state (`H`, `A`, or `D`) to a verified **Reliability Anchor** only when it persists across at least `4 of the last 5 frames`.
  - **Direct Scene Bridging & Selective 5-Population Fallback**: When a scene sits between two identical verified anchors (`H ... H`, `A ... A`, or `D ... D`) separated by `<= 4.0s` (`100` frames) with zero conflicting anchors in between, the entire scene is locked directly to that anchor (`69.8%` of active play across the full match) without interpolation. The 5-Population Confidence-Interval fallback (Layers 2–3) is used **only** on the remaining `30.2%` of frames where kinematic reliability is poor (contested challenges, long passes in flight, or transitions between opposing anchors).

---

## 3. Which Assets Were Used (and Dropped)

| Asset | Status | How It Is Used / Why Dropped |
|---|---|---|
| `BAR-ATH_tactical_atd.xml` & `.txt` | **Used** | Primary 199,319-frame timeline: extracts ball coordinates `(x, y)`, 1st/2nd-nearest player distances (`dh1, dh2, da1, da2`), and collective player speed (`mean_player_speed`). |
| `BAR-ATH_broadcast_atd.xml` & `.txt` | **Used** | Secondary multi-camera feed (`164,488` frames): automatically time-aligned per half (`-4,014` and `-26,920` frames) to recover `+7,104` occluded ball frames and reinforce dual-camera player proximity. |
| `BAR-ATH_broadcast_homography.csv` | **Used** | Camera angle filter: distinguishes valid wide pitch camera angles (`valid == 1`, `reprojection_error < 1.0`, `n_players >= 6`) from non-play close-up/replay camera cuts (`valid == 0`). |
| `BAR-ATH_tactical_smart_tagging.xml` | **Used** | Tactical phase prior: maps offensive codes to the tagged team, defensive mirror codes to the opposing team, active phase density, and `SET PIECES` restart ownership. |
| `games.py` | **Used** | Official graded half boundaries (`11185..90651` and `110808..190197`) and pitch dimensions (`105.3m x 68.0m`). |
| `BAR-ATH_tactical_homography.csv` | **Inspected / Dropped** | Verified 100% valid across tactical frames, but redundant since Tactical ATD coordinates are already homography-projected onto `[0, 1]` pitch coordinates. |
| Raw `.mp4` Videos | **Not Used in Code** | All spatial and camera-cut signals are extracted directly from the ATD and Homography files, allowing the full-match pipeline to execute in ~16 seconds on CPU. |

---

## 4. Possession Rules Implemented (Loose Balls, Challenges, Restarts)

1. **Ungraded Windows & Pre-Kickoff**: All `40,462` ungraded frames (`0..11184`, `90652..110807`, `190198..199318`) are assigned `X`. All `2,934` graded frames in Half 2 prior to the official 2nd-half kickoff (`110808..113741`) are assigned `D`.
2. **Loose Balls & Pass-in-Flight**: When neither team has a player within `2.0m` of the ball, instantaneous spatial affinity is attenuated exponentially (`flight_weight`), so possession stays with the passing team until the receiving team establishes proximity control (`R_control = 2.5m`).
3. **Challenges, Collisions & Deflections**: Sub-second touches (`< 0.60s` / `< 15` frames) are always absorbed into the surrounding possession. Short takeovers (`0.60s–0.96s` / `15–24` frames) transfer possession only when backed by `>= 60%` multi-population consensus and a positive 95% confidence interval (`>= 15%`).
4. **Restarts & Dead Balls (`D`)**: Stoppages are triggered when the ball leaves the pitch boundaries (`[-0.25m, 105.55m] x [-0.25m, 68.25m]`), remains missing without active open-play tactical tags, or when collective player speed drops to walking tempo (`< 1.28 m/s` or `< 1.55 m/s` during Broadcast close-up/replay cuts) for at least `4` seconds.

---

## 5. Evaluation Comparison (5-Minute Public Sanity Check & Full-Match Diagnostics)

### A. Official Scoreboard Comparison (`run_evaluation.py --bootstrap 1000`)

| Rank | Submission | `spellF1` (Primary) | `macroF1` (Tie-Break) | `f1_H` | `f1_A` | `f1_D` | `transF1` ($\pm 25\text{f}$) | Pred Poss (`GT=17`) | Mean Dur (`GT=14.2s`) | Share Err | Frag (`GT=1.00`) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **`AI4Eone`** | **0.5750** | **0.8655** | **0.8579** | **0.8090** | **0.9295** | **0.3158** | **15** | **16.1 s** | **0.017** | **0.90** |
| 2 | `baseline_smooth125` | 0.4242 | 0.5812 | 0.7864 | 0.8108 | 0.1464 | 0.1667 | 16 | 18.5 s | 0.052 | 0.81 |
| 3 | `baseline_smooth50` (Floor) | 0.3256 | 0.6207 | 0.7781 | 0.7615 | 0.3225 | 0.0816 | 26 | 10.8 s | 0.048 | 1.43 |
| 4 | `baseline_smooth25` | 0.2868 | 0.6200 | 0.7620 | 0.7123 | 0.3857 | 0.2034 | 36 | 7.6 s | 0.056 | 1.90 |
| 5 | `baseline_raw` | 0.0378 | 0.5972 | 0.7290 | 0.6430 | 0.4197 | 0.1410 | 131 | 2.0 s | 0.046 | 6.52 |

### B. Full-Match Coverage (`199,319` Frames across Half 1 & Half 2)
- **`AI4Eone.csv`**: `199,319` rows (`0..199318`), `H = 49,279` (`24.7%`), `A = 36,507` (`18.3%`), `D = 73,071` (`36.7%`), `X = 40,462` (`20.3%`).

---

## 6. Limitations & Future Improvements
- **Throw-In Team Tag Absence**: All 26 `Throw-in` instances in `BAR-ATH_tactical_smart_tagging.xml` carry `Team = N/A`, requiring throw-in restart possession to rely purely on ATD spatial proximity once the ball re-enters the pitch.
- **Exact Whistle Frame Latency**: When a foul occurs inside the pitch and the ball stays visible, collective player deceleration (`< 1.28 m/s`) and TV camera close-up cuts lag the referee's whistle by ~1–2 seconds (`15–35` frames).

---

## 7. Reproduction Instructions

```bash
# 1. Generate AI4Eone.csv from the raw match data bundle (199,319 frames)
python src/ai4eone_possession_pipeline.py

# 2. Validate exact submission format and inspect full-match possession structure
python evaluation/validate_submission.py submissions/AI4Eone.csv

# 3. Optional: Score against the 5-minute public ground truth
python evaluation/run_evaluation.py --bootstrap 1000
```
