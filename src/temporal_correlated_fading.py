from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
"""
Experiment 1: Large-Scale Temporal Correlated Fading and Burst Erasure Simulation
Target: 1,000,000 Slots (10^6 Monte Carlo Realizations)
Evaluates burst loss statistics, drift distortion, and Lyapunov virtual queue stability.
"""

import numpy as np
import json, os, time

np.random.seed(42)

def run_large_scale_temporal_simulation(n_slots=1000000, tau_c_ms=2.5, T_slot_ms=1.0):
    t_start = time.time()
    print(f"[*] Starting 1,000,000-slot simulation (tau_c = {tau_c_ms} ms, T_slot = {T_slot_ms} ms)...")
    
    # Inter-slot correlation under Taylor's frozen turbulence hypothesis
    rho = np.exp(- (T_slot_ms / tau_c_ms)**(5.0/3.0))
    print(f"[*] Inter-slot correlation rho = {rho:.4f}")
    
    # Target operating point marginal & joint outages under strong turbulence (Rytov = 2.5)
    # Mode +1 marginal outage: 2.48% (thresh = -1.963)
    # Mode -2 marginal outage: 2.52% (thresh = -1.956)
    # Cross-modal correlation: r_cross = 0.22
    r_cross = 0.22
    
    # Fast vectorized generation of correlated bivariate Gauss-Markov series in blocks
    block_size = 100000
    n_blocks = n_slots // block_size
    
    z1 = np.empty(n_slots, dtype=np.float32)
    z2 = np.empty(n_slots, dtype=np.float32)
    
    cur_z1 = np.random.randn()
    cur_z2 = r_cross * cur_z1 + np.sqrt(1 - r_cross**2) * np.random.randn()
    
    sqrt_1_minus_rho2 = np.sqrt(1.0 - rho**2)
    sqrt_1_minus_rcross2 = np.sqrt(1.0 - r_cross**2)
    
    idx = 0
    for b in range(n_blocks):
        e1 = np.random.randn(block_size).astype(np.float32)
        e2_raw = np.random.randn(block_size).astype(np.float32)
        e2 = r_cross * e1 + sqrt_1_minus_rcross2 * e2_raw
        
        # Sequentially propagate AR(1) state within block using loop (numba-like fast in numpy)
        for i in range(block_size):
            cur_z1 = rho * cur_z1 + sqrt_1_minus_rho2 * e1[i]
            cur_z2 = rho * cur_z2 + sqrt_1_minus_rho2 * e2[i]
            z1[idx] = cur_z1
            z2[idx] = cur_z2
            idx += 1
            
    # Apply outage thresholds
    thresh1 = -1.963
    thresh2 = -1.956
    
    outage_p1 = (z1 < thresh1)
    outage_m2 = (z2 < thresh2)
    
    outage_single = outage_p1
    outage_dual = outage_p1 & outage_m2
    
    p_out_single = float(np.mean(outage_single))
    p_out_dual = float(np.mean(outage_dual))
    
    print(f"[*] Computed outages over {n_slots:,} slots:")
    print(f"    - Single Anchor Outage (Mode +1): {p_out_single:.5f} ({p_out_single*100:.3f}%)")
    print(f"    - Dual Anchor Outage  ((-2, +1)): {p_out_dual:.6f} ({p_out_dual*100:.4f}%)")
    print(f"    - Reliability Gain: {p_out_single / p_out_dual:.2f}x")
    
    # Analyze burst erasure distributions
    def get_burst_lengths(series):
        # Find start and end indices of runs of True
        diff = np.diff(np.concatenate(([False], series, [False])).astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return ends - starts
        
    bursts_s = get_burst_lengths(outage_single)
    bursts_d = get_burst_lengths(outage_dual)
    
    hist_single = {f"burst_len_ge_{l}": int(np.sum(bursts_s >= l)) for l in range(1, 11)}
    hist_dual = {f"burst_len_ge_{l}": int(np.sum(bursts_d >= l)) for l in range(1, 11)}
    
    # Cumulative probability of burst lengths
    prob_burst_single = {f"P_burst_ge_{l}": float(np.mean(bursts_s >= l)) for l in range(1, 11)}
    prob_burst_dual = {f"P_burst_ge_{l}": float(np.mean(bursts_d >= l)) if len(bursts_d) > 0 else 0.0 for l in range(1, 11)}
    
    # Fast vectorized Drift Distortion and Lyapunov Virtual Queue
    # D_drift(L) = min(0.42, 0.12 * (1 + 0.45 * L))
    D_base = 0.12
    D_repeat_max = 0.42
    
    def simulate_drift_and_queue(outage_series):
        n = len(outage_series)
        distortions = np.empty(n, dtype=np.float32)
        Z_trace = np.empty(n, dtype=np.float32)
        
        cur_L = 0
        Z = 0.0
        delta_target = 1e-4
        
        for t in range(n):
            if outage_series[t]:
                cur_L += 1
                a = 1.0
            else:
                cur_L = 0
                a = 0.0
            distortions[t] = min(D_repeat_max, D_base * (1.0 + 0.45 * cur_L))
            Z = max(0.0, Z + a - delta_target)
            Z_trace[t] = Z
            
        return float(np.mean(distortions)), float(np.mean(Z_trace)), float(np.max(Z_trace)), Z_trace[::100].tolist()
        
    mean_d_s, mean_z_s, max_z_s, trace_z_s = simulate_drift_and_queue(outage_single)
    mean_d_d, mean_z_d, max_z_d, trace_z_d = simulate_drift_and_queue(outage_dual)
    
    elapsed = time.time() - t_start
    print(f"[*] Simulation completed in {elapsed:.2f} seconds.")
    
    results = {
        "metadata": {
            "total_slots": n_slots,
            "tau_c_ms": tau_c_ms,
            "T_slot_ms": T_slot_ms,
            "rho": float(rho),
            "execution_time_sec": round(elapsed, 2),
            "random_seed": 42
        },
        "outage_summary": {
            "P_out_single_anchor": p_out_single,
            "P_out_dual_anchor": p_out_dual,
            "gain_factor": float(p_out_single / p_out_dual)
        },
        "burst_counts": {
            "single_anchor": hist_single,
            "dual_anchor": hist_dual
        },
        "burst_probabilities": {
            "single_anchor": prob_burst_single,
            "dual_anchor": prob_burst_dual
        },
        "burst_summary": {
            "mean_burst_len_single": float(np.mean(bursts_s)),
            "max_burst_len_single": int(np.max(bursts_s)),
            "mean_burst_len_dual": float(np.mean(bursts_d)) if len(bursts_d) > 0 else 0.0,
            "max_burst_len_dual": int(np.max(bursts_d)) if len(bursts_d) > 0 else 0
        },
        "distortion_and_queue": {
            "mean_distortion_single": mean_d_s,
            "mean_distortion_dual": mean_d_d,
            "distortion_reduction_pct": float((mean_d_s - mean_d_d) / mean_d_s * 100.0),
            "mean_Z_backlog_single": mean_z_s,
            "max_Z_backlog_single": max_z_s,
            "mean_Z_backlog_dual": mean_z_d,
            "max_Z_backlog_dual": max_z_d,
            "queue_stable_dual": bool(mean_z_d < 100.0)
        },
        "sampled_queue_trajectories": {
            "subsample_step": 100,
            "trace_single": trace_z_s[:1000],
            "trace_dual": trace_z_d[:1000]
        }
    }
    
    return results

if __name__ == "__main__":
    res = run_large_scale_temporal_simulation(n_slots=1000000, tau_c_ms=2.5)
    os.makedirs("../data", exist_ok=True)
    out_path = str(DATA_DIR / "exp1_temporal_correlated_1m_results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=4)
    print(f"[+] Saved 1,000,000-slot results to {out_path}")
