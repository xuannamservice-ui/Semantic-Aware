from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
"""
Experiment 2: High-Resolution Hardware Constraints & Laser RIN Sensitivity Sweep
Evaluates 210 operating points across RIN in [-165, -135] dB/Hz and Excess Insertion Loss in [0, 4.5] dB.
"""

import numpy as np
import json, os, csv

def run_high_resolution_hardware_sweep():
    print("[*] Running High-Resolution Hardware RIN & Sorter Insertion Loss Sweep...")
    
    rin_vals = [-165.0, -160.0, -155.0, -150.0, -145.0, -140.0, -135.0]  # dB/Hz
    loss_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]        # dB
    
    target_urllc = 1e-4
    
    configs = [
        {"name": "Square (M=8, Nr=8)", "short_name": "M8_Nr8", "d": 0.5, "base_outage": 0.146},
        {"name": "Oversampled (M=8, Nr=11)", "short_name": "M8_Nr11", "d": 2.0, "base_outage": 2.3e-3},
        {"name": "Throttled (M=6, Nr=11)", "short_name": "M6_Nr11", "d": 3.0, "base_outage": 9.2e-5}
    ]
    
    grid = []
    
    for cfg in configs:
        cname = cfg["name"]
        sname = cfg["short_name"]
        d = cfg["d"]
        base_Pout = cfg["base_outage"]
        
        for rin in rin_vals:
            for loss in loss_vals:
                # 1. Optical excess loss penalty (optical power P scales as 10^(-loss/10), electrical SNR scales as P^2)
                snr_loss_penalty_db = 2.0 * loss
                
                # 2. Laser RIN penalty relative to balanced nominal level (-155 dB/Hz)
                if rin > -155.0:
                    snr_rin_penalty_db = (rin - (-155.0)) * 0.85
                else:
                    snr_rin_penalty_db = (rin - (-155.0)) * 0.25
                    
                total_snr_drop_db = snr_loss_penalty_db + snr_rin_penalty_db
                snr_ratio = 10.0**(-total_snr_drop_db / 10.0)
                
                # High-SNR asymptotic outage scaling: P_out ≈ base_Pout * (snr_ratio)^(-d)
                p_out = min(1.0, base_Pout * (snr_ratio**(-d)))
                
                meets_urllc = (p_out <= target_urllc)
                
                grid.append({
                    "config": cname,
                    "short_name": sname,
                    "diversity_order": d,
                    "laser_rin_db_hz": rin,
                    "excess_loss_db": loss,
                    "total_snr_penalty_db": round(float(total_snr_drop_db), 2),
                    "system_outage": float(f"{p_out:.4e}"),
                    "meets_urllc_1e4": bool(meets_urllc)
                })
                
    # Operating boundaries
    boundaries = {}
    for cfg in configs:
        cname = cfg["name"]
        pts = [g for g in grid if g["config"] == cname and g["meets_urllc_1e4"]]
        if pts:
            max_rin = max(p["laser_rin_db_hz"] for p in pts if p["excess_loss_db"] == 0.0)
            max_loss = max(p["excess_loss_db"] for p in pts if p["laser_rin_db_hz"] == -155.0)
            boundaries[cname] = {
                "urllc_achievable": True,
                "max_tolerable_rin_db_hz": max_rin,
                "max_tolerable_loss_db": max_loss,
                "status": f"Meets 10^-4 target up to {max_loss} dB insertion loss and RIN <= {max_rin} dB/Hz"
            }
        else:
            boundaries[cname] = {
                "urllc_achievable": False,
                "max_tolerable_rin_db_hz": None,
                "max_tolerable_loss_db": None,
                "status": "Requires generative semantic error concealment to conceal outages."
            }
            
    summary = {
        "metadata": {
            "configurations": len(configs),
            "rin_values_tested": len(rin_vals),
            "loss_values_tested": len(loss_vals),
            "total_points": len(grid)
        },
        "operating_boundaries": boundaries,
        "grid_data": grid
    }
    
    # Save CSV as well
    os.makedirs("../data", exist_ok=True)
    csv_path = str(DATA_DIR / "exp2_hardware_rin_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        writer.writeheader()
        writer.writerows(grid)
        
    print(f"[+] Saved CSV data to {csv_path}")
    return summary

if __name__ == "__main__":
    res = run_high_resolution_hardware_sweep()
    json_path = str(DATA_DIR / "exp2_hardware_rin_sensitivity.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=4)
    print(f"[+] Saved JSON summary to {json_path}")
