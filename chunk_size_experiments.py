#!/usr/bin/env python
# coding: utf-8

# In[1]:


import simulator     
import demo_validation  


# In[19]:


import sys
sys.path.append(".")   

from simulator import (
    SimConfig,
    DecodePrioritizingScheduler,
    PrefillPrioritizingScheduler,
    HybridScheduler,
    run_trials_for_scheduler,
    export_results_to_csv,
    print_steady_state_report,
    plot_comprehensive_metrics,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[70]:


#demo2 base config
def make_demo2_base_cfg() -> SimConfig:
    """
    Demo2-like base configuration.
    We will only tweak arrival_time_window and chunked prefill settings.
    """
    cfg = SimConfig(
        num_gpus=1,
        max_sequences=1000,
        max_tokens=2048,

        # Prompt length distribution
        prompt_dist_type="mixture",
        L_values=(32, 128, 512),
        L_probs=(0.6, 0.3, 0.1),

        # Output length distribution
        output_dist_type="mixture",
        B_values=(64, 256),
        B_probs=(0.5, 0.5),

        # Arrival process
        lam=2.0,
        n_queries=3000,
        warmup=0,
        warmup_time=0.0,

        # Service-time model
        service_time_model="deterministic",
        c_ms=10.0,
        a_ms=0.5,
        b0=0,
    )
    return cfg



# In[72]:


#sanity check (no chunking, small tmax)

cfg = make_demo2_base_cfg()
cfg.arrival_time_window = 200.0
cfg.enable_chunked_prefill = False

scheduler = DecodePrioritizingScheduler()

res = run_trials_for_scheduler(
    cfg,
    scheduler,
    n_trials=3,
    seed0=0,
    max_workers=None,
    return_runs=False,
    verbose=True,
)

res["summary"]   # just to see it works


# In[74]:


#run_chunk_setting (decode scheduler, with CIs)
def run_chunk_setting(
    enable_chunked_prefill: bool,
    prefill_chunk_size: int,
    tmax: float = 200.0,
    n_trials: int = 10,
    seed0: int = 0,
    verbose: bool = False,
):
    """
    Run n_trials for one (enable_chunked_prefill, prefill_chunk_size) setting
    using demo2 config as base, with DecodePrioritizingScheduler.

    Returns a dict: one row with mean metrics + 95% CI.
    """

    cfg = make_demo2_base_cfg()
    cfg.arrival_time_window = tmax
    cfg.enable_chunked_prefill = enable_chunked_prefill
    cfg.prefill_chunk_size = prefill_chunk_size

    scheduler = DecodePrioritizingScheduler()

    res = run_trials_for_scheduler(
        cfg,
        scheduler,
        n_trials=n_trials,
        seed0=seed0,
        max_workers=None,   
        return_runs=False,
        verbose=verbose,
    )
    summ = res["summary"]

    row = {
        "scheduler":              summ.get("scheduler", "decode"),
        "enable_chunked_prefill": int(enable_chunked_prefill),
        "prefill_chunk_size":     prefill_chunk_size if enable_chunked_prefill else np.nan,
        "tmax":                   float(tmax),
        "n_trials":               int(summ.get("n_trials", n_trials)),

        # Means
        "throughput_qps": float(summ.get("mean(throughput_qps)", 0.0) or 0.0),
        "mean_ttft_s":    float(summ.get("mean(mean_ttft_s)", 0.0) or 0.0),
        "mean_tbt_s":     float(summ.get("mean(mean_tbt_s)", 0.0) or 0.0),
        "p95_rts_s":      float(summ.get("mean(p95_rts_s)", 0.0) or 0.0),

        # 95% CIs (optional but useful)
        "ci95_low_throughput_qps":  float(summ.get("ci95_low(throughput_qps)", 0.0) or 0.0),
        "ci95_high_throughput_qps": float(summ.get("ci95_high(throughput_qps)", 0.0) or 0.0),
        "ci95_low_mean_ttft_s":     float(summ.get("ci95_low(mean_ttft_s)", 0.0) or 0.0),
        "ci95_high_mean_ttft_s":    float(summ.get("ci95_high(mean_ttft_s)", 0.0) or 0.0),
        "ci95_low_mean_tbt_s":      float(summ.get("ci95_low(mean_tbt_s)", 0.0) or 0.0),
        "ci95_high_mean_tbt_s":     float(summ.get("ci95_high(mean_tbt_s)", 0.0) or 0.0),
        "ci95_low_p95_rts_s":       float(summ.get("ci95_low(p95_rts_s)", 0.0) or 0.0),
        "ci95_high_p95_rts_s":      float(summ.get("ci95_high(p95_rts_s)", 0.0) or 0.0),
    }

    if verbose:
        print(
            f"[decode] chunked={enable_chunked_prefill}, "
            f"chunk={prefill_chunk_size} | "
            f"QPS={row['throughput_qps']:.3f}, "
            f"TTFT={row['mean_ttft_s']:.4f}, "
            f"TBT={row['mean_tbt_s']:.4f}, "
            f"P95_RTS={row['p95_rts_s']:.4f}"
        )

    return row


# In[76]:


#small chunk-size sweep for testing (DECODE scheduler)
def sweep_chunk_sizes(
    chunk_sizes,
    tmax: float = 200.0,
    n_trials: int = 10,
    seed0: int = 0,
):
    """
    Run:
      - baseline (no chunked prefill)
      - chunked prefill for each chunk size

    Returns a DataFrame (one row per configuration).
    """
    rows = []

    # Baseline: no chunking 
    rows.append(
        run_chunk_setting(
            enable_chunked_prefill=False,
            prefill_chunk_size=chunk_sizes[0],
            tmax=tmax,
            n_trials=n_trials,
            seed0=seed0,
        )
    )

    # Chunking ON for each chunk size
    for i, cs in enumerate(chunk_sizes):
        rows.append(
            run_chunk_setting(
                enable_chunked_prefill=True,
                prefill_chunk_size=cs,
                tmax=tmax,
                n_trials=n_trials,
                seed0=seed0 + 1000 * (i + 1),  
            )
        )

    df = pd.DataFrame(rows)
    return df



# In[78]:


#small test sweep
test_chunk_sizes = [16, 64, 256]

df_test = sweep_chunk_sizes(
    test_chunk_sizes,
    tmax=200.0,   
    n_trials=10,
    seed0=0,
)

df_test



# In[92]:


# plotting helpers
import os

def plot_metric_vs_chunk(df, metric, ylabel=None, save_dir=None):
    df_chunk = df[df["enable_chunked_prefill"] == 1].copy()
    df_chunk = df_chunk.sort_values("prefill_chunk_size")

    base_val = df[df["enable_chunked_prefill"] == 0][metric].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(base_val, linestyle="--", label="no chunk", alpha=0.8)

    xs = df_chunk["prefill_chunk_size"].to_numpy(dtype=float)
    ys = df_chunk[metric].to_numpy(dtype=float)
    ax.plot(xs, ys, marker="o", label="chunked prefill")

    lo_col = f"ci95_low_{metric}"
    hi_col = f"ci95_high_{metric}"
    if lo_col in df_chunk.columns and hi_col in df_chunk.columns:
        lows = df_chunk[lo_col].to_numpy(dtype=float)
        highs = df_chunk[hi_col].to_numpy(dtype=float)
        ax.fill_between(xs, lows, highs, alpha=0.2)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("prefill_chunk_size (tokens, log2)")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{metric}_vs_chunk.png"),
                    dpi=200, bbox_inches="tight")

    plt.show()


# In[88]:


# real sweep
real_chunk_sizes = [8, 16, 32, 64, 128, 256, 512]

df_real = sweep_chunk_sizes(
    real_chunk_sizes,
    tmax=5000.0, 
    n_trials=100,
    seed0=0,
)

df_real.to_csv("data/chunk_size_sweep_decode_tmax5000.csv", index=False)
df_real



# In[94]:


# Plots for the REAL experiment (df_real)
for m, label in [
    ("throughput_qps", "Throughput (qps)"),
    ("mean_ttft_s", "Mean TTFT (s)"),
    ("mean_tbt_s", "Mean TBT (s)"),
    ("p95_rts_s", "P95 RTS (s)"),
]:
    plot_metric_vs_chunk(df_real, m, ylabel=label, save_dir="figs")

