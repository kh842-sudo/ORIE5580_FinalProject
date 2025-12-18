# run_hybrid_ratio.py
# run_hybrid_ratio.py
# Direction 3: sweep HybridScheduler decode_seq_ratio and visualize tradeoffs.
# IMPORTANT: Do NOT modify simulator.py or demo_validation.py.

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from simulator import (
    SimConfig,
    HybridScheduler,
    sweep_time_window_for_scheduler,
    export_results_to_csv,
    print_steady_state_report,
    plot_comprehensive_metrics,
)

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def make_hybrid_scheduler(r: float) -> HybridScheduler:
    """
    Create a HybridScheduler with a stable, unique name to avoid collisions/overwrites.
    """
    s = HybridScheduler(r)
    s.name = f"hybrid_r{r:.2f}"
    return s

def parse_ratios(ratios_str: str) -> list:
    """
    Parse ratios from either:
      - "0.1,0.2,0.3"
      - "0.1:0.9:0.1" (start:end:step, inclusive end if close)
    """
    ratios_str = ratios_str.strip()
    if ":" in ratios_str:
        start_s, end_s, step_s = ratios_str.split(":")
        start, end, step = float(start_s), float(end_s), float(step_s)
        vals = []
        x = start
        # include end with tolerance
        while x <= end + 1e-12:
            vals.append(round(x, 10))
            x += step
        return vals
    return [float(x.strip()) for x in ratios_str.split(",") if x.strip()]

def steady_summary_for_ratio(results: dict, r: float) -> dict:
    """
    results[name] = (t_vals, summaries)
    Return the last summary dict for scheduler of ratio r.
    """
    name = f"hybrid_r{r:.2f}"
    t_vals, summaries = results[name]
    return summaries[-1]

def plot_steady_metrics_vs_ratio(results: dict, ratios: list, out_png: str, title: str) -> None:
    """
    Create a 2x2 plot at steady state (last time-window point) vs ratio.
    """
    ensure_dir(os.path.dirname(out_png))

    metric_keys = [
        ("mean(mean_ttft_s)", "Mean TTFT (s)"),
        ("mean(mean_tbt_s)", "Mean TBT (s)"),
        ("mean(p95_rts_s)", "P95 RTS (s)"),
        ("mean(throughput_qps)", "Throughput (qps)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (key, ylabel) in zip(axes, metric_keys):
        y = []
        for r in ratios:
            s = steady_summary_for_ratio(results, r)
            val = s.get(key, np.nan)
            try:
                y.append(float(val))
            except Exception:
                y.append(np.nan)

        ax.plot(ratios, y, marker="o")
        ax.set_xlabel("decode_seq_ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_png}")

# -----------------------------
# Demo configs (match your earlier driver style)
# -----------------------------
def make_demo2_cfg(lam: float) -> SimConfig:
    """
    Demo2-style config (mixture L/B, no chunked prefill).
    """
    return SimConfig(
        num_gpus=1,
        max_sequences=1000,
        max_tokens=2048,

        prompt_dist_type="mixture",
        L_values=(32, 128, 512),
        L_probs=(0.6, 0.3, 0.1),

        output_dist_type="mixture",
        B_values=(64, 256),
        B_probs=(0.5, 0.5),

        lam=lam,

        # if your SimConfig uses different names, keep these defaults harmless
        n_queries=3000,
        warmup=0,
        warmup_time=0.0,

        service_time_model="deterministic",
        c_ms=10.0,
        a_ms=0.5,
        b0=0,

        enable_chunked_prefill=False,
        prefill_chunk_size=64,
    )

def make_demo3_cfg(lam: float) -> SimConfig:
    """
    Demo3-style config (lognormal prompt, geometric output, chunked prefill enabled).
    """
    return SimConfig(
        num_gpus=1,
        max_sequences=1000,
        max_tokens=2048,

        lam=lam,

        prompt_dist_type="lognormal",
        L_lognorm_mu=3.2764,
        L_lognorm_sigma=1.5201,

        output_dist_type="geometric",
        B_geom_p=0.00560,
        B_max=1024,

        c_ms=45.5,
        a_ms=0.30,

        enable_chunked_prefill=True,
        prefill_chunk_size=64,
    )

# -----------------------------
# Core runner
# -----------------------------
def run_sweep_for_cfg(
    cfg: SimConfig,
    ratios: list,
    t_max: float,
    n_points: int,
    n_trials: int,
    tag: str,
    out_data_dir: str,
    out_fig_dir: str,
) -> None:
    ensure_dir(out_data_dir)
    ensure_dir(out_fig_dir)

    results = {}
    for r in ratios:
        sched = make_hybrid_scheduler(r)
        t_vals, summaries = sweep_time_window_for_scheduler(
            cfg,
            sched,
            t_max=t_max,
            n_points=n_points,
            n_trials=n_trials,
            seed0=0,
            max_workers=None,
            verbose_each=False,
        )
        results[sched.name] = (t_vals, summaries)

    # 1) Export time-series metrics
    ts_csv = os.path.join(out_data_dir, f"{tag}_time_series.csv")
    export_results_to_csv(results, filename=ts_csv)
    print(f"[Saved CSV] {ts_csv}")

    # 2) Print + export steady-state report
    steady_csv = os.path.join(out_data_dir, f"{tag}_steady_state.csv")
    print_steady_state_report(results, csv_path=steady_csv)
    print(f"[Saved CSV] {steady_csv}")

    # 3) Dashboard time-series plots (TTFT/TBT/RTS/Throughput vs t_max)
    # NOTE: plot_comprehensive_metrics writes to figs/ by default inside simulator
    #       but also respects filename arg; we pass a file name only.
    dashboard_png = dashboard_png = f"{tag}_dashboard.png"
    plot_comprehensive_metrics(results, filename=dashboard_png, demo_name=tag)
    print(f"[Saved] {dashboard_png}")

    # 4) Steady-state tradeoff plots (metrics vs ratio)
    steady_vs_ratio_png = os.path.join(out_fig_dir, f"{tag}_steady_vs_ratio.png")
    plot_steady_metrics_vs_ratio(
        results,
        ratios,
        out_png=steady_vs_ratio_png,
        title=f"{tag}: Steady-state Metrics vs decode_seq_ratio (t_max={t_max})",
    )

def main():
    parser = argparse.ArgumentParser(description="Direction 3: Hybrid decode ratio sweep (demo2/demo3).")
    parser.add_argument("--scenario", type=str, default="demo3",
                        choices=["demo2", "demo3", "both"],
                        help="Which scenario config to run.")
    parser.add_argument("--tmax", type=float, default=1000.0,
                        help="arrival_time_window t_max (e.g., 200/1000/5000).")
    parser.add_argument("--npoints", type=int, default=50,
                        help="Number of time-window sweep points.")
    parser.add_argument("--ntrials", type=int, default=50,
                        help="Trials per time point (parallelizable).")
    parser.add_argument("--lam", type=float, default=2.0,
                        help="Arrival rate lambda.")
    parser.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help='Ratios list "0.1,0.2,..." or range "0.1:0.9:0.1".')
    parser.add_argument("--out_data", type=str, default="data", help="Output folder for CSVs.")
    parser.add_argument("--out_figs", type=str, default="figs", help="Output folder for figures.")

    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    # keep ratios within (0,1) just in case
    ratios = [r for r in ratios if 0.0 < r < 1.0]
    if len(ratios) == 0:
        raise ValueError("No valid ratios parsed. Example: --ratios 0.1:0.9:0.1")

    if args.scenario in ("demo2", "both"):
        cfg2 = make_demo2_cfg(lam=args.lam)
        tag2 = f"hybrid_ratio_demo2_tmax{int(args.tmax)}"
        run_sweep_for_cfg(
            cfg=cfg2,
            ratios=ratios,
            t_max=args.tmax,
            n_points=args.npoints,
            n_trials=args.ntrials,
            tag=tag2,
            out_data_dir=args.out_data,
            out_fig_dir=args.out_figs,
        )

    if args.scenario in ("demo3", "both"):
        cfg3 = make_demo3_cfg(lam=args.lam)
        tag3 = f"hybrid_ratio_demo3_tmax{int(args.tmax)}"
        run_sweep_for_cfg(
            cfg=cfg3,
            ratios=ratios,
            t_max=args.tmax,
            n_points=args.npoints,
            n_trials=args.ntrials,
            tag=tag3,
            out_data_dir=args.out_data,
            out_fig_dir=args.out_figs,
        )

    print("\nDone. Outputs saved under:")
    print(f"  - {args.out_data}/")
    print(f"  - {args.out_figs}/")

if __name__ == "__main__":
    main()

