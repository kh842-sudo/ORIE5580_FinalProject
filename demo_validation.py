from simulator import (
    SimConfig, 
    DecodePrioritizingScheduler, 
    PrefillPrioritizingScheduler,
    HybridScheduler,
    run_trials_for_scheduler,
    sweep_time_window_for_scheduler,
    export_results_to_csv,
    print_steady_state_report,
    plot_comprehensive_metrics
)

def demo2_tmax():
    cfg = SimConfig(
        num_gpus=1,
        max_sequences=1000,
        max_tokens=2048,

        prompt_dist_type='mixture',
        L_values=(32, 128, 512),
        L_probs=(0.6, 0.3, 0.1),

        output_dist_type='mixture',
        B_values=(64, 256),
        B_probs=(0.5, 0.5),

        lam=2.0,
        n_queries=3000,
        warmup=0,
        warmup_time=0.0,
        service_time_model='deterministic',
        c_ms=10.0,
        a_ms=0.5,
        b0=0,

    )

    schedulers = [
        DecodePrioritizingScheduler(),
        PrefillPrioritizingScheduler(),
        HybridScheduler(0.5),
    ]

    t_max = 5000.0
    n_points = 100
    n_trials = 100

    results = {}
    for s in schedulers:
        t_values, summaries = sweep_time_window_for_scheduler(
            cfg, s,
            t_max=t_max,
            n_points=n_points,
            n_trials=n_trials,
            seed0=0,
            max_workers=None,
            verbose_each=False,
        )
        results[s.name] = (t_values, summaries)

    export_results_to_csv(results, filename="data/demo2_time_series_results.csv")
    print_steady_state_report(results, csv_path="data/demo2_final_results.csv")
    plot_comprehensive_metrics(
        results, 
        filename="demo2_dashboard.png",
        demo_name="Scheduler Comparison under Simplified Scenario"
    )

def demo3_tmax():
    cfg = SimConfig(
        num_gpus=1,
        lam=2.0,
        max_sequences=1000,
        max_tokens=2048,
        prompt_dist_type='lognormal',
        L_lognorm_mu=3.2764,
        L_lognorm_sigma=1.5201,
        B_geom_p=0.00560,
        c_ms=45.5,
        a_ms=0.30,
        b0 = 64,
        enable_chunked_prefill=True,

    )

    schedulers = [
        DecodePrioritizingScheduler(),
        PrefillPrioritizingScheduler(),
        HybridScheduler(0.5),
    ]

    t_max = 5000.0
    n_points = 100
    n_trials = 100

    results = {}
    for s in schedulers:
        t_values, summaries = sweep_time_window_for_scheduler(
            cfg, s,
            t_max=t_max,
            n_points=n_points,
            n_trials=n_trials,
            seed0=0,
            max_workers=None,
            verbose_each=False,
        )
        results[s.name] = (t_values, summaries)

    export_results_to_csv(results, filename="data/demo3_time_series_results.csv")
    print_steady_state_report(results, csv_path="data/demo3_final_results.csv")
    plot_comprehensive_metrics(
        results, 
        filename="demo3_dashboard.png",
        demo_name="Scheduler Comparison under Realistic Scenario"
    )


# ## Validation: M/M/1
def validate_with_queueing_theory():
    val_cfg = SimConfig(
        num_gpus=1,
        lam=1.0,
        max_sequences=1,
        max_tokens=10000,

        prompt_dist_type='mixture',
        L_values=(1,),
        L_probs=(1.0,),

        # prefill-only
        output_dist_type='mixture',
        B_values=(0,),
        B_probs=(1.0,),

        c_ms=10.0,
        a_ms=0.0,
        b0=0,
        service_time_model='exp_mm1',

        n_queries=10000,
        warmup=1000,
        warmup_time=100.0,
        seed=42,
    )

    scheduler = PrefillPrioritizingScheduler()  # or DecodePrioritizing, same

    out = run_trials_for_scheduler(
        val_cfg, scheduler,
        n_trials=1,
        verbose=False,
        return_runs=False
    )
    summ = out["summary"]

    # Theory
    mean_service_ms = val_cfg.c_ms + val_cfg.a_ms * max(0, 1 - val_cfg.b0)
    mu = 1000.0 / mean_service_ms
    rho = val_cfg.lam / mu
    theory_response = 1.0 / (mu * (1 - rho))

    print(f"\nTheoretical:")
    print(f"Utilization ρ = λ/μ = {val_cfg.lam}/{mu:.2f} = {rho:.4f}")
    print(f"Mean Response Time E[T] = {theory_response:.4f} s")

    print(f"\nSimulation Results (Monte Carlo over trials):")
    print("Mean:", f"{summ["mean(mean_rts_s)"]:.4f}")
    print("95% CI: (" + f"{summ["ci95_low(mean_rts_s)"]:.4f}" + ", " + f"{summ["ci95_high(mean_rts_s)"]:.4f}" + ")")

    error = abs(summ['mean(mean_rts_s)'] - theory_response) / theory_response * 100
    print(f"Relative Error = {error:.2f}%")

    return summ

if __name__ == "__main__":
    # scheduler comparison
    demo2_tmax()
    demo3_tmax()
    
    # validation with queueing theory
    validate_with_queueing_theory()