# LLM Serving System Simulator README

A discrete-event simulator for analyzing scheduling policies in LLM serving systems, with support for chunked prefill, hybrid scheduling, and comprehensive performance metrics.



## Overview

This simulator models token-level processing in LLM inference systems, supporting:
- **Three scheduling policies**: Decode-prioritizing, Prefill-prioritizing, and Hybrid
- **Chunked prefill**: Enables interleaving prefill and decode phases
- **Configurable workloads**: Mixture or log-normal prompt distributions, geometric or mixture output distributions
- **Monte Carlo experiments**: Parallel trial execution with confidence intervals



## Key Configuration Parameters

### SimConfig Fields

```python
SimConfig(
    # System resources
    num_gpus=1,
    max_sequences=1000,    # max concurrent sequences
    max_tokens=2048,       # token budget per iteration
    
    # Workload
    lam=2.0,              # arrival rate (queries/sec)
    prompt_dist_type='mixture',  # or 'lognormal'
    output_dist_type='mixture',  # or 'geometric'
    
    # Service time model
    c_ms=10.0,            # fixed overhead (ms)
    a_ms=0.5,             # per-token cost (ms)
    b0=0,                 # KV cache threshold
    
    # Chunked prefill
    enable_chunked_prefill=False,
    prefill_chunk_size=64,
    
    # Simulation scale
    n_queries=3000,
    arrival_time_window=5000.0,
)
```



## Schedulers

1. **DecodePrioritizingScheduler**: Always schedules decode before prefill
2. **PrefillPrioritizingScheduler**: Always schedules prefill before decode
3. **HybridScheduler(decode_seq_ratio)**: Allocates `decode_seq_ratio` fraction of sequences to decode



## Metrics

All experiments report:
- **Throughput**: Queries per second
- **TTFT**: Mean time to first token (latency)
- **TBT**: Mean time between tokens
- **P95 RTS**: 95th percentile response time
- **Confidence intervals**: 95% CI from Monte Carlo trials



## Simulation

### Time Window Sweep

```python
from simulator import sweep_time_window_for_scheduler

t_values, summaries = sweep_time_window_for_scheduler(
    cfg, scheduler,
    t_max=5000.0,
    n_points=100,
    n_trials=50,
)
```

### Parallel Execution

The simulator automatically uses multiprocessing for trials. Control with:

```python
results = run_trials_for_scheduler(
    cfg, scheduler,
    n_trials=100,
    max_workers=8,  # or None for all CPUs
)
```



## Project Structure

```
.
├── main report.ipynb            # Main Report
├── simulator.py                 # Core simulator engine
├── demo_validation.py           # Validation and demos
├── chunk_size_experiments.py    # Chunking experiments
├── run_hybrid_ratio.py          # Hybrid ratio sweep
├── interactive_visualization.py # Annimation code used in the technical appendix
├── data/                        # CSV outputs
├── figs/                        # Figure outputs
```



## Notes

- First run may take time due to Monte Carlo sampling
- Increase `n_trials` for more precise confidence intervals
- Use smaller `tmax` and `n_points` for faster iteration
- Results are reproducible via `seed` parameter



## Reproducing Results

### Validation and Basic Demos

Run M/M/1 queueing theory validation and scheduler comparisons:
* Demo1: Validation with M/M/1 queueing theory
* Demo 2: Simplified scenario (mixture distributions, no chunking)
* Demo 3: Realistic scenario (log-normal + geometric, with chunking)

```bash
python demo_validation.py
```

**Outputs:**
- `data/demo2_time_series_results.csv` - Simplified scenario metrics over time
- `data/demo3_time_series_results.csv` - Realistic scenario metrics over time
- `demo2_dashboard.png` - Visualization for Demo 2
- `demo3_dashboard.png` - Visualization for Demo 3
- Console output with M/M/1 validation results



### Stability Detection Experiments

#### Demo 2 — Stability Boundary Detection (Low Variability)

Run stability sweeps for Demo 2 with fixed-length distributions.

##### Run experiment

```bash
python demo4_demo2.py
```

This script automatically:

- Sweeps arrival rate λ
- Runs Monte Carlo trials per test point
- Applies RTS-based stability detection
- Computes stability boundaries for each scheduler

##### Outputs

- `data/demo2_stability_results.csv`
  Stability classification and metrics per λ
- `data/demo2_capacity_summary.csv`
  Final stability boundary (λ*) per scheduler
- `figs/demo2_scheduler_comparison.png`
  Scheduler comparison plots

##### Optional: Visualization

To regenerate all figures used in Demo 2 analysis:

```bash
jupyter notebook demo2_visualization.ipynb
```

Outputs:

- `figs/demo2_time_series.png`
- `figs/demo2_stability_vs_lambda.png`
- `figs/demo2_capacity_bar.png`

------

#### Demo 3 — Stability Boundary Detection (High Variability)

Demo 3 evaluates scheduler robustness under long-tailed workloads.

##### Case 1: Chunked-Prefill Enabled (b0 = 64)

```bash
python demo4_demo3.py
```

This run:

- Uses LogNormal prompt distribution
- Enables Chunked-Prefill (b0 = 64)
- Sweeps λ and detects stability boundaries

Outputs:

- `data/demo3_b0_64_stability_results.csv`
- `data/demo3_b0_64_capacity_summary.csv`
- `figs/demo3_b0_64_scheduler_comparison.png`

##### Case 2: Chunked-Prefill Disabled (b0 = 0)

```bash
python demo4_demo3_b0_0.py
```

This run isolates the effect of chunking by disabling it entirely.

Outputs:

- `data/demo3_b0_0_stability_results.csv`
- `data/demo3_b0_0_capacity_summary.csv`

##### Optional: Visualization

To regenerate Demo 3 plots:

```bash
jupyter notebook demo3_visualization.ipynb
```

Outputs:

- `figs/demo3_time_series.png`
- `figs/demo3_capacity_comparison.png`
- `figs/demo3_chunking_impact.png`

------

#### Recommended Reproduction Order

1. **Run Demo 2** to verify baseline stability behavior
2. **Run Demo 3 (b0 = 64)** to observe long-tailed workload effects
3. **Run Demo 3 (b0 = 0)** to isolate chunked-prefill impact
4. (Optional) Run visualization notebooks to regenerate figures

------

#### Minimal Sanity Check (Quick Run)

If you only want to verify the pipeline runs correctly:

```bash
python demo4_demo2.py
```



### Chunked Prefill Experiments

Explore the impact of different chunk sizes (Jupyter notebook or script):

```bash
# Run as notebook
jupyter notebook chunk_size_experiments.py

# Or execute as script
python chunk_size_experiments.py
```

**Key experiments:**
- `sweep_chunk_sizes()`: Test baseline vs. various chunk sizes
- Chunk sizes tested: 8, 16, 32, 64, 128, 256, 512 tokens
- Metrics: Throughput, TTFT (Time to First Token), TBT (Time Between Tokens), P95 response time

**Outputs:**
- `data/chunk_size_sweep_decode_tmax5000.csv` - Full results with confidence intervals
- `figs/*.png` - Individual metric plots vs chunk size



### Hybrid Scheduler Ratio Sweep

Systematically explore hybrid scheduler tradeoffs:

```bash
# Demo 3 scenario with default ratios
python run_hybrid_ratio.py --scenario demo3 --tmax 1000 --ntrials 50

# Demo 2 scenario with custom ratio range
python run_hybrid_ratio.py --scenario demo2 --tmax 5000 --ntrials 100 \
  --ratios "0.1:0.9:0.1" --lam 2.5

# Both scenarios
python run_hybrid_ratio.py --scenario both --tmax 1000 --ntrials 50
```

**Arguments:**
- `--scenario`: `demo2`, `demo3`, or `both`
- `--tmax`: Time window for arrival process (default: 1000.0)
- `--npoints`: Number of time sweep points (default: 50)
- `--ntrials`: Monte Carlo trials per point (default: 50)
- `--lam`: Arrival rate λ (default: 2.0)
- `--ratios`: Either comma-separated `"0.1,0.3,0.5"` or range `"0.1:0.9:0.1"` (start:end:step)

**Outputs:**
- `data/hybrid_ratio_demo*_time_series.csv` - Time-series metrics
- `data/hybrid_ratio_demo*_steady_state.csv` - Final steady-state summary
- `hybrid_ratio_demo*_dashboard.png` - Time-series dashboard
- `figs/hybrid_ratio_demo*_steady_vs_ratio.png` - Metrics vs ratio plot