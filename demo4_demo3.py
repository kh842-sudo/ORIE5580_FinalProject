"""
Demo 4: Demo 3 实验（独立运行）

配置：
- Decode-First:  λ ∈ [0.001, 0.006, 0.011, 0.016, 0.021, 0.026, 0.031, 0.036, 0.041] (极低负载, 9点)
- Prefill-First: λ ∈ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] (7点)
- Hybrid:        λ ∈ [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0] (高负载, 9点)

参数：
- t_max = 200s
- n_points = 40
- n_trials = 80
- decode step = 0.005, 其他 step = 0.5

预计时间: ~7.5 小时 (总计25点)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import copy
from typing import List, Dict, Any, Tuple

# ==================== 导入 simulator.py ====================
sys.path.append('/mnt/user-data/uploads')
from simulator import (
    SimConfig,
    DecodePrioritizingScheduler,
    PrefillPrioritizingScheduler,
    HybridScheduler,
    sweep_time_window_for_scheduler,
    save_figure,
)

os.makedirs('data', exist_ok=True)
os.makedirs('figs', exist_ok=True)


# ==================== 不稳定性检测函数 ====================

def detect_instability_by_block_monotonicity(
    t_values: np.ndarray,
    summaries: List[Dict[str, Any]],
    metric: str = 'mean_rts_s',
    n_blocks: int = 10,
    significance_level: float = 0.05,
    min_relative_increase: float = 0.10,
) -> Tuple[bool, Dict[str, Any]]:
    """检测延迟是否不收敛（系统不稳定）"""
    
    n_points = len(summaries)
    if n_points < n_blocks:
        return False, {"error": "数据点不足"}
    
    metric_key = f"mean({metric})"
    values = np.array([s[metric_key] for s in summaries])
    
    # 分块
    block_size = n_points // n_blocks
    block_means = []
    block_times = []
    
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else n_points
        block_mean = np.mean(values[start:end])
        block_time = np.mean(t_values[start:end])
        block_means.append(block_mean)
        block_times.append(block_time)
    
    block_means = np.array(block_means)
    block_times = np.array(block_times)
    
    # Mann-Kendall 检验
    n = len(block_means)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(block_means[j] - block_means[i])
    
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p_value = 1 - stats.norm.cdf(z)
    has_significant_upward_trend = (p_value < significance_level) and (z > 0)
    
    # 相对增长
    relative_increase = (block_means[-1] - block_means[0]) / (block_means[0] + 1e-9)
    has_large_increase = relative_increase > min_relative_increase
    
    # 线性回归
    slope, intercept, r_value, p_value_regression, std_err = stats.linregress(
        block_times, block_means
    )
    has_positive_slope = slope > 0 and p_value_regression < significance_level
    
    # 判断不稳定
    is_unstable = has_significant_upward_trend and has_large_increase
    
    diagnostics = {
        'metric': metric,
        'n_blocks': n_blocks,
        'block_means': block_means.tolist(),
        'block_times': block_times.tolist(),
        'mk_statistic': float(s),
        'mk_z_score': float(z),
        'mk_p_value': float(p_value),
        'has_significant_trend': has_significant_upward_trend,
        'relative_increase': float(relative_increase),
        'has_large_increase': has_large_increase,
        'linear_slope': float(slope),
        'linear_p_value': float(p_value_regression),
        'has_positive_slope': has_positive_slope,
        'is_unstable': is_unstable,
    }
    
    return is_unstable, diagnostics


def analyze_congestion_pattern(
    t_values: np.ndarray,
    summaries: List[Dict[str, Any]],
    n_blocks: int = 10,
) -> Dict[str, Any]:
    """研究拥塞形态：TTFT、TBT、RTS 哪个优先爆炸"""
    
    metrics_to_check = {
        'TTFT': 'mean_ttft_s',
        'TBT': 'mean_tbt_s',
        'RTS': 'mean_rts_s',
    }
    
    results = {}
    increases = {}
    unstable_flags = {}
    
    for name, metric in metrics_to_check.items():
        is_unstable, diag = detect_instability_by_block_monotonicity(
            t_values, summaries, metric=metric, n_blocks=n_blocks
        )
        results[name] = diag
        increases[name] = diag['relative_increase']
        unstable_flags[name] = is_unstable
    
    first_unstable = None
    if any(unstable_flags.values()):
        for metric in ['TTFT', 'TBT', 'RTS']:
            if unstable_flags[metric]:
                if first_unstable is None or increases[metric] > increases[first_unstable]:
                    first_unstable = metric
    
    return {
        'first_to_explode': first_unstable if first_unstable else 'None',
        'max_increase_metric': max(increases, key=increases.get),
        'ttft_increase': increases['TTFT'],
        'tbt_increase': increases['TBT'],
        'rts_increase': increases['RTS'],
        'ttft_unstable': unstable_flags['TTFT'],
        'tbt_unstable': unstable_flags['TBT'],
        'rts_unstable': unstable_flags['RTS'],
        'detailed_results': results,
    }


# ==================== Lambda 扫描 ====================

def lambda_sweep(
    base_config: SimConfig,
    scheduler,
    lambda_range: Tuple[float, float, float],
    t_max: float,
    n_points: int,
    n_trials: int,
    experiment_name: str,
) -> Dict[float, Dict[str, Any]]:
    """Lambda 扫描"""
    
    print("\n" + "="*80)
    print(f"开始实验: {experiment_name}")
    print(f"Scheduler: {scheduler.name}")
    print("="*80)
    
    lambda_min, lambda_max, lambda_step = lambda_range
    lambda_values = np.arange(lambda_min, lambda_max + lambda_step/2, lambda_step).tolist()
    
    print(f"\nLambda 范围: {[f'{l:.3f}' for l in lambda_values]}")
    print(f"参数: t_max={t_max}s, n_points={n_points}, n_trials={n_trials}\n")
    
    results = {}
    
    for i, lam in enumerate(lambda_values):
        print(f"[{i+1}/{len(lambda_values)}] λ = {lam:.3f} queries/s", end=" ... ")
        
        cfg = copy.deepcopy(base_config)
        cfg.lam = lam
        
        t_vals, summs = sweep_time_window_for_scheduler(
            cfg, scheduler,
            t_max=t_max,
            n_points=n_points,
            n_trials=n_trials,
            seed0=int(lam * 10000),
            max_workers=None,
            verbose_each=False,
        )
        
        is_unstable, diag = detect_instability_by_block_monotonicity(
            t_vals, summs, metric='mean_rts_s', n_blocks=10
        )
        
        congestion = analyze_congestion_pattern(t_vals, summs, n_blocks=10)
        
        results[lam] = {
            't_values': t_vals,
            'summaries': summs,
            'is_unstable': is_unstable,
            'diagnostics': diag,
            'congestion': congestion,
        }
        
        last = summs[-1]
        status = "✗ UNSTABLE" if is_unstable else "✓ STABLE"
        print(f"{status} | QPS={last['mean(throughput_qps)']:.3f} | RTS={last['mean(mean_rts_s)']:.2f}s")
    
    return results


# ==================== 数据保存 ====================

def save_results(results, experiment_name):
    """保存结果"""
    rows = []
    for lam in sorted(results.keys()):
        data = results[lam]
        last = data['summaries'][-1]
        n_arr = last['mean(arried_queries)']
        n_com = last['mean(completed_queries)']
        row = {
            'Lambda': lam,
            'Stable': 'Yes' if not data['is_unstable'] else 'No',
            'QPS': last['mean(throughput_qps)'],
            'Arrived': n_arr,
            'Done': n_com,
            'Completion_Rate': n_com / n_arr if n_arr > 0 else 0,
            'TTFT_Avg': last['mean(mean_ttft_s)'],
            'TBT_Avg': last['mean(mean_tbt_s)'],
            'RTS_Avg': last['mean(mean_rts_s)'],
            'RTS_P95': last['mean(p95_rts_s)'],
        }
        rows.append(row)
    
    filename = f"data/{experiment_name}_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"  ✓ 结果数据: {filename}")


def find_lambda_star(results):
    """找到稳定性边界 λ*"""
    lambda_vals = sorted(results.keys())
    stable_flags = [not results[l]['is_unstable'] for l in lambda_vals]
    
    stable_lambdas = [l for l, s in zip(lambda_vals, stable_flags) if s]
    unstable_lambdas = [l for l, s in zip(lambda_vals, stable_flags) if not s]
    
    if not stable_lambdas:
        return None, None
    
    if not unstable_lambdas:
        return max(stable_lambdas), 0.0
    
    lambda_star_lower = max(stable_lambdas)
    lambda_star_upper = min(unstable_lambdas)
    
    if lambda_star_lower > lambda_star_upper:
        lambda_star = lambda_star_upper
        confidence_width = 0.0
    else:
        lambda_star = (lambda_star_lower + lambda_star_upper) / 2
        confidence_width = lambda_star_upper - lambda_star_lower
    
    return lambda_star, confidence_width


# ==================== Demo 3 配置 ====================

def get_demo3_base_config():
    """Demo 3 基础配置"""
    return SimConfig(
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
        
        enable_chunked_prefill=True,
    )


# Demo 3 的 Lambda 范围
DEMO3_LAMBDA_RANGES = {
    'decode_prioritizing': (0.001, 0.04, 0.005),   # [0.001, 0.006, 0.011, ..., 0.036, 0.041] - 9点
    'prefill_prioritizing': (0.5, 3.5, 0.5),       # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] - 7点
    'hybrid': (8.0, 12.0, 0.5),                    # [8.0, 8.5, 9.0, ..., 11.5, 12.0] - 9点
}


# ==================== 主程序 ====================

if __name__ == "__main__":
    
    # 参数设置
    T_MAX = 200.0
    N_POINTS = 40
    N_TRIALS = 80
    
    print("\n" + "="*80)
    print("Demo 3: LogNormal/Geometric 分布 + Chunked-Prefill")
    print("="*80)
    
    print("\n参数设置:")
    print(f"  t_max: {T_MAX}s")
    print(f"  n_points: {N_POINTS}")
    print(f"  n_trials: {N_TRIALS}")
    
    print("\n实验计划:")
    print("  - Decode-First:  λ ∈ [0.001, 0.006, 0.011, ..., 0.036, 0.041] (step=0.005, 9点)")
    print("  - Prefill-First: λ ∈ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] (step=0.5, 7点)")
    print("  - Hybrid:        λ ∈ [8.0, 8.5, 9.0, ..., 11.5, 12.0] (step=0.5, 9点)")
    
    demo3_points = 9 + 7 + 9  # decode: 9点, prefill: 7点, hybrid: 9点
    simulation_speedup = 15
    estimated_hours = demo3_points * N_TRIALS * T_MAX / 3600 / simulation_speedup
    
    print(f"\n预计总时间: ~{estimated_hours:.1f} 小时")
    print(f"  (总计 {demo3_points} 个 λ 点 × {N_TRIALS} trials × {T_MAX:.0f}s)")
    
    # ==================== 运行实验 ====================
    
    all_results = {}
    base_config = get_demo3_base_config()
    
    schedulers = [
        ('decode_prioritizing', DecodePrioritizingScheduler()),
        ('prefill_prioritizing', PrefillPrioritizingScheduler()),
        ('hybrid', HybridScheduler(0.5)),
    ]
    
    for sched_name, scheduler in schedulers:
        lambda_range = DEMO3_LAMBDA_RANGES[sched_name]
        
        results = lambda_sweep(
            base_config=base_config,
            scheduler=scheduler,
            lambda_range=lambda_range,
            t_max=T_MAX,
            n_points=N_POINTS,
            n_trials=N_TRIALS,
            experiment_name=f"Demo3_{scheduler.name}",
        )
        
        # 找到 λ*
        lambda_star, conf_width = find_lambda_star(results)
        
        print("\n" + "="*80)
        print(f"实验完成: Demo3_{scheduler.name}")
        if lambda_star:
            if conf_width > 0:
                print(f"稳定性边界: λ* = {lambda_star:.3f} ± {conf_width/2:.3f} queries/s")
            else:
                print(f"稳定性边界: λ* > {lambda_star:.3f} queries/s")
        else:
            print("未找到稳定性边界（所有λ都不稳定）")
        print("="*80)
        
        # 保存数据
        save_results(results, f"Demo3_{scheduler.name}")
        
        all_results[f"Demo3_{scheduler.name}"] = {
            'results': results,
            'lambda_star': lambda_star,
            'confidence_width': conf_width,
        }
    
    # ==================== 打印总结 ====================
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    
    print(f"\n{'实验名称':<35} | {'λ*':<10} | {'置信区间':<15}")
    print("-"*80)
    
    for exp_name in sorted(all_results.keys()):
        data = all_results[exp_name]
        lambda_star = data['lambda_star']
        conf_width = data['confidence_width']
        
        if lambda_star:
            if conf_width > 0:
                print(f"{exp_name:<35} | {lambda_star:>8.3f}  | ±{conf_width/2:>6.3f}")
            else:
                print(f"{exp_name:<35} | >{lambda_star:>7.3f}  | {'N/A':>12}")
        else:
            print(f"{exp_name:<35} | {'N/A':>8}  | {'N/A':>12}")
    
    print("="*80)
    print("\n✓ 所有结果已保存:")
    print("  - 数据: data/Demo3_*_results.csv")
    
    print("\n" + "="*80)
    print("Demo 3 实验完成！")
    print("="*80)
