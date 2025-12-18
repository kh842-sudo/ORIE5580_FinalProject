from __future__ import annotations
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
import heapq
import math
import multiprocessing as mp
import random
from dataclasses import dataclass, field
import numpy as np
import csv
import matplotlib.pyplot as plt

@dataclass
class SimConfig:
    # GPU
    num_gpus: int = 1

    # Batching limit
    max_sequences: int = 32
    max_tokens: int = 2048

    # Arrival
    lam: float = 2.0  # λ (queries/sec)

    # Prompt length dist
    prompt_dist_type: str = 'mixture'  # 'lognormal' or 'mixture'

    # LogNormal
    L_lognorm_mu: float = 3.2764
    L_lognorm_sigma: float = 1.5201

    # Mixture
    L_values: Tuple[int, ...] = (32, 128, 512)
    L_probs: Tuple[float, ...] = (0.6, 0.3, 0.1)

    # Output budget dist
    output_dist_type: str = 'geometric'
    # Geometric
    B_geom_p: float = 0.00560
    B_max: int = 1024

    # Mixture
    B_values: Tuple[int, ...] = (64, 256)
    B_probs: Tuple[float, ...] = (0.5, 0.5)

    # Chunked-Prefill
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int = 64

    # Deterministic service time model
    c_ms: float = 45.5          # fixed cost
    a_ms: float = 0.30          # marginal cost
    b0: int = 64  # b0

    service_time_model: str = 'deterministic'

    # simulation scale
    n_queries: int = 10000
    warmup: int= 0
    warmup_time: int = 0
    seed: int = 0

    arrival_time_window: float = 5000.0

def sample_prompt_length(cfg: SimConfig, rng: random.Random) -> int:
    """
    Sampling prompt length L_i

    Supported:
    - 'lognormal': LogNormal distribution
    - 'mixture': Mixed discrete distribution
    """
    if cfg.prompt_dist_type == 'lognormal':
        val = int(np.random.lognormal(cfg.L_lognorm_mu, cfg.L_lognorm_sigma))
        return max(1, val)

    elif cfg.prompt_dist_type == 'mixture':
        return rng.choices(cfg.L_values, weights=cfg.L_probs, k=1)[0]

    else:
        raise ValueError(f"Unknown prompt_dist_type: {cfg.prompt_dist_type}")


def sample_output_budget(cfg: SimConfig, rng: random.Random, L: int = None) -> int:
    """
    Sampling output budget B_i
    Use Geometric or Mixture distribution
    """
    if cfg.output_dist_type == 'mixture':
        return rng.choices(cfg.B_values, weights=cfg.B_probs, k=1)[0]

    elif cfg.output_dist_type == 'geometric':
        b = int(np.random.geometric(cfg.B_geom_p))
        return min(b, cfg.B_max)

    else:
        raise ValueError(f"Unknown output_dist_type: {cfg.output_dist_type}")


@dataclass
class Query:
    """Complete status of a single user request"""
    qid: int
    arrival_t: float
    L: int  # prompt token 总数
    B: int  # 输出 token 预算

    # Prefill 状态
    prefill_left: int = field(init=False)
    prefill_chunks: List[int] = field(default_factory=list, init=False)
    current_chunk_idx: int = 0

    # Decode 状态
    decode_left: int = field(init=False)

    # 时间戳
    prefill_done_t: Optional[float] = None
    done_t: Optional[float] = None
    started_decode: bool = False
    decode_start_t: Optional[float] = None
    record_first: bool=False

    def __post_init__(self):
        self.prefill_left = self.L
        self.decode_left = self.B

    def initialize_chunks(self, chunk_size: int):
        """Initialize chunked-prefill"""
        self.prefill_chunks = []
        remaining = self.L
        while remaining > 0:
            take = min(remaining, chunk_size)
            self.prefill_chunks.append(take)
            remaining -= take
        self.current_chunk_idx = 0

    def get_next_prefill_chunk(self) -> Optional[int]:
        """Get the next prefill chunk to process"""
        if self.current_chunk_idx < len(self.prefill_chunks):
            return self.prefill_chunks[self.current_chunk_idx]
        return None

    def advance_prefill_chunk(self):
        """Mark the current chunk as processed"""
        if self.current_chunk_idx < len(self.prefill_chunks):
            self.current_chunk_idx += 1


@dataclass(order=True)
class Event:
    """Event Classes in Discrete Event Simulation"""
    t: float
    etype: str
    payload: object = field(compare=False)


@dataclass
class GPUWorker:
    """The status of a single GPU"""
    gpu_id: int
    busy: bool = False
    batch_type: Optional[str] = None
    jobs: List[Tuple[Query, int, str]] = field(default_factory=list)
    finish_t: Optional[float] = None


def service_time_ms(workload: int, cfg: SimConfig, rng: random.Random) -> float:
    """
    S(b) = c + a * max(0, b - b0)
    """

    if workload <= 0:
        return 0.0

    # S(b) = c + a * max(0, b - b0)
    eff_workload = max(0, workload - cfg.b0)

    if cfg.service_time_model == 'exp_mm1':
        mean = cfg.c_ms + eff_workload * cfg.a_ms
        U = rng.random()
        return -mean * math.log(max(U, 1e-12))

    if cfg.service_time_model == 'deterministic':
        return cfg.c_ms + eff_workload * cfg.a_ms

    else:
        raise ValueError(f"Unknown service_time_model: {cfg.service_time_model}")


class Scheduler:
    """Scheduler"""
    name: str = "base"

    def pick_batch(
        self,
        prefill_q: List[Query],
        decode_q: List[Query],
        cfg: SimConfig,
    ) -> Tuple[str, List[Tuple[Query, int, str]]]:
        """
        Select the next batch

        Constraints:
        - max_sequences: Maximum number of sequences to process simultaneously
        - max_tokens: Maximum number of tokens to process simultaneously

        Returns:
            batch_type: 'prefill' 或 'decode'
            jobs: [(query, count, job_type), ...]
        """
        raise NotImplementedError


class DecodePrioritizingScheduler(Scheduler):
    """
    Decode Prioritizing Scheduler (Request-level batching)
    """
    name = "decode_prioritizing"

    def pick_batch(self, prefill_q, decode_q, cfg):
        jobs: List[Tuple[Query, int, str]] = []

        # Prioritize decoding: Package all decode tasks
        if decode_q:
            n_sequences = 0
            used_tokens = 0
            for q in decode_q:
                if n_sequences >= cfg.max_sequences or used_tokens + 1 > cfg.max_tokens:
                    break
                jobs.append((q, 1, 'decode'))
                n_sequences += 1
                used_tokens += 1

            return 'decode', jobs

        # Prefill only when the decode queue is empty
        if prefill_q:
            remaining_tokens = cfg.max_tokens
            remaining_seqs = cfg.max_sequences

            if cfg.enable_chunked_prefill:
                for q in prefill_q:
                    if remaining_seqs <= 0 or remaining_tokens <= 0:
                        break

                    chunk_size = q.get_next_prefill_chunk()
                    if not chunk_size:
                        continue

                    # Strict blocking: Stop completely as soon as the current piece cannot be stuffed in; 
                    # do not cut it into smaller pieces or look at the next one
                    if chunk_size > remaining_tokens:
                        break  

                    # If it can get this far, it can fit; just put it in
                    jobs.append((q, chunk_size, 'prefill'))
                    remaining_tokens -= chunk_size
                    remaining_seqs -= 1
            
            else:
                for q in prefill_q:
                    if remaining_seqs <= 0 or remaining_tokens <= 0:
                        break
                    
                    if q.prefill_left <= 0:
                        continue

                    # Strict blocking
                    if q.prefill_left > remaining_tokens:
                        break

                    # If it can get this far, it can fit; just put it in
                    jobs.append((q, q.prefill_left, 'prefill'))
                    remaining_tokens -= q.prefill_left
                    remaining_seqs -= 1


            if jobs:
                return 'prefill', jobs


        return 'prefill', []


class PrefillPrioritizingScheduler(Scheduler):
    """
    Prefill Prioritizing Scheduler (Iteration-level batching)
    """
    name = "prefill_prioritizing"

    def pick_batch(self, prefill_q, decode_q, cfg):
        jobs: List[Tuple[Query, int, str]] = []

        # Prioritize prefill: Greedy packing
        if prefill_q:
            remaining_tokens = cfg.max_tokens
            remaining_seqs = cfg.max_sequences

            if cfg.enable_chunked_prefill:
                for q in prefill_q:
                    if remaining_tokens <= 0 or remaining_seqs <= 0:
                        break
                    chunk_size = q.get_next_prefill_chunk()
                    if chunk_size:
                        take = min(chunk_size, remaining_tokens)
                        jobs.append((q, take, 'prefill'))
                        remaining_tokens -= take
                        remaining_seqs -= 1
            else:
                for q in prefill_q:
                    if remaining_tokens <= 0 or remaining_seqs <= 0:
                        break
                    take = min(q.prefill_left, remaining_tokens)
                    jobs.append((q, take, 'prefill'))
                    remaining_tokens -= take
                    remaining_seqs -= 1

            return 'prefill', jobs

        # Decoding is only performed when the prefill queue is empty
        if decode_q:
            n_sequences = 0
            used_tokens = 0
            for q in decode_q:
                if n_sequences >= cfg.max_sequences or used_tokens + 1 > cfg.max_tokens:
                    break
                jobs.append((q, 1, 'decode'))
                n_sequences += 1
                used_tokens += 1

            return 'decode', jobs

        return 'idle', []


class HybridScheduler(Scheduler):
    """
    Hybrid Scheduler (Token-budgeted hybrid)
    """
    name = "hybrid"
    def __init__(self, decode_seq_ratio: float = 0.5):
        self.decode_seq_ratio = decode_seq_ratio

    def pick_batch(self, prefill_q, decode_q, cfg):
        jobs: List[Tuple[Query, int, str]] = []

        # Allocate sequence budget
        decode_seq_budget = int(cfg.max_sequences * self.decode_seq_ratio)
        prefill_seq_budget = cfg.max_sequences - decode_seq_budget
        remaining_tokens = cfg.max_tokens

        # 1) First allocate decode
        used_decode_seqs = 0
        used_decode_tokens = 0
        for q in decode_q:
            if used_decode_seqs >= decode_seq_budget or used_decode_tokens + 1 > remaining_tokens:
                break
            jobs.append((q, 1, 'decode'))
            used_decode_seqs += 1
            used_decode_tokens += 1

        remaining_tokens -= used_decode_tokens 

        # 2) The remaining space is allocated to prefill
        used_prefill_seqs = 0

        if cfg.enable_chunked_prefill:
            for q in prefill_q:
                if used_prefill_seqs >= prefill_seq_budget or remaining_tokens <= 0:
                    break
                chunk_size = q.get_next_prefill_chunk()
                if chunk_size:
                    take = min(chunk_size, remaining_tokens)
                    jobs.append((q, take, 'prefill'))
                    remaining_tokens -= take
                    used_prefill_seqs += 1
        else:
            for q in prefill_q:
                if used_prefill_seqs >= prefill_seq_budget or remaining_tokens <= 0:
                    break
                take = min(q.prefill_left, remaining_tokens)
                jobs.append((q, take, 'prefill'))
                remaining_tokens -= take
                used_prefill_seqs += 1

        # Determine batch type
        has_prefill = any(t == 'prefill' for _, _, t in jobs)
        has_decode = any(t == 'decode' for _, _, t in jobs)

        if has_prefill and has_decode:
            return 'mixed', jobs
        elif has_decode:
            return 'decode', jobs
        elif has_prefill:
            return 'prefill', jobs
        else:
            return 'prefill', []


# ==================== 仿真器 ====================

class LLMSimulator:
    """Discrete event simulator for LLM Serving systems"""

    def __init__(self, cfg: SimConfig, scheduler: Scheduler):
        self.cfg = cfg
        self.scheduler = scheduler
        self.rng = random.Random(cfg.seed)
        np.random.seed(cfg.seed)

        self.events: List[Event] = []
        self.prefill_queue: List[Query] = []
        self.decode_queue: List[Query] = []

        self.gpus: List[GPUWorker] = [
            GPUWorker(gpu_id=i) for i in range(cfg.num_gpus)
        ]

        self.queries: List[Query] = []
        self.completed: int = 0
        self.time: float = 0.0
        # Time window
        self.total_queries: int = 0   
        # Time series statistics
        self.ts_t: List[float] = []
        self.ts_throughput: List[float] = []
        self.ts_ttft: List[float] = []
        self.ts_tbt: List[float] = []
        self.ts_queue_length: List[float] = []

        self._last_ts_t = 0.0
        self._last_ts_completed = 0

    def _schedule_event(self, t: float, etype: str, payload=None):
        heapq.heappush(self.events, Event(t, etype, payload))

    def _init_arrivals(self):
        """Generate arrival events for all requests within a fixed time window"""
        t = 0.0
        qid = 0
        T_max = self.cfg.arrival_time_window

        while True:
            # Sample the next arrival interval
            U = self.rng.random()
            dt = -math.log(max(U, 1e-12)) / self.cfg.lam
            t += dt

            # Stop if the time window is exceeded
            if t > T_max:
                break

            L = sample_prompt_length(self.cfg, self.rng)
            B = sample_output_budget(self.cfg, self.rng, L)

            q = Query(qid=qid, arrival_t=t, L=L, B=B)

            if self.cfg.enable_chunked_prefill:
                q.initialize_chunks(self.cfg.prefill_chunk_size)

            self.queries.append(q)
            self._schedule_event(t, 'arrival', q)

            qid += 1

        # Record the actual number of requests generated
        self.total_queries = len(self.queries)


    def run(self):
        """Run Simulation"""
        self._init_arrivals()

        while self.events:
            ev = heapq.heappop(self.events)

            self.time = ev.t

            if ev.etype == 'arrival':
                self._handle_arrival(ev.payload)
            elif ev.etype == 'gpu_finish':
                gpu_id = ev.payload
                self._handle_gpu_finish(gpu_id)
                self._maybe_dispatch()
            elif ev.etype == 'tick':
                self._handle_tick()
            else:
                raise ValueError(f"Unknown event type: {ev.etype}")

            if self.total_queries > 0 and self.completed >= self.total_queries:
                break


        return self._finalize_stats()

    def _handle_arrival(self, q: Query):
        self.prefill_queue.append(q)
        self._maybe_dispatch()

        if self.time - self._last_ts_t >= 1.0:
            self._schedule_event(self.time, 'tick')

    def _maybe_dispatch(self):
        """Attempt to dispatch new batches to idle GPUs"""
        while True:
            idle_gpu_id = None
            for gpu in self.gpus:
                if not gpu.busy:
                    idle_gpu_id = gpu.gpu_id
                    break

            if idle_gpu_id is None:
                return

            btype, jobs = self.scheduler.pick_batch(
                self.prefill_queue, self.decode_queue, self.cfg
            )

            if not jobs:
                return

            self._dispatch_batch(idle_gpu_id, btype, jobs)

    def _dispatch_batch(self, gpu_id: int, btype: str, jobs: List[Tuple[Query, int, str]]):
        """Dispatch the batch to the specified GPU"""
        gpu = self.gpus[gpu_id]
        gpu.busy = True
        gpu.batch_type = btype
        #gpu.jobs=jobs.copy()
        gpu.jobs = jobs

        # Calculate service time
        if btype == 'prefill':
            # Prefill: workload = token num
            n_tokens = sum(tok for _, tok, jt in jobs if jt == 'prefill')
            st_ms = service_time_ms(n_tokens, self.cfg, self.rng)
        elif btype == 'decode':
            # Decode: workload = sequence num
            n_sequences = sum(1 for _, _, jt in jobs if jt == 'decode')
            st_ms = service_time_ms(n_sequences, self.cfg, self.rng)
        else:  # mixed
            # Mixed: take max
            n_tokens = sum(tok for _, tok, jt in jobs if jt == 'prefill')
            n_sequences = sum(1 for _, _, jt in jobs if jt == 'decode')
            st_prefill = service_time_ms(n_tokens, self.cfg, self.rng) if n_tokens > 0 else 0
            st_decode = service_time_ms(n_sequences, self.cfg, self.rng) if n_sequences > 0 else 0
            st_ms = max(st_prefill, st_decode)

        st_s = st_ms / 1000.0
        finish_t = self.time + st_s

        gpu.finish_t = finish_t
        self._schedule_event(finish_t, 'gpu_finish', gpu_id)

        # Remove the scheduled query from the queue
        used_qids = {q.qid for q, _, jtype in jobs}
        decode_used_qids = {q.qid for q, _, jt in jobs if jt == 'decode'}
        self.decode_queue = [q for q in self.decode_queue if q.qid not in decode_used_qids]
        self.prefill_queue = [q for q in self.prefill_queue if q.qid not in used_qids]

    def _handle_gpu_finish(self, gpu_id: int):
        """Handling GPU batch completion events"""
        gpu = self.gpus[gpu_id]
        jobs = gpu.jobs

        requeue_prefill = []
        requeue_decode = []

        for q, tok, job_type in jobs:
            if job_type == 'prefill':
                q.prefill_left -= tok

                if self.cfg.enable_chunked_prefill:
                    q.advance_prefill_chunk()

                if q.prefill_left <= 0 and q.prefill_done_t is None:
                    q.prefill_done_t = self.time
                    q.started_decode = True
                    
                    if q.decode_left > 0:
                        self.decode_queue.append(q)
                    else:
                        q.done_t = self.time
                        self.completed += 1
                else:
                # Regardless of whether it's chunked or not, as long as prefilling isn't finished, 
                # it needs to be requeued.
                    if q.prefill_left > 0:
                        requeue_prefill.append(q)

            elif job_type == 'decode':
                q.decode_left -= tok
                if not q.record_first:
                    q.record_first=True
                    q.decode_start_t=self.time
                if q.decode_left > 0:
                    requeue_decode.append(q)
                else:
                    q.done_t = self.time
                    self.completed += 1
      # Continue the strategy of "prioritizing currently active requests"
        if requeue_decode:
            for q in reversed(requeue_decode):
                self.decode_queue.insert(0, q)

        if requeue_prefill:
            for q in reversed(requeue_prefill):
                self.prefill_queue.insert(0, q)

        gpu.busy = False
        gpu.batch_type = None
        gpu.jobs = []
        gpu.finish_t = None

    def _handle_tick(self):
        """Regularly collect time series indicators"""
        if self.time - self._last_ts_t < 1.0:
            return

        dt = self.time - self._last_ts_t
        n_done = self.completed - self._last_ts_completed
        throughput = n_done / dt if dt > 0 else 0

        self.ts_t.append(self.time)
        self.ts_throughput.append(throughput)

        # Calculate the TTFT and TBT of the most recently completed request
        recent_queries = [
            q for q in self.queries
            if q.done_t is not None and self._last_ts_t <= q.done_t < self.time
        ]

        if recent_queries:
            ttfts = [q.prefill_done_t - q.arrival_t for q in recent_queries if q.prefill_done_t]
            tbts = [
                (q.done_t - q.prefill_done_t) / q.B if q.B > 0 else 0
                for q in recent_queries if q.prefill_done_t and q.done_t
            ]

            mean_ttft = np.mean(ttfts) if ttfts else 0
            mean_tbt = np.mean(tbts) if tbts else 0
        else:
            mean_ttft = 0
            mean_tbt = 0

        self.ts_ttft.append(mean_ttft)
        self.ts_tbt.append(mean_tbt)
        self.ts_queue_length.append(len(self.prefill_queue) + len(self.decode_queue))

        self._last_ts_t = self.time
        self._last_ts_completed = self.completed

        # Schedule the next tick
        self._schedule_event(self.time + 1.0, 'tick')

    def _finalize_stats(self):
        """Calculate the final statistical indicators"""
        # Filter warmup
        valid_arrivals = [q for q in self.queries if q.arrival_t >= self.cfg.warmup_time]
        valid_queries = [
            q for q in self.queries
            if q.done_t is not None
            and q.arrival_t >= self.cfg.warmup_time
        ]

        if not valid_queries:
        # Even without a valid request, ensure that all keys used downstream exist
            return {
                'scheduler': self.scheduler.name,
                'num_gpus': self.cfg.num_gpus,
                'mean_ttft_s': 0.0,
                'p95_ttft_s': 0.0,
                'mean_tbt_s': 0.0,
                'p95_tbt_s': 0.0,
                'mean_rts_s': 0.0,
                'p95_rts_s': 0.0,
                'throughput_qps': 0.0,
                'sim_time_s': self.time,
                'n_arrived': len(valid_arrivals),
                'n_completed': 0,
                'enable_chunked_prefill': self.cfg.enable_chunked_prefill,
            }

        # TTFT: Time To First Token
        ttfts = [q.decode_start_t - q.arrival_t for q in valid_queries if q.decode_start_t]

        # TBT: Time Between Tokens 
        tbts = [
            (q.done_t - q.decode_start_t) / q.B
            for q in valid_queries
            if q.B > 0 and q.decode_start_t and q.done_t
        ]
        rts = [q.done_t - q.arrival_t for q in valid_queries]

        # Throughout
        t_first = min(q.arrival_t for q in valid_queries)
        t_last = max(q.done_t for q in valid_queries)
        elapsed = t_last - t_first
        throughput = len(valid_queries) / elapsed if elapsed > 0 else 0

        return {
            'scheduler': self.scheduler.name,
            'num_gpus': self.cfg.num_gpus,
            'mean_ttft_s': np.mean(ttfts) if ttfts else 0,
            'p95_ttft_s': np.percentile(ttfts, 95) if ttfts else 0,
            'mean_tbt_s': np.mean(tbts) if tbts else 0,
            'p95_tbt_s': np.percentile(tbts, 95) if tbts else 0,
            'mean_rts_s': np.mean(rts) if rts else 0,
            'p95_rts_s': np.percentile(rts, 95) if rts else 0,
            'throughput_qps': throughput,
            'sim_time_s': self.time,
            'n_arrived': len(valid_arrivals),
            'n_completed': len(valid_queries),
            'enable_chunked_prefill': self.cfg.enable_chunked_prefill,
        }


def run_experiment(cfg: SimConfig, scheduler: Scheduler, verbose: bool = False, seed: Optional[int] = None):
    """
    Run a single simulation experiment (optionally covering the seed)
    """
    cfg_local = copy.deepcopy(cfg)
    if seed is not None:
        cfg_local.seed = seed

    sim = LLMSimulator(cfg_local, scheduler)
    stats = sim.run()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scheduler: {stats['scheduler']}")
        print(f"{'='*60}")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")

    return stats, sim


def mean_ci95(x: np.ndarray):
    """
    95% CI using normal approx (z=1.96):
    mean ± 1.96 * std/sqrt(n)
    Returns: (mean, half_width, low, high)
    """
    n = len(x)
    m = float(x.mean())
    if n <= 1:
        return m, 0.0, m, m
    s = float(x.std(ddof=1))
    se = s / math.sqrt(n)
    hw = 1.96 * se
    return m, hw, m - hw, m + hw


def _one_trial_single(args) -> Dict[str, Any]:
    cfg, scheduler, seed = args
    cfg_local = copy.deepcopy(cfg)
    cfg_local.seed = seed
    stats, _ = run_experiment(cfg_local, scheduler, verbose=False)
    return stats


def run_trials_for_scheduler(
    cfg: SimConfig,
    scheduler: Scheduler,
    n_trials: int = 100,
    seed0: int = 0,
    max_workers: Optional[int] = None,
    return_runs: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single scheduler in parallel n_trials times and return a summary such as the mean/CI
    """
    if max_workers is None:
        max_workers = min(os.cpu_count(), n_trials)

    tasks = [(cfg, scheduler, seed0 + k) for k in range(n_trials)]
    runs: List[Dict[str, Any]] = []

    mp_ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as ex:
        futures = [ex.submit(_one_trial_single, t) for t in tasks]
        for fut in as_completed(futures):
            runs.append(fut.result())

    def arr(key):
        return np.array([r[key] for r in runs], dtype=float)

    mean_ttft = arr("mean_ttft_s")
    p95_ttft  = arr("p95_ttft_s")
    mean_tbt  = arr("mean_tbt_s")
    p95_tbt   = arr("p95_tbt_s")
    thr       = arr("throughput_qps")
    mean_rts  = arr("mean_rts_s")
    p95_rts   = arr("p95_rts_s")
    n_arr     = arr("n_arrived")
    n_com     = arr("n_completed")

    m_rts, hw_rts, lo_rts, hi_rts = mean_ci95(mean_rts)
    m_p95_rts, hw_p95_rts, lo_p95_rts, hi_p95_rts = mean_ci95(p95_rts)

    m_ttft, hw_ttft, lo_ttft, hi_ttft = mean_ci95(mean_ttft)
    m_tbt,  hw_tbt,  lo_tbt,  hi_tbt  = mean_ci95(mean_tbt)
    m_thr,  hw_thr,  lo_thr,  hi_thr  = mean_ci95(thr)
    m_p95_ttft, hw_p95_ttft, lo_p95_ttft, hi_p95_ttft = mean_ci95(p95_ttft)
    m_p95_tbt,  hw_p95_tbt,  lo_p95_tbt,  hi_p95_tbt  = mean_ci95(p95_tbt)
    m_Narr, hw_Narr, lo_Narr, hi_Narr = mean_ci95(n_arr)
    m_Ncom, hw_Ncom, lo_Ncom, hi_Ncom = mean_ci95(n_com)

    summary = {
        "scheduler": scheduler.name,
        "n_trials": len(runs),

        "mean(mean_ttft_s)": m_ttft,
        "ci95_low(mean_ttft_s)": lo_ttft,
        "ci95_high(mean_ttft_s)": hi_ttft,
        "std(mean_ttft_s)": float(mean_ttft.std(ddof=1)),

        "mean(p95_ttft_s)": m_p95_ttft,
        "ci95_low(p95_ttft_s)": lo_p95_ttft,
        "ci95_high(p95_ttft_s)": hi_p95_ttft,
        "std(p95_ttft_s)": float(p95_ttft.std(ddof=1)),

        "mean(mean_tbt_s)": m_tbt,
        "ci95_low(mean_tbt_s)": lo_tbt,
        "ci95_high(mean_tbt_s)": hi_tbt,
        "std(mean_tbt_s)": float(mean_tbt.std(ddof=1)),

        "mean(p95_tbt_s)": m_p95_tbt,
        "ci95_low(p95_tbt_s)": lo_p95_tbt,
        "ci95_high(p95_tbt_s)": hi_p95_tbt,
        "std(p95_tbt_s)": float(p95_tbt.std(ddof=1)),

        "mean(mean_rts_s)": m_rts,
        "ci95_low(mean_rts_s)": lo_rts,
        "ci95_high(mean_rts_s)": hi_rts,
        "std(mean_rts_s)": float(mean_rts.std(ddof=1)),

        "mean(p95_rts_s)": m_p95_rts,
        "ci95_low(p95_rts_s)": lo_p95_rts,
        "ci95_high(p95_rts_s)": hi_p95_rts,
        "std(p95_rts_s)": float(p95_rts.std(ddof=1)),
        
        "mean(throughput_qps)": m_thr,
        "ci95_low(throughput_qps)": lo_thr,
        "ci95_high(throughput_qps)": hi_thr,
        "std(throughput_qps)": float(thr.std(ddof=1)),

        "mean(arried_queries)": m_Narr,
        "ci95_low(arried_queries)": lo_Narr,
        "ci95_high(arried_queries)": hi_Narr,
        "std(arried_queries)": float(n_arr.std(ddof=1)),

        "mean(completed_queries)": m_Ncom,
        "ci95_low(completed_queries)": lo_Ncom,
        "ci95_high(completed_queries)": hi_Ncom,
        "std(completed_queries)": float(n_com.std(ddof=1)),

    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scheduler: {scheduler.name} | trials={len(runs)}")
        print(f"{'='*60}")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k:28s}: {v:.4f}")
            else:
                print(f"  {k:28s}: {v}")

    return {
        "runs": runs if return_runs else None,
        "summary": summary,
    }


# ==================== 扫 t_max + 画图 & 存图 ====================

def save_figure(fig, filename: str, save_dir: str = "figs"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {path}")
    plt.close(fig)


def sweep_time_window_for_scheduler(
    cfg: SimConfig,
    scheduler: Scheduler,
    t_max: float,
    n_points: int = 100,
    n_trials: int = 100,
    seed0: int = 0,
    max_workers: Optional[int] = None,
    verbose_each: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Scan different arrival_time_windows (0, t_max], 
    run n_trials times for each point, and return a summary list
    """
    # Avoid t=0 (no request)
    t_values = np.linspace(t_max / n_points, t_max, n_points)
    summaries: List[Dict[str, Any]] = []

    for i, t in enumerate(t_values):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.arrival_time_window = float(t)

        if verbose_each:
            print(f"\n=== t_max = {t:.3f} s ({i+1}/{n_points}) ===")

        out = run_trials_for_scheduler(
            cfg_i,
            scheduler,
            n_trials=n_trials,
            seed0=seed0 + i * 1000,  # Each t has a seed interval
            max_workers=max_workers,
            return_runs=False,
            verbose=verbose_each,
        )
        summaries.append(out["summary"])

    return t_values, summaries


def plot_metric_vs_t(
    t_values: np.ndarray,
    summaries: List[Dict[str, Any]],
    metric: str,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
):
    """
    metric:
      - 'mean_ttft_s'
      - 'p95_ttft_s'
      - 'mean_rts_s'
      - 'p95_rts_s'
      - 'throughput_qps'
      - 'mean_tbt_s'
      - 'p95_tbt_s'
    read from summary dict:
      - mean_key = f"mean({metric})"
      - low_key  = f"ci95_low({metric})"
      - high_key = f"ci95_high({metric})"
    """
    mean_key = f"mean({metric})"
    low_key = f"ci95_low({metric})"
    high_key = f"ci95_high({metric})"

    means = np.array([s[mean_key] for s in summaries])
    lows = np.array([s[low_key] for s in summaries])
    highs = np.array([s[high_key] for s in summaries])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.figure

    if label is None:
        label = metric

    ax.plot(t_values, means, label=label)
    ax.fill_between(t_values, lows, highs, alpha=0.2)

    ax.set_xlabel("arrival_time_window t_max (s)")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend()
    return fig, ax

def export_results_to_csv(results, filename):
    rows = []
    
    for scheduler_name, (t_vals, summaries) in results.items():
        for t, s in zip(t_vals, summaries):
            row = {
                "scheduler": scheduler_name,
                "t_max": t,
                "throughput_qps": s.get("mean(throughput_qps)"),
                "n_arrived": s.get("mean(arried_queries)"),
                "n_completed": s.get("mean(completed_queries)"),
                
                "mean_ttft": s.get("mean(mean_ttft_s)"),
                "mean_tbt": s.get("mean(mean_tbt_s)"),
                "mean_rts": s.get("mean(mean_rts_s)"),
                
                "p95_ttft": s.get("mean(p95_ttft_s)"),
                "p95_tbt": s.get("mean(p95_tbt_s)"),
                "p95_rts": s.get("mean(p95_rts_s)"),
            }
            rows.append(row)
            
    if not rows:
        print("No data to export.")
        return

    fieldnames = [
        "scheduler", "t_max", 
        "throughput_qps", "n_arrived", "n_completed", "completion_rate",
        "mean_ttft", "mean_tbt", "mean_rts",
        "p95_ttft", "p95_tbt", "p95_rts"
    ]
    
    actual_fieldnames = [f for f in fieldnames if f in rows[0].keys()]

    with open(filename, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=actual_fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(rows)
    
    print(f"Results exported to {filename}")

import csv
from pathlib import Path

def print_steady_state_report(results, csv_path: str | None = None):
    header = (
        f"{'Scheduler':<25} | "
        f"{'QPS':<8} | {'Arrived':<8} | {'Done':<8} "
        f"{'TTFT(Avg)':<10} | {'TBT(Avg)':<10} | {'RTS(P95)':<10}"
    )

    rows = []

    print("\n" + "="*115)
    print("Results:")
    print("="*115)
    print(header)
    print("-" * 115)

    for name, (t_vals, summaries) in results.items():
        # Take the data from the last time point
        s = summaries[-1]

        n_arr_v = float(s.get("mean(arried_queries)", 0) or 0)
        n_com_v = float(s.get("mean(completed_queries)", 0) or 0)
        qps_v   = float(s.get("mean(throughput_qps)", 0) or 0)
        ttft_v  = float(s.get("mean(mean_ttft_s)", 0) or 0)
        tbt_v   = float(s.get("mean(mean_tbt_s)", 0) or 0)
        rts95_v = float(s.get("mean(p95_rts_s)", 0) or 0)

        n_arr = f"{n_arr_v:.1f}"
        n_com = f"{n_com_v:.1f}"
        qps   = f"{qps_v:.2f}"
        ttft  = f"{ttft_v:.4f}"
        tbt   = f"{tbt_v:.4f}"
        rts95 = f"{rts95_v:.4f}"

        print(
            f"{name:<25} | "
            f"{qps:<8} | {n_arr:<8} | {n_com:<8} | "
            f"{ttft:<10} | {tbt:<10} | {rts95:<10}"
        )

        rows.append({
            "Scheduler": name,
            "QPS": qps_v,
            "Arrived": n_arr_v,
            "Done": n_com_v,
            "TTFT_Avg": ttft_v,
            "TBT_Avg": tbt_v,
            "RTS_P95": rts95_v,
        })

    print("="*115 + "\n")

    # Write CSV if path provided
    if csv_path:
        csv_path = str(csv_path)
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["Scheduler", "QPS", "Arrived", "Done", "TTFT_Avg", "TBT_Avg", "RTS_P95"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[Saved CSV] {csv_path}")



def plot_comprehensive_metrics(results, filename, demo_name):
    """
    Plot a 2x2 composite indicator panel
    [Mean TTFT]  [Mean TBT]
    [P95 RTS]    [Throughput]
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    plots_config = [
        (0, 0, "mean_ttft_s",     "Mean TTFT",          "Seconds"),
        (0, 1, "mean_tbt_s",      "Mean TBT",           "Seconds"),
        (1, 0, "p95_rts_s",       "P95 Response Time",  "Seconds"),
        (1, 1, "throughput_qps",  "Throughput",         "Queries/s"),
    ]

    for row, col, metric, title, ylabel in plots_config:
        ax = axes[row, col]
        for name, (t_vals, summaries) in results.items():
            plot_metric_vs_t(t_vals, summaries, metric=metric, ax=ax, label=name)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        ax.legend(fontsize='small', loc='best')

        if row == 1:
            ax.set_xlabel("Time Window $t_{max}$ (s)")

    plt.suptitle(f"{demo_name}: Performance by Time", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    save_figure(fig, filename)