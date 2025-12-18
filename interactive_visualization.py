"""
Interactive Visualizations for Demo2
Provides animated batch timelines and queue dynamics dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from simulator import (
    SimConfig, 
    DecodePrioritizingScheduler, 
    PrefillPrioritizingScheduler,
    HybridScheduler,
    Query,
    sample_prompt_length,
    sample_output_budget,
    service_time_ms,
    Event,
    GPUWorker
)
import random
import heapq
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class BatchEvent:
    """Records a batch execution event for visualization"""
    start_time: float
    end_time: float
    batch_type: str  # 'prefill' or 'decode'
    gpu_id: int
    queries: List[int]  # query IDs
    workload: int  # tokens for prefill, sequences for decode
    
@dataclass 
class QueueSnapshot:
    """Records queue state at a point in time"""
    time: float
    prefill_queue_size: int
    decode_queue_size: int
    gpu_busy: bool
    total_queries_completed: int
    total_queries_arrived: int

class InteractiveSimulator:
    """Extended simulator that captures events for visualization"""
    
    def __init__(self, cfg: SimConfig, scheduler):
        self.cfg = cfg
        self.scheduler = scheduler
        self.batch_events: List[BatchEvent] = []
        self.queue_snapshots: List[QueueSnapshot] = []
        self.queries: Dict[int, Query] = {}
        
    def run_and_capture(self, max_time: float = 50.0, seed: int = 42):
        """Run simulation and capture events for visualization"""
        cfg = self.cfg
        rng = random.Random(seed)
        np.random.seed(seed)
        
        # Initialize
        event_queue = []
        prefill_q: List[Query] = []
        decode_q: List[Query] = []
        gpu = GPUWorker(gpu_id=0)
        
        qid_counter = 0
        completed_queries = []
        current_t = 0.0
        
        # Generate initial arrivals
        t = 0.0
        while t < max_time:
            dt = -math.log(rng.random()) / cfg.lam if cfg.lam > 0 else float('inf')
            t += dt
            if t >= max_time:
                break
                
            L = sample_prompt_length(cfg, rng)
            B = sample_output_budget(cfg, rng, L)
            q = Query(qid=qid_counter, arrival_t=t, L=L, B=B)
            
            if cfg.enable_chunked_prefill:
                q.initialize_chunks(cfg.prefill_chunk_size)
                
            self.queries[qid_counter] = q
            heapq.heappush(event_queue, Event(t, 'arrival', q))
            qid_counter += 1
        
        # Simulation loop
        snapshot_interval = 0.5  # Take snapshots every 0.5 seconds
        next_snapshot_time = 0.0
        
        while event_queue:
            ev = heapq.heappop(event_queue)
            current_t = ev.t
            
            # Take periodic snapshots
            while next_snapshot_time <= current_t:
                snapshot = QueueSnapshot(
                    time=next_snapshot_time,
                    prefill_queue_size=len(prefill_q),
                    decode_queue_size=len(decode_q),
                    gpu_busy=gpu.busy,
                    total_queries_completed=len(completed_queries),
                    total_queries_arrived=qid_counter
                )
                self.queue_snapshots.append(snapshot)
                next_snapshot_time += snapshot_interval
            
            if ev.etype == 'arrival':
                q = ev.payload
                prefill_q.append(q)
                
            elif ev.etype == 'gpu_done':
                gpu_id = ev.payload
                
                # Process completed jobs
                for (q, count, job_type) in gpu.jobs:
                    if job_type == 'prefill':
                        q.prefill_left -= count
                        if cfg.enable_chunked_prefill:
                            q.advance_prefill_chunk()
                            
                        if q.prefill_left <= 0:
                            q.prefill_done_t = current_t
                            if q.B > 0:
                                decode_q.append(q)
                            else:
                                q.done_t = current_t
                                completed_queries.append(q)
                        else:
                            # Re-queue for next prefill chunk
                            prefill_q.append(q)
                            
                    elif job_type == 'decode':
                        if not q.started_decode:
                            q.started_decode = True
                            q.decode_start_t = current_t
                        
                        q.decode_left -= count
                        if q.decode_left <= 0:
                            q.done_t = current_t
                            completed_queries.append(q)
                        else:
                            decode_q.append(q)
                
                # Free GPU
                gpu.busy = False
                gpu.batch_type = None
                gpu.jobs = []
                gpu.finish_t = None
            
            # Try to schedule next batch
            if not gpu.busy and (prefill_q or decode_q):
                batch_type, jobs = self.scheduler.pick_batch(prefill_q, decode_q, cfg)
                
                if jobs:
                    # Remove scheduled queries from queues
                    scheduled_qids = {q.qid for q, _, _ in jobs}
                    prefill_q = [q for q in prefill_q if q.qid not in scheduled_qids]
                    decode_q = [q for q in decode_q if q.qid not in scheduled_qids]
                    
                    # Calculate workload
                    if batch_type == 'prefill':
                        workload = sum(count for _, count, _ in jobs)
                    else:  # decode
                        workload = len(jobs)
                    
                    # Calculate service time
                    service_ms = service_time_ms(workload, cfg, rng)
                    service_s = service_ms / 1000.0
                    finish_t = current_t + service_s
                    
                    # Update GPU state
                    gpu.busy = True
                    gpu.batch_type = batch_type
                    gpu.jobs = jobs
                    gpu.finish_t = finish_t
                    
                    # Record batch event
                    batch_event = BatchEvent(
                        start_time=current_t,
                        end_time=finish_t,
                        batch_type=batch_type,
                        gpu_id=0,
                        queries=[q.qid for q, _, _ in jobs],
                        workload=workload
                    )
                    self.batch_events.append(batch_event)
                    
                    # Schedule completion
                    heapq.heappush(event_queue, Event(finish_t, 'gpu_done', 0))
        
        return completed_queries


def create_animated_batch_timeline(sim: InteractiveSimulator):
    """
    Create an animated visualization of batch scheduling over time.
    Returns the animation object for display in Jupyter notebook.
    Shows prefill (blue) and decode (orange) batches on a timeline.
    """
    import math
    
    batch_events = sim.batch_events
    if not batch_events:
        print("No batch events to visualize")
        return None
    
    max_time = max(be.end_time for be in batch_events)
    
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(14, 6))
        
        def init():
            ax.clear()
            ax.set_xlim(0, max_time)
            ax.set_ylim(-1, 2)
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('GPU 0', fontsize=12)
            ax.set_yticks([0.5])
            ax.set_yticklabels(['GPU 0'])
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_title(f'Batch Timeline - {sim.scheduler.name}', fontsize=14, fontweight='bold')
            return []
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(0, max_time)
            ax.set_ylim(-0.5, 1.5)
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('GPU', fontsize=12)
            ax.set_yticks([0.5])
            ax.set_yticklabels(['GPU 0'])
            ax.grid(True, alpha=0.3, axis='x')
            
            # Calculate current time based on frame
            current_time = (frame / 100) * max_time
            
            # Draw completed and in-progress batches
            for be in batch_events:
                if be.start_time <= current_time:
                    # Determine how much to show
                    display_end = min(be.end_time, current_time)
                    width = display_end - be.start_time
                    
                    if width > 0:
                        color = '#3498db' if be.batch_type == 'prefill' else '#e74c3c'
                        alpha = 0.7 if be.end_time <= current_time else 0.9
                        
                        rect = Rectangle(
                            (be.start_time, 0.2), width, 0.6,
                            facecolor=color, edgecolor='black', alpha=alpha, linewidth=1
                        )
                        ax.add_patch(rect)
                        
                        # Add text label
                        if width > 0.5:  # Only add text if rectangle is wide enough
                            mid_x = be.start_time + width / 2
                            label = f"{be.batch_type[0].upper()}\n{len(be.queries)}q"
                            ax.text(mid_x, 0.5, label, 
                                ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Draw current time marker
            ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current Time')
            
            # Create legend
            prefill_patch = mpatches.Patch(color='#3498db', label='Prefill Batch', alpha=0.7)
            decode_patch = mpatches.Patch(color='#e74c3c', label='Decode Batch', alpha=0.7)
            ax.legend(handles=[prefill_patch, decode_patch], loc='upper right')
            
            ax.set_title(f'Batch Timeline - {sim.scheduler.name} (t={current_time:.2f}s)', 
                        fontsize=14, fontweight='bold')
            
            return []
        
        # Create animation
        n_frames = 100
        anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, 
                            interval=100, blit=False, repeat=True)
        
        plt.close(fig)
        return anim


def create_queue_dynamics_dashboard(sim: InteractiveSimulator):
    """
    Create an animated dashboard showing:
    - Queue lengths over time (prefill vs decode)
    - GPU utilization
    - Cumulative queries completed
    Returns the animation object for display in Jupyter notebook.
    """
    import math
    
    snapshots = sim.queue_snapshots
    if not snapshots:
        print("No queue snapshots to visualize")
        return None
    
    times = [s.time for s in snapshots]
    prefill_sizes = [s.prefill_queue_size for s in snapshots]
    decode_sizes = [s.decode_queue_size for s in snapshots]
    gpu_busy = [1 if s.gpu_busy else 0 for s in snapshots]
    completed = [s.total_queries_completed for s in snapshots]
    arrived = [s.total_queries_arrived for s in snapshots]
    
    max_time = max(times)
    max_queue = max(max(prefill_sizes), max(decode_sizes)) + 5
    
    with plt.ioff():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        def init():
            for ax in axes.flat:
                ax.clear()
            return animate(0)
        
        def animate(frame):
            for ax in axes.flat:
                ax.clear()
            
            # Calculate how much data to show
            show_idx = int((frame / 100) * len(times))
            show_idx = max(1, show_idx)  # At least show one point
            
            t_show = times[:show_idx]
            
            # Plot 1: Queue Lengths
            ax1 = axes[0, 0]
            ax1.plot(t_show, prefill_sizes[:show_idx], label='Prefill Queue', color='#3498db', linewidth=2)
            ax1.plot(t_show, decode_sizes[:show_idx], label='Decode Queue', color='#e74c3c', linewidth=2)
            ax1.fill_between(t_show, prefill_sizes[:show_idx], alpha=0.3, color='#3498db')
            ax1.fill_between(t_show, decode_sizes[:show_idx], alpha=0.3, color='#e74c3c')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Queue Size')
            ax1.set_title('Queue Lengths Over Time', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, max_time)
            ax1.set_ylim(0, max_queue)
            
            # Plot 2: GPU Utilization
            ax2 = axes[0, 1]
            ax2.fill_between(t_show, gpu_busy[:show_idx], alpha=0.6, color='#2ecc71', label='GPU Busy')
            ax2.plot(t_show, gpu_busy[:show_idx], color='#27ae60', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('GPU Status')
            ax2.set_title('GPU Utilization', fontweight='bold')
            ax2.set_xlim(0, max_time)
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Idle', 'Busy'])
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Cumulative Completions
            ax3 = axes[1, 0]
            ax3.plot(t_show, arrived[:show_idx], label='Arrived', color='#95a5a6', 
                    linewidth=2, linestyle='--')
            ax3.plot(t_show, completed[:show_idx], label='Completed', color='#27ae60', linewidth=2)
            ax3.fill_between(t_show, completed[:show_idx], alpha=0.3, color='#27ae60')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Number of Queries')
            ax3.set_title('Query Progress', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, max_time)
            
            # Plot 4: System State Summary (text-based)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            if show_idx > 0:
                current_snap = snapshots[show_idx - 1]
                summary_text = f"""
    System State at t={current_snap.time:.2f}s

    Scheduler: {sim.scheduler.name}

    Queue Status:
    • Prefill Queue: {current_snap.prefill_queue_size} queries
    • Decode Queue: {current_snap.decode_queue_size} queries
    • Total Queued: {current_snap.prefill_queue_size + current_snap.decode_queue_size}

    Progress:
    • Queries Arrived: {current_snap.total_queries_arrived}
    • Queries Completed: {current_snap.total_queries_completed}
    • Completion Rate: {100 * current_snap.total_queries_completed / max(1, current_snap.total_queries_arrived):.1f}%

    GPU Status: {'BUSY' if current_snap.gpu_busy else 'IDLE'}
                """
                ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                        fontsize=11, verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            fig.suptitle(f'Queue Dynamics Dashboard - {sim.scheduler.name}', 
                        fontsize=14, fontweight='bold', y=0.995)
            
            return []
        
        # Create animation
        n_frames = 100
        anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                            interval=100, blit=False, repeat=True)
        
        plt.close(fig)
        return anim

def run_demo2_interactive():
    import math
    
    print("="*60)
    print("Demo2 Interactive Visualization")
    print("="*60)
    
    # Configure simulation
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
    
    # Run simulations
    sims = {}
    for scheduler in schedulers:
        
        sim = InteractiveSimulator(cfg, scheduler)
        completed = sim.run_and_capture(max_time=50.0, seed=42)
        
        sims[scheduler.name] = sim

    return sims


if __name__ == "__main__":
    # Example: Run simulations and display info
    sims = run_demo2_interactive()