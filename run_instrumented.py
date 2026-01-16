# run_instrumented.py
"""Run the real wikidata pipeline with memory instrumentation."""
import psutil
import os
import sys
import threading
import time
from pathlib import Path

# Memory monitoring in background thread
class MemoryMonitor:
    def __init__(self, log_path="memory_log.parquet", interval=5.0):
        self.log_path = Path(log_path)
        self.interval = interval
        self.running = False
        self.records = []
        self.thread = None
    
    def _get_memory(self):
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        sys_mem = psutil.virtual_memory()
        return {
            "timestamp": time.time(),
            "rss_gb": mem.rss / 1024**3,
            "vms_gb": mem.vms / 1024**3,
            "sys_used_gb": sys_mem.used / 1024**3,
            "sys_avail_gb": sys_mem.available / 1024**3,
            "sys_percent": sys_mem.percent,
        }
    
    def _monitor_loop(self):
        import polars as pl
        while self.running:
            mem = self._get_memory()
            self.records.append(mem)
            print(f"[MEM] RSS={mem['rss_gb']:.2f}GB VMS={mem['vms_gb']:.2f}GB Avail={mem['sys_avail_gb']:.1f}GB ({mem['sys_percent']:.0f}%)", file=sys.stderr)
            sys.stderr.flush()
            
            # Save periodically (every 10 samples)
            if len(self.records) % 10 == 0:
                pl.DataFrame(self.records).write_parquet(self.log_path)
            
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[MEM] Monitor started, logging to {self.log_path}", file=sys.stderr)
    
    def stop(self):
        import polars as pl
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.records:
            pl.DataFrame(self.records).write_parquet(self.log_path)
            print(f"[MEM] Saved {len(self.records)} samples to {self.log_path}", file=sys.stderr)


def main():
    monitor = MemoryMonitor(log_path="pipeline_memory.parquet", interval=5.0)
    monitor.start()
    
    try:
        # Import and run the real pipeline
        from wikidata.main import run
        run()
    except KeyboardInterrupt:
        print("\n[MEM] Interrupted by user", file=sys.stderr)
    except Exception as e:
        print(f"\n[MEM] Pipeline error: {e}", file=sys.stderr)
        raise
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()