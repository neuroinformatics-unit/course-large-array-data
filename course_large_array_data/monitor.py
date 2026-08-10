import threading
import time
import tracemalloc

from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from IPython.display import display


@register_cell_magic
def monitor(_, cell):
    tracemalloc.start()
    start = time.time()
    stop_event = threading.Event()
    handle = display("starting...", display_id=True)

    def report_loop():
        while not stop_event.wait(0.5):  # poll every 0.5s
            current, peak = tracemalloc.get_traced_memory()
            elapsed = time.time() - start
            handle.update(
                f"[t={elapsed:6.2f}s] current={current / 1e6:.1f}MB "
                f"peak={peak / 1e6:.1f}MB"
            )

    t = threading.Thread(target=report_loop, daemon=True)
    t.start()
    try:
        get_ipython().run_cell(cell)
    finally:
        stop_event.set()
        t.join()

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("\n--- Monitoring report ---")
    print(
        f"[memory] current: {current / 1e6:.1f} MB, peak: {peak / 1e6:.1f} MB"
    )
    print(f"[speed] time elapsed: {elapsed:.3f}s")
