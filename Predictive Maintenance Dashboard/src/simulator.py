import time
import os
import threading
import random
from typing import Callable, Dict, Any, Optional
from queue import Queue


class RealTimeSimulator:
    """Simple real-time sensor simulator producing realistic-ish values every second."""
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._thread = None
        self._stop = threading.Event()
        # Product ID counter and lock
        self._counter_lock = threading.Lock()
        self._counter_file = None
        self._counter = None
        self._init_counter()

    def _generate(self) -> Dict[str, Any]:
        # Select operating mode with weighted probabilities
        # Healthy: 70%, Warning: 20%, Critical: 10%
        mode = random.choices(['healthy', 'warning', 'critical'], weights=[0.7, 0.2, 0.1], k=1)[0]

        if mode == 'healthy':
            # Healthy Mode ranges
            air = round(random.uniform(295.0, 302.0), 2)
            proc = round(random.uniform(305.0, 312.0), 2)
            rpm = round(random.uniform(1300, 2200), 2)
            torque = round(random.uniform(20, 55), 2)
            tool = round(random.uniform(0, 120), 2)
        elif mode == 'warning':
            # Warning Mode ranges
            air = round(random.uniform(300.0, 304.0), 2)
            proc = round(random.uniform(310.0, 316.0), 2)
            rpm = round(random.uniform(2000, 2600), 2)
            torque = round(random.uniform(50, 70), 2)
            tool = round(random.uniform(100, 200), 2)
        else:
            # Critical Mode ranges
            air = round(random.uniform(302.0, 304.0), 2)
            proc = round(random.uniform(315.0, 324.0), 2)
            rpm = round(random.uniform(2400, 2886), 2)
            torque = round(random.uniform(65, 77), 2)
            tool = round(random.uniform(180, 253), 2)

        pid = self._next_product_id()
        return {
            'Product_ID': pid,
            'Operating_Mode': mode,
            'Type': random.choice(['L', 'M', 'H']),
            'Air_temperature_K': air,
            'Process_temperature_K': proc,
            'Rotational_speed_rpm': rpm,
            'Torque_Nm': torque,
            'Tool_wear_min': tool,
            'timestamp': time.time()
        }

    def _init_counter(self):
        """Initialize persistent counter stored in a small file under models/."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            self._counter_file = os.path.join(models_dir, 'product_id_counter.txt')
            if os.path.exists(self._counter_file):
                with open(self._counter_file, 'r') as f:
                    content = f.read().strip()
                    try:
                        self._counter = int(content)
                    except Exception:
                        self._counter = 10000
            else:
                # start before first ID so first call produces 10001
                self._counter = 10000
                with open(self._counter_file, 'w') as f:
                    f.write(str(self._counter))
        except Exception:
            # fallback to in-memory counter
            self._counter_file = None
            self._counter = 10000

    def _next_product_id(self) -> str:
        """Increment and persist the product id counter and return a string like 'M10001'."""
        with self._counter_lock:
            self._counter += 1
            try:
                if self._counter_file:
                    with open(self._counter_file, 'w') as f:
                        f.write(str(self._counter))
            except Exception:
                pass
            return f"M{self._counter}"

    def start(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None, queue: Optional[Queue] = None):
        """
        Start the simulator. If `queue` is provided, samples will be put into the queue
        (thread-safe) for the main Streamlit thread to consume. `callback` is optional
        and will be called in the simulator thread (avoid UI operations in callback).
        """
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()

        def run():
            while not self._stop.is_set():
                sample = self._generate()
                try:
                    if queue is not None:
                        queue.put(sample)
                    if callback is not None:
                        # callback should not touch Streamlit UI directly
                        try:
                            callback(sample)
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


if __name__ == '__main__':
    def print_cb(s):
        print(s)

    sim = RealTimeSimulator(1.0)
    sim.start(print_cb)
    time.sleep(5)
    sim.stop()
