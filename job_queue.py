import time
from threading import Thread 

class JobQueue:
    def __init__(self, num_devices, max_job_per_device):
        self.num_devices = num_devices
        self.max_job_per_device = max_job_per_device
        self.device_counters = [0] * num_devices
        self.running_processes = [None] * num_devices
        self.wait_list = []

    def get_thread(self, callable):
        # callable : device -> None

        while True:
            for device in range(self.num_devices):
                if self.device_counters[device] < self.max_job_per_device:
                    self.device_counters[device] += 1
                    break
            else:
                time.sleep(1)
                continue
            break
        def job():
            try:
                callable(device)
            finally:
                self.device_counters[device] -= 1

        return Thread(target=job)

    def map(self, callable_list):
        for callable in callable_list:
            thread = self.get_thread(callable)
            thread.start()
            self.wait_list.append(thread)
        self.wait_all()

    def wait_all(self):
        for thread in self.wait_list:
            print(thread)
            thread.join()
