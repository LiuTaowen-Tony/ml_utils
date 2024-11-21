import time
from threading import Thread 
import subprocess

class JobQueue:
    def __init__(self, device_list, max_job_per_device):
        self.device_list = device_list
        self.num_devices = len(device_list)
        self.max_job_per_device = max_job_per_device
        self.device_counters = [0] * self.num_devices
        self.wait_list = []

    def cmd_to_thread(self, cmd):
        # callable : device -> None

        while True:
            for device in self.device_list:
                if self.device_counters[device] < self.max_job_per_device:
                    self.device_counters[device] += 1
                    break
            else:
                time.sleep(1)
                continue
            break
        def job():
            try:
                subprocess.run(f"CUDA_VISIBLE_DEVICES={device} {cmd}", shell=True)
            finally:
                self.device_counters[device] -= 1

        return Thread(target=job)

    def map(self, cmd_list):
        for cmd in cmd_list:
            thread = self.cmd_to_thread(cmd)
            thread.start()
            self.wait_list.append(thread)
        self.wait_all()

    def wait_all(self):
        for thread in self.wait_list:
            print(thread)
            thread.join()
