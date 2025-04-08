# logging resources usage
# I saw errors like this: 
#   Memory cgroup out of memory: Killed process 2891 (uvicorn) 
#   total-vm:1415724kB, anon-rss:1348964kB, file-rss:0kB, 
#   shmem-rss:0kB, UID:0 pgtables:2720kB oom_score_adj:0
# when using `sudo dmesg -T | grep -i "killed process"`
# after docker crashed
#
# Another tricky thing is, to log cpu percent of the current
# process, we have to use non-blocking mode. After all, if we
# block the current process there is nothing to measure. Then
# we either need to pass a process object around or to use it
# as a global variable. Neither is appealing so wrap in a class

import psutil
import os
import logging

class Logger:

    def __init__(self, log_filename = None):
        self.process = psutil.Process(os.getpid())

        if log_filename is None:
            logging.basicConfig(
                format = '%(asctime)s - %(message)s',
                level = logging.INFO
            )
        else:
            logging.basicConfig(
                filename = log_filename,
                filemode = 'a',
                format = '%(asctime)s - %(message)s',
                level = logging.INFO
            )
    
    def set_proc_avg_cpu_percent_start_point(self):
        _ = self.process.cpu_percent(interval = None)
    
    def log_proc_avg_cpu_percent(self, note = ""):
        avg_cpu_percent = self.process.cpu_percent()
        log_msg = (
            f"{note}\n"
            f"  AVG PROCESS CPU USAGE: {avg_cpu_percent:.2f}%"
        )
        logging.info(log_msg)
    
    def log_proc_memory(self, note = ""):
        mem_rss_MB = self.process.memory_info().rss / 1024 / 1024
        log_msg = (
            f"{note}\n"
            f"  PROCESS MEMORY USAGE: {mem_rss_MB:.2f} MB (RSS)"
        )
        logging.info(log_msg)
    
    def set_system_avg_cpu_percent_start_point(self):
        self.system_process_list = list(psutil.process_iter())
        _ = sum(p.cpu_percent(interval = None) for p in self.system_process_list)
    
    def log_system_avg_cpu_percent(self, note = ""):
        avg_cpu_percent = sum(
            get_proc_cpu_pct_safe(p) 
            for p in self.system_process_list
        )
        log_msg = (
            f"{note}\n"
            f"  AVG SYSTEM CPU USAGE: {avg_cpu_percent:.2f}%"
        )
        logging.info(log_msg)
    
    def log_system_memory(self, note = ""):
        mem_rss_MB = sum(
            p.memory_info().rss
            for p in self.system_process_list
        ) / 1024 / 1024
        log_msg = (
            f"{note}\n"
            f"  SYSTEM MEMORY USAGE: {mem_rss_MB:.2f} MB (RSS)"
        )
        logging.info(log_msg)

def get_proc_cpu_pct_safe(proc):
    try:
        cpu_pct = proc.cpu_percent(interval = None)
        return cpu_pct
    except psutil.NoSuchProcess:
        return 0
    except psutil.AccessDenied:
        return 0