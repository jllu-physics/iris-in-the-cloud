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
import datetime

class Logger:

    def __init__(self, log_filename = None):
        self.process = psutil.Process(os.getpid())
        self.system_process_iter = psutil.process_iter()

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
        mem_rss = self.process.memory_info().rss
        log_msg = (
            f"{note}\n"
            f"  PROCESS MEMORY USAGE: {mem_rss:.2f} MB (RSS)"
        )
        logging.info(log_msg)
    
    def set_system_avg_cpu_percent_start_point(self):
        _ = sum(p.cpu_percent(interval = None) for p in self.system_process_iter)
    
    def log_system_avg_cpu_percent(self, note = ""):
        avg_cpu_percent = sum(
            p.cpu_percent(interval=None) 
            for p in self.system_process_iter
        )
        log_msg = (
            f"{note}\n"
            f"  AVG SYSTEM CPU USAGE: {avg_cpu_percent:.2f}%"
        )
        logging.info(log_msg)
    
    def log_system_memory(self, note = ""):
        mem_rss = sum(
            p.memory_info().rss
            for p in self.system_process_iter
        )
        log_msg = (
            f"{note}\n"
            f"  SYSTEM MEMORY USAGE: {mem_rss:.2f} MB (RSS)"
        )
        logging.info(log_msg)