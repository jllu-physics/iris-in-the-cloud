import os
import csv
import datetime

TIME_STR_FORMAT = "%Y-%m-%d_%H-%M-%S"

def append_to_csv(new_row, csv_filename):
    with open(csv_filename, 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)

def if_file_exist(filename, path = '.'):
    full_file_path = os.path.join(path, filename)
    return os.path.exists(full_file_path)

def get_available_filename(base_filename, path):
    if not if_file_exist(base_filename, path):
        return base_filename
    parts = base_filename.split('.')
    i = 1
    while True:
        new_filename = '.'.join([parts[0] + '_' + str(i),] + parts[1:])
        if not if_file_exist(new_filename, path):
            return new_filename
        else:
            i += 1

def get_current_time_str(format = TIME_STR_FORMAT):
    return datetime.datetime.now().strftime(format)