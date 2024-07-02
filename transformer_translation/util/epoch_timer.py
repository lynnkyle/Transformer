import time
import datetime


def get_time_diff(start_time):
    end_time = time.time()
    time_diff = end_time - start_time
    return time.strftime("%H:%M:%S", time.gmtime(time_diff))


if __name__ == '__main__':
    start = time.time()
    time.sleep(2)
    print(get_time_diff(start))
