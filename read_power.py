import os
import time


def read_sysfs():
    # sysfs node of GPU power in mW
    filename = '/sys/bus/i2c/drivers/ina3221x/1-0040/iio_device/in_power0_input'
    with open(filename, 'r') as f:
        power = f.read();
    power = power.replace('\n', '')
    return power

def main():
    pairs = list()
    for i in range(5000):
        start = time.time()
        power = read_sysfs()
        end = time.time()
        time_passed = (end - start) * 1e3 # ms
        pairs.append((power, time_passed))
    import pickle
    with open('power_time_pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
