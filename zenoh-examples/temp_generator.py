from zenoh import Zenoh
import random
import time

random.seed()

def read_temp():
    return random.randint(15, 30)

def run_sensor_loop(w):
    # read and produce e temperature every second
    while True:
        t = read_temp()
        w.put('/myhome/kitcken/temp', t)
        time.sleep(1)

if __name__ == "__main__":
    z = Zenoh({})
    w = z.workspace('/')
    run_sensor_loop(w)
