from tcprl_module import *
import signal
import os

# Agent selection: 1 - DeepSARSA, 2 - A2C, 3 - A3C
if __name__ == "__main__":
    pool = normal_execute(2, True, 0, 0)
    try:
        while True:
            # waiting
            time.sleep(1)
    except KeyboardInterrupt:
        for p in pool:
            p.terminate()
            p.join()


