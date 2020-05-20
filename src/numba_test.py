import time
from numba import jit

@jit
def ex_jit():
    start = time.time()

    for i in range(1000000):
        if i%100000 == 0:
            print(i)
        for j in range(1000000):
            for k in range(1000000):
                for l in range(1000000):
                    a = 2
                    b = 3
                    c = pow(a,b,2)

    end = time.time()

    return end-start

print("time : " + str(ex_jit()))