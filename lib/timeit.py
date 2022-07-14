import time

def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        with open("output_times.txt", "a") as f:
            print('Function', func.__name__, 'time:', round((te -ts)*1000,1), 'ms', file=f)
            print(file=f)
        return result
    return timed