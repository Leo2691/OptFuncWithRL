import numpy as np

def Rastrigin_env(vals):
    """"
        Rastrigin function:
        https://en.wikipedia.org/wiki/Rastrigin_function
    """
    r = 10 * len(vals)
    s = sum([np.power(val, 2) - 10 * np.cos(2 * np.pi * val) for val in vals])

    reward = r + s

    return reward

def Rosenbrock_env(vals):
    """"
        Rosenbrock function:
        https://en.wikipedia.org/wiki/Rosenbrock_function
    """

    s = sum([4 * (( vals[i + 1] - vals[i] **2 ) ** 2) + (1 - vals[i]) ** 2 for i in np.arange(len(vals) - 1)])

    reward = s

    return reward
