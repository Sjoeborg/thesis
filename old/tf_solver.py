import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from numerical import H_4, evolve
import timeit

E = 100
L = 14000
H = tf.convert_to_tensor(-1j/(2*E) * H_4(0,0), dtype = tf.complex64)
i_state = tf.constant([1., 0., 0., 0.], dtype=tf.complex64)

def tf_evolve(t,state):
    return tf.linalg.matvec(H, state)

def tf_solver():
    results = tfp.math.ode.DormandPrince().solve(tf_evolve, initial_time=0, initial_state=i_state,
                                   solution_times=[5.06*L])
    y0 = results.states[0]
    return y0.numpy()

def sp_solver(L):
    sp_solver = solve_ivp(fun = evolve,t_eval=[5.06*L], t_span = [0, 5.06*L], method='BDF', y0 = i_state, args=('e','e', E, 4, 0))
    return sp_solver.y

def pooler():
    p = Pool(8)
    L_range = np.linspace(L/2, L, 10)
    results_pooled = p.map(sp_solver, L_range)
    return results_pooled
def nonPool():
    L_range = np.linspace(L/2, L, 10)
    result = [sp_solver(x) for x in L_range]
    return result


#print(timeit.timeit('tf_solver()', number=10,setup="from __main__ import tf_solver"))
#print(timeit.timeit('sp_solver()', number=10,setup="from __main__ import sp_solver"))

print(timeit.timeit('pooler()', number=10,setup="from __main__ import pooler"))
print(timeit.timeit('nonPool()', number=10,setup="from __main__ import nonPool"))
