"""On veut étudier la façon de calibrer cett utilité. 
"""

# %%
import utility as ut
import numpy as np
import matplotlib.pyplot as plt
#%%
u = ut.get_cvx_utility(3)
r = np.linspace(-5,5,100)

us = {}
for β in [1,2,3]:
    u = ut.get_cvx_utility(β)
    plt.plot(r,u(r).value)

plt.legend([1,2,3])
plt.grid()
plt.plot()




# %%
