import utility
import matplotlib.pyplot as plt
import numpy as np

u = utility.get_utility()
r = np.linspace(-5,5,100)
plt.plot(r, u(r))
