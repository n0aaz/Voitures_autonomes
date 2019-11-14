import matplotlib.pyplot as plt
import numpy as np

x=np.array([0.001*a for a in range(-3000,3000)])

def sigmoid(x):
	return 1/(1+np.exp(-x))

plt.plot(x,sigmoid(x))
plt.xlabel('x')
plt.ylabel('sig(x)')
plt.grid()
plt.savefig('sigmoid.png')
plt.show()

