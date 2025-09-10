import numpy as np
import matplotlib.pyplot as plt
import geostrokit as qg

plt.ion()

N = 256 + 1 

x = np.linspace(0,1,N)
y = np.linspace(0,1,N)

xx,yy = np.meshgrid(x,y)

Delta = x[1] - x[0]

psif = np.sin(2*np.pi*xx)*np.sin(6*np.pi*yy)
sol_exact = -np.sin(2*np.pi*xx)*np.sin(6*np.pi*yy)*(1/((6*np.pi)**2 + (2*np.pi)**2))

sol = qg.solve_mg(psif, Delta)

plt.figure()
plt.imshow(sol)


plt.figure()
plt.imshow(sol_exact)

plt.figure()
plt.imshow(sol-sol_exact)

