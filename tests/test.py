import numpy as np
import matplotlib.pyplot as plt

plt.ion()



def restrict_node(v):
  '''
  restrict 'v' to the coarser grid
  '''
  si = v.shape
  nl = si[0]
  nx = ny = (si[-1]-1)//2
  v_c = np.zeros([nl,nx+1,ny+1])
  
  v_c[:,1:-1,1:-1] = (0.25*v[:,2:-2:2,2:-2:2] +
                      0.125*(v[:,1:-3:2,2:-2:2] +
                             v[:,3:-1:2,2:-2:2] +
                             v[:,2:-2:2,1:-3:2] +
                             v[:,2:-2:2,3:-1:2]) + 
                      0.0625*(v[:,1:-3:2,1:-3:2] + 
                              v[:,1:-3:2,3:-1:2] +
                              v[:,3:-1:2,1:-3:2] +
                              v[:,3:-1:2,3:-1:2]))

  return v_c


def prolong_node(v):
  '''
  interpolate 'v' to the fine grid (node version)
  '''
  si = v.shape
  nl = si[0]
  nx = ny = (si[-1]-1)

  v_f = np.zeros([nl,2*nx+1,2*ny+1])

  v_f[:,2:-2:2,2:-2:2] = v[:,1:-1,1:-1]

  v_f[:,1:-1:2,2:-2:2] = 0.5*(v[:,:-1,1:-1] + v[:,1:,1:-1])
#  v_f[:,-2,2:-2:2] = 0.5*(v[:,-1,1:-1] + v[:,-2,1:-1]) # side

  v_f[:,2:-2:2,1:-1:2] = 0.5*(v[:,1:-1,:-1] + v[:,1:-1,1:])
#  v_f[:,2:-2:2,-2] = 0.5*(v[:,-1,1:-1] + v[:,1:-1,-2]) # side

  v_f[:,1:-1:2,1:-1:2] = 0.25*(v[:,1:,1:] + v[:,1:,:-1] + v[:,:-1,1:] + v[:,:-1,:-1])


  return v_f



N = 256 + 1 

x = np.linspace(0,1,N)
y = np.linspace(0,1,N)

xx,yy = np.meshgrid(x,y)

psif = np.sin(2*np.pi*xx)*np.sin(6*np.pi*yy)

psic = restrict_node(restrict_node(restrict_node(restrict_node(psif[None,:,:]))))

# plt.figure()
# plt.imshow(psif)


# plt.figure()
# plt.imshow(psic[0])


psif2 = prolong_node(prolong_node(prolong_node(prolong_node(psic) )))

plt.figure()
plt.imshow(psif2[0])
