__author__ = 'zhengyuh'
from Warp import Warp
import numpy as np
'''
This is a problem, with fixed left [0,0,0],
and fixed right [x,y, theta]. We have two
parameters x, y, theta, which are strain,
deflection and bending angle
ux     \in [-0.1,0.1]
uy     \in [-0.1,0.1]
theta  = 0.0
And we will predict the force at right
[Fx, Fy, M_theta]

This problem to some degrees, explores the constitutive
law of the warp.
'''

def build_basis():
    theta = 0.0
    wn = 1e6
    MAXITE = 1000
    k = 3


    warp = Warp('sine beam',[0.0,0.0,0.0],wn,k, MAXITE)
    nEquations = warp.nEquations

    n,m = 10,10
    pars = np.zeros(2,n*m)
    disps = np.zeros([nEquations,n*m])
    fs = np.zeros(6,n*m)
    ress = np.zeros(n*m)

    u_x_range = np.linspace(-0.1,0.1,n)
    u_y_range = np.linspace(-0.1,0.1,m)
    for i in range(n):
        u_x = u_x_range[i]
        for j in range(m):
            u_y = u_y_range[j]
            warp.reset_par([u_x,u_y,theta])
            d,res = warp.fem_calc()
            f = warp.compute_force(d)

            pars[:,i*n + m] = [u_x,u_y]
            disps[:,i*n + m] = d
            fs[:,i*n + m] = f
            ress[:,i*n + m] = res

    np.save('pars', pars)
    np.save('disps', disps)
    np.save('fs', fs)
    np.save('ress', ress)




if __name__ == '__main__':
    build_basis()
