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



def rom_assembly(K,V,q):
    # force is Kq + V^Tf_n(Vq)
    # Hessian is Kq + V^T df_n V q


    nNodes = self.nNodes
    nElements = self.nElements
    nEquations = self.nEquations
    nDoF = self.nDoF
    nWeft = self.nWeft
    ID = self.ID
    LM = self.LM
    EBC = self.EBC
    g = self.g
    elements = self.elements
    wn = self.wn
    wefts = self.wefts

    #Step 1: Evaluate Vq

    u = np.dot(V,q)

    #Step 2: Evaluate f_n(u), df_n(u)

    disp = np.empty([nDoF, nNodes])

    for i in range(nNodes):
        for j in range(nDoF):
            disp[j,i] = u[ID[j,i]] if EBC[j,i] == 0 else g[j,i]

    #Step 4: Allocate dP and ddP
    dP = np.zeros(nDoF*nNodes)
    ddP = np.zeros([nEquations,nEquations])

    #Setp 5: Assemble K and F
    contact_dist = self.contact_dist

    contact_dist.fill(np.inf)

    g_min = np.inf

    for i in range(nWeft):
        xm, rm = wefts[0:2, i], wefts[2, i]
        closest_e = -1
        for e in range(nElements):
            ele = elements[e]
            da, db = disp[:, e], disp[:, e + 1]
            contact, penalty, info = ele.penalty_term(da, db, xm, rm, wn)
            if (contact and info[1] < g_min):
                closest_e, g_min = e, info[1]
                #print('closest_e is ' , closest_e, 'g_min is ', g_min,' penalty is ', penalty)


        if (closest_e >= 0):

            ele = elements[closest_e]
            da, db = disp[:, closest_e], disp[:, closest_e + 1]
            contact, penalty, info = ele.penalty_term(da, db, xm, rm, wn)

            if(info[1] < contact_dist[closest_e]):
                contact_dist[closest_e] = info[1]

            print('Weft ', i , ' contacts element', closest_e, ' local coordinate is ',
                  info[0], ' distance is ', info[1], ' side is ',info[2])

            #print('closest_e is ' , closest_e, 'info is ', info,' penalty is ', penalty)
            _, f_contact, k_contact = penalty
            # Step 3b: Get Global equation numbers
            P = LM[:, closest_e]

            # Step 3c: Eliminate Essential DOFs
            I = (P >= 0)
            P = P[I]

            # Step 3d: Insert k_e, f_e, f_g, f_h
            ddP[np.ix_(P, P)] += k_contact[np.ix_(I, I)]
            dP[P] += f_contact[I]



    dPi = np.dot(K,q) + np.dot(V.T, dP)
    ddPi = K + np.dot(V.T, np.dot(ddP,V))
    return dPi, ddPi

def rom_fem_calc(self,V):

#step 1 build stiff matrix V^TKV

    nEquations = self.nEquations


    nDoF = self.nDoF

    q = np.zeros(nEquations)
#step 2 dPi = V^TKVq - V^Tf_n(Vq)
#       ddPi = V^TKV - V^T df_n V



    dPi,ddPi = rom_assembly(u)

    res0 = np.linalg.norm(dPi)

    MAXITE = self.MAXITE
    EPS = 1e-8
    found = False
    dt_max = 0.5
    T = 0
    for ite in range(MAXITE):


        dPi,ddPi = rom_assembly(K, V, q)

        res = np.linalg.norm(dPi)

        dq = np.linalg.solve(ddPi,dPi)

        ################################
        # Time stepping
        ###############################

        du_abs = np.repeat(np.sqrt(du[0:-1:nDoF]**2 + du[1:-1:nDoF]**2) + 1e-12, nDoF)

        gap_lower_bound = self.compute_gap_lower_bound()

        if(ite < 1000):
            dt = min(dt_max[0], self.r/np.max(du_abs)/10.0)

        else:
            dt = np.min(np.minimum(dt_max, gap_lower_bound/du_abs))


        q -=  dt*dq




        print('Ite/MAXITE: ', ite, ' /', MAXITE, 'In fem_calc res is', res,' dt is ', dt )
        if(res < EPS):# or res < EPS*res0):
            found = True
            break
        T += dt
    if(not found):
        print("Newton cannot converge in fem_calc")
    print('T is ', T)
    return u,res

if __name__ == '__main__':
    build_basis()
