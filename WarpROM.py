__author__ = 'zhengyuh'

from Warp import Warp
import numpy as np
import matplotlib.pyplot as plt

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
class WarpROM(Warp):

    def __init__(self,type,par,wn,k, MAXITE,n,m,err):
        Warp.__init__(self,type,par,wn,k, MAXITE)
        self.n = n
        self.m = m
        self.err = err

    def _build_basis(self, files = None):
        '''
        Off-line process to compute basis for the
        problem
        :return:
        '''
        if(files is None):
            nEquations = self.nEquations

            n,m = self.n, self.m
            pars = np.zeros([2,n*m])
            disps = np.zeros([nEquations,n*m])
            fs = np.zeros([6,n*m])
            ress = np.zeros(n*m)

            u_x_range = np.linspace(-0.1,0.1,n)
            u_y_range = np.linspace(-0.1,0.1,m)
            for i in range(n):
                u_x = u_x_range[i]
                for j in range(m):
                    u_y = u_y_range[j]
                    self.reset_par([u_x,u_y,theta])
                    d, res = self.fem_calc()
                    f = self.compute_force(d)

                    pars[:,i*n + j] = [u_x,u_y]
                    disps[:,i*n +j] = d
                    fs[:,i*n + j] = f
                    ress[i*n + j] = res

            np.save('pars', pars)
            np.save('disps', disps)
            np.save('fs', fs)
            np.save('ress', ress)
        else:
            pars = np.load(files[0])
            disps = np.load(files[1])
            fs = np.load(files[2])
            ress = np.load(files[3])


        V,s,U = np.linalg.svd(disps,False)

        energy = s*s/np.dot(s,s)
        for i in range(1,len(s)):
            energy[i] = energy[i-1] + energy[i]
        energy = 1.0 - energy

        basis_n = np.argmax(energy < self.err) + 1
        '''
        plt.figure()
        plt.plot(np.arange(len(energy)) + 1, energy, '-o', markersize = 2)
        plt.xlabel('k')
        plt.ylabel(r'$1 - E_{POD}(k)$')
        #plt.show()
        '''
        plt.figure()
        plt.plot(s,'-ro')
        plt.show()


        self.V = V[:,0: basis_n]
        self.basis_n = basis_n
        #todo determine basis_n
        return basis_n




    def _rom_assembly_linear(self):
        nEquations = self.nEquations
        nElements = self.nElements
        LM = self.LM

        # Step 2: Allocate K,  F,  dP and ddP
        K = np.zeros([nEquations, nEquations])
        F = np.zeros(nEquations);

        # Step 3: Assemble K and F

        for e in range(nElements):
            [k_e, f_e, f_g] = self._linear_beam_arrays(e);
            # Step 3b: Get Global equation numbers
            P = LM[:, e]

            # Step 3c: Eliminate Essential DOFs
            I = (P >= 0)
            P = P[I]

            # Step 3d: Insert k_e, f_e, f_g, f_h
            K[np.ix_(P, P)] += k_e[np.ix_(I, I)]
            F[P] += f_e[I] + f_g[I]

        #todo reduced stiff matrix

        V = self.V
        self.K = np.dot(V.T,np.dot(K,V))
        self.F = np.dot(V.T, F)


    def _rom_assembly(self,q):
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
        V = self.V
        K = self.K
        F = self.F
        #Step 1: Evaluate Vq

        u = np.dot(V,q)

        #Step 2: Evaluate f_n(u), df_n(u)

        disp = np.empty([nDoF, nNodes])

        for i in range(nNodes):
            for j in range(nDoF):
                disp[j,i] = u[ID[j,i]] if EBC[j,i] == 0 else g[j,i]

        #Step 4: Allocate dP and ddP
        dP = np.zeros(nEquations)
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



        dPi = np.dot(K,q) - F  +  np.dot(V.T, dP)
        ddPi = K + np.dot(V.T, np.dot(ddP,V))
        return dPi, ddPi

    def rom_fem_calc(self):

    #step 1 build stiff matrix V^TKV

        nEquations = self.nEquations


        nDoF = self.nDoF

        basis_n = self.basis_n


        V = self.V

        q = np.zeros(basis_n)
    #step 2 dPi = V^TKVq - V^Tf_n(Vq)
    #       ddPi = V^TKV - V^T df_n V



        dPi,ddPi = self._rom_assembly(q)

        res0 = np.linalg.norm(dPi)

        MAXITE = self.MAXITE
        EPS = 1e-8
        found = False
        dt_max = 0.5
        T = 0
        for ite in range(MAXITE):


            dPi,ddPi = self._rom_assembly(q)

            res = np.linalg.norm(dPi)

            dq = np.linalg.solve(ddPi,dPi)

            ################################
            # Time stepping
            ###############################

            du = np.dot(V, dq)

            du_abs = np.repeat(np.sqrt(du[0:-1:nDoF]**2 + du[1:-1:nDoF]**2) + 1e-12, nDoF)




            dt = min(dt_max, self.r/np.max(du_abs)/10.0)


            q -=  dt*dq




            print('Ite/MAXITE: ', ite, ' /', MAXITE, 'In fem_calc res is', res,' dt is ', dt )
            if(res < EPS):# or res < EPS*res0):
                found = True
                dPi,ddPi = self._rom_assembly(q)
                break
            T += dt
        if(not found):
            print("Newton cannot converge in fem_calc")
        print('T is ', T)

        u = np.dot(V,q)
        return u,res


def basis_visualize():
    #############Test hybrid solutions
    disps = np.load('disps.npy')
    pars = np.load('pars.npy')
    for i in range(100):
        par = np.array([pars[0,i], pars[1,i], 0.0])
        warp.reset_par(par )
        warp.visualize_result(disps[:,i], 2)

if __name__ == '__main__':
    u_x, u_y, theta = 0.2, -0.2, 0.0
    wn = 1e6
    MAXITE = 2000
    k = 3
    err = 1e-5
    warp = WarpROM('sine beam', [u_x, u_y, theta], wn, k, MAXITE,10,10,err)
    #build basis
    basis_n = warp._build_basis(['pars.npy', 'disps.npy', 'fs.npy', 'ress.npy'])
    #basis_n = warp._build_basis( )

    print('basis number is  ', basis_n)

    #build linear part reduced matrix
    warp._rom_assembly_linear()


    warp.reset_par([u_x,u_y,theta])

    u,res = warp.rom_fem_calc()

    warp.visualize_result(u)







