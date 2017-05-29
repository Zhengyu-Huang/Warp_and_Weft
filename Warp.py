import numpy as np
from Beam import *
import matplotlib.pyplot as plt

class Warp:
    def __init__(self,type):

        '''
                      Name            Description
                    -------         --------------
                    nDim            Number of Spatial Dimensions
                    nNodes          Number of nodes in the mesh
                    nElements       Number of elements in the mesh
                    nNodesElement   Number of nodes per element
                    nDoF            Number of DoF per node
                    nEquations      Number of equations to solve
                    elements        list contains nElements beam element class
                    ID              Array of global eq. numbers, destination array (ID)
                    EBC             EBC = 1 if DOF is on Essential B.C.
                    IEN             Array of global node numbers
                    LM              Array of global eq. numbers, location matrix (LM)
                    C               Material Constant at each element
                    f               Distributed Load at each node, an array(nDof, nNodes)
                    g               Essential B.C. Value at each node is an array(nDoF, nNodes)
                    h               Natural B.C. Value at each node
        '''

        self.nDim = nDim = 2
        self.nDoF = nDoF = 3
        self.nNodesElement = 2

        #build elements
        self.type = type
        if(type == 'horizontal beam'):
            self._horizontal_beam_data()
        elif(type == 'slanting beam'):
            self._slanting_beam_data()




        #construct  element nodes array
        #IEM(i,e) is the global node id of element e's node i
        self.IEM = np.zeros([self.nNodesElement, self.nElements], dtype = 'int')
        self.IEM[0,:] = np.arange(self.nElements)
        self.IEM[1,:] = np.arange(1, self.nElements + 1)

        #construct destination array
        #ID(d,n) is the global equation number of node n's dth freedom, -1 means no freedom
        self.ID = np.zeros([self.nDoF, self.nNodes],dtype = 'int') - 1
        self.ID[:,1:self.nNodes] = np.reshape(np.arange(self.nDoF*(self.nNodes - 1)), (3,-1), order='F')

        #construct Local matrix
        #LM(d,e) is the global equation number of element e's d th freedom
        self.LM = np.zeros([self.nNodesElement*self.nDoF, self.nElements],dtype = 'int')
        for i in range(self.nDoF):
            for j in range(self.nNodesElement):
                for k in range(self.nElements):
                    self.LM[j*self.nDoF + i, k] = self.ID[i,self.IEM[j,k]]



    def _horizontal_beam_data(self):
        '''
        g is dirichlet boundary condition
        f is the internal force
        '''
        nDoF = self.nDoF
        nDim = self.nDim




        self.nElements = nElements = 5
        self.elements = elements = []
        self.nNodes = nNodes = self.nElements + 1
        self.nEquations = self.nDoF * (self.nNodes - 1)

        E = 1.0e4
        r = 0.1
        self.Coord = Coord = np.zeros([nDoF, nNodes])
        Coord[0,:] = np.linspace(0,1.0,nNodes)

        for e in range(nElements):
            Xa0,Xb0 = np.array([Coord[0,e],Coord[1,e],Coord[2,e]]),np.array([Coord[0,e+1],Coord[1,e+1],Coord[2,e]])
            elements.append(LinearEBBeam(Xa0, Xb0,E,r))


        # Essential bounary condition
        self.g = np.zeros([nDoF, nNodes])
        self.EBC = np.zeros([nDoF,nNodes],dtype='int')
        self.EBC[:,0] = 1

        # Force
        fx,fy,m = 0.0,1, 0.0
        self.f = np.zeros([nDoF, nNodes])
        self.f[:, -1] = fx, fy, m

        # Weft info
        self.nWeft = nWeft = 1
        self.wefts = wefts = np.zeros([nDim+1, nWeft]) # (x,y,r)
        wefts[:,0] = 0.4,0.22,0.1

        #Penalty parameters
        self.wn = 1e8

    def _sine_beam_data(self):
        '''
        g is dirichlet boundary condition
        f is the internal force
        '''
        nDoF = self.nDoF
        nDim = self.nDim




        self.nElements = nElements = 5
        self.elements = elements = []
        self.nNodes = nNodes = self.nElements + 1
        self.nEquations = self.nDoF * (self.nNodes - 1)

        # Weft info
        self.nWeft = nWeft = 1
        self.wefts = wefts = np.zeros([nDim + 1, nWeft])  # (x,y,r)
        wefts[:, 0] = 0.4, 0.22, 0.1

        # Penalty parameters
        self.wn = 1e8


        E = 1.0e4
        r = 0.1
        #The curve is A*sin(w*(x - pi/2.0))
        self.Coord = Coord = np.zeros([nDim, nNodes])
        Coord[0, :] = np.linspace(0, 2*np.pi, nNodes)
        Coord[1, :] = A*np.sin(w*(x - pi/2.0))
        Coord[1, :] = A*w*np.cos(w*(x - pi/2.0))
        for e in range(nElements):
            Xa0,Xb0 = np.array([Coord[0,e],Coord[1,e],0]),np.array([Coord[0,e+1],Coord[1,e+1],0])
            elements.append(LinearEBBeam(Xa0, Xb0,E,r))


        # Essential bounary condition
        self.g = np.zeros([nDoF, nNodes])
        self.EBC = np.zeros([nDoF,nNodes],dtype='int')
        self.EBC[:,0] = 1

        # Force
        fx,fy,m = 0.0,1, 0.0
        self.f = np.zeros([nDoF, nNodes])
        self.f[:, -1] = fx, fy, m




    def assembly(self,d):
        '''
        :param u: displacement of all freedoms
        :return: dPi and Pi
        Pi = Ku - F + \sum f_c^i
        dPi = K + \sum df_c^i
        '''

        #Step 1: Access required global variables
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


        #Step 2: Allocate K,  F,  dP and ddP
        K = np.zeros([nEquations,nEquations])
        F = np.zeros(nEquations);


        #Step 3: Assemble K and F

        for e in range(nElements):

            [k_e,f_e,f_g] = self._linear_beam_arrays(e);
            #Step 3b: Get Global equation numbers
            P = LM[:,e]

            #Step 3c: Eliminate Essential DOFs
            I = (P >= 0);
            P = P[I];

            #Step 3d: Insert k_e, f_e, f_g, f_h
            K[np.ix_(P,P)] += k_e[np.ix_(I,I)]
            F[P] += f_e[I] + f_g[I]




        disp = np.empty([nDoF, nNodes])

        for i in range(nNodes):
            for j in range(nDoF):
                disp[j,i] = d[ID[j,i]] if EBC[j,i] == 0 else g[j,i]

        #Step 4: Allocate dP and ddP
        dP = np.zeros(nEquations)
        ddP = np.zeros([nEquations,nEquations])

        #Setp 5: Assemble K and F
        closest_e = -1
        g_min = np.inf

        for i in range(nWeft):
            xm, rm = wefts[0:2, i], wefts[2, i]
            for e in range(nElements):
                ele = elements[e]
                da, db = disp[:, e], disp[:, e + 1]
                contact, penalty, info = ele.penalty_term(da, db, xm, rm, wn)
                if (contact and info[1] < g_min):
                    closest_e, g_min = e, info[1]


            if (closest_e >= 0):
                ele = elements[closest_e]
                da, db = disp[:, closest_e], disp[:, closest_e + 1]
                contact, penalty, info = ele.penalty_term(da, db, xm, rm, wn)
                _, f_contact, k_contact = penalty
                # Step 3b: Get Global equation numbers
                P = LM[:, closest_e]

                # Step 3c: Eliminate Essential DOFs
                I = (P >= 0)
                P = P[I]

                # Step 3d: Insert k_e, f_e, f_g, f_h
                ddP[np.ix_(P, P)] += k_contact[np.ix_(I, I)]
                dP[P] += f_contact[I]



        dPi = np.dot(K,d) - F + dP
        ddPi = K + ddP

        return dPi, ddPi

    def _linear_beam_arrays(self,e):
        '''
        :param e:
        :return: k_e stiffmatrix, f_e f_g
        '''
        nNodesElement = self.nNodesElement
        nDoF = self.nDoF
        g = self.g
        f = self.f
        IEM = self.IEM
        ele = self.elements[e]
        k_e = ele.stiffmatrix()

        #Point force
        f_e = np.reshape(f[:,IEM[:,e]], (nNodesElement*nDoF), order='F')

        #Dirichlet boundary
        g_e = np.reshape(g[:,IEM[:,e]], (nNodesElement*nDoF), order='F')
        f_g = -np.dot(k_e,g_e)

        return k_e, f_e, f_g





    def fem_calc(self):
        nEquations = self.nEquations

        d = np.zeros(nEquations)

        dPi,ddPi = self.assembly(d)

        res0 = np.linalg.norm(dPi)

        MAXITE = 10000
        EPS = 1e-8
        found = False
        alpha = 0.05
        for ite in range(MAXITE):

            dPi,ddPi = self.assembly(d)
            d =  d - alpha*np.linalg.solve(ddPi,dPi)

            res = np.linalg.norm(dPi)
            print('In fem_calc res is', res)
            if(res < EPS or res < EPS*res0):
                found = True
                break

        if(not found):
            print("Newton cannot converge in fem_calc")

        return d

    def visualize_result(self, d, k=2):
        '''
        :param d: displacement of all freedoms
        :param k: visualize points for each beam elements
        '''
        ID = self.ID
        nDim = self.nDim
        nNodes = self.nNodes
        nDoF = self.nDoF
        nElements = self.nElements
        elements = self.elements
        EBC = self.EBC
        g = self.g
        disp = np.empty([nDoF, nNodes])

        for i in range(nNodes):
            for j in range(nDoF):
                disp[j,i] = d[ID[j,i]] if EBC[j,i] == 0 else g[j,i]

        coord_ref, coord_cur =  np.empty([nDim,(k - 1)*nElements + 1]), np.empty([nDim,(k - 1)*nElements + 1])
        for e in range(nElements):
            ele = elements[e]
            X0 , X = ele.visualize(disp[:,e],disp[:,e+1], k, fig = 0)
            coord_ref[:, (k-1)*e:(k-1)*(e+1) + 1] = X0
            coord_cur[:, (k-1)*e:(k-1)*(e+1) + 1] = X




        plt.plot(coord_ref[0,:], coord_ref[1,:], '-o', label='ref')
        plt.plot(coord_cur[0,:], coord_cur[1,:],'-o', label='current')

        wefts = self.wefts
        plt.plot(wefts[0, :], wefts[1, :], 'o', label='weft')
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

        return


if __name__ == "__main__":
    warp = Warp('horizontal beam')
    #warp.assembly()
    d = warp.fem_calc()
    warp.visualize_result(d,2)

