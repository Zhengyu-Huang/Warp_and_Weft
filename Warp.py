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

        E = 1.0
        r = 1.0
        self.Coord = Coord = np.zeros([nDim, nNodes])
        Coord[0,:] = np.linspace(0,1.0,nNodes)

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

    def assembly(self):

        #Step 1: Access required global variables
        nElements = self.nElements
        nEquations = self.nEquations
        nDoF = self.nDoF
        ID = self.ID
        LM = self.LM


        #Step 2: Allocate K and F
        K = np.zeros([nEquations,nEquations])
        F = np.zeros([nEquations,1]);

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

        return K,F

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
        f_e = np.reshape(f[:,IEM[:,e]], (nNodesElement*nDoF,1), order='F')

        #Dirichlet boundary
        g_e = np.reshape(g[:,IEM[:,e]], (nNodesElement*nDoF,1), order='F')
        f_g = -np.dot(k_e,g_e)

        return k_e, f_e, f_g

    def _closest_node(self, xm):
        '''
        find the closest node on the beams
        :param xm: master node
        :return: g_n, f_n, K_n
        '''
        nNodes = self.nNodes
        nElements = self.nElements
        for e in range(nElements):
            return




    def fem_calc(self):
        K,F = self.assembly()
        d = np.linalg.solve(K,F)
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

