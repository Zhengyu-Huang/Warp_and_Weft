import numpy as np
from Beam import *
import matplotlib.pyplot as plt

class Warp:
    def __init__(self):

        '''
                      Name            Description
                    -------         --------------
                    nDim            Number of Spatial Dimensions
                    nNodes          Number of nodes in the mesh
                    nElements       Number of elements in the mesh
                    nNodesElement   Number of nodes per element
                    nDoF            Number of DoF per node
                    nEquations      Number of equations to solve
                    Coord           Array of nodal coordinates
                    ID              Array of global eq. numbers, destination array (ID)
                    EBC             EBC = 1 if DOF is on Essential B.C.
                    IEN             Array of global node numbers
                    LM              Array of global eq. numbers, location matrix (LM)
                    C               Material Constant at each element
                    f               Distributed Load at each node, an array(nDof, nNodes)
                    g               Essential B.C. Value at each node is an array(nDoF, nNodes)
                    h               Natural B.C. Value at each node
        '''

        self._horizontal_beam_data()




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

        self.nDim = nDim = 2
        self.nDoF = nDoF = 3
        self.nNodesElement = 2

        self.nElements = nElements = 5

        self.nNodes = nNodes = self.nElements + 1
        self.nEquations = self.nDoF * (self.nNodes - 1)


        # Young's module
        self.E = np.empty(nElements)
        self.E.fill(1.0)

        # Cross section area
        self.A = np.empty(nElements)
        self.A.fill(1.0)

        # Moment inertial
        self.I = np.empty(nElements)
        self.I.fill(1.0)

        # Initial condition
        self.Coord = np.zeros([nDim,nNodes])
        self.Coord[0,:] = np.linspace(0,1.0,nNodes)

        # Essential bounary condition
        self.g = np.zeros([nDoF, nNodes])
        self.EBC = np.zeros([nDoF,nNodes],dtype='int')
        self.EBC[:,0] = 1

        # Force
        fx,fy,m = 1.0, 0.0, 0.0
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
            print(I,P)
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
        E,A,I,Xa,Xb = self.E[e],self.A[e],self.I[e],self.Coord[:,IEM[0,e]],self.Coord[:,IEM[1,e]]
        k_e = LinearBeam(E,A,I,Xa,Xb)

        #Point force
        f_e = np.reshape(f[:,IEM[:,e]], (nNodesElement*nDoF,1), order='F')

        #Dirichlet boundary
        g_e = np.reshape(g[:,IEM[:,e]], (nNodesElement*nDoF,1), order='F')
        f_g = -np.dot(k_e,g_e)

        return k_e, f_e, f_g

    def fem_calc(self):
        K,F = self.assembly()
        d = np.linalg.solve(K,F)
        self.visualize_result(d,False)
    def visualize_result(self, d, highOrder = True):
        ID = self.ID
        nNodes = self.nNodes
        EBC = self.EBC
        g = self.g
        if(highOrder):
            return
        else:

            x,y = np.empty(nNodes),np.empty(nNodes)

            for i in range(nNodes):
                print(ID[0,i],ID[1,i])
                x[i] = self.Coord[0, i]  +  d[ID[0,i]] if EBC[0,i] == 0 else g[0,i]
                y[i] = self.Coord[1, i]  +  d[ID[1,i]] if EBC[1,i] == 0 else g[1,i]

            plt.plot(x,y,'-o',label='solution')
            plt.plot(self.Coord[0,:], self.Coord[1,:],'-o', label='reference')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()

        return


if __name__ == "__main__":
    warp = Warp()
    #warp.assembly()
    warp.fem_calc()

