import numpy as np
from Beam import *
import matplotlib.pyplot as plt

class Warp:
    def __init__(self,type,par,wn,k, MAXITE):

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
        self.type = type
        self.par = par
        #Penalty parameters
        self.wn = wn
        self.MAXITE = MAXITE
        self.k = k
        self.nDim = nDim = 2
        self.nDoF = nDoF = 3
        self.nNodesElement = 2

        #build elements

        if(type == 'straight beam'):
            self._straight_beam_data( )
        elif(type == 'sine beam0'):
            self._sine_beam_data0( )
        elif(type == 'sine beam'):
            self._sine_beam_data( )

        self.nEquations = (self.EBC == 0).sum()


        #construct  element nodes array
        #IEM(i,e) is the global node id of element e's node i
        self.IEM = np.zeros([self.nNodesElement, self.nElements], dtype = 'int')
        self.IEM[0,:] = np.arange(self.nElements)
        self.IEM[1,:] = np.arange(1, self.nElements + 1)

        #construct destination array
        #ID(d,n) is the global equation number of node n's dth freedom, -1 means no freedom
        self.ID = np.zeros([self.nDoF, self.nNodes],dtype = 'int') - 1
        eq_id = 0
        for i in range(self.nNodes):
            for j in range(self.nDoF):
                if(self.EBC[j,i] == 0):
                    self.ID[j,i] = eq_id
                    eq_id += 1
        #construct Local matrix
        #LM(d,e) is the global equation number of element e's d th freedom
        self.LM = np.zeros([self.nNodesElement*self.nDoF, self.nElements],dtype = 'int')
        for i in range(self.nDoF):
            for j in range(self.nNodesElement):
                for k in range(self.nElements):
                    self.LM[j*self.nDoF + i, k] = self.ID[i,self.IEM[j,k]]


        #contact information
        self.contact_dist = contact_info = np.empty(self.nElements)




    def _straight_beam_data(self ):
        '''
        g is dirichlet boundary condition
        f is the internal force
        '''
        nDoF = self.nDoF
        nDim = self.nDim




        self.nElements = nElements = 5
        self.elements = elements = []
        self.nNodes = nNodes = self.nElements + 1


        E = 1.0e4
        r = 0.1
        self.Coord = Coord = np.zeros([nDoF, nNodes])
        Coord[0,:] = np.linspace(0,1.0,nNodes)
        Coord[1,:] = np.linspace(0,1.0,nNodes)
        for e in range(nElements):
            Xa0,Xb0 = np.array([Coord[0,e],Coord[1,e],Coord[2,e]]),np.array([Coord[0,e+1],Coord[1,e+1],Coord[2,e]])
            elements.append(LinearEBBeam(e, Xa0, Xb0,E,r))


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



    def _sine_beam_data0(self ):
        '''
        g is dirichlet boundary condition
        f is the internal force
        '''
        nDoF = self.nDoF
        nDim = self.nDim




        self.nElements = nElements = 50
        self.elements = elements = []
        self.nNodes = nNodes = self.nElements + 1




        #Young's module
        E = 1.0e8
        #beam radius
        self.r = r = 0.02
        #The curve is h*sin(k*x - pi/2.0)
        k = 3
        h = 0.1
        self.Coord = Coord = np.zeros([nDoF, nNodes])
        Coord[0, :] = np.linspace(0, 2*np.pi, nNodes)         # x
        Coord[1, :] = h*np.sin(k*Coord[0, :] - np.pi/2.0) + h   # y
        Coord[2, :] = h*k*np.cos(k*Coord[0, :] - np.pi/2.0) # rotation theta
        for e in range(nElements):
            Xa0,Xb0 = np.array([Coord[0,e],Coord[1,e],Coord[2, e]]),np.array([Coord[0,e+1],Coord[1,e+1], Coord[2, e+1]])
            elements.append(LinearEBBeam(e,Xa0, Xb0,E,r))

        # no wefts

        self.nWeft = nWeft = 0
        self.wefts = np.zeros([nDim + 1, nWeft])  # (x,y,r)

        # Penalty parameters
        self.wn = 1e7



        # Essential bounary condition
        self.g = np.zeros([nDoF, nNodes])
        self.EBC = np.zeros([nDoF,nNodes],dtype='int')
        self.EBC[:,0] = 1

        self.EBC[:,-1] = 1

        self.g[:,-1] = self.par

        # Force
        #fx,fy,m = 0.1, -0.1, 0.0
        self.f = np.zeros([nDoF, nNodes])
        #self.f[:, -1] = fx, fy, m


    def _sine_beam_data(self ):
        '''
        g is dirichlet boundary condition
        f is the internal force
        '''
        nDoF = self.nDoF
        nDim = self.nDim




        self.nElements = nElements = 50
        self.elements = elements = []
        self.nNodes = nNodes = self.nElements + 1




        #Young's module
        E = 1.0e8
        #beam radius
        self.r = r = 0.02
        #The curve is h*sin(k*x - pi/2.0)
        k = self.k
        h = 0.1
        self.Coord = Coord = np.zeros([nDoF, nNodes])
        Coord[0, :] = np.linspace(0, 2*np.pi, nNodes)         # x
        Coord[1, :] = h*np.sin(k*Coord[0, :] - np.pi/2.0) + h   # y
        Coord[2, :] = h*k*np.cos(k*Coord[0, :] - np.pi/2.0) # rotation theta
        for e in range(nElements):
            Xa0,Xb0 = np.array([Coord[0,e],Coord[1,e],Coord[2, e]]),np.array([Coord[0,e+1],Coord[1,e+1], Coord[2, e+1]])
            elements.append(LinearEBBeam(e, Xa0, Xb0,E,r))

        # Weft info
        rWeft = r
        self.nWeft = nWeft = 2*k-1
        self.wefts = wefts = np.zeros([nDim + 1, nWeft])  # (x,y,r)
        for i in range(nWeft):
            wefts[:,i] = np.pi*(i+1.0)/k, h, rWeft






        # Essential bounary condition
        self.g = np.zeros([nDoF, nNodes])
        self.EBC = np.zeros([nDoF,nNodes],dtype='int')
        self.EBC[:,0] = 1

        self.EBC[:,-1] = 1

        self.g[:,-1] = self.par

        # Force
        #fx,fy,m = 0.1, -0.1, 0.0
        self.f = np.zeros([nDoF, nNodes])
        #self.f[:, -1] = fx, fy, m
        #self.f[:, nElements//2] = fx, fy, m


    def reset_par(self,par):
        self.par = par
        self.g[:,-1] = self.par

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



        dPi = np.dot(K,d) - F + dP
        ddPi = K + ddP
        return dPi, ddPi


    def compute_force(self,d):
        '''
        :param u: displacement of all freedoms
        :return: return the force at each Dirichlet freedom
        F_total = Ku - F + \sum f_c^i
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
        K = np.zeros([nDoF*nNodes,nDoF*nNodes])
        F = np.zeros(nDoF*nNodes);


        #Step 3: Assemble K and F

        for e in range(nElements):



            [k_e,f_e,f_g] = self._linear_beam_arrays(e);
            #Step 3b: Get Global equation numbers
            P = np.arange(e*nDoF,(e+2)*nDoF)


            #Step 3d: Insert k_e, f_e, f_g, f_h
            K[np.ix_(P,P)] += k_e

            #Step 3b: Get Global equation numbers


            #Step 3c: Eliminate Essential DOFs
            I = (LM[:,e] >= 0);
            P = P[I];

            F[P] += f_e[I]




        disp = np.empty([nDoF, nNodes])

        for i in range(nNodes):
            for j in range(nDoF):
                disp[j,i] = d[ID[j,i]] if EBC[j,i] == 0 else g[j,i]

        #Step 4: Allocate dP and ddP
        dP = np.zeros(nDoF*nNodes)

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
                P = np.arange(closest_e*nDoF,(closest_e+2)*nDoF)



                # Step 3d: Insert k_e, f_e, f_g, f_h
                dP[P] += f_contact



        F_total = np.dot(K,disp.flatten('F')) - F + dP


        return F_total[(EBC==1).flatten('F')]


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


    def compute_gap_lower_bound(self):
        nEquations = self.nEquations

        nElements = self.nElements

        nNodesElements = self.nNodesElement

        nDoF = self.nDoF

        r = self.r

        LM = self.LM

        gap_lower_bound = np.empty(nEquations)
        gap_lower_bound.fill(r)

        contact_dist = self.contact_dist

        for e in range(nElements):
            if contact_dist[e] < 2*r:
                e_dist = 2*r  - contact_dist[e]

                for i in range(nNodesElements):
                    for j in range(nDoF):
                        eq_id = LM[i*nDoF + j,e]
                        if eq_id >= 0:
                            gap_lower_bound[eq_id] =  min(gap_lower_bound[eq_id], e_dist)


        return gap_lower_bound

    def fem_calc(self):
        nEquations = self.nEquations


        nDoF = self.nDoF

        u = np.zeros(nEquations)

        dPi,ddPi = self.assembly(u)

        res0 = np.linalg.norm(dPi)

        MAXITE = self.MAXITE
        EPS = 1e-8
        found = False
        dt_max = np.empty(nEquations)
        dt_max.fill(0.5)
        T = 0
        for ite in range(MAXITE):
            print(self.wn)

            dPi,ddPi = self.assembly(u)

            res = np.linalg.norm(dPi)

            du = np.linalg.solve(ddPi,dPi)

            ################################
            # Time stepping
            ###############################

            du_abs = np.repeat(np.sqrt(du[0:-1:nDoF]**2 + du[1:-1:nDoF]**2) + 1e-12, nDoF)

            gap_lower_bound = self.compute_gap_lower_bound()

            if(ite < 1000):
                dt = min(dt_max[0], self.r/np.max(du_abs)/10.0)

            else:
                dt = np.min(np.minimum(dt_max, gap_lower_bound/du_abs))
            u =  u - dt*du

            #print('dPi ', np.reshape(dPi,(3,-1),order='F'))
            #print('du is ', np.reshape(du,(3,-1),order='F'))


            print('Ite/MAXITE: ', ite, ' /', MAXITE, 'In fem_calc res is', res,' dt is ', dt )
            if(res < EPS):# or res < EPS*res0):
                found = True
                break
            T += dt
        if(not found):
            print("Newton cannot converge in fem_calc")
        print('T is ', T)
        return u,res

    def visualize_result(self, u, k=2):
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
                disp[j,i] = u[ID[j,i]] if EBC[j,i] == 0 else g[j,i]

        coord_ref, coord_cur =  np.empty([nDim,(k - 1)*nElements + 1]), np.empty([nDim,(k - 1)*nElements + 1])
        for e in range(nElements):
            ele = elements[e]
            X0 , X = ele.visualize(disp[:,e],disp[:,e+1], k, fig = 0)
            coord_ref[:, (k-1)*e:(k-1)*(e+1) + 1] = X0
            coord_cur[:, (k-1)*e:(k-1)*(e+1) + 1] = X




        plt.plot(coord_ref[0,:], coord_ref[1,:], '-o', label='ref',markersize = 2)
        plt.plot(coord_cur[0,:], coord_cur[1,:],'-o', label='current',markersize = 2)

        wefts = self.wefts
        plt.plot(wefts[0, :], wefts[1, :], 'o', label='weft',markersize = 2)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

        return


if __name__ == "__main__":
    u_x,u_y,theta = -0.1,-0.1,0.0
    wn = 1e6
    MAXITE = 2000
    k = 3
    warp = Warp('sine beam',[u_x,u_y,theta],wn,k, MAXITE)
    #warp.assembly()
    d,res = warp.fem_calc()


    f = warp.compute_force(d)
    print('Dirichlet freedom force is ', f)

    warp.visualize_result(d,2)
