import numpy as np
import matplotlib.pyplot as plt

class LinearEBBeam:
    def __init__(self,Xa0, Xb0,E,r):
        '''
        :param Xa: initial position, x_a, y_a, theta_a
        :param Xb: initial position, x_b, y_b, theta_b
        :param E: Young's module
        :param r: radius
        :return:
        '''
        # Young's module
        self.E = E
        # radius
        self.r = r
        # Cross section area
        self.A = np.pi*r**2
        # Moment inertial
        self.I = np.pi*r**4 / 4.0


        # Initial condition
        self.Xa0 = Xa0
        self.Xb0 = Xb0
        # length
        self.L = L = np.sqrt((Xa0[0] - Xb0[0])**2 + (Xa0[1] - Xb0[1])**2)
        # local coordinate rotation angle
        self.s,self.c = (Xb0[1] - Xa0[1])/L, (Xb0[0] - Xa0[0])/L

    def basis(self, xi,d=0):
        '''
        This is the physics of the problem
        Local Basis
        initial is x \in [0,L]
        xi = 2x/L - 1 \in [-1,1]
        N1 = (1 - xi)/2                 x translation
        N2 = (1 - xi)^2*(2 + xi)/4      y translation
        N3 =  L*(1 - xi)^2*(1 + xi)/8   y bending
        N4 = (1 + xi)/2                 x translation
        N5 =  (1 + xi)^2*(2 - xi)/4     y translation
        N6 = -L*(1 + xi)^2*(1 - xi)/8   y bending
        u_x = u1 N1 + u4 N4
        u_x(-1) = u1,  u_x(1) = u4
        u_y = u2 N2 + u3 N3 + u5 N5 + u6 N6
        u_y(-1) = u2,  u_y(1) = u5, u_y'(-1) = u3*L/2, u_y'(1) =  u6*L/2,
        return value of 6 basis
        '''
        L = self.L
        B = np.array([[(1. - xi)/2.,0,0,(1 + xi)/2.,0.0,0.0],
                      [0., (1. - xi)**2*(2 + xi)/4.,  L*(1 - xi)**2*(1 + xi)/8.,0.0, (1 + xi)**2*(2. - xi)/4., -L*(1 + xi)**2*(1 - xi)/8.]])

        if d == 0:
            return B

        dB = np.array([[-1.0/2.0, 0,0,1/2.0,0.0,0.0],
              [0.0, (3*xi**2 - 3)/4.0, L*(3*xi**2 - 2*xi - 1)/8.,0.0,(-3*xi**2 + 3)/4.0,-L*(-3*xi**2 - 2*xi + 1)/8.0]])

        if d == 1:
            return B, dB

        ddB = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 3*xi/2.0, L*(3*xi - 1)/4.,0.0,-3*xi/2.0,L*(3*xi + 1)/4.]])

        if d == 2:
            return B, dB, ddB





    def stiffmatrix(self):
        '''
        return linear beam element's stiff matrix in the global coordinate
        '''

        L = self.L
        I = self.I
        E = self.E
        A = self.A
        Xa0, Xb0 = self.Xa0, self.Xb0

        #stiffness matrix in initial configuration
        K = np.array([[E*A/L,   0,          0,                -E*A/L,            0,             0],
                           [ 0,     12*E*I/L**3,  6*E*I/L**2,    0    ,  -12*E*I/L**3,     6*E*I/L**2],
                           [ 0,     6*E*I/L**2,   4*E*I/L,       0,      -6*E*I/L**2,      2*E*I/L],
                           [-E*A/L,   0,          0,             E*A/L,            0,             0],
                           [0,      -12*E*I/L**3, -6*E*I/L**2,   0    ,   12*E*I/L**3,    -6*E*I/L**2],
                           [0,      6*E*I/L**2,   2*E*I/L,       0,       -6*E*I/L**2,     4*E*I/L]])

        s,c = (Xb0[1] - Xa0[1])/L, (Xb0[0] - Xa0[0])/L
        R = np.array([[c, -s, 0,0,0,0],
        [s,c,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,c,-s,0],
        [0,0,0,s,c,0],
        [0,0,0,0,0,1]])
        # stiff matrix in reference configuration is  R*K*R^T
        return np.dot(R, np.dot(K,R.T))

    def _normal_in_local(self, d_, xi):
        '''
        compute left normal vector in the local coordinate
        :param d_: 6 entries array, local displacement
        :param xi: local coordinate
        '''

        x0_ = np.array([0.,0.,self.Xa0[2], self.L, 0.,self.Xb0[2]]) #initial position in local coordinates

        B, dB = self.basis(xi,d = 1)

        xs_, dxs_ = np.dot(B,x0_ + d_), np.dot(dB,x0_ + d_)

        et_ = dxs_ / np.linalg.norm(dxs_)

        en_ = np.array([-et_[1], et_[0]])

        return en_

    def _tangential_in_local(self, d_, xi):
        '''
        compute tangential vector in the local coordinate(tangent is in  XaXb direction)
        :param d_: 6 entries array, local displacement
        :param xi: local coordinate
        '''

        x0_ = np.array([0.,0.,self.Xa0[2], self.L, 0.,self.Xb0[2]]) #initial position in local coordinates

        B, dB = self.basis(xi,d = 1)

        xs_, dxs_ = np.dot(B,x0_ + d_), np.dot(dB,x0_ + d_)

        et_ = dxs_ / np.linalg.norm(dxs_)

        return et_




    def _distance(self, d_ , xi,  xm_, side):
        '''
        find the distance (xm0 - xs(xi))*en*side
        :param xi: beam position parameter
        :param x0_: initial position
        :param d_ : displacement
        :param xm_: master node relative position
        :param side: -1 right side is position distance;
                      1 left side is position distance
        :return:
        '''

        x0_ = np.array([0.,0.,self.Xa0[2], self.L, 0.,self.Xb0[2]]) #initial position of the master node

        B, dB = self.basis(xi,d = 1)

        xs_, dxs_ = np.dot(B,x0_ + d_), np.dot(dB, x0_ + d_)

        et = dxs_ / np.linalg.norm(dxs_)

        en = np.array([-et[1], et[0]])

        return side*np.dot(xm_ - xs_, en)

    def _rel_position_in_local(self,xm):
        Xa0 = self.Xa0
        s,c = self.s, self.c
        R = np.array([[c, s],[-s,c]])

        xm_ = np.dot(R, xm - Xa0[0:2])
        return xm_

    def _disp_in_local(self,da,db):
        s,c = self.s, self.c
        R = np.array([[c, s],[-s,c]])

        #master node, displacement in local coordinates
        d_ = np.zeros(6)
        d_[0:2] = np.dot(R, da[0:2])
        d_[2] = da[-1]
        d_[3:5] = np.dot(R, db[0:2])
        d_[5] = db[-1]

        return d_

    def penalty_term(self,da,db,xm,rm,wn,side):
        '''
        compute the closest point distance in initial configuration, gn is the signed distance
        gn = side*(xm -xs)*en
        :param da: beam current left node displacement x,y,theta
        :param db: beam current left node displacement x,y,theta
        :param xm: rigid master node position
        :param side: master node should on the side(1 left -1 right) of the beam
        if side = 1, left is positive right is the wall
        if side = -1, right is positive, left is the wall
        :return: bool: true, find contact, false not find
                 a tuple, including penalty function information
                    P: 1/2*wn*(gn - rm - r)**2 if gn - rm -r <0  else 0
                    f: contact force dP if P > 0
                    K: Hessian matrix ddP, if P > 0
                 a tuple, including contact point information
                    xi_c: closest node local coordinate
                    gn: signed distance
        '''
        Xa0, Xb0, L = self.Xa0, self.Xb0, self.L
        s,c = (Xb0[1] - Xa0[1])/L, (Xb0[0] - Xa0[0])/L


        d_ = self._disp_in_local(da, db)

        xm_ = self._rel_position_in_local(xm)

        x0_ = np.array([0.,0.,Xa0[2], L, 0.,Xb0[2]]) #initial position of the master node
        #compute closest points by Newton iteration

        MAXITE = 50
        EPS = 1e-14
        xi_c = 0.5
        found = False
        B, dB = self.basis(xi_c,d = 1)
        xs_,dxs_ = np.dot(B,x0_) + np.dot(B,d_), np.dot(dB,d_)
        f0 = np.fabs(np.dot(xm_ - xs_, dxs_))

        for ite in range(MAXITE):
            B, dB, ddB = self.basis(xi_c,d = 2)
            xs_,dxs_,ddxs_ = np.dot(B,x0_ + d_), np.dot(dB,x0_ + d_), np.dot(ddB,x0_ + d_)
            f = np.dot(xm_ - xs_, dxs_)
            df = np.dot(xm_ - xs_, ddxs_) - np.dot(dxs_, dxs_)
            xi_c =  xi_c - f/df

            if(np.fabs(f) < EPS or np.fabs(f) < EPS*f0):
                found = True
                break

        if(not found):
            print("Newton cannot converge in finding closest points")




        if(xi_c > 1 or xi_c < -1 or not found):
            print("found closest points but xi_c is", xi_c)
            dist_a = self._distance(d_, -1.0, xm_, side)
            dist_b = self._distance(d_,  1.0, xm_, side)
            xi_c = -1.0 if(dist_a < dist_b) else 1.0




        #compute contact force

        gn = self._distance(d_, xi_c,xm_,side)
        P = wn*(gn - rm - self.r)**2/2.0

        if(gn < rm + self.r):
            #compute normal penalty force in local coordinate
            #f = -wm*gn*en
            en_ = self._normal_in_local(d_, xi_c)
            et_ = self._tangential_in_local(d_, xi_c)
            B,dB,ddB = self.basis(xi_c,d = 2)
            f_ = - wn * (gn - rm - self.r)*np.dot(en_, B)*side


            #compute Kn_
            xs_,dxs_, ddxs_ = np.dot(B,x0_ + d_), np.dot(dB,x0_ + d_), np.dot(ddB,x0_ + d_)
            dxi_ = (-np.dot(dxs_,B) + np.dot(xm_ - xs_,dB))/(np.dot(dxs_, dxs_) - np.dot(xm_ - xs_, ddxs_)) #1 by 6

            den_ = -np.outer(et_, np.dot(en_, dB) + np.dot(en_,ddxs_)*dxi_)/np.linalg.norm(dxs_) # 2 by 6

            enTB_ = np.dot(en_, B)

            K_ = wn*(np.outer(enTB_, enTB_) - side*(gn- rm - self.r)*(np.dot(B.T ,den_) + np.outer(np.dot(en_,dB), dxi_))) # 6 by 6


            R = np.array([[c, -s, 0,0,0,0],
                          [s,c,0,0,0,0],
                          [0,0,1,0,0,0],
                          [0,0,0,c,-s,0],
                          [0,0,0,s,c,0],
                          [0,0,0,0,0,1]])
            f = np.dot(R,f_)
            K = np.dot(R, np.dot(K_, R.T))
            return True, (P ,f, K), (xi_c, gn)

        else:
            return False, (), ()


    def visualize(self,da,db,k = 2, fig = 1):
        Xa0, Xb0 = self.Xa0, self.Xb0
        X0 = np.vstack([np.linspace(Xa0[0],Xb0[0],k),np.linspace(Xa0[1],Xb0[1],k)])
        #plot initial configuration
        da0,db0 = np.array([0,0,Xa0[-1]]), np.array([0,0,Xb0[-1]])
        u0 = self.visualize_helper(da0,db0,k)
        plt.figure(fig)
        plt.plot(X0[0,:]+u0[0,:],X0[1,:]+u0[1,:],label='ref')
        u = self.visualize_helper(da,db,k)
        plt.plot(X0[0,:]+u[0,:],X0[1,:]+u[1,:],label='current')



    def visualize_helper(self,da, db, k):
        c,s = self.c, self.s
        R = np.array([[c, -s],
                      [s, c]])

        d_= self._disp_in_local(da, db)
        xi = np.linspace(-1.0,1.0,k)
        u_ = np.zeros([2,k])
        u = np.zeros([2,k])
        for i in range(k):
            B = self.basis(xi[i],d=0)
            u_[:,i] = np.dot(B,d_)
            u = np.dot(R,u_)
        return u




def test_basis():

    Xa0 = np.array([0.0,0.0,0.0])
    Xb0 = np.array([1.0,1.0,0.0])
    E = 1.0
    r = 0.1
    myBeam = LinearEBBeam(Xa0, Xb0,E,r)
    xi = -1.0
    EPS = 0.001
    B, dB, ddB = myBeam.basis(xi,d = 2)
    Bp, dBp = myBeam.basis(xi + EPS,d = 1)
    Bm, dBm = myBeam.basis(xi - EPS,d = 1)
    print('Basis test\n', Bp - Bm - 2*EPS*dB,'\n', dBp - dBm - 2*EPS*ddB)

def test_visualization():
    Xa0 = np.array([0.0,0.0,0.0])
    Xb0 = np.array([1.0,1.0,0.0])
    E = 1.0
    r = 0.1
    myBeam = LinearEBBeam(Xa0, Xb0,E,r)
    d = np.array([0.2, 0.1, 0.8, 0., 0., 0.3])
    da = d[0:3]
    db = d[3:6]
    myBeam.visualize(da,db, k = 10,fig = 1)
    plt.show()

def test_derivative():
    Xa0 = np.array([0.0,0.0,0.0])
    Xb0 = np.array([1.0,0.0,0.0])
    E = 1.0
    r = 0.1
    myBeam = LinearEBBeam(Xa0, Xb0, E, r)
    d = np.array([0.0, 0.3, 0.1, 0.0, 0.5, 0.5])
    da = d[0:3]
    db = d[3:6]

    myBeam.visualize(da,db, k = 10,fig = 1)
    plt.show()

    xm = np.array([2.5, 0.5])
    side = 1
    wn = 1.0
    success, penalty, info  = myBeam.penalty_term(da, db, xm, r, wn, side)
    if not success:
        return
    P, f, K = penalty
    print('contact point local coordinate is', info[0], 'signed distance is ', info[1])

    EPS= 0.01



    #finite difference test
    #np.dot(f, dp-dm) = (wn*gn*gn/2.0)_p - (wn*gn*gn/2.0)_m
    #fp - fm = np.dot(K,dp-dm)
    eps = np.array([EPS, EPS,EPS,EPS,EPS,EPS])
    dp = d + eps
    dap = dp[0:3]
    dbp = dp[3:6]

    success, penalty_p, info_p = myBeam.penalty_term(dap,dbp,xm,r, wn, side)
    if not success:
        return
    Pp, fp, Kp = penalty_p

    dm = d - eps
    dam = dm[0:3]
    dbm = dm[3:6]
    success, penalty_m, info_m = myBeam.penalty_term(dam,dbm,xm,r, wn, side)
    if not success:
        return
    Pm, fm, Km = penalty_m

    error_dP =  Pp -Pm - np.dot(f, dp-dm)
    error_ddP = fp - fm - np.dot(K,dp-dm)

    np.set_printoptions(precision=16)
    print('first derivative error of Penalty is ', error_dP)
    print('second derivative error of Penalty is ', error_ddP)









if __name__ == "__main__":
    test_derivative()

















