import numpy as np


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
        self.L = np.sqrt((Xa0[0] - Xb0[0])**2 + (Xa0[1] - Xb0[1])**2)
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
        return linear beam element's stiff matrix
        :param E : Young's module
        :param A : cross section area
        :param I : moment inertial
        :param Xa: start point at initial configuration
        :param Xb: end point at initial configuration
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
        return np.dot(R, np.dot(K,R.transpose()))

    def _normal_in_local(self, d_, xi, side):
        '''
        compute normal in global coordinate
        :param xi: local coordinate
        :param side: -1 right normal, 1 left normal
        :return:
        '''

        x0_ = np.array([0.,0.,self.Xa0[2], self.L, 0.,self.Xb0[2]]) #initial position in local coordinates

        B, dB = self.basis(xi,d = 1)

        xs_, dxs_ = np.dot(B,x0_ + d_), np.dot(dB,x0_ + d_)

        et_ = dxs_ / np.linalg.norm(dxs_)

        en_ = side*np.array([-et_[1], et_[0]])

        return en_

    def _tangential_in_local(self, d_, xi, side):
        '''
        compute normal in global coordinate
        :param xi: local coordinate
        :param side: -1 right normal, 1 left normal
        :return:
        '''

        x0_ = np.array([0.,0.,self.Xa0[2], self.L, 0.,self.Xb0[2]]) #initial position in local coordinates

        B, dB = self.basis(xi,d = 1)

        xs_, dxs_ = np.dot(B,x0_ + d_), np.dot(dB,x0_ + d_)

        et_ = dxs_ / np.linalg.norm(dxs_)

        return side*et_




    def _distance(self, d_ , xi,  xm_, side):
        '''
        find the distance (xm0 - xs(xi))*en
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
        L = self.L
        Xa0, Xb0 = self.Xa0, self.Xb0
        s,c = (Xb0[1] - Xa0[1])/L, (Xb0[0] - Xa0[0])/L
        R = np.array([[c, -s],[s,c]])

        xm_ = np.dot(R, xm - Xa0[0:2])
        return xm_

    def _disp_in_local(self,da,db):
        L = self.L
        Xa0, Xb0 = self.Xa0, self.Xb0
        s,c = (Xb0[1] - Xa0[1])/L, (Xb0[0] - Xa0[0])/L
        R = np.array([[c, -s],[s,c]])

        #master node, displacement in local coordinates
        d_ = np.zeros(6)
        d_[0:2] = np.dot(R, da[0:2])
        d_[2] = da[-1]
        d_[3:5] = np.dot(R, db[0:2])
        d_[5] = db[-1]

        return d_

    def closest_points_distance(self,da,db,xm,rm,wn,side):
        '''
        compute the closest point distance in initial configuration, Xa is the original point
        :param Xa: beam initial left node position x,y,theta
        :param Xb: beam initial right node position x,y,theta
        :param xa: beam current left node displacement x,y,theta
        :param xb: beam current left node displacement x,y,theta
        :param xm: rigid master node position
        :param side: master node should on the side(-1 right 1 left) of the beam
        :return: gn
        '''
        Xa0, Xb0, L = self.Xa0, self.Xb0, self.L
        s,c = (Xb0[1] - Xa0[1])/L, (Xb0[0] - Xa0[0])/L
        R = np.array([[c, -s, 0,0,0,0],
        [s,c,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,c,-s,0],
        [0,0,0,s,c,0],
        [0,0,0,0,0,1]])

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
            dist_a = self._distance(-1.0, d_,xm_,side)
            dist_b = self._distance(1.0, d_,xm_,side)
            xi_c = -1.0 if(dist_a < dist_b) else 1.0




        #compute contact force

        gn = self._distance(d_, xi_c,xm_,side)

        if(gn < rm + self.r):
            #compute normal penalty force in local coordinate
            #f = -wm*gn*en
            en_ = self._normal_in_local(d_, xi_c, side)
            et_ = self._tangential_in_local(d_, xi_c, side)
            B,dB,ddB = self.basis(xi_c,d = 2)
            f_ = - wn * (gn - rm - self.r)*np.dot(en_, B)
            #f = np.dot(R,f_)

            #compute Kn_
            xs_,dxs_, ddxs_ = np.dot(B,x0_ + d_), np.dot(dB,x0_ + d_), np.dot(ddB,x0_ + d_)
            dxi_ = (-np.dot(dxs_,B) + np.dot(xm_ - xs_,dB))/(np.dot(dxs_, dxs_) - np.dot(xm_ - xs_, ddxs_)) #1 by 6

            den_ = -np.outer(et_, np.dot(en_, dB) + np.dot(en_,ddxs_)*dxi_)/np.linalg.norm(dxs_) # 2 by 6

            enTB_ = np.dot(en_, B)
            print('enTB is ',enTB_)
            K = wn*(np.outer(enTB_, enTB_) - (gn- rm - self.r)*(np.dot(B.T ,den_) + np.outer(np.dot(en_,dB), dxi_))) # 6 by 6

            return xi_c, gn ,f_, K , dxi_,  en_, den_, xs_








if __name__ == "__main__":
    '''
    Test ClosestPointsDistance(Xa,Xb,da,db,xm,side):
    horizontal beam
    slanting beam
    '''

    Xa0 = np.array([0.0,0.0,0.0])
    Xb0 = np.array([1.0,0.0,0.0])
    E = 1.0
    r = 0.1
    horizontalBeam = LinearEBBeam(Xa0, Xb0,E,r)



    d = np.array([0.2, 0.1, 0.3, 1., 0.4, 0.5])
    da = d[0:3]
    db = d[3:6]

    xm = np.array([1.0, 0.1])
    side = 1
    wn = 1.0
    xi_c, gn ,f, K, dxi, en, den,x =horizontalBeam.closest_points_distance(da,db,xm,r, wn, side)

    #todo add rotaion



    EPS= 0.001



    #finite difference test
    #np.dot(f, dp-dm) = (wn*gn*gn/2.0)_p - (wn*gn*gn/2.0)_m
    #fp - fm = np.dot(K,dp-dm)
    eps = np.array([EPS, EPS,EPS,EPS,EPS,EPS])
    dp = d + eps
    dap = dp[0:3]
    dbp = dp[3:6]

    xi_cp, gnp ,fp, Kp, dxi_p,  en_p, den_p,x_p =horizontalBeam.closest_points_distance(dap,dbp,xm,r, wn, side)


    dm = d - eps
    dam = dm[0:3]
    dbm = dm[3:6]
    xi_cm, gnm ,fm, Km, dxi_m,  en_m, den_m,x_m =horizontalBeam.closest_points_distance(dam,dbm,xm,r, wn, side)


    error_dxi = (xi_cp - xi_cm - np.dot(dxi, dp - dm))
    error_den = (en_p - en_m - np.dot(den, dp - dm))
    error_dP = (wn*(gnp - 2*r)**2/2.0) - (wn*(gnm-2*r)**2/2.0) - np.dot(f, dp-dm)
    error_ddP = fp - fm - np.dot(K,dp-dm)

    np.set_printoptions(precision=16)
    print('first derivative error of dxi is ', error_dxi)
    print('first derivative error of den is ', error_den)
    print('first derivative error of Penalty is ', error_dP)
    print('second derivative error of Penalty is ', error_ddP)

    #basis test
    '''
    xi = -1.0
    B, dB, ddB = horizontalBeam.basis(xi,d = 2)
    Bp, dBp = horizontalBeam.basis(xi + EPS,d = 1)
    Bm, dBm = horizontalBeam.basis(xi - EPS,d = 1)
    print('Basis test\n', Bp - Bm - 2*EPS*dB,'\n', dBp - dBm - 2*EPS*ddB)
    '''




















