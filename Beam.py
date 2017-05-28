import numpy as np



def LinearBeamBasis(xi,L,d=0):
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





def LinearBeamStiffMatrix(E,A,I,Xa,Xb):
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

    L = np.sqrt((Xa[0] - Xb[0])**2 + (Xa[1] - Xb[1])**2)

    #stiffness matrix in initial configuration
    K = np.array([[E*A/L,   0,          0,                -E*A/L,            0,             0],
                       [ 0,     12*E*I/L**3,  6*E*I/L**2,    0    ,  -12*E*I/L**3,     6*E*I/L**2],
                       [ 0,     6*E*I/L**2,   4*E*I/L,       0,      -6*E*I/L**2,      2*E*I/L],
                       [-E*A/L,   0,          0,             E*A/L,            0,             0],
                       [0,      -12*E*I/L**3, -6*E*I/L**2,   0    ,   12*E*I/L**3,    -6*E*I/L**2],
                       [0,      6*E*I/L**2,   2*E*I/L,       0,       -6*E*I/L**2,     4*E*I/L]])

    s,c = (Xb[1] - Xa[1])/L, (Xb[0] - Xa[0])/L
    R = np.array([[c, -s, 0,0,0,0],
    [s,c,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,c,-s,0],
    [0,0,0,s,c,0],
    [0,0,0,0,0,1]])
    # stiff matrix in reference configuration is  R*K*R^T
    return np.dot(R, np.dot(K,R.transpose()))

def Distance(xi, x0_, d_, xm_, side):
    '''
    find the distance (xm0 - xs(xi))*en
    :param xi: beam position parameter
    :param x0_: initial position
    :param d_ : displacement
    :param xm_: master node
    :param side: -1 right side is position distance;
                  1 left side is position distance
    :return:
    '''
    L = x0_[3]

    B, dB = LinearBeamBasis(xi,L,d = 1)

    xs_, dxs_ = np.dot(B,x0_) + np.dot(B,d_), np.dot(dB,d_)

    et = dxs_ / np.linalg.norm(dxs_)

    en = np.array([-et[1], et[0]])

    return side*np.dot(xm_ - xs_, en)


def ClosestPointsDistance(Xa,Xb,da,db,xm,side):
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
    L = np.sqrt((Xa[0] - Xb[0])**2 + (Xa[1] - Xb[1])**2)
    s,c = (Xb[1] - Xa[1])/L, (Xb[0] - Xa[0])/L
    R = np.array\
        ([[c, -s],
        [s,c]])

    #master node, displacement in initial configuration
    xm_ = np.dot(R, xm - Xa[0:2])
    d_ = np.zeros(6)
    d_[0:2] = np.dot(R, da[0:2])
    d_[2] = da[-1]
    d_[3:5] = np.dot(R, db[0:2])
    d_[5] = db[-1]
    x0_ = np.array([0.,0.,Xa[2], L, 0.,Xb[2]]) #initial position of the master node
    #compute closest points by Newton iteration

    MAXITE = 50
    EPS = 1e-8
    xi_c = 0.5
    found = False
    B, dB = LinearBeamBasis(xi_c,L,d = 1)
    xs_,dxs_ = np.dot(B,x0_) + np.dot(B,d_), np.dot(dB,d_)
    f0 = np.fabs(np.dot(xm_ - xs_, dxs_))

    for ite in range(MAXITE):
        B, dB, ddB = LinearBeamBasis(xi_c,L,d = 2)
        xs_,dxs_,ddxs_ = np.dot(B,x0_ + d_), np.dot(dB,x0_ + d_), np.dot(ddB,x0_ + d_)
        f = np.dot(xm_ - xs_, dxs_)
        df = np.dot(xm_ - xs_, ddxs_) - np.dot(dxs_, dxs_)
        xi_c =  xi_c - f/df

        if(np.fabs(f) < EPS or np.fabs(f) < EPS*f0):
            found = True
            break

    if(not found):
        print("Newton cannot converge in finding closest points")




    if(xi_c <= 1 and xi_c >= -1 and found):
        return xi_c, Distance(xi_c,x0_, d_, xm_,side)
    else:
        print("found closest points but xi_c is", xi_c)
        dist_a = Distance(-1.0,x0_,d_,xm_,side)
        dist_b = Distance(1.0,x0_, d_,xm_,side)
        if(dist_a < dist_b):
            return -1.0, dist_a
        else:
            return 1.0, dist_b



    #compute contact force
    #compute 




if __name__ == "__main__":
    '''
    Test ClosestPointsDistance(Xa,Xb,da,db,xm,side):
    horizontal beam
    slanting beam
    '''
    Xa = np.array([1.0,1.0,0.0])
    Xb = np.array([2.0,1.0,0.0])
    da = np.array([0.0,0.0,0.0])
    db = np.array([0.0,-1.0,0.0])

    xm = np.array([1.5,0.0])
    side = 1
    xi_c, g =ClosestPointsDistance(Xa,Xb,da,db,xm,side)
    print(xi_c, g)




















