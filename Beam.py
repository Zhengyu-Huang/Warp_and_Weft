import numpy as np

def LinearBeam(E,A,I,Xa,Xb):
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




