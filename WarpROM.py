__author__ = 'zhengyuh'
'''
This is a problem, with fixed left [0,0,0],
and fixed right [x,y, theta]. We have two
parameters x, y, theta, which are strain,
deflection and bending angle
x     \in [-0.2,0.2]
y     \in [-0.2,0.2]
theta \in [-30, 30]
And we will predict the force at right
[Fx, Fy, M_theta]

This problem to some degrees, explores the constitutive
law of the warp.
'''

def build_basis():



