'''
Created on May 28, 2012
@author: deepak

Perform various manipulations on spinors (length 2 array), pauli matrices, vectors
and matrices.
Source adapted from spinfoam_4simplex.py by Yasha Neiman
Ref: http://arxiv.org/abs/1109.3946, "Parity and reality properties of the EPRL spinfoam"
#-------------------
 
'''
#! /usr/bin/python
# begin Yasha Neiman's code
import sys
import getopt

from math import *
from numpy import *
from numpy.linalg import *


#-----------------------------------------------
# Functions on general tensors (numpy arrays)
#-----------------------------------------------

def complex_zeros (dim):
    """Returns a vector with 'dim' complex components."""
    return zeros(dim, dtype=complex)

def average (tensors):
    """Returns the arithmetic average of a list of tensors."""
    return reduce(add, tensors) / len(tensors)
  
#------------------------------------------------------
# Functions on general vectors (numpy rank-1 arrays)
#------------------------------------------------------

def normalize (v):
    """Returns a unit vector in the direction of the given one. For a zero vector, returns zero."""
    norm_v = norm(v)
    return copy(v) if norm_v == 0 else v/norm_v
  
#---------------------------------------------
# Functions on 3d vectors (length-3 arrays)
#---------------------------------------------

def area_normal (x1, x2, x3):
    """Returns the area normal to a given triangle, with right-hand orientation."""
    return cross(x2 - x1, x3 - x2) / 2

def area_normal_outgoing (triangle_points, source_point):
    """Returns the area normal to a triangle whose vertices are in the list 'triangle_points', 
    oriented to point away from the given 'source_point'."""
    raw_normal = area_normal(*triangle_points)
    outward_vector = triangle_points[0] - source_point
    return copysign(1, dot(raw_normal, outward_vector)) * raw_normal

def cos_theta (v):
    """Returns the cosine of the spherical theta angle defined by the given vector. 
    For a zero vector, returns 1."""
    norm_v = norm(v)
    return 1 if norm_v == 0 else v[2]/norm(v)

def cos_phi (v):
    """Returns the cosine of the azimuthal phi angle defined by the given vector. 
    For a vector with vanishing xy components, returns 1."""
    rho = norm(v[0:2])
    return 1 if rho == 0 else v[0]/rho

#----------------------------------------------------
# Functions on Minkowski vectors (length-4 arrays)
#----------------------------------------------------

def minkowski_vector_from_3vector (v):
    """Upcasts a 3-vector into a 4-vector with a zero time component."""
    mink = zeros(4)
    mink.put([1,2,3], v)
    return mink

def unit_time_vector ():
    """Returns the unit vector (1,0,0,0) along the time axis."""
    return array([1,0,0,0])

def raise_or_lower_4index (v):
    """Flips the sign of a 4-vector's time component.
    Represents multiplication by the mostly-plus Minkowski metric."""
    return array([-v[0], v[1], v[2], v[3]])

def minkowski_product (u, v):
    """Returns the Minkowski scalar product (mostly-plus signature) of two vectors."""
    return dot(raise_or_lower_4index(u), v)

#---------------------------------------------
# Functions on 2d spinors (length-2 arrays)
#---------------------------------------------

def component_ratio (z):
    """Returns the spinor's first component divided by the second."""
    return z[0] / z[1]

def vector_to_unit_spinor (v):
    """Returns the unit spinor (e^{-i*phi/2}cos(theta/2), e^{i*phi/2}sin(theta/2)) 
    pointing along the given vector. For a zero vector, returns a zero spinor."""
    if norm(v) == 0:
        return [0, 0]
   
    cos_theta_v = cos_theta(v)
    cos_theta_half = sqrt((1. + cos_theta_v)/2.)
    sin_theta_half = sqrt((1. - cos_theta_v)/2.)
  
    cos_phi_v = cos_phi(v)  
    cos_phi_half = copysign(sqrt((1. + cos_phi_v)/2.), v[1])
    sin_phi_half = sqrt((1. - cos_phi_v)/2.)
    exp_phi_half = cos_phi_half + 1j*sin_phi_half

    return array([cos_theta_half*exp_phi_half.conjugate(), sin_theta_half*exp_phi_half])

def J_conjugate (z):
    """Performs the J conjugation (parity transformation) on a spinor."""
    return array([-z[1].conjugate(), z[0].conjugate()])
  
def spinor_product (w, z):
    """Returns the antisymmetric scalar product w[1]*z[0] - w[0]*z[1] of two spinors."""
    return -cross(w, z)

#--------------------------------------------------
# Functions on general matrices (rank-2 arrays)
#--------------------------------------------------

def complex_identity (dim):
    """Returns a dim*dim identity matrix with complex components."""
    return identity(dim, dtype=complex)

#--------------------------------------------------
# Functions on SU(2) matrices (2x2 arrays)
#--------------------------------------------------

# A vector containing the three Pauli matrices
pauli_matrices = array([[[0, 1], [1, 0]], \
                        [[0, -1j], [1j, 0]], \
                        [[1, 0], [0, -1]]], dtype=complex)

def matrix_from_3vector (v):
    """Returns the dot product of the vector with the Pauli matrices."""
    return tensordot(v, pauli_matrices, axes = ([0], [0]))  

def SU2_around_vector (axis_vector, angle):
    """Returns the spin-1/2 rotation matrix for rotation around a given vector by a given angle."""
    cos_half = cos(angle/2.)
    sin_half = sin(angle/2.)
  
    return cos_half * complex_identity(2) - 1j*sin_half * matrix_from_3vector(normalize(axis_vector))

#--------------------------------------------------
# Functions on SL(2,C) matrices (2x2 arrays)
#--------------------------------------------------

def SL2C_boost_along_vector (axis_vector, boost):
    """Returns the spin-1/2 rotation matrix for a boost along a given vector by a given angle."""
    cosh_half = cosh(boost/2.)
    sinh_half = sinh(boost/2.)
  
    return cosh_half * complex_identity(2)  + sinh_half * matrix_from_3vector(normalize(axis_vector))

# end Yasha Neiman's code'