#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import abs, all, arctan2, array, ascontiguousarray, cross, dot, diag, isclose, logspace, log10, shape, unique
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from re import search
import sys
import os

def is_number(s):
  try:
    a = array(s)
    return all([float(entry) for entry in a.flat])
  except TypeError:
    return False
    
def is_arraylike(s):
  return isinstance(s, (collections.Sequence, np.ndarray))
  
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    import re
    def tryint(s):
      try:
          return int(s)
      except:
          return s
        
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect. This happens in place.
    """
    l.sort(key=alphanum_key)

def cross2d(v, w):
    """Defines a sort-of cross-product for two dimensions. In fact, 
    it simply returns the z component of the standard cross-product."""
    return v[0]*w[1] - v[1]*w[0]

def linesIntersection(l0, l1, m0, m1):
    """Determines the intersection of two lines
The two lines are defined as
l = l0 + d * (l1 - l0)
m = m0 + e * (m1 - m0)
The point P where the lines intersect can be described by either coefficient d and e.

Returns:
    (P, d, e)"""
    denominator = cross2d(l1 - l0, m1 - m0)
    if isclose(abs(denominator), 0):
        print("Vectors are parallel")
    else:   
        d = cross2d(m0 - l0, m1 - m0)/denominator
        e = cross2d(m0 - l0, l1 - l0)/denominator
        P = l0 + d * (l1 - l0)
        return (P, d, e)

def lineCircleIntersection(l0, l1, c, r):
    """Determines the intersection of a line and a circle
Line defined as x = l0 + d * (l1 - l0)
Circle defined as all points x satisfying |x - c|^2 = r^2
    The points P where the line and circle intersect can also be described by distance parameter d.
    """
    D = []

    l = l1 - l0
    a = np.dot(l,l)
    b = 2*(np.dot(l, l0 - c))
    c = np.dot(l0 - c, l0 - c) - r**2
    
    arg = b**2 - 4*a*c
    if arg >= 0:
        root = np.sqrt(arg)    
        if (root == 0):
            D.append(-b/(2*a))
        elif(root > 0):
            D.append((-b - root)/(2*a))
            D.append((-b + root)/(2*a))
    
    P = [l0 + d * (l1 - l0) for d in D]
    return (P,D)

def vectorAngle(vector1, vector2):
  """
  angle(vector1, vector2) calculates the angle between vector 1 and vector 2 in 3d. The resulting angles range between 0 and pi.
  """
  # vector1 = array(vector1)
  # vector2 = array(vector2)
  return arctan2(norm(cross(vector1, vector2)), dot(vector1, vector2))

def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def quatMultiplication(q1, q2):
  """
  Multiplies two quaternions with each other.
  
  See https://en.wikipedia.org/wiki/Quaternion#Quaternions_and_the_geometry_of_R3
  (Stolen from hoomd)
  """
  s = q1[0]
  v = q1[1:]
  t = q2[0]
  w = q2[1:]
  q = np.empty((4,), dtype=np.float64)
  q[0] = s*t - np.dot(v, w)
  q[1:] = s*w + t*v + np.cross(v,w)
  return q
  
# Rotate a vector by a unit quaternion
# Quaternion rotation per http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# (requires numpy)
# \param q rotation quaternion
# \param v 3d vector to be rotated
# \returns q*v*q^{-1}
def quatRotation(q, v):
  """
  Rotate a vector by a unit quaternion.
  (Stolen from hoomd)
  """
  v = np.asarray(v)
  q = np.asarray(q)
  # assume q is a unit quaternion
  w = q[0]
  r = q[1:]
  vnew = np.empty((3,), dtype=v.dtype)
  vnew = v + 2*np.cross(r, np.cross(r,v) + w*v)
  return vnew
  
# https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
def quatToRotationMatrix(q):
  w = q[0]
  x = q[1]
  y = q[2]
  z = q[3]
  
  n = dot(q,q)
  Q = diag((1,1,1))
  if n != 0:
    s = 2./n
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    
    Q = np.array([
        [1. - yy-zz,      xy-wz,      xz+wy], 
        [   xy+wz,   1. - xx-zz,      yz-wx], 
        [   xz-wy,        yz+wx, 1. - xx-yy]
      ])
    
  return Q

def extent(r):
  """ Calculates the minimum and maximum values in the array and outputs them in a format ready for using to constrain the axis in a plot."""
  return np.vstack((np.amin(r, axis=0), np.amax(r, axis=0))).T.flatten()
    

def movavg(x, N):
  """Calculates the moving average of x with window width N.
Reference:
https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/30141358#30141358
  """
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N
  
def blockavg(x,N):
  """Group and average x in blocks of length N, disregarding the end of the dataset if it doesn't fill a whole block.
"""
  nBlocks = np.int(x.shape[0]/N)
  blocked = x[:N*nBlocks].reshape((nBlocks, -1, N))
  avg = np.average(blocked, axis=-1)
  # err = std
  if len(x.shape) == 1:
    avg = avg.reshape((nBlocks,))
  return avg  

# normalize list of vectors        
def normalize(l):
    return np.array(list(map(lambda l0 : l0/np.linalg.norm(l0), l)))
    
# compute pbc distance between r1 and r2 (r1,r2 are posititions or arrays of positions)
def distance(r1, r2, lbox):
    return r2 - r1 - np.around((r2-r1)/lbox)*lbox

def pbc(r, lbox):
  """returns positions wrapped around periodic boundaries
  """
  return np.fmod(r + lbox, lbox)

# minimum image convention
def mic(data, lbox):
  flip = np.abs(data) > 0.5 * lbox    
  data[flip] -= lbox * np.sign(data[flip])
  return data

def com(R, lbox):
    """ Calculates the centers of mass of the molecules contained in R, assuming that the first index iterates over the molecules, and assuming that the positions are for a square system of side length lbox with periodic boundary conditions.
    """
    nCells, nElements, nDim = np.shape(R)
    com = [cell[0] + np.sum(distance(cell[0], cell[1:], lbox), axis=0)/nElements for cell in R]
    return np.array(com)  

def nematicOrientation(angle, maxAngle=1):
    """ Calculates the nematic orientation of a particle from its orientation angle under the assumption that a 180 degree flip doesn't change anything. Assumes that the angle is in the range [-maxAngle, maxAngle]
    """
    return np.fmod(angle - np.array(angle/maxAngle*2).astype('int32'), 0.5)

# Undo the wrapping of the trajectories at the periodic boundaries
def unwrap_trajectories(rt, lbox):
  increments = rt[1:] - rt[:-1]
  unwrappedIncrements = mic(increments, lbox)

  unwrapped = np.zeros_like(rt)
  unwrapped[0] = rt[0]
  unwrapped[1:] = rt[0] + np.cumsum(increments, axis=0)
  return unwrapped  
  
# compute pbc distance between r1 and r2 (r1,r2 are posititions or arrays of positions)
def distance(r1, r2, lbox):
    return r2 - r1 - np.around((r2-r1)/lbox)*lbox
  

def find_nearest(array, value):
    """ Returns array index and value of the value in the array that is closest to the provided value.
    """
    idx = (abs(array-value)).argmin()
    return idx, array[idx]

def find_root_interp(x, y, value, kind='linear'):
  """ Creates an interpolating function f(x) = y, and then returns x_root and f(x_root)-value for which f(x_root) = value
  """
  f = interp1d(x, y - value, kind=kind)
  res = root_scalar(f, bracket=[x[0], x[-1]])
  return res.root, f(res.root) 

def secant(F, x1, x2, epsilon = 1e-4, maxSteps = 100):
    """Seeks for a root of function F the secant procedure, requires two initial guesses x1 and x2.
    root, isConverged = secant(F, x1, x2, epsilon = 1e-4, maxSteps = 100)"""
    isConverged = False
    steps = 0
    F1 = F(x1)
    F2 = F(x2)
    print(x1, x2, F1, F2)

    while (not isConverged):
        # sort such that we always keep the guess with F(guess) closer to 0:
        if (np.abs(F1) > np.abs(F2)):
            x1, x2 = x2, x1
            F1, F2 = F2, F1

        # get new guess
        d = F1 * (x2 - x1)/(F2 - F1)
        x2 = x1
        F2 = F1
        x1 -= d
        F1 = F(x1)

        print(x1, x2, F1, F2)

        if(np.abs(F1) < epsilon): # TODO do a better convergence criterion
            isConverged = True
        elif(steps >= maxSteps):   
            break         
    return x1, isConverged

def unique_rows(a):
  """
  Implementation of unique for obtaining unique rows in a matrix
  """
  a = ascontiguousarray(a)
  unique_a = unique(a.view([('', a.dtype)]*a.shape[1]))
  return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def unique_tuples(a, b):
  Tuples = [(aValue, bValue) for aValue, bValue in zip(a, b)]
  uniqueTuples = unique_rows(Tuples)
  
def logRange(stop, start=1, maxLength = 100):
  """
  array = logRange(stop, start=1, maxLength = 100)
  Returns array of maximum length maxLength of logarithmically spaced integers 
  between start and stop.
  """
  return unique(logspace(0, log10(stop), maxLength).astype('int'))


def arrayMidPoints(array):
  """
  midPoints = arrayMidPoints(array)
  
  Simply returns the average of each consecutive pair of entries for all entries of an array.
  """
  return (array[1:]+array[:-1])/2.

def findAllWithEnding(ending, path='.'):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ending):
                result.append(os.path.join(root, file))
    return result  

def findStringInFile(string, filename):
    file = open(filename, "r")

    lines = []
    for line in file:
        if search(string, line):
              lines.append(line)
    if len(lines) == 1:
        lines = lines[0]
    return lines

def extractNumbersFromString(string):
    l = []
    for t in string.split():
        try:
            l.append(float(t))
        except ValueError:
            pass
    # if len(l) == 1:
        # l = l[0]
    return l  
    
    
def stepFunction(x, xp, fp):
    """   
    Returns the one-dimensional step function to a series of known function
    values fp at discrete points xp.
    """
    ## TODO Do it with a single loop over all indices, should be 
    ## much much faster
    return array([max(fp[xp <= value]) for value in x])
    