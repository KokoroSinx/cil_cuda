#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import glob
from numpy import *
from helpers import *
import pylab as pl


def WCAPotential(r, eps = 0.1, sigma = 1.):
  result = zeros([len(r),])
  ind = find(r < 1.12246205 * sigma)
  rho = (sigma/r[ind])**6
  result[ind] = 4*eps * rho * (rho - 1) + eps
  return result

def hardSphereDiameter(sigma_mf, energy, epsilon = 0.1):
    """
    sigma: soft-sphere diameter, from WCA pair potential
    energy: energy of the particle
    """
    return 2. * sigma_mf * (0.5 + sqrt(energy/(8.*epsilon)))**(-1./6.)
    #return 2. * sigma * (sqrt(energy/(8.*epsilon)))**(-1./6.)

def read_positions(fname_glob):
  """
  Reads positional data from all files described by fname_glob, attempts
  to put it all into one timeline and returns it as the tuple [t,r].
  t[t_index] is a one-dimensional tuple,
  r[t_index][particle_index,dim_index] is multidimensional
  """
  files = create_file_list(fname_glob)
  return read_positions_from_files(files)

def read_positions_from_files(files):
  """
  Reads positional data from files passed as the only argument, attempts
  to put it all into one timeline and returns it as the tuple [t,r].
  t[t_index] is a one-dimensional tuple,
  r[t_index][particle_index,dim_index] is multidimensional
  """
  t = []
  r = []
  for f in files:
    # the first row contains only the time as a single number,
    # the rest is in three columns. Because loadtxt automatically
    # assumes that the structure of the first line will hold for
    # all lines in the document, I have to read in the first line
    # separately from the rest.
    with open(f, "r") as filehandle:
      time = extractNumbersFromString(filehandle.readline())[0]
    # tmp = loadtxt(f)
      t.append(time)
    
    r.append(loadtxt(f,skiprows=1))
    [npart,dim] = shape(r[-1])
    if dim != 3 and dim != 2:
      sys.exit('error: Data has wrong shape. read_pos assumes a time in the first line, then three columns of positional data')
  # Sort the list according to ascending times
  t,r = (list(entry) for entry in zip(*sorted(zip(t,r))))
  t = array(t)
  r = array(r)
  return t, r


def read_configuration(fname_glob):
  """
  Reads positional and velocity data from all files described by fname_glob, attempts
  to put it all into one timeline and returns it as the tuple [t,r,v].
  DOES NOT INCLUDE ACTUAL TIMES, RIGHT NOW, t CARRIES ONLY AN INCREMENTING INDEX!
  t[t_index] is a one-dimensional tuple,
  r[t_index][particle_index,dim_index] is multidimensional
  v[t_index][particle_index,dim_index] is multidimensional
  """
  files = create_file_list(fname_glob)
  t = []
  r = []
  v = []
  incr = 0
  for f in files:
    # do not attempt to read times for now.
    t.append(incr)
    incr = incr + 1

    tmp = loadtxt(f)
    [npart,width] = shape(tmp)
    if width != 6:
      sys.exit('error: Data has wrong shape. read_configuration assumes three columns of positional data and three columns of velocity data.')
    r.append(tmp[:,:3])
    v.append(tmp[:,3:])

  # Sort the list according to ascending times
  #t,r = (list(entry) for entry in zip(*sorted(zip(t,r))))
  return t, r, v

def read_msd(fname, dim):
  """
  Reads MSD data and returns the arrays t,msd.
  """
  assert type(dim) is int
  if not dim in [2,3]:
    sys.exit('error: Require that the spatial dimension parameter dim is either 2 or 3.')
    
  tmp = loadtxt(fname)
  [length, width] = shape(tmp)
  msd = zeros([length,3])
  if width:
      t = tmp[:,0]
      if dim == 3:
          msd[:,0] = sum(tmp[:,1:4],1)
          msd[:,1] = sum(tmp[:,4:7],1)
          msd[:,2] = sum(tmp[:,7:9],1)
      elif dim == 2:
          msd[:,0] = sum(tmp[:,1:3],1)
          msd[:,1] = sum(tmp[:,3:5],1)
          msd[:,2] = sum(tmp[:,5:7],1)
  else:
      sys.exit('error: Data has wrong shape.')
  return t, msd

def read_msd_energydist(fname):
  """
  Reads energy-distributed MSD data and returns the arrays time,energy,msd.
  """
  # Function assumes three columns: time, energy, and msd. Since msd is a function of both,
  # every combination of a time and a energy value comes up. The main purpose of the function
  # is to figure out all the unique time and energy values and produce a two-dimensional msd array.
  tmp = loadtxt(fname)
  [length, width] = shape(tmp)
  if width:
    if width == 3:
      t = tmp[:,0]
      e = tmp[:,1]
      msd = tmp[:,2]
    else:
      sys.exit('error: "read_msd_energydist" assumes three columns: time, energy and MSD. Please check the format of %s.' %(fname))
  else:
    sys.exit('error: Some grave error with the data format of "%s".' %(fname))

  # Produce time and energy arrays consisting of all different values but each value only once.
  tu = unique(t)
  eu = unique(e)
  eu_size = size(eu)
  # Reconstruct the two-dimensional msd array
  msdu = ones((size(tu),eu_size)) * (-100)
  k = 0
  for time in tu:
    ind = find(t == time)
    # Normally the number of identical time values is supposed to match the number of energies
    if size(ind) == eu_size:
      msdu[k,:] = msd[ind,:]
    elif int(size(ind)/2) == eu_size:
      sys.exit('What? More identival time values than energies?')
      #msdu[k,:] = (msd[ind[:eu_size],:] + msd[ind[eu_size:],:])/2
    else:
      sys.exit('error: MSD data in "%s" seems corrupted, energy-histogram appears to vary over the course of the file.' %(fname))
    k = k+1
  return tu, eu, msdu


def read_sofq(fname):
  tmp = loadtxt(fname)
  [length, width] = shape(tmp)
  if width:
    q = tmp[1:,0]
    if width == 8:
      sofq = {'aa' : tmp[1:,1],
      'ab' : tmp[1:,2],
      'ac' : tmp[1:,3],
      'bb' : tmp[1:,4],
      'bc' : tmp[1:,5],
      'cc' : tmp[1:,6],
      'mult' : tmp[1:,7],
      'q0': tmp[0,:]}
    else:
      sys.exit('error: Function assumes eight data columns for structure factor of terniary mixture.')
  else:
    sys.exit('error: Some grave error with the data format of "%s".' %(fname))
  return q, sofq

#def read_fsqt(fname):
  #"""
  #Reads the self-intermediate scattering function data given in fname and returns it as the tuple
  #[t,F,Chi] with up to three components included in F and Chi.
  #"""
  #tmp = loadtxt(fname)
  #[length, width] = shape(tmp)
  #if width:
    #if width == 7:
      #t = tmp[:,0]
      #F = tmp[:,1:4]
      #Chi = tmp[:,4:]
    #else:
      #sys.exit('error: Function assumes seven data columns for intermediate scattering function of terniary mixture.')
  #else:
    #sys.exit('error: Some grave error with the data format of "%s".' %(fname))
  #return t,F,Chi
  
  
def read_ngp(fname, dim=2):
  """
  Reads MSD and MQD data and returns the arrays t, msd, mqd, ngp.
  """
  if not os.path.isfile(fname):
    sys.exit('error: %s is not a file' %fname)
    
  tmp = loadtxt(fname)
  [length, width] = shape(tmp)

  msd = zeros([length,3])
  mqd = zeros([length,3])
  ngp = zeros([length,1])
  
  if not dim == 2:
    sys.exit('error: "read_ngp" assumes two spatial dimensions for now.')
  
  if width:
    t = tmp[:,0]
    if width == 10:
        msd[:,0] = sum(tmp[:,1:3],1)
        msd[:,1] = sum(tmp[:,3:5],1)
        msd[:,2] = sum(tmp[:,5:7],1)
        
        mqd[:,0] = tmp[:,7]
        mqd[:,1] = tmp[:,8]
        mqd[:,2] = tmp[:,9]
        
        # if mqd[0] == 0:
          # ngp[1:] = mqd[1:]/msd[1:]**2 * 0.5 - 1.
        # else:
        ngp = mqd/msd**2 * 0.5 - 1.
    else:
      sys.exit('error: read_ngp assumes 10 data columns, you have provided file {} with {}. Maybe the file does not contain the MQD?'.format(fname, width))  
  else:
      sys.exit('error: Data has wrong shape ')
  return t, msd, mqd, ngp



def read_fsqt(fname):
  """
  Reads the self-intermediate scattering function data given in fname and returns it as the tuple
  [t, F, Chi, q] with up to three components included in F and Chi.
  """
  import re

  ##
  
  ## Read data

  try:
    tmp = loadtxt(fname)
  except ValueError:
    sys.exit('error: It seems that the file %s did not contain readable data.' %fname)
  
  tmpshape = shape(tmp)
  if size(tmpshape) < 2:
    sys.exit('error: Seemingly no data in file "%s".' %(fname))
  [length, width] = tmpshape
  if width:
    if width == 7:
      t = tmp[:,0]
      F = tmp[:,1:4]
      Chi = tmp[:,4:]
    else:
      sys.exit('error: Function assumes seven data columns for intermediate scattering function of terniary mixture.')
  else:
    sys.exit('error: Some grave error with the data format of "%s".' %(fname))

    
  ## Read wavenumber interval
  q = [-1, -1]
  for line in open(fname):
    # remove leading and trailing whitespace
    li=line.strip()
    if li.startswith("#"):
      if "qmin" in li:
        q[0] = [float(s) for s in re.findall("\d+.\d+", li)][0]
      if "qmax" in li:
        q[1] = [float(s) for s in re.findall("\d+.\d+", li)][0]
  if -1 in q:
    sys.exit('error: Could not determine valid q-value')


  return t, F, Chi, q

  
def read_fsqt_fromFiles(fname_glob):
  """ Reads in a filename glob and returns the t, Fsqts and wave numbers qs it finds
      (Fsqt only for type b of binary mixtures).
  """
  files = create_file_list(fname_glob)
  ## TODO: Read in necessary parameters
  # TODO Enter parameters file and read out sigma_bb
  Fs = []
  qs = []
  t = 0
  for f in files:
    t, F, Chi, q = read_fsqt(f)
    Fs.append(F[:,1])
    qs.append(q)
  Fs = reshape(Fs, shape(Fs))
  #qs = array(qs)
  #print shape(qs), qs, q

  return t, Fs, qs



def fsqt_longTimeLimitFromFiles(fname_glob):
  t, Fs, qs = read_fsqt_fromFiles(fname_glob)
  print (shape(Fs))
  return qs, average(Fs[:,-6:-1], axis=1)

  

def read_fsqt_hoefling(fname):
  """
  Reads the self-intermediate scattering function data given in fname, assuming the data structure
  of Felix Hoeflings Lorentz model code, and returns it as the tuple
  [t,q,F].
  """
  tmp = loadtxt(fname)
  [length, width] = shape(tmp)
  if width:
    if width == 4:
      t = tmp[:,0]
      q = tmp[:,1]
      F = tmp[:,2]
    else:
      sys.exit('error: Function assumes seven data columns for intermediate scattering function of terniary mixture.')
  else:
    sys.exit('error: Some grave error with the data format of "%s".' %(fname))
  
  #shape(t)
  #shape(q)
  #shape(F)
  
  tu = unique(t)
  qu = unique(q)
  sizequ = size(qu)
  Fu = ones((size(tu),sizequ)) * (-100)
  k = 0
  for time in tu:
    ind = find(t == time)
    if size(ind) == sizequ:
      Fu[k,:] = F[ind,:]
    elif int(size(ind)/2) == sizequ:
      Fu[k,:] = (F[ind[:sizequ],:] + F[ind[sizequ:],:])/2
    else:
      sys.exit('error: Fsqt data in "%s" seems corrupted, q-grid appears to vary over the course of the file.' %(fname))
    k = k+1
  return tu, qu, Fu
  #return t,q,F


  
  
def read_vh_self(fname):
  """
  Reads the self-van-Hove data given in fname and returns it as the tuple
  [t,r,vH].
  """
  tmp = loadtxt(fname)
  [length, width] = shape(tmp)
  if not width:
    sys.exit('error: Function needs at least three columns of data: t,r,vH.')
  t = tmp[:,0]
  r = tmp[:,1]
  vH= tmp[:,2:]
  
  tu = unique(t)
  ru = unique(r)
  vHu = ones((size(tu),size(ru),width-2)) * (-100)
  k = 0
  for time in tu:
    ind = find(t == time)
    vHu[k,:,:] = vH[ind,:]
    k = k+1
  return tu,ru,vHu

def read_vh_self_legacy(fname):
  """
  Reads the self-van-Hove data given in fname and returns it as the tuple
  [t,r,vH].
  """

  return -1
  #tmp = loadtxt(fname)
  #[length, width] = shape(tmp)
  #if not width:
    #sys.exit('error: Function needs at least three columns of data: t,r,vH.')
  #t = tmp[:,0]
  #r = tmp[:,1]
  #vH= tmp[:,2:]

  #tu = unique(t)
  #ru = unique(r)
  #vHu = ones((size(tu),size(ru),width-2)) * (-100)
  #k = 0
  #for time in tu:
    #ind = pl.find(t == time)
    #vHu[k,:,:] = vH[ind,:]
    #k = k+1
  #return tu,ru,vHu

def read_energy(fname):
  """
  Reads the energy data given in fname and returns it as the tuple
  [t,ekin,epot,etot].
  """
  tmp = loadtxt(fname)
  [length, width] = shape(tmp)
  t = tmp[:,0]
  ekin = tmp[:,1]
  epot = tmp[:,2]
  if width == 3:
    etot = ekin+epot
  else:
    etot = tmp[:,3]
  return t,ekin,epot,etot

def read_energy_histogram(fname):
    """
    Reads the histogram of particle energies data given in fname and returns it as the tuple
    [t,energy,probability].
    """
    tmp = loadtxt(fname)
    [length, width] = shape(tmp)
    if not width or width != 3:
      sys.exit('error: Function needs three columns of data: t, energy, histogram')
    t = tmp[:,0]
    energy = tmp[:,1]
    probability = tmp[:,2]

    tu = unique(t)
    energyu = unique(energy)
    probabilityu = ones((size(tu),size(energyu))) * (-100)
    k = 0
    for time in tu:
      ind = find(t == time)
      probabilityu[k,:] = probability[ind]
      k = k+1
    return tu,energyu, probabilityu
  
def saveResult(path, times, distances, function, header = None, fmt = '%.3e'):
  """
  Saves a function f(t,r) to a text file
  """

  results = zeros(array(function.shape) + (1,1))
  results[0,0] = nan
  results[1:,0] = times
  results[0,1:] = distances
  results[1:,1:] = function
  savetxt(path, results, header = header, fmt=fmt)
  # os.chmod(path, 0766)
  
def bin_y_according_to_x(x, y, xMin=None, xMax=None, xDelta=1.):
    """ Group data in y into bins according to their corresponding x data, and then return the mean and standard deviation.
    """
    # TODO Make sure to handle the case of a single data point in y correctly, e.g. by simply inserting its value and corresponding x into the output.
    # Debugging:
    # x = centerDistances
    # y = cellAreas
    # xMin = 0
    # xMax=None
    # xDelta = 1.

    if xMin == None:
        xMin = np.min(x)
    if xMax == None:
        xMax = np.max(x)
    xBinEdges = np.arange(xMin, xMax + xDelta, xDelta)
    xBinCenters = (xBinEdges[1:]+xBinEdges[:-1])/2
    binIdxs = np.digitize(x, xBinEdges)
    counts = np.nan * xBinCenters
    yBinnedMean = np.nan * xBinCenters
    yBinnedMedian = np.nan * xBinCenters
    yBinnedSTD = np.nan * xBinCenters
    for idx in range(1,len(xBinEdges)):
        selected = (binIdxs == idx)
        if np.any(selected):
            counts[idx-1] = np.sum(selected)
            yBinnedMean[idx-1] = np.mean(y[selected])
            yBinnedMedian[idx-1] = np.median(y[selected])
            yBinnedSTD[idx-1] = np.std(y[selected])
    return xBinCenters, yBinnedMean, yBinnedMedian, yBinnedSTD, counts
  
def derive(x,f, option='forward'):
    """
    derive(x,f,option) calculates the difference quotient df/dx. Optionally, one can set
    whether to employ the forward, backward or central quotient with the option strings
    'forward', 'backward', 'central'.
    """
    xshape = shape(x)
    xlen = xshape[0]
    if (size(xshape) > 1):
        xwidth = xshape[1]
    fshape = shape(f)
    flen = fshape[0]
    if (size(fshape) > 1):
        fwidth = fshape[1]
        if flen < fwidth:
            f = transpose(f)
            fshape = shape(f)
            flen = fshape[0]
            fwidth = fshape[1]
    if xlen != flen:
        sys.exit("error: both arguments need to be of the same length")
        
    # Backward difference quotient
    if option == 'backward':
      if 'fwidth' in locals():
          #df = zeros([flen,fwidth])
          df = zeros([flen,fwidth])*nan
          for row in range(fwidth):
              df[1:,row] = (f[1:,row] - f[:-1,row]) / (x[1:] - x[:-1])
          #df[0,:] = df[1,:]
      else:
          df = zeros(flen)
          df[1:] = (f[1:] - f[:flen-1]) / (x[1:] - x[:flen-1])
          df[0] = nan

    # Forward difference quotient
    elif option == 'forward':
      if 'fwidth' in locals():
          df = zeros([flen,fwidth])*nan
          for row in range(fwidth):
              df[:-1,row] = (f[1:,row] - f[:-1,row]) / (x[1:] - x[:-1])
          #df[-1,:] = df[-2,:]
      else:
          df = zeros(flen)*nan
          df[:-1] = (f[1:] - f[:flen-1]) / (x[1:] - x[:flen-1])
          #df[-1] = nan

    # Central difference quotient
    elif option == 'central':
      if 'fwidth' in locals():
          df = zeros([flen,fwidth])*nan
          for row in range(fwidth):
              df[1:-1,row] = (f[2:,row] - f[:-2,row]) / (x[2:] - x[:-2])
          #df[0,:] = df[1,:]
          #df[-1,:] = df[-2,:]
      else:
          df = zeros(flen)*nan
          df[1:-1] = (f[2:] - f[:flen-2]) / (x[2:] - x[:flen-2])
          #df[0] = nan
          #df[-1] = nan
    else:
      sys.exit("error: incorrect third argument set. Use either 'forward', 'backward', or 'central'.")
    return df

def second_derive(x,f):
    """
    second_derive(x,f) calculates the numerical approximation of the 2nd derivative d^2f/dx^2.
    x does not need to be uniform.
    """
    xshape = shape(x)
    xlen = xshape[0]
    if (size(xshape) > 1):
        xwidth = xshape[1]
    fshape = shape(f)
    flen = fshape[0]
    if (size(fshape) > 1):
        fwidth = fshape[1]
        if flen < fwidth:
            f = transpose(f)
            fshape = shape(f)
            flen = fshape[0]
            fwidth = fshape[1]
    if xlen != flen:
        sys.exit("error: both arguments need to be of the same length")
        
    df1 = derive(x,f,'backward')
    df2 = derive(x,f,'forward')
    
    diff = zeros(shape(x))*nan
    diff[1:-1] = x[2:]
    diff[1:] = (diff[1:]-x[:-1])/2.

    if 'fwidth' in locals():
      d2f = zeros([flen,fwidth])*nan       
      for row in range(fwidth):
        d2f = (df2-df1)/diff
    else:
      d2f = (df2-df1)/diff
    return d2f

def log_derive(x,f, option='forward'):
    """
    log_derive(x,f,option) calculates the logarithmic difference quotient d(log10(f))/d(log10(x)). Optionally, one can set whether to employ the forward, backward or central quotient with the option strings 'forward', 'backward', 'central'.
    """
    #ind = find(f==0)
    #if len(ind) == 0:
        #sys.exit "warning: function values contain zero(s)"
    return derive(log10(x),log10(f),option)


def calc_msd(t,r):
  msd = []
  [npart,dim] = shape(r[0])
  for pos in r:
    msd.append(sum(pos-r[0])**2/npart)
  return msd

# Calculates the msd, mqd and ngp. Optimised for the experimenal data in Schnyder et al, PRE (2017) which has particles leaving the observation window sometimes. When data is outside the window, -1 are inserted into the raw data as placeholders, which this function filters out. 
def calculate_msd_mqd_ngp(ts, pos, timeOrigins = [0,]):  
  msd = zeros_like(ts)
  mqd = zeros_like(ts)
  ngp = zeros_like(ts)
  degeneracy = zeros_like(ts)

  for p in pos:
    actualData = all(p != -1, axis = 1)
    p = p[actualData]
    for timeOrigin in timeOrigins:
      if timeOrigin < len(p):
          
        rsquared = sum((p[timeOrigin:] - p[timeOrigin])**2, axis = 1)
        degeneracy[:len(rsquared)] += 1.
        msd[:len(rsquared)] += rsquared
        mqd[:len(rsquared)] += rsquared**2

  ## Normalize
  ind = degeneracy.nonzero()
  msd[ind] /= degeneracy[ind]
  mqd[ind] /= degeneracy[ind]

  ## Determine the NGP
  ngp[0] = nan
  ngp[msd.nonzero()] = mqd[msd.nonzero()]/msd[msd.nonzero()]**2/2. - 1
  
  return msd, mqd, ngp, degeneracy  


def create_file_list(file_wildcard):
    # list files matching the file_wildcard in respect to the current directory
    files = [f for f in glob.glob(file_wildcard) if os.path.isfile(f)]
    # sort list of files in place
    sort_nicely(files)
    return files


def create_folder_list(folder_wildcard):
    # list folders matching the folder_wildcard in respect to the current directory
    folders = [f for f in glob.glob(folder_wildcard) if not os.path.isfile(f)]
    # sort list of files in place
    sort_nicely(folders)
    return folders


def create_file_list_spread_with_delta(fname_glob, delta):
  """
  create_file_list_spread_with_delta(fname_glob, delta):
  Creates a sorted file list for files matching fname_glob assuming that they contain a timestamp in their name. It only returns files which are at least delta timesteps spaced apart.
  """

  files = create_file_list(fname_glob)
  #ignore files starting with '.' using list comprehension
  files = [filename for filename in files if os.path.basename(filename)[0] != '.']
  # Exclude parameter files
  files = [filename for filename in files if (len(filename) >= 3 and filename[3] != '_')]

  files_with_spread = []
  next_allowed_timestamp = 0
  for filename in files:
    f_key = alphanum_key(filename) # separate file names into individual parts.
    # typically the timestamp is the last element of the filename
    timestamp = f_key[-2]
    # correct for the stupid case of the first file, which has no timestamp
    if (timestamp == "run"):
      timestamp = 0
    elif (not is_number(timestamp)):
      continue

    if(timestamp >= next_allowed_timestamp):
      files_with_spread.append(filename)
      next_allowed_timestamp = timestamp + delta

  return files_with_spread

#def periodicBoundaryPositions(ps, lBox, diameter = 1.):
  #ps_prb = ones(shape(ps))
  #if shape(ps)[1] == 2:
    #xs_prb, ys_prb = periodicBoundaryPositions(ps[:,0], ps[:,1], lBox,)
    #ps_prb[:,0] = xs_prb
    #ps_prb[:,1] = ys_prb
  #else:
    #return array([])
  #return ps_prb
  
def distanceWithPBC(v1, v2, lBox):
  distance = v1 - v2
  distance -= floor(distance/lBox) * lBox
  return linalg.norm(distance)

def periodicBoundaryPositions(xs,ys, lBox, diameter = 1.):
  """
  xs_pb, ys_pb, originals = periodicBoundaryPositions(xs,ys, lBox, diameter = 1.)

  Returns particle positions by transforming (xs,ys) assuming periodic boundary
  conditions with box size lBox in both directions.
  Additionally it duplicates all those particles which are close enough
  to the boundaries to show on the other side as well.
  Originals provides the index of the original point
  """
  xs_pb = xs%lBox
  ys_pb = ys%lBox
  radius = diameter/2.

  # with duplicated particles near the boundaries
  xs_wd = []
  ys_wd = []
  originals = []

  for x,y in zip(xs_pb,ys_pb):
    reference = len(xs_wd)
    xs_wd.append(x)
    ys_wd.append(y)
    originals.append(reference)

    # each particle may be duplicated up to three times,
    # e.g. a particle on one corner of the box would show up
    # at all four corners.
    if x <= radius:
      xs_wd.append(x+lBox)
      ys_wd.append(y)
      originals.append(reference)
      
      if y <= radius:
        xs_wd.append(x+lBox)
        ys_wd.append(y+lBox)
        originals.append(reference)
        
      elif y >= lBox-radius:
        xs_wd.append(x+lBox)
        ys_wd.append(y-lBox)
        originals.append(reference)

    if x >= lBox - radius:
      xs_wd.append(x-lBox)
      ys_wd.append(y)
      originals.append(reference)
      
      if y <= radius:
        xs_wd.append(x-lBox)
        ys_wd.append(y+lBox)
        originals.append(reference)
        
      elif y >= lBox-radius:
        xs_wd.append(x-lBox)
        ys_wd.append(y-lBox)
        originals.append(reference)

    if y <= radius:
      xs_wd.append(x)
      ys_wd.append(y+lBox)
      originals.append(reference)
      
    if y >= lBox-radius:
      xs_wd.append(x)
      ys_wd.append(y-lBox)
      originals.append(reference)

  ## Testing that the same position does not occur twice
    #for x,y in zip(xs,ys):
  #if ((x,y) not in xy_set):
    #xy_set.append((x,y))  

  return xs_wd, ys_wd, array(originals)


def occupied_area_on_lattice(points, radii, lbox, latticeSpacing=5):
  """ Approximate the 2D area covered by circles centered around 'points' with radii given by 'radii' by counting the number of square lattice sites which are touched by at least one circle. The square lattice sites have edge length 'latticeSpacing' and the lattice is expected to go from 0 to 'lbox' in both directions.
  'points' is expected to be in the shape = (nCircles, 1, 2), whereas 'radii' has shape (nCircles, 1).
  """           
  # Simply count lattice sites that contain a circle center. This is fastest but very imprecise for the too large or too small lattice spacing
  centerLatticeIdxs = unique_rows((points//latticeSpacing)[:,0,:])    
  
  # More precisely, check which circles are touching which lattice squares:
  # To reduce the number of lattice sites that need to be looked at, calculate the center lattice positions, find the largest radius, and use that to make a rough envelope
  maxRadiusInLatticeSpacings = int(np.ceil(np.max(radii)/latticeSpacing))
  smallestIdx = np.min(centerLatticeIdxs) - maxRadiusInLatticeSpacings
  largestIdx = np.max(centerLatticeIdxs) + maxRadiusInLatticeSpacings

  occupiedLatticeIdxs = []
  for xLattice in np.arange(smallestIdx, largestIdx+1):
    for yLattice in np.arange(smallestIdx, largestIdx+1):
      latticePos = np.array((xLattice, yLattice)) * latticeSpacing
      dists = points[:,0,:] - latticePos                 
      # Calculate the distance of the circle to the closest point on the lattice square if the circle center is outside the square or 0 if the center is inside: 
      # The lattice site's extent is [0,latticeSpacing] in both directions.
      # If the x coordinate of the circle is within the square's x-extent, then the x coordinate of the closest point is set to be the same as that of the circle. Else, it is the closest x coordinate possible. Same goes for the y coordinate. Once I have the closest point, I can simply calculate the distance between the circle center and the closest point. If that special distance is smaller than the circle radius, the circle is touching the square.
      closestPoints = np.copy(dists)
      closestPoints[dists[:,0] < 0, 0] = 0
      closestPoints[dists[:,0] > latticeSpacing, 0] = latticeSpacing
      closestPoints[dists[:,1] < 0, 1] = 0
      closestPoints[dists[:,1] > latticeSpacing, 1] = latticeSpacing
      atLeastOneCellIsTouchingTheBox = np.any(np.linalg.norm(dists - closestPoints, axis=1) <= radii[...,0])
      if atLeastOneCellIsTouchingTheBox:
        occupiedLatticeIdxs.append((xLattice, yLattice))
  occupiedLatticeIdxs = unique_rows(np.array(occupiedLatticeIdxs))
  occupiedLatticeSites = (occupiedLatticeIdxs + 0.5) * latticeSpacing
  
  latticeSiteArea = latticeSpacing**2
  occupiedLatticeArea = occupiedLatticeIdxs.shape[0] * latticeSiteArea
  return occupiedLatticeArea, occupiedLatticeSites