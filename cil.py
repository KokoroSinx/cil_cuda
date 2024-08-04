import json
import numpy as np
import matplotlib.mlab as ml
import gsd.fl
import gsd.pygsd
import gsd.hoomd
import h5py
import glob
import os
import sys

from matplotlib.patches import CirclePolygon

from helpers import sort_nicely, is_number, rotation
import utils

# TODO Fix this function, since it doesn't seem to work anymore. Ah, I think this was only meant for the "Last.h5" type files.
def getSnapshotFromFileAsDictionary(filename):
  with h5py.File(filename,'r') as f:
    # Transfer snapshot into dictionary
    snapshot = {}
    for key in f['/'].keys():
      snapshot[key] = f[key][...]
  return snapshot

def saveDictionaryToH5(snapshot, output_file_name):
  with h5py.File(output_file_name, 'w') as new_f:
    for key, data in snapshot.items():
      new_f.create_dataset(key, data=data)

def saveSnapshotToH5(snapshot, output_file_name):
  with h5py.File(output_file_name, 'w') as new_f:
    for array in snapshot.keys():
      snapshot.copy(array, new_f)

def extractSnapshotFromH5(filename, snapshot_number, output_file_name=None):
  """
  extractSnapshotFromH5(filename, snapshot_number, output_file_name=None)
  
  Loads a h5 file filename and exports the snapshot with snapshot_number for later use as a starting configuration to a file output_file_name. Stores also all parameters of the original simulation in output_file_name.
  
  If output_file_name is not provided, it's generated from the snapshot_number.
  """

  if not output_file_name:
    output_file_name = "data.t_{}.h5".format(snapshot_number)

  with h5py.File(filename,'r') as f:
    snapshot_ids = list(f["snapshots/"].keys())
    sort_nicely(snapshot_ids)

    numFrames = len(snapshot_ids)
    if snapshot_number > numFrames:
      sys.exit("Requested snapshot number is too large for the file. Choose a number <= {}".format(numFrames))

    snapshot = f["snapshots/" + snapshot_ids[snapshot_number]]

    ## Create new file and copy the snapshot over, group by group
    with h5py.File(output_file_name, 'w') as new_f:
      for array in snapshot.keys():
        snapshot.copy(array, new_f)

      ## Save also all parameters
      snapshot.copy(f["params"], new_f)

def generateSnapshotFromPositions(R, diameters):
  """Create a valid snapshot dictionary with keys ['R', 'F', 'SigmaCore', 'State', 'T0'] from cell positions R with R.shape = [nCells, nElementsPerCell, DIM] and diameters as a float or list/array with diameters.shape = [nElementsPerCell] or [nCells, nElementsPerCell].

  The result has with the following shapes:
    R.shape = [nCells, nElementsPerCell, DIM]
    F.shape = [nCells, nElementsPerCell, DIM]
    SigmaCore.shape = [nCells, nElementsPerCell]
    State.shape = [nCells]
    T0.shape = [nCells]
  Datatypes: R, F, SigmaCore are float32, State and T0 are int32."""
  nCells, nElementsPerCell, DIM = np.shape(R)

  F = np.zeros_like(R)
  if (np.shape(diameters) == ()):
    sigmaCore = np.ones([nCells, nElementsPerCell]) * diameters
  elif (np.shape(diameters) == (nCells,)):
    sigmaCore = np.array(diameters).reshape((-1,1))
  elif (np.shape(diameters) == (nElementsPerCell,)):
    sigmaCore = np.tile(diameters, nCells).reshape([nCells, -1])
  elif (np.shape(diameters) == (nCells, nElementsPerCell)):
    sigmaCore = diameters
  else:
    raise ValueError('Diameters are not in expected shape. Either must be given for all cells or just for one cell')

  snapshot = {}
  snapshot['R'] = R
  snapshot['F'] = F
  snapshot['SigmaCore'] = sigmaCore
  snapshot['State'] = np.ones([nCells, ], dtype=np.int32)
  snapshot['T0'] = np.zeros([nCells, ], dtype=np.int32)

  return snapshot

def getTrajectories(filename, startTime=0, stopTime=None, step=None):
  """
  timesteps, positions, velocities = getTrajectories(filename, startTime=0, stopTime=None, step=None)

  Gets simulation data from gsd output file for the snapshots defined by  timesteps = range(startTime, stopTime, step).
  Positions and velocities are in the shape (nTimes, nElements, 2).
  Works only for CONSERVED PARTICLE NUMBER.
  """
  with gsd.fl.GSDFile(filename, mode='rb', application="cil", schema=None, schema_version=None) as fp:
    ft = gsd.hoomd.HOOMDTrajectory(fp)
    numFrames = fp.nframes

    if not stopTime:
      stopTime = numFrames

    sampleTimes = np.arange(startTime, stopTime, step)
    N = ft[0].particles.N

    t = np.zeros((len(sampleTimes), ), dtype=np.int64)
    rt = np.zeros((len(sampleTimes), N, 2))
    vt = np.zeros((len(sampleTimes), N, 2))

    # fill buffer (r0,r1,...,ri)
    for tIndex, sampleTime in enumerate(sampleTimes):
      t[tIndex] =  ft[int(sampleTime)].configuration.step
      rt[tIndex,...] = ft[int(sampleTime)].particles.position[:,:2]
      vt[tIndex,...] = ft[int(sampleTime)].particles.velocity[:,:2]

    # nTimes, nCells, dim = rt.shape
    # rt = rt.reshape((nTimes, int(nCells/2), 2, dim))
    # vt = vt.reshape((nTimes, int(nCells/2), 2, dim))

    return t, rt, vt

def getTrajectoriesFromH5(filename, start=0, stop=None, step=None):
  """
  sampleTimes, positions, velocities = getTrajectoriesFromH5(filename, startTime=0, stopTime=None, step=None)

  Gets simulation data from hdf5 output file from John's simulation for the
  snapshots with indices defined by range(start, stop, step). Outputs the times
  associated with these snapshots as sampleTimes.
  Positions and velocities are in the shape (nTimes, nCells, nParticles, nDim).
  """
  with h5py.File(filename,'r') as datafile:
    snapshots = list(datafile["snapshots/"].keys())
    sort_nicely(snapshots)

    numFrames = len(snapshots)
    if not stop:
      stop = numFrames

    samples = np.arange(start, stop, step)
    # This assumes that the number of cells is monotonically increasing
    N, nElements, nDim = datafile["snapshots/" + snapshots[-1]].get('R').shape
    rt = np.zeros((len(samples), N, nElements, nDim))
    vt = np.zeros((len(samples), N, nElements, nDim))

    rt[:] = np.NAN
    vt[:] = np.NAN

    ## Fill arrays with the trajectories
    for tIndex, sampleTime in enumerate(samples):
      numCells = int(datafile["snapshots/" + snapshots[int(sampleTime)]].get('R').shape[0])
      rt[tIndex, :numCells, ...] = datafile["snapshots/" + snapshots[int(sampleTime)]].get('R')[...]
      vt[tIndex, :numCells, ...] = datafile["snapshots/" + snapshots[int(sampleTime)]].get('F')[...]

    dt = datafile["/params"].attrs['dt'][0]
    timestep = dt * datafile["snapshots/" + snapshots[1]].attrs['ts'][0]

    return samples*timestep, rt, vt

def getStatesFromH5(filename, start=0, stop=None, step=None):
  """
  sampleTimes, states, T0s = getStatesFromH5(filename, startTime=0, stopTime=None, step=None)

  Gets simulation data from hdf5 output file from John's simulation for the
  snapshots with indices defined by range(start, stop, step). Outputs the times
  associated with these snapshots as sampleTimes.
  States and times T0s are in the shape (nTimes, nCells).
  """
  with h5py.File(filename,'r') as datafile:
    snapshots = list(datafile["snapshots/"].keys())
    sort_nicely(snapshots)

    numFrames = len(snapshots)
    if not stop:
      stop = numFrames

    samples = np.arange(start, stop, step)
    # This assumes that the number of cells is monotonically increasing
    N = datafile["snapshots/" + snapshots[-1]].get('R').shape[0]
    states = np.zeros((len(samples), N))
    T0s = np.zeros((len(samples), N))

    states[:] = np.NAN
    T0s[:] = np.NAN

    ## Fill arrays with the trajectories
    for tIndex, sampleTime in enumerate(samples):
      numCells = datafile["snapshots/" + snapshots[int(sampleTime)]].get('State').shape[0]
      states[tIndex, :numCells] = datafile["snapshots/" + snapshots[int(sampleTime)]].get('State')[...]
      T0s[tIndex, :numCells] = datafile["snapshots/" + snapshots[int(sampleTime)]].get('T0')[...]

    dt = datafile["/params"].attrs['dt'][0]
    timestep = dt * datafile["snapshots/" + snapshots[1]].attrs['ts'][0]

    return samples*timestep, states, T0s
        
def getRundIDRange(wildcards):
  runIDs = [np.inf,0]
  for wildcard in wildcards:
      for runFolder in [d for d in glob.glob(wildcard) if os.path.isdir(d)]:
          try:
              params = inputParams(runFolder + "/cil.json")
              seed = params['RUNTIME']['RNG']['SEED']
              runIDs[0] = min(runIDs[0], seed)
              runIDs[1] = max(runIDs[1], seed)
              
          except OSError:
              print("Whooopsie, run {} doesn't seem to have a correct data file.".format(runFolder))
  return runIDs
  
# read json file
def inputParams(jfile):
    with open(jfile) as jp:
        jp = json.load(jp)
        units = jp['UNITS']
        units['ENERGY']   = units['MOTILITY']*units['LENGTH']**2
        units['TIME']     = units['LENGTH']**2 * units['FRICTION'] / units['ENERGY']
        units['FORCE']    = units['ENERGY']/units['LENGTH']
        units['DSPATIAL'] = units['LENGTH']**2 / units['TIME']
        units['DANGULAR'] = 1.0/units['TIME']
        return jp


# compute pbc distance between r1 and r2 (r1,r2 are posititions or arrays of positions)
def distance(r1, r2, lbox):
    return r2 - r1 - np.around((r2-r1)/lbox)*lbox


# wrap positions around periodic boundaries
def pbc(r, lbox):
    return np.fmod(r + lbox, lbox)

# minimum image convention, same as distance function above
def mic(data, lbox):
  flip = np.abs(data) > 0.5 * lbox
  data[flip] -= lbox * np.sign(data[flip])
  return data
  
  
def com(R, lbox):
    """ Calculates the centers of mass of the molecules contained in the snapshot R, assuming that the first index iterates over the molecules, and assuming that the positions are for a square system of side length lbox with periodic boundary conditions.
    """
    nCells, nElements, nDim = np.shape(R)
    com = [cell[0] + np.sum(distance(cell[0], cell[1:], lbox), axis=0)/nElements for cell in R]
    return pbc(np.array(com),lbox)

# # Calculate trajectories and relevant cell averages from the disk coordinates rt and velocities vt. Assumes shape r[t, cell, disk, dim] and v[t, cell, disk, dim] with constant number of disks.
# ## Angles of cell positions are expressed with respect to x-axis
# # TODO Apply periodic boundary conditions to everything, with mic and distance functions.
# def cellData_JJM(rt, vt):
#   cellCenters =    np.average(rt, axis=2)
#   cellExtensions = rt[:,:,1,:] - rt[:,:,0,:]
#   cellVelocities = np.average(vt, axis=2)
#
#   cellSpeeds = np.linalg.norm(cellVelocities, axis = 2)
#   cellAngles = np.arctan2(cellExtensions[...,1], cellExtensions[...,0])
#   return cellCenters, cellExtensions, cellVelocities, cellSpeeds, cellAngles

def cellData(rt, vt, lbox):
  """ cellCenters, cellExtensions, cellVelocities, cellSpeeds, cellAngles = cellData(rt, vt, lbox)
  
  Calculates trajectories and relevant cell averages from the disk coordinates rt and velocities vt. Assumes constant number of cells and shape r[t, cell, disk, dim] and v[t, cell, disk, dim].
  """
  # nTimes, *rest, nDim = rt.shape
  #
  # if len(rest) == 1:
  #     if rest[0] == nElems:
  #         nCells = 1
  #     elif rest[0]%nElems == 0: # i.e. rest = nCells * nElems
  #         nCells = int(rest[0]/nElems)
  #     else:
  #         raise ValueError('Unexpected trajectory shape, maybe nElems is incorrect?')
  # elif len(rest) == 2:
  #     nCells = rest[0]
  #     if rest[1] != nElems:
  #         raise ValueError('Unexpected trajectory shape, maybe nElems is incorrect?')
  #
  # pos = rt.reshape((nTimes, nCells, nElems, nDim))
  # vel = vt.reshape((nTimes, nCells, nElems, nDim))

  cellCenters = np.array([com(p, lbox) for p in rt])
  cellExtensions = distance(rt[:,:,0,:], rt[:,:,-1,:], lbox)
  cellVelocities = np.average(vt, axis=2)

  cellSpeeds = np.linalg.norm(cellVelocities, axis=-1)
  cellAngles = np.arctan2(cellExtensions[...,1], cellExtensions[...,0])
  return cellCenters, cellExtensions, cellVelocities, cellSpeeds, cellAngles

# Calculate trajectories and relevant cell averages from the disk coordinates rt and velocities vt. Assumes shape r[t, disk, dim] and v[t, disk, dim] with constant number of disks.
## Angles of cell positions are expressed with respect to x-axis


# def cellData(rt, vt):
#   cellCenters =    (rt[:,1::2,:] + rt[:,::2,:])/2
#   cellExtensions =  rt[:,1::2,:] - rt[:,::2,:]
#   cellVelocities = (vt[:,1::2,:] + vt[:,::2,:])/2
#
#   cellSpeeds = np.linalg.norm(cellVelocities, axis = 2)
#   cellAngles = np.arctan2(cellExtensions[...,1], cellExtensions[...,0])
#   return cellCenters, cellExtensions, cellVelocities, cellSpeeds, cellAngles


# Calculate trajectories and relevant cell averages from the disk coordinates r and velocities v. Assumes shape r[cell, disk, dim] and v[cell, disk, dim]
## Angles of cell positions are expressed with respect to x-axis
## TODO Needs to account for periodic boundaries
def cellDataForSnapshot_JJM(r, v):
  cellCenters = np.average(r, axis=1)
  cellExtensions = r[:,1,:] - r[:,0,:]
  cellVelocities = np.average(v, axis=1)
  cellSpeeds = np.linalg.norm(cellVelocities, axis = 1)
  cellAngles = np.arctan2(cellExtensions[...,1], cellExtensions[...,0])
  return cellCenters, cellExtensions, cellVelocities, cellSpeeds, cellAngles



# # Calculate trajectories and relevant cell averages from the disk coordinates r and velocities v. Assumes shape r[disk, dim] and v[disk, dim]
# ## Angles of cell positions are expressed with respect to x-axis
# def cellDataForSnapshot(r, v):
#   cellCenters = (r[1::2,:] + r[::2,:])/2
#   cellExtensions = r[1::2,:] - r[::2,:]
#   cellAngles = np.arctan2(cellExtensions[...,1], cellExtensions[...,0])
#
#   cellVelocities = (v[1::2,:] + v[::2,:])/2
#   cellSpeeds = np.linalg.norm(cellVelocities, axis = 1)
#   return cellCenters, cellExtensions, cellVelocities, cellSpeeds, cellAngles

# def cellDataFromTwoSnapshots(s1, s2, timestep):
#   ## determine cell centers, only keep cells that were present in both snapshots
#   oldCenters = (s1[0::2,:] + s1[1::2,:])/2.
#   newCenters = (s2[0::2,:] + s2[1::2,:])/2.
#   newCenters = newCenters[:len(oldCenters),:]
#
#   ## Calculate cell speed from two snapshots
#   centerVelocities = (newCenters - oldCenters)/timestep
#   centerSpeeds = np.linalg.norm(centerVelocities, axis=1)
#   return newCenters, centerVelocities, centerSpeeds



def createRandomConfiguration(templateCell, nCells, criterion, lbox, tries=5e3):
  """Takes "templateCell" and tries to place it "nCells"-times in random places and random orientations inside the square box of size "lbox" (dim=2), and aborts the attempt after "tries". Cell particles must have a distance of "criterion" from any other cell particle.
  """
  R = []
  tried = 0
  while ((len(R) < nCells) and (tried < tries)):
    tried += 1
    angle = 2.*np.pi*np.random.random()
    position = np.random.random(2)*lbox
    new_cell = (rotation(angle) @ templateCell.T).T + position
    new_cell = pbc(new_cell, lbox)

    # Test for overlap with previous cells
    accept = True
    for disk in new_cell:
      for old_cell in R:
        for disk2 in old_cell:
          if(np.linalg.norm(distance(disk, disk2,lbox)) < criterion):
            accept = False
    if (accept):
      R.append(new_cell)
  return R


def createRandomConfiguration_circle(templateCell=np.array([[0.0, 0.0]]), nCells=10, criterion=1, radius=10, tries=5e3):
  """Takes "templateCell" and tries to place it "nCells"-times in random places and random orientations strictly inside a circle of size "radius" (dim=2) centered on the origin, and aborts the attempt after "tries". Cell particles must have a distance of "criterion" from any other cell particle. We assume that periodic boundary conditions can be ignored in this case.
  """

  # To properly sample space in the circle, we first generate positions uniformly in the square immediately enveloping the circle, and then reject positions that lie outside the circle
  R = []
  tried = 0
  while ((len(R) < nCells) and (tried < tries)):
    tried += 1
    angle = 2.*np.pi*np.random.random()
    position = (np.random.random(2)*2 - 1)*radius
    new_cell = (rotation(angle) @ templateCell.T).T + position

    # Test for overlap with previous cells and with the circle
    accept = True
    for disk in new_cell:
      # print(f"New disk at {disk}")
      if np.linalg.norm(disk) >= radius:
         accept = False
        #  break
      for old_cell in R:
        for disk2 in old_cell:
          if (np.linalg.norm(disk-disk2) < criterion):
            accept = False
            # break
        # if not accept:
          #  break
    if (accept):
      R.append(new_cell)

  # print(len(R), tried, tries, accept)
  return R

def unwrap_trajectories(rt, lbox):
  """
  Undoes the wrapping of the trajectories at the periodic boundaries
  """
  increments = rt[1:] - rt[:-1]
  unwrappedIncrements = mic(increments, lbox)

  unwrapped = np.zeros_like(rt)
  unwrapped[0] = rt[0]
  unwrapped[1:] = rt[0] + np.cumsum(increments, axis=0)
  return unwrapped


def flattenPositionsSomewhat(pos):
  """
  Transforms positions with shape [times, cells, disks per cell, dim] to [times, cells, disks per cell * dim] and [cells, disks per cell, dim] to [cells, disks per cell * dim]
  """
  return pos.reshape([*pos.shape[:-2], -1])
  
def reshapeForCells(pos):
  """
  Transforms positions with shape [times, cells, disks per cell, dim] to [times, disks, dim] and [cells, disks per cell, dim] to [disks, dim]
  """
  ps = pos.shape
  if pos.ndim == 3:
    return pos.reshape([ps[-3]*ps[-2],ps[-1]])
  if pos.ndim == 4:
    return pos.reshape([-1, ps[-3]*ps[-2],ps[-1]])
  else:
    raise ValueError("Input array needs to have dimension 3 or 4")



# compute local order parameter
def orderParameter(pos, vel, rcut):
    phi = np.zeros(vel.shape)
    for ri, i in zip(pos, range(pos.shape[0])):
        rij    = distance(ri, pos, lbox)
        drij   = np.linalg.norm(rij, axis=1)
        neighbors = (drij < rcut)
        phi[i] = np.sum(vel[neighbors], axis=0)/np.count_nonzero(neighbors)
    return phi
    
def polarOrderParameterFromCellAngles(cellAngles):
    """Calculate polar order parameter from cellAngles"""
    return np.sqrt(np.average(np.cos(cellAngles))**2 + np.average(np.sin(cellAngles))**2)    
    
def nematicOrderParameterFromCellAngles(cellAngles):
    """Calculate nematic order parameter from cellAngles"""
    return np.sqrt(np.average(np.cos(2*cellAngles))**2 + np.average(np.sin(2*cellAngles))**2)    
    
# Strip out all coordinates that are not inside visibleBox
def onlyVisible(xs, ys, visibleBox):
  xs = np.array(xs)
  ys = np.array(ys)
  visibleInd = np.where(xs >= visibleBox[0]-sigma/2.) and np.where(xs <= visibleBox[1]+sigma) and np.where(ys >= visibleBox[2]-sigma/2.) and np.where(ys <= visibleBox[3]+sigma/2.)
  return xs[visibleInd], ys[visibleInd]




def kymographs(positions, speeds, bins = [], mask = []):
  """
  bins, kymograph, kymograph_density = kymographs(positions, speeds, bins = [], mask = [])

  Calculates the speed and density kymographs from the positions and speeds.
  Assumes that the positions and speeds are in the shape (nTimes, nParticles),
  i.e. that the positions are 1-dimensional already.
  """
  
  ## Perform a test whether the shape is as expected
  if np.ndim(positions) != 2 or np.ndim(speeds) != 2:
    sys.exit("Shape of input is wrong. Need arrays to be 2-dimensional")
    
  ## Make a standard choice for the binning
  if len(bins) == 0:
    bins = np.linspace(positions.min(), positions.max(), 101)
  binSpacing = bins[1] - bins[0]

  ## If no mask is supplied, select all entries for analysis
  if len(mask) == 0:
    mask = np.ones(np.shape(positions)).astype('bool')

  kymograph = np.empty((positions.shape[0], len(bins)-1))
  kymograph[:] = np.nan
  kymograph_density = np.zeros((positions.shape[0], len(bins)-1))

  for tIndex in range(positions.shape[0]):
    # One way of quickly determining the density kymograph
    # dummyBins, kymograph_density[t] = np.histogram(angles[0], bins = nBins, range = (-1,1))

    for binIndex, left in enumerate(bins[:-1]):
      ind = (positions[tIndex][mask[tIndex]] > left) & (positions[tIndex][mask[tIndex]] <= left + binSpacing)
      if np.any(ind):
        kymograph[tIndex, binIndex] = np.average(speeds[tIndex][mask[tIndex]][ind])
        kymograph_density[tIndex, binIndex] = np.sum(ind)

  return bins, kymograph, kymograph_density

## Adds a cell, consisting of four coordinates to an existing axis
def addCellToAx(ax, cell, sigmas, color='0.8', edgecolor='none', 
                linewidth=-1., alpha=1, zorder=-1.1, resolution=20, clip_on=True):
  properties = {
    'edgecolor': edgecolor,
    'linewidth' : linewidth,
    'zorder' : zorder,
    'facecolor': color,
    'resolution': resolution,
    'alpha' : alpha,
    'clip_on'	: clip_on,
  }
  if isinstance(color, (list, tuple, np.ndarray)):
        properties['facecolor'] = color[0]
        
  ax.add_artist(CirclePolygon(cell[:2], sigmas[0]/2., **properties))
  # ax.add_artist(CirclePolygon(cell[:2], sigmas[0]/2., edgecolor='0.7', linewidth = 0.2, zorder = -1.1, \
  # facecolor='none'))

  if isinstance(color, (list, tuple, np.ndarray)):
        properties['facecolor'] = color[1]

  ax.add_artist(CirclePolygon(cell[2:], sigmas[1]/2, **properties))
#   ax.add_artist(CirclePolygon(cell[2:], sigmas[1]/2, edgecolor='0.7', linewidth = 0.2, zorder = -1.1, \
  # facecolor='none'))
  
def plotCells(ax, cells, sigmas, color='0.8', edgecolor='none', linewidth=-1., resolution = 10):
  for cell in cells:
    addCellToAx(ax, cell, sigmas, color, 
                edgecolor=edgecolor,
                linewidth = linewidth, 
                resolution = resolution)
  