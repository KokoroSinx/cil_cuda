import numpy as np
import gsd.hoomd
import json
import functools

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

# wrap around pbc boundaries
def pbc(r, lbox):
    return np.fmod(r + lbox, lbox)

# compute pbc distance r = r2 - r1
def distance(r1, r2, lbox):
    r12 = r2-r1
    return r12 - np.around(r12/lbox)*lbox

# Normalize vector
def normalize(x):
    return x / np.linalg.norm(x)

# map func over data lists
def maparray(func, *data):
    return np.array(list(map(func, *data)))

# compose functions i.e., f(g(h(x))) = compose(f, g, h)
# https://mathieularose.com/function-composition-in-python/
def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# correlate data frames
# input : (x0, x1, x2, ..., xn)
# output: (f(x0,xn), f(x0,x1), f(x0,x2), ..., f(x0,xn))
#def correlate(func, data):
#    def correlate0(x0, xt):
#        return maparray(lambda a,b: func(a,b), x0, xt)
#    return maparray(lambda xt: correlate0(data[0], xt), data)

# update circular buffer
# input:   databuffer = (x0,x1,x2,...,xn), newdata = y
# output:  databuffer = (x1,x2,...,xn,y)
def updateBuffer(databuffer, newdata):
    databuffer = np.roll(databuffer, -1, axis=0)
    databuffer[-1] = newdata
    return databuffer

#def cellOrientation(cellPos, box):
#    return maparray(compose(normalize, functools.partial(orientationFunc, box=box)), pos)

#compute two-point trajectory data
def calculTwoPoint(fname, dataFunc, conf):
    with gsd.hoomd.open(fname, mode='rb') as fp:
        istart,iend = conf['tskip'], len(fp) - conf['tau']
        data0 = dataFunc(fp[istart],fp[istart+conf['tau']], conf)
        data  = np.zeros((iend-istart,)+data0.shape)
        data[0] = data0
        for i in range(istart+1, iend):
            data[i-istart] = dataFunc(fp[i], fp[i+conf['tau']], conf)
        return data

#compute single-point trajectory data
def calculData(fname, dataFunc, conf):
    with gsd.hoomd.open(fname, mode='rb') as fp:
        istart, iend = conf['tskip'], len(fp)
        data0 = dataFunc(fp[istart], conf)
        data  = np.zeros((iend-istart,)+data0.shape)
        data[0] = data0
        for i in range(istart+1, iend):
            data[i-istart] = dataFunc(fp[i], conf)
        return data

# update partial average
def blockCumul0(count, avg, newdata):
    avg = (avg * count + newdata)/(count + 1)
    return count + 1, avg

# update cumulative block average and variance
def blockCumul(count, avg, sig, newdata):
    inewcount = 1.0 /(count + 1)
    sig = count*inewcount*(sig + inewcount*(avg - newdata)**2)
    avg = (avg*count + newdata)*inewcount
    return count+1, avg, sig

# Update average/variance with newdata using different blocksize per data element
# blockSize : n
# count     : (2,n)
# avg       : (2, n)
# sig       : n
# newdata   : n
def nblockAvgUpdate(blocksize, count, avg, sig, newdata):
    count[0], avg[0] = blockCumul0(count[0], avg[0], newdata)
    block = (blocksize == count[0])
    if(np.any(block)):
        count[1,block], avg[1,block], sig[block] = blockCumul(count[1,block], avg[1,block], sig[block], avg[0,block])
        count[0,block], avg[0,block] = 0, 0.0
    return count, avg, sig
# copute block average error given number of samples and variance
def nblockAvgError(count, sig):
    norm, pick = np.ones_like(count, dtype=np.float), count > 1
    norm[pick] = 1.0/np.sqrt(count[pick] - 1)
    error      = np.sqrt(sig)*norm
    return error, error*norm/np.sqrt(2.0)

# Update average/variance with newdata using single blocksize for whole data range
def blockAvgUpdate(blocksize, count, avg, sig, newdata):
    count[0], avg[0] = blockCumul0(count[0], avg[0], newdata)
    if(blocksize == count[0]):
        count[1], avg[1], sig = blockCumul(count[1], avg[1], sig, avg[0])
        count[0], avg[0] = 0, 0.0
    return count, avg, sig
# copute block average error given number of samples and variance
def blockAvgError(count, sig):
    norm = 1.0
    if count > 1:
        norm = 1.0 / np.sqrt(count - 1)
    error= np.sqrt(sig)*norm
    return error, error*norm/np.sqrt(2.0)
    return error, error*norm/np.sqrt(2.0)

#compute block average
def blockAvg(data, blocksize):
    count, avg, sig = np.zeros((2,1), dtype=np.int64), np.zeros((2,1)), np.zeros(1)
    for xi in data:
        count, avg, sig = blockAvgUpdate(blocksize, count, avg, sig, xi)
    err, derr = blockAvgError(count[1], sig)
    return count[1,0], avg[1,0], err[0], derr[0]
#compute block average as a function of blocksize (2^n)
def blockAvgScan(data):
    size  = np.power(2, np.arange(0, int(np.log2(len(data))) + 1, dtype=np.int64))
    shape = (2,) + size.shape
    count, avg, sig = np.zeros(shape, dtype=np.int64), np.zeros(shape), np.zeros(size.shape)
    for xi in data:
        count, avg, sig = nblockAvgUpdate(size, count, avg, sig, xi)
    err, derr = nblockAvgError(count[1], sig)
    return np.log2(size), count[1], avg[1], err, derr


# Compute mean square displacement
def calculMSD(fname, posFunc, distFunc, msdFunc, conf):
    with gsd.hoomd.open(name=fname, mode='rb') as fp:
        #fill buffer
        istart,iend = conf['tskip'], conf['tskip']+conf['tau']
        pos0        = posFunc(fp[istart], conf)
        buffer      = np.zeros((conf['tau'],) + pos0.shape)
        buffer[0]   = pos0
        for i in range(istart+1, iend):
            it   = i - istart
            pos  = posFunc(fp[i], conf)
            buffer[it] = buffer[it-1] + distFunc(pos0, pos)
            pos0 = pos
        istart, iend = iend, len(fp) - conf['tau']
        
        #compute msd
        msd0 = msdFunc(buffer)
        msdCnt = np.zeros(2, dtype=np.int64)
        msdAvg, msdSig = np.zeros((2,)+msd0.shape), np.zeros(msd0.shape)
        msdCnt, msdAvg, msdSig = blockAvgUpdate(conf['blocksize'], msdCnt, msdAvg, msdSig, msd0)
        for i in range(istart, iend):
            pos  = posFunc(fp[i], conf)
            buffer = updateBuffer(buffer, buffer[-1]+distFunc(pos0, pos))
            msd0 = msdFunc(buffer)
            msdCnt, msdAvg, msdSig = blockAvgUpdate(conf['blocksize'], msdCnt, msdAvg, msdSig, msd0)
            pos0 = pos
        msdErr, msdDErr = blockAvgError(msdCnt[1], msdSig)
        return msdCnt[1], np.transpose([msdAvg[1], msdErr, msdDErr])

def calculCt(fname, dataFunc, corrFunc, conf):
    with gsd.hoomd.open(name=fname, mode='rb') as fp:

        # fill buffer
        istart, iend = conf['tskip'], conf['tskip']+conf['tau']
        x0           = dataFunc(fp[istart], conf)
        buffer       = np.zeros((conf['tau'],) + x0.shape)
        buffer[0]    = x0
        for i in range(istart+1, iend):
            buffer[i - istart] = dataFunc(fp[i], conf)
        istart, iend = iend, len(fp) - conf['tau']
        
        # compute correlation
        # np.average(correlate(corrFunc, buffer), axis=1)
        ct0   = corrFunc(buffer)
        ctCnt = np.zeros(2, dtype=np.int64)
        ctAvg, ctSig = np.zeros((2,)+ct0.shape), np.zeros(ct0.shape)
        ctCnt, ctAvg, ctSig = blockAvgUpdate(conf['blocksize'], ctCnt, ctAvg, ctSig, ct0)
        for i in range(istart, iend):
            buffer = updateBuffer(buffer, dataFunc(fp[i], conf))
            ct0 = corrFunc(buffer)
            ctCnt, ctAvg, ctSig = blockAvgUpdate(conf['blocksize'], ctCnt, ctAvg, ctSig, ct0)
        ctErr, ctDErr = blockAvgError(ctCnt[1], ctSig)
        return ctCnt[1], np.transpose([ctAvg[1], ctErr, ctDErr])

