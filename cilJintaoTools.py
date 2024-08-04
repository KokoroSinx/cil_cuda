#!/usr/bin/env python
import numpy as np
import os
from pathlib import Path

import h5py
import gsd.hoomd
import matplotlib.pyplot as plt
import matplotlib as mpl

import cil
import simulationtools as st
from figuresetup import setAxisLabels, letterLabels

# TODO Expand similarly as averageVelocities_JJM.py for averaging over multiple simulations

# This script has to be run from the main folder of the simulation.

# Prepare output file
# outputfile = f'quantities_v{version}.dat'
# f = open(outputfile, 'w')
# header = ""
# header += "# Output of nematicOrderParameter.py, v.{}\n".format(version)
# header += "# Nematic order parameter \n"
# # header += "# for the following systems (given by wildcards)\n"
# # for wildcard in wildcards:
# #   header += "#  {}\n".format(wildcard)
# header += "# Data format: #1 time #2 average nematic order\n"
# f.write(header)


# TODO:
# DONE Read the trajectory data from the h5 file
# DONE find out what all the other cell data means
# DONE determine snapshot timestamps
# DONE analyse colony size
# DONE analyse cell sizes
# DONE analyse division times
# DONE spatial plot of cell sizes
# DONE spatial plot of activity/proliferation rate
# - Output
# - Average with other simulation data

version = "0.3.2"
version_gsd = "gsd_0.2"

def analyse_folder(folder, datafile, paramfile):
    folder = Path(folder)

    # TODO Iterate over the folders:
    # TODO Make sure that all parameters except seed match

    latticeSpacing = 5 # for colony area estimation


    # Store all the calculated results in a  dictionary for storage and averaging
    data = {}
    data['version'] = version

    params = cil.inputParams(folder / paramfile)
    seed = params['RUNTIME']['RNG']['SEED']
    unitLength = params['UNITS']['LENGTH']
    dt = params['MD']['DT']['MANUAL']
    gts = params['MD']['GTS']
    frames = params['MD']['FRAMES']
    lbox = np.array(params['MD']['BOX'])
    nElem = params['SPECIES']['C1']['NELEM']
    tauDiv = params['SPECIES']['C1']['DIVISION']['TAU_DIVISION']['AVG']
    sigmaMin = params['SPECIES']['C1']['DIVISION']['CYCLE_SIGMA_MIN']
    v1 = params['SPECIES']['C1']['DIVISION']['V1']


    # time between snapshots in units of division times
    snapshotInterval = dt * gts / tauDiv

    with h5py.File(folder / datafile, 'r+') as f:
        snapshots = f['snapshots']
        max_range = len(snapshots)
        timesteps = []
        times = []
        numbers = []
        cellAreas = []
        areaMean = []
        thetas = []
        rhos = []
        pressures = []
        T0s = []

        distances_radial = []
        areas_radial = []
        rhos_radial = []
        pressures_radial = []
        counts_radial = []

        # Calculate average colony densities: nondimensionalised density (via max. colony radius, i.e. the maximum distance a cell has from the colony center) and nondimensionalised density (via lattice area).
        densitiesFromMaxColonyRadius = []
        densitiesFromLatticeArea = []

        # For now, calculate the colony origin only at the beginning of the simulation.
        # origin = np.array(snapshots['t_0']['R'][0][0])
        origin = np.mean(snapshots['t_0']['R'], axis=0)

        for i in range(0, max_range, 1):
            name = 't_' + str(i)
            snapshot = snapshots[name]
            timesteps.append(i*gts)
            times.append(i*snapshotInterval)
            pos = np.array(snapshots[name]['R'])
            cellDiameters = np.array(snapshot['SigmaCore'])
            # tempR = snapshot['R']
            # tempVel = snapshot['Vel']
            numbers.append(len(cellDiameters))
            area = np.pi*((cellDiameters/2)**2)
            cellAreas.append(np.array(area))
            areaMean.append(np.mean(area))
            thetas.append(np.array(snapshot['Theta']))
            rhos.append(np.array(snapshot['RHO']))
            pressures.append(np.array(snapshot['PR']))
            T0s.append(np.array(snapshot['T0']))

            # RADIAL ANALYSIS OF THE COLONY
            # 1) CALCULATE THE DISTANCE OF EACH CELL FROM THE COLONY CENTER
            # shift the colony to the center of the box (I sometimes accidentally don't put the first cell into the center when initialising the sim):
            # TODO Simply calculate the COM for each snapshot and center around that. Could be useful especially for motile colonies
            # newCom = np.average(pos, axis=0)
            newCom = lbox/2
            pos = pos - origin + newCom
            pos = cil.pbc(pos, lbox)
            centerDistances = np.linalg.norm(pos - newCom, axis=2).flatten()
            colonyRadius = np.max(centerDistances)
            # 2) BIN THE DATA 
            distanceBinCenters, quantityBinned, quantityBinnedMedian, quantityBinnedSTD, counts = st.bin_y_according_to_x(
                centerDistances, rhos[-1], xMin=0, xDelta=2)
            distances_radial.append(distanceBinCenters)
            counts_radial.append(counts)
            rhos_radial.append(quantityBinned)

            distanceBinCenters, quantityBinned, quantityBinnedMedian, quantityBinnedSTD, counts = st.bin_y_according_to_x(
                centerDistances, cellAreas[-1], xMin=0, xDelta=2)
            areas_radial.append(quantityBinned)

            distanceBinCenters, quantityBinned, quantityBinnedMedian, quantityBinnedSTD, counts = st.bin_y_according_to_x(
                centerDistances, pressures[-1], xMin=0, xDelta=2)
            pressures_radial.append(quantityBinned)

            densitiesFromMaxColonyRadius.append(numbers[-1]/(np.pi*colonyRadius**2))
            occupiedLatticeArea, occupiedLatticeSites = st.occupied_area_on_lattice(points=pos, radii=cellDiameters/2, lbox=np.max(lbox), latticeSpacing=latticeSpacing)
            densitiesFromLatticeArea.append(numbers[-1]/occupiedLatticeArea)

    times = np.array(times)
    areaMean = np.array(areaMean)
    data['t'] = times
    data['cellNumber'] = numbers
    data['cellAreaMean'] = areaMean
    data['colonyArea'] = numbers*areaMean
    data['densityFromMaxColonyRadius'] = densitiesFromMaxColonyRadius
    data['densityFromLatticeArea'] = densitiesFromLatticeArea
    
    # POSTPROCESS THE RADIAL DISTRIBUTIONS SO THEY FIT INTO ARRAYS
    # assumes that the radial distances all start at the same xMin and have equal xDelta
    distances_radial_final = distances_radial[0]
    for distances_slice in distances_radial:
        if len(distances_radial_final) < len(distances_slice):
            distances_radial_final = distances_slice
    # distances_radial_final = distances_radial[-1]
    shape_final = [len(distances_radial), np.shape(distances_radial_final)[0]]
    counts_radial_final = np.zeros(shape_final)
    for idx, counts_slice in enumerate(counts_radial):
        counts_radial_final[idx][:len(counts_slice)] = counts_slice
    np.nan_to_num(counts_radial_final, copy=False)
    rhos_radial_final = np.empty(shape_final)
    rhos_radial_final[:] = np.nan
    for idx, rhos_slice in enumerate(rhos_radial):
        rhos_radial_final[idx][:len(rhos_slice)] = rhos_slice
    areas_radial_final = np.empty(shape_final)
    areas_radial_final[:] = np.nan
    for idx, areas_slice in enumerate(areas_radial):
        areas_radial_final[idx][:len(areas_slice)] = areas_slice
    pressures_radial_final = np.empty(shape_final)
    pressures_radial_final[:] = np.nan
    for idx, pressures_slice in enumerate(pressures_radial):
        pressures_radial_final[idx][:len(pressures_slice)] = pressures_slice

    data['countsRadial'] = counts_radial_final
    data['distanceBinCentersRadial'] = distances_radial_final
    data['areaRadial'] = areas_radial_final
    data['rhoRadial'] = rhos_radial_final
    data['pressureRadial'] = pressures_radial_final

    # Cell size distributions over time (not normalised yet)
    cellAreaBinEdges = np.arange(0.5, 16, 0.5)
    cellAreaBinCenters = (cellAreaBinEdges[1:]+cellAreaBinEdges[:-1])/2.
    cellAreaDistributions = []
    for ca in cellAreas:
        histogram, _ = np.histogram(ca, bins=cellAreaBinEdges, density=True)
        cellAreaDistributions.append(histogram)
    data['cellAreaBinEdges'] = cellAreaBinEdges
    data['cellAreaBinCenters'] = cellAreaBinCenters
    data['cellAreaDistributions'] = cellAreaDistributions

    # Calculate histograms of cell activty
    rhoBinEdges = np.arange(0, 1.1, 0.025)
    rhoBinCenters = (rhoBinEdges[1:]+rhoBinEdges[:-1])/2.
    rhoDistributions = []
    for rho in rhos:
        # print(rho)
        histogram, _ = np.histogram(rho, bins=rhoBinEdges, density=True)
        rhoDistributions.append(histogram)
    data['rhoBinEdges'] = rhoBinEdges
    data['rhoBinCenters'] = rhoBinCenters
    data['rhoDistributions'] = rhoDistributions


    # For each division, record area at division, duration since the last division, and the colony age when the division happened
    # TODO Write this as a function directly acting on the bare h5 file.
    divisionAreas = []
    divisionTimes = []
    divisionColonyAge = []
    for idx in range(10, len(times)):
        # ... the length difference between two consecutive snapshots tells me how many divisions there were
        nDivisionsInLastInterval = len(thetas[idx]) - len(thetas[idx-1])

        if nDivisionsInLastInterval > 0:
            # ... infer division events from where theta wraps back to 0
            divisionEvents = thetas[idx][:-nDivisionsInLastInterval] < thetas[idx-1]
            # ... and read off birth time from T0 
            birthTimesteps = T0s[idx-1][divisionEvents]
            # .. and how old the cell was at division 
            divisionTimesteps = np.ones((nDivisionsInLastInterval,))*timesteps[idx-1] # lower limit for the timestep at which the cells divided
            # Extrapolate with an Euler integration step when the thetas would have hit 1 and thus triggered a division
            extrapolatedTimes = np.minimum((1.0 - thetas[idx-1][divisionEvents]) / rhos[idx-1][divisionEvents], 0.1)
            [divisionTimes.append(time) for time in extrapolatedTimes + (
                divisionTimesteps - birthTimesteps)*dt/tauDiv]
            # ... also calculate the premitotic sizes (lower bound)
            [divisionAreas.append(ca) for ca in cellAreas[idx-1][divisionEvents]] 
            # ... and record the colony age
            [divisionColonyAge.append(extrapolatedTime + times[idx-1])
             for extrapolatedTime in extrapolatedTimes]
    # ... now bin the data so that we don't have to save everything. We need: binEdges, binCenters, how many divisions in each bin, mean and stddev of the distribution in the bin.
    data['divisionAreas'] = divisionAreas
    data['divisionTimes'] = divisionTimes
    data['divisionColonyAge'] = divisionColonyAge

    # Calculate histograms of all division rates over the whole simulation
    # Caveat: this might not be a particularly well defined quantity. Perhaps look at this only over smaller time intervals?
    divisionBinEdges = np.arange(0, 10, 0.125)
    divisionBinCenters = (divisionBinEdges[1:]+divisionBinEdges[:-1])/2.
    # divisionDistributions = []
    divisionTimeDistribution, _ = np.histogram(divisionTimes, bins=divisionBinEdges, density=True)
    # rhoDistributions.append(histogram)
    data['divisionBinEdges'] = divisionBinEdges
    data['divisionBinCenters'] = divisionBinCenters
    data['divisionTimeDistribution'] = divisionTimeDistribution
    return data


def analyse_folder_gsd(folder, datafile, paramfile):
    folder = Path(folder)
    data = {}
    data['version'] = version_gsd

    params = cil.inputParams(folder / paramfile)
    dt = params['MD']['DT']['MANUAL']
    gts = params['MD']['GTS']
    tauDiv = params['SPECIES']['C1']['DIVISION']['TAU_DIVISION']['AVG']
    lbox = params['MD']['BOX']

    # time between snapshots in units of division times
    snapshotInterval = dt * gts / tauDiv

    with gsd.hoomd.open(folder / datafile, "rb") as fp:
        trajectoryLength = len(fp)
        snapshot_indices = range(0, trajectoryLength)
        timesteps = []
        times = []
        numbers = []
        cellAreas = []
        areaMean = []
        rhos = []
        T0s = []
        for snapshot_idx in snapshot_indices:
            si = fp.read_frame(snapshot_idx)
            timesteps.append(si.configuration.step)
            # print(si.configuration.step)
            times.append(dt*si.configuration.step)
            cellDiameters = np.array(si.particles.diameter)
            numbers.append(si.particles.N)
            area = np.pi*((cellDiameters/2)**2)
            cellAreas.append(np.array(area))
            areaMean.append(np.mean(area))
            rhos.append(np.array(si.particles.charge))
            T0s.append(si.particles.mass)
    times = np.array(times)
    areaMean = np.array(areaMean)
    data['t'] = times
    data['cellNumber'] = numbers
    data['cellAreaMean'] = areaMean
    data['colonyArea'] = numbers*areaMean

    # Cell size distributions over time (not normalised yet)
    cellAreaBinEdges = np.arange(0.5, 16, 0.5)
    cellAreaBinCenters = (cellAreaBinEdges[1:]+cellAreaBinEdges[:-1])/2.
    cellAreaDistributions = []
    for ca in cellAreas:
        histogram, _ = np.histogram(ca, bins=cellAreaBinEdges, density=True)
        cellAreaDistributions.append(histogram)
    data['cellAreaBinEdges'] = cellAreaBinEdges
    data['cellAreaBinCenters'] = cellAreaBinCenters
    data['cellAreaDistributions'] = cellAreaDistributions

    # Calculate histograms of cell activity
    rhoBinEdges = np.arange(0, 1.1, 0.025)
    rhoBinCenters = (rhoBinEdges[1:]+rhoBinEdges[:-1])/2.
    rhoDistributions = []
    for rho in rhos:
        histogram, _ = np.histogram(rho, bins=rhoBinEdges, density=True)
        rhoDistributions.append(histogram)
    data['rhoBinEdges'] = rhoBinEdges
    data['rhoBinCenters'] = rhoBinCenters
    data['rhoDistributions'] = rhoDistributions

    return data

def average_quantities(folders):
    data_av = {}
    data_av['folders'] = []
    data = np.load(Path(folders[0]) / "quantities.npz")
    data_av['folders'].append(folders[0])
    for key, value in data.items():
        data_av[key] = value
    counter = 1
    data_av['cellNumber'] = data_av['cellNumber'].astype(float)

    for folder in folders[1:]:
        data = np.load(Path(folder) / "quantities.npz")
        data_av['folders'].append(folder)
        for key in ['version', 't', 'cellAreaBinEdges', 'cellAreaBinCenters', 'rhoBinEdges', 'rhoBinCenters', 'divisionBinEdges', 'divisionBinCenters']:
            if np.any(data[key] != data_av[key]):
                sys.exit("Keys don't match")
        for key in ['divisionAreas', 'divisionTimes', 'divisionColonyAge']:
            data_av[key] = np.concatenate((data_av[key], data[key]))
        for key in ['cellNumber', 'cellAreaMean', 'colonyArea', 'cellAreaDistributions', 'rhoDistributions', 'divisionTimeDistribution']:
            data_av[key] += data[key]
        counter += 1

    for key in ['cellNumber', 'cellAreaMean', 'colonyArea', 'cellAreaDistributions', 'rhoDistributions', 'divisionTimeDistribution']:
        data_av[key] /= counter
    return data_av


def plot_quantities_from_dict(folder, data, plotIdxs=[70, 100, 150, 200], unitArea = 1., figurefilename="Areas_speeds_divisions.pdf"):
    times = data['t']
    colonyArea = data['colonyArea']*unitArea

    # Check if the plotIdxs are actually in the data
    idxs = [plotIdx for plotIdx in plotIdxs if plotIdx <= len(times)]  # time indices for the distributions

    folder = Path(folder)
    if not os.path.exists(folder / "figures"):
        os.makedirs(folder / "figures")

    fig, Axs = plt.subplots(nrows=4, ncols=3, figsize=[12, 12], dpi=300)
    axs = Axs[0]
    # Colony area over time
    axs[0].semilogy(times, colonyArea)
    setAxisLabels(axs[0], "Time $t$", "Colony area $A$")
    # (average) Colony radius over time
    colonyRadius = np.sqrt(colonyArea/np.pi)
    axs[1].semilogy(times, colonyRadius)
    setAxisLabels(axs[1], "Time $t$", "Colony radius $R = \sqrt{A/\pi}$")
    axs[2].plot(times, data['cellNumber'])
    setAxisLabels(axs[2], "Time $t$", "Cell Number $N$")


    axs = Axs[1]
    # Colony radial speed
    colonyRadiusSpeed = st.derive(times, colonyRadius)
    axs[0].plot(times, colonyRadiusSpeed, label='full data')
    axs[0].plot(st.movavg(times, 10), st.movavg(
        colonyRadiusSpeed, 10), 'k', label='moving average')
    setAxisLabels(axs[0], "Time $t$", "$dR/dt$")
    axs[0].legend()

    # # Average cell area over time
    axs[1].plot(times, data['cellAreaMean']*unitArea)
    # Median area over time
    # cellAreaMedians = [np.median(ca) for ca in cellAreas]
    # axs[1].plot(times, cellAreaMedians)
    setAxisLabels(axs[1], "Time $t$", "Mean cell area")

    # Cell size distributions over time (not normalised yet)
    for idx in idxs:
        axs[2].plot(data['cellAreaBinCenters']*unitArea, data['cellAreaDistributions'][idx], '.-',
                    label=f"${np.average(times[idx]):.1f}$")
    # # average over m last samples
    # m = 3
    # axs[2].plot(binCenters, np.average(cellAreaDistributions[-m:], axis=0), label=f"$t = {np.average(times[-m:])}$")
    axs[2].legend(loc='upper center', title="$t=$", ncol=2)
    setAxisLabels(axs[2], "Cell area $a$", "Cell area distribution $p(a)$")
    
    axs = Axs[2]
    # Plot division times as a function of areas
    if 'divisionAreas' in data:
        divtimesplot = axs[0].scatter(data['divisionAreas']*unitArea, data['divisionTimes'],
                                      c=data['divisionColonyAge'], s=1, rasterized=True)
        axs[0].set_yscale('log')
        plt.colorbar(divtimesplot, ax=axs[0], label='Colony age at division')
    else:
        axs[0].text(
            0.5, 0.5, "divisionAreas are\nnot contained in the data.", ha='center', transform=axs[0].transAxes)
    setAxisLabels(axs[0], "Premitotic cell area $a$",
                "Division time $t_{div}$")

    for idx in idxs:
        axs[1].plot(data['rhoBinCenters'], data['rhoDistributions'][idx], '.-',
                    label=f"${np.average(times[idx]):.1f}$")
    axs[1].legend(loc='upper center', title="$t=$", ncol=2)
    setAxisLabels(axs[1], "Cell activity $r$",
                  "Cell activity distribution $p(r)$", )

    if 'divisionTimeDistribution' in data:
        axs[2].plot(data['divisionBinCenters'],
                    data['divisionTimeDistribution'], '.-')
    else:
        axs[2].text(
            0.5, 0.5, "divisionTimeDistribution are\nnot contained in the data.", ha='center', transform=axs[2].transAxes)
    setAxisLabels(
            axs[2], "Division time $t_{div}$", "Division time distribution $p(t_{div})$")
    
    axs = Axs[3]
    ax = axs[0]
    dataShape = np.shape(data['rhoRadial'])
    times_p = np.repeat(data['t'], dataShape[-1]).reshape((-1, dataShape[-1]))
    plotMappable = ax.scatter(data['rhoRadial'], data['pressureRadial'], c=times_p)
    plt.colorbar(plotMappable, label='Colony age $t$')
    ax.set_xlabel("Radially averaged cell cycle activity $r$")
    ax.set_ylabel("Radially averaged pressure $p$")
    # ax.set_title("Radially averaged pressure vs activity")

    if 'densityFromMaxColonyRadius' in data:
        axs[1].plot(times, data['densityFromMaxColonyRadius']/unitArea, '.-', label="max. colony radius")
    if 'densityFromLatticeArea' in data:
        axs[1].plot(times, data['densityFromLatticeArea']/unitArea, '.-', label="lattice area")
    axs[1].plot(times, 1/(data['cellAreaMean']*unitArea), '.-', label="$1/\langle$cell area$\\rangle$")
    setAxisLabels(
            axs[1], "Time $t$", "Density: cells/(unit area)")
    axs[1].legend(loc='lower right', title="calculated from", ncol=1)

    letterLabels(Axs, x=0.05, y=0.95, fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(folder) / "figures" / figurefilename)


def plot_radial_quantities(folder, data, unitArea = 1., figurefilename="radial.pdf"):
    """Plot the radial distributions in a naive way"""
    folder = Path(folder)
    if not os.path.exists(folder / "figures"):
        os.makedirs(folder / "figures")
    
    extent = [data['distanceBinCentersRadial'].min(), data['distanceBinCentersRadial'].max(), data['t'].max(), data['t'].min()]
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=[5, 7], dpi=300)
    areaMappable = axs[0].imshow(data['areaRadial'], aspect="auto", extent=extent)
    plt.colorbar(areaMappable)
    rhoMappable = axs[1].imshow(data['rhoRadial'], aspect="auto", extent=extent)
    plt.colorbar(rhoMappable)
    pressureMappable = axs[2].imshow(data['pressureRadial'], aspect="auto", extent=extent)
    plt.colorbar(pressureMappable)
    countsMappable = axs[3].imshow(data['countsRadial'], aspect="auto", extent=extent)
    plt.colorbar(countsMappable)
    for ax, title in zip(axs, ["Avg. cell area", "Avg. activity", "Avg. pressure", "no. of cells per pixel"]):
        ax.set_xlabel("Distance from colony center")
        ax.set_ylabel("Time")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(Path(folder) / "figures" / figurefilename)

# def plot_p_vs_r(folder, data, unitArea = 1., figurefilename="p_vs_r.pdf"):
#     """Plot p vs r using the pre-computed radial distributions"""
#     folder = Path(folder)
#     if not os.path.exists(folder / "figures"):
#         os.makedirs(folder / "figures")
    
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5, 7], dpi=300)
#     dataShape = np.shape(data['rhoRadial'])
#     times = np.repeat(data['t'], dataShape[-1]).reshape((-1, dataShape[-1]))
#     plotMappable = ax.scatter(data['rhoRadial'], data['pressureRadial'], c=times)
#     plt.colorbar(plotMappable, label='Colony age $t$')
#     ax.set_xlabel("Cell cycle activity $r$")
#     ax.set_ylabel("Pressure $p$")
#     ax.set_title("Radially averaged pressure vs activity")
#     fig.tight_layout()
#     fig.savefig(Path(folder) / "figures" / figurefilename)

#同时读取两个dict来绘图
def plot_quantities_from_two_dicts(folder, data1, data2, plotIdxs=[70, 100, 150, 200], unitArea=1., figurefilename="Areas_speeds_divisions_with_two_dict.pdf"):
    times1 = data1['t']
    times2 = data2['t']
    colonyArea1 = data1['colonyArea'] * unitArea
    colonyArea2 = data2['colonyArea'] * unitArea

    # Check if the plotIdxs are actually in the data
    idxs1 = [plotIdx for plotIdx in plotIdxs if plotIdx <= len(times1)]  # time indices for the distributions
    idxs2 = [plotIdx for plotIdx in plotIdxs if plotIdx <= len(times2)]  # time indices for the distributions

    folder = Path(folder)
    if not os.path.exists(folder / "figures"):
        os.makedirs(folder / "figures")

    fig, Axs = plt.subplots(nrows=4, ncols=3, figsize=[12, 12], dpi=300)
    axs = Axs[0]
    # Colony area over time
    axs[0].semilogy(times1, colonyArea1, label='Data 1')
    axs[0].semilogy(times2, colonyArea2, label='Data 2')
    setAxisLabels(axs[0], "Time $t$", "Colony area $A$")
    axs[0].legend()
    # (average) Colony radius over time
    colonyRadius1 = np.sqrt(colonyArea1/np.pi)
    colonyRadius2 = np.sqrt(colonyArea2/np.pi)
    axs[1].semilogy(times1, colonyRadius1, label='Data 1')
    axs[1].semilogy(times2, colonyRadius2, label='Data 2')
    setAxisLabels(axs[1], "Time $t$", "Colony radius $R = \sqrt{A/\pi}$")
    axs[1].legend()
    axs[2].plot(times1, data1['cellNumber'], label='Data 1')
    axs[2].plot(times2, data2['cellNumber'], label='Data 2')
    setAxisLabels(axs[2], "Time $t$", "Cell Number $N$")
    axs[2].legend()

    axs = Axs[1]
    # Colony radial speed
    colonyRadiusSpeed1 = st.derive(times1, colonyRadius1)
    colonyRadiusSpeed2 = st.derive(times2, colonyRadius2)
    axs[0].plot(times1, colonyRadiusSpeed1, label='Data 1')
    axs[0].plot(times2, colonyRadiusSpeed2, label='Data 2')
    axs[0].plot(st.movavg(times1, 10), st.movavg(
        colonyRadiusSpeed1, 10), 'k', label='Data 1 moving average')
    axs[0].plot(st.movavg(times2, 10), st.movavg(
        colonyRadiusSpeed2, 10), 'r', label='Data 2 moving average')
    setAxisLabels(axs[0], "Time $t$", "$dR/dt$")
    axs[0].legend()

    # # Average cell area over time
    axs[1].plot(times1, data1['cellAreaMean']*unitArea, label='Data 1')
    axs[1].plot(times2, data2['cellAreaMean']*unitArea, label='Data 2')
    setAxisLabels(axs[1], "Time $t$", "Mean cell area")
    axs[1].legend()

    # Cell size distributions over time (not normalised yet)
    for idx1, idx2 in zip(idxs1, idxs2):
        axs[2].plot(data1['cellAreaBinCenters']*unitArea, data1['cellAreaDistributions'][idx1], '.-',
                    label=f"Data 1 $t={np.average(times1[idx1]):.1f}$")
        axs[2].plot(data2['cellAreaBinCenters']*unitArea, data2['cellAreaDistributions'][idx2], '.-',
                    label=f"Data 2 $t={np.average(times2[idx2]):.1f}$")
    axs[2].legend(loc='upper center', title="$t=$", ncol=2)
    setAxisLabels(axs[2], "Cell area $a$", "Cell area distribution $p(a)$")

    # Remaining plotting code for other quantities...
    # ...

    letterLabels(Axs, x=0.05, y=0.95, fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(folder) / "figures" / figurefilename)


#何腾蛟的想法到此结束
def plot_snapshots(folder, datafile, paramfile, plotIdxs=[70, 100, 150]):
    folder = Path(folder)
    lenPlotIdxs = np.shape(plotIdxs)[0]
    if not os.path.exists(folder / "figures"):
        os.makedirs(folder / "figures")
    fig, Axs = plt.subplots(nrows=2, ncols=lenPlotIdxs+1, figsize=[2.5*lenPlotIdxs+1, 5], gridspec_kw={
                            'width_ratios': [*np.ones(lenPlotIdxs,), 0.06]}, dpi=300)
    cmap = mpl.colormaps['viridis']

    params = cil.inputParams(folder / paramfile)
    dt = params['MD']['DT']['MANUAL']
    gts = params['MD']['GTS']
    lbox = params['MD']['BOX']
    tauDiv = params['SPECIES']['C1']['DIVISION']['TAU_DIVISION']['AVG']
    sigmaMin = params['SPECIES']['C1']['DIVISION']['CYCLE_SIGMA_MIN']
    v1 = params['SPECIES']['C1']['DIVISION']['V1']
    # time between snapshots in units of division times
    snapshotInterval = dt * gts / tauDiv

    with h5py.File(folder / datafile, 'r+') as f:
        snapshots = f['snapshots']

        sigmaMin = params['SPECIES']['C1']['DIVISION']['CYCLE_SIGMA_MIN']

        for colIdx, time in enumerate(plotIdxs):
            name = 't_' + str(time)
            pos = snapshots[name]['R']
            sigmas = np.array(snapshots[name]['SigmaCore'])
            activities = np.array(snapshots[name]['RHO'])

            import matplotlib.collections
            patches = [plt.Circle(cellPos[0], sigma/2)
                       for cellPos, sigma, activity in zip(pos, sigmas, activities)]
            for label, rowIdx, colors in zip(('Cell activity $r$', 'Cell area $A/A_1$'), range(2), (cmap(activities), cmap((sigmas/sigmaMin)**2./v1))):
                ax = Axs[rowIdx, colIdx]
                ax.set_title(f"{label} at $t = {time*snapshotInterval}$")
                coll = matplotlib.collections.PatchCollection(
                    patches, facecolors=colors)
                coll.set_rasterized(True)
                ax.add_collection(coll)
                # ax.set_rasterized(True)

                # colors = cmap((sigmas/sigmaMin)**2./v1)
                # coll_size = matplotlib.collections.PatchCollection(
                # patches, facecolors=colors)
                # ax.add_collection(coll_size)

                ax.set_xlim([0, lbox[0]])
                ax.set_ylim([0, lbox[1]])
                ax.axis('equal')

                cax = Axs[rowIdx, -1]
                plt.colorbar(coll, ax=ax, cax=cax, label=label)
    # Axs[1,-1].axis('off')
    letterLabels(Axs[:2, :-1], x=0.05, y=0.95, fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / "figures/Heterogeneity.pdf")


# # TODO Implement the plotting code so that both gsd and h5 files simply call into a common function
# def plot_snapshot_dicts(folder, plotTimes, snapshots, sigmaMin, v1, lbox):
#     """
#     Plot snapshot dictionaries into a file called "folder/figures/Heterogeneity.pdf" corresponding to the times listed in plotTimes. Each dictionary snapshot must contain positions snapshot['pos'], diameters snapshot['sigma'] and activities snapshot['rho'].
#     """
#     # TODO Specify the shapes of positions, etc.

#     folder = Path(folder)
#     lenPlotIdxs = np.shape(plotTimes)[0]
#     if not os.path.exists(folder / "figures"):
#         os.makedirs(folder / "figures")
#     fig, Axs = plt.subplots(nrows=2, ncols=lenPlotIdxs+1, figsize=[2.5*lenPlotIdxs+1, 5], gridspec_kw={
#                             'width_ratios': [*np.ones(lenPlotIdxs,), 0.06]}, dpi=300)
#     cmap = mpl.colormaps['viridis']

#     # params = cil.inputParams(folder / paramfile)
#     # dt = params['MD']['DT']['MANUAL']
#     # gts = params['MD']['GTS']
#     # lbox = params['MD']['BOX']
#     # tauDiv = params['SPECIES']['C1']['DIVISION']['TAU_DIVISION']['AVG']
#     # sigmaMin = params['SPECIES']['C1']['DIVISION']['CYCLE_SIGMA_MIN']
#     # v1 = params['SPECIES']['C1']['DIVISION']['V1']
#     # # time between snapshots in units of division times
#     # snapshotInterval = dt * gts / tauDiv

#     for colIdx, time in enumerate(plotTimes):
#         name = 't_' + str(time)
#         pos = snapshots[name]['pos']
#         sigmas = np.array(snapshots[name]['sigma'])
#         activities = np.array(snapshots[name]['rho'])

#         import matplotlib.collections
#         patches = [plt.Circle(cellPos[0], sigma/2)
#                     for cellPos, sigma, activity in zip(pos, sigmas, activities)]
#         for label, rowIdx, colors in zip(('Cell activity $r$', 'Cell area $A/A_1$'), range(2), (cmap(activities), cmap((sigmas/sigmaMin)**2./v1))):
#             ax = Axs[rowIdx, colIdx]
#             ax.set_title(f"{label} at $t = {time}$")
#             coll = matplotlib.collections.PatchCollection(
#                 patches, facecolors=colors)
#             coll.set_rasterized(True)
#             ax.add_collection(coll)
#             # ax.set_rasterized(True)

#             # colors = cmap((sigmas/sigmaMin)**2./v1)
#             # coll_size = matplotlib.collections.PatchCollection(
#             # patches, facecolors=colors)
#             # ax.add_collection(coll_size)

#             ax.set_xlim([0, lbox[0]])
#             ax.set_ylim([0, lbox[1]])
#             ax.axis('equal')

#             cax = Axs[rowIdx, -1]
#             plt.colorbar(coll, ax=ax, cax=cax, label=label)
#     # Axs[1,-1].axis('off')
#     letterLabels(Axs[:2, :-1], x=0.05, y=0.95, fontsize=12)
#     fig.tight_layout()
#     fig.savefig(folder / "figures/Heterogeneity.pdf")


# def plot_snapshots_h5(folder, datafile, paramfile, plotIdxs=[70, 100, 150]):
#     folder = Path(folder)
#     lenPlotIdxs = np.shape(plotIdxs)[0]
#     if not os.path.exists(folder / "figures"):
#         os.makedirs(folder / "figures")
#     fig, Axs = plt.subplots(nrows=2, ncols=lenPlotIdxs+1, figsize=[2.5*lenPlotIdxs+1, 5], gridspec_kw={
#                             'width_ratios': [*np.ones(lenPlotIdxs,), 0.06]}, dpi=300)
#     cmap = mpl.colormaps['viridis']

#     params = cil.inputParams(folder / paramfile)
#     dt = params['MD']['DT']['MANUAL']
#     gts = params['MD']['GTS']
#     lbox = params['MD']['BOX']
#     tauDiv = params['SPECIES']['C1']['DIVISION']['TAU_DIVISION']['AVG']
#     sigmaMin = params['SPECIES']['C1']['DIVISION']['CYCLE_SIGMA_MIN']
#     v1 = params['SPECIES']['C1']['DIVISION']['V1']
#     # time between snapshots in units of division times
#     snapshotInterval = dt * gts / tauDiv

#     with h5py.File(folder / datafile, 'r+') as f:
#         snapshots = f['snapshots']
#         sigmaMin = params['SPECIES']['C1']['DIVISION']['CYCLE_SIGMA_MIN']
#         snapshots_for_plotting = []
#         for colIdx, time in enumerate(plotIdxs):
#             snapshot_for_plotting = {}
#             name = 't_' + str(time)
#             snapshot_for_plotting['pos'] = snapshots[name]['R']
#             sigmas = np.array(snapshots[name]['SigmaCore'])
#             activities = np.array(snapshots[name]['RHO'])

#             import matplotlib.collections
#             patches = [plt.Circle(cellPos[0], sigma/2)
#                        for cellPos, sigma, activity in zip(pos, sigmas, activities)]
#             for label, rowIdx, colors in zip(('Cell activity $r$', 'Cell area $A/A_1$'), range(2), (cmap(activities), cmap((sigmas/sigmaMin)**2./v1))):
#                 ax = Axs[rowIdx, colIdx]
#                 ax.set_title(f"{label} at $t = {time*snapshotInterval}$")
# ``````                
#         plot_snapshot_dicts(folder, plotTimes, snapshots, sigmaMin, v1, lbox)

