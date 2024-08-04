#!/usr/bin/env python
import numpy as np
import os
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib as mpl
import cil
import simulationtools as st
import cilJintaoTools as cjt
from figuresetup import setAxisLabels, letterLabels


# TODO Use a regular expression to generate the foldernames
folders = [
 "/Users/he/Downloads/CIL/Program",
]

datafile = "output/cil.h5"
datafile_gsd = "output/cil.gsd"
paramfile = "Match_Puliafito_2D.json"

unitArea = 30

# # ANALYSE AND AVERAGE
for folder in folders:
    folder = Path(folder)
    data = cjt.analyse_folder(folder, datafile, paramfile)
    np.savez(folder / "quantities_v3.2.npz",
             **data)  # data is a dict here
    # data = cjt.analyse_folder_gsd(folder, datafile_gsd, paramfile)
    # np.savez(folder / "quantities_gsd.npz",
    #          **data)  # data is a dict here

# data_av = cjt.average_quantities(folders)
# np.savez("quantities_averaged.npz", **data_av)  # data is a dict here

# plotIdxs = [50, 100, 150,]
# plotIdxs = [0, 25, 50, 75, 100]
plotIdxs = [0, 10, 20, 30]

# PLOT SNAPSHOTS AND DATA
for folder in folders:
    folder = Path(folder)

    data = np.load(folder / "quantities_v3.2.npz")
    cjt.plot_quantities_from_dict(
        folder, data, plotIdxs=plotIdxs, unitArea=unitArea)
    # Plot the radial distributions in a naive way, only works with the analysis of h5 files
    # cjt.plot_radial_quantities(folder, data, figurefilename="radial.pdf")
    
    # data_gsd = np.load(folder / "quantities_gsd.npz")
    # cjt.plot_quantities_from_dict(
    #     folder, data_gsd, plotIdxs=plotIdxs, unitArea=unitArea, figurefilename="Areas_speeds_divisions_gsd.pdf")
    # cjt.plot_snapshots(folder, datafile, paramfile, plotIdxs=plotIdxs)



# # # # PLOT AVERAGED DATA
# # data_av = np.load("quantities_averaged.npz")
# # plotIdxs = [70, 100, 150, 200, 400]
# # cjt.plot_quantities_from_dict("", data_av, plotIdxs)