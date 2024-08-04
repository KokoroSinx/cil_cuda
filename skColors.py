import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# Custom colorcycles

# only a small selection of very useful colorblind safe colors https://personal.sron.nl/~pault/
tol_yellow = "#DDAA33"
tol_red = [187/255, 85/255, 102/255]
tol_blue = [0/255, 68/255, 136/255]

# IBM colorblind safe colors
IBM_yellow = "#FFB000"
IBM_orange = "#FE6100"
IBM_pink = "#DC267F"
IBM_purple = "#785EF0"
IBM_blue = "#648FFF"


# NZZ colors, https://nzzdev.github.io/Storytelling-Styleguide/#/colors
nzz_Nacht = "#374e8e"
nzz_Lagune = "#1b87aa"
nzz_Aquamarin = "#4fbbae"
nzz_Moos = "#006d64"
nzz_Pesto = "#478c5b"
nzz_Guacamole = "#93a345"
nzz_Nachos = "#e3b13e"
nzz_Mandarine = "#df7c18"
nzz_Sugo = "#ce4631"
nzz_Chianti = "#ac004f"
nzz_Amethyst = "#ae49a2"
nzz_Flieder = "#a07bde"
nzz_Himmel = "#8aabfd"
nzz_Schokolade = "#704600"
nzz_Sand = "#a08962"
nzz_LatteMacchiato = "#d5cdb9"
nzz_Aubergine = "#383751"
nzz_Regen = "#7e7e8f"
nzz_Nebel = "#cdcdd1"
cc_nzz = (nzz_Nacht, nzz_Lagune, nzz_Aquamarin, nzz_Moos, nzz_Pesto, nzz_Guacamole, nzz_Nachos, nzz_Mandarine, nzz_Sugo, nzz_Chianti, nzz_Amethyst, nzz_Flieder, nzz_Himmel , nzz_Schokolade, nzz_Sand, nzz_LatteMacchiato, nzz_Aubergine, nzz_Regen, nzz_Nebel)

# NYC Subway colors, see http://www.jsvine.com/mta-colors/
nyc_red = "#E00034"
nyc_blue = "#2850AD"
nyc_green = "#009B3A"
nyc_purple = "#6E267B"
nyc_sand = "#CE8E00"
nyc_brown = "#6E3219"
nyc_navy = "#060983"
nyc_cyan = "#00A1DE"
nyc_pink = "#B933AD"
nyc_grass = "#6CBE45"
nyc_caramel = "#996633"
nyc_orange = "#FF6319"
nyc_yellow = "#FCCC0A"
cc_nyc = (nyc_red, nyc_blue, nyc_green, nyc_purple, nyc_sand, nyc_brown, nyc_navy, nyc_cyan, nyc_pink, nyc_grass, nyc_caramel, nyc_orange, nyc_yellow)

cc_light = ("#3B76FF", "#E0476C", "#4A9B6A", "#8C4398",
          "#CEA242",  "#994623","#382F98", "#48C1FE", "#F26CE3", "#94E66B",
          "#99714C", "#FF9046",)

cc_veryLight = ("#88ABFF", "#DBB8C0", "#CCDBC5", "#D6BBDB",
          "#DBBC77",  "#DBB7A9","#8183DB", "#9AE1FE", "#DBB3D7", "#81EDAA", 
          "#DBCDC0", "#DBA68C",)

          
# Optimized sequential colors for colorblinds
cc_blind = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",)

# Colors which increase in grey value
cc_greyequiv = ("#D8C8E3", "#9BBDDB", "#7CAB45", "#897B31", "#8B3525", "#440031")

cc_nyc_long = ("#2850AD", "#E00034", "#009B3A", "#6E267B",
          "#CE8E00", "#6E3219","#060983", "#00A1DE", "#B933AD", "#6CBE45",
          "#996633", "#FF6319", "#923D97","#0039A6","#00985F",  "#FCCC0A",
          "#FF6319", "#EE352E", "#00933C",
          "#C60C30", "#0039A6", "#A626AA",)

def plotAllColorMaps():
  colormaps = [
    cc_nzz,
    cc_nyc, 
    cc_light, 
    cc_veryLight, 
    cc_blind,
    cc_greyequiv,
    cc_nyc_long]
  names = [
    "cc_nzz",
    "cc_nyc", 
    "cc_light", 
    "cc_veryLight", 
    "cc_blind",
    "cc_greyequiv",
    "cc_nyc_long"
    ]
  
  nColors = max(len(colormap) for colormap in colormaps)
  
  fig, axs = plt.subplots(len(colormaps), 1, figsize=(3, float(len(colormaps))/2.), sharex=True)
  fig.subplots_adjust(top=0.95, bottom=0.01, left=0.3, right=0.99)
  
  for (name, colormap, ax) in zip(names, colormaps, axs):
    for position, color in enumerate(colormap):
      ax.plot([position+0.5, position+0.5], [0,1], color = color, lw = 2)
      
    pos = list(ax.get_position().bounds)      
    x_text = pos[0] - 0.01
    y_text = pos[1] + pos[3]/2.
    fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.axis([-0.5, nColors, 0, 1])
    ax.axis('off')
  # plt.tight_layout()
  plt.savefig("skColors.pdf")
  plt.close()

def interpolateBetweenColors(color1, color2, ratio):
  """
  Produces a color between color1 and color2 as the ratio between color2 and color1.
  """
  return [c1 + (c2 - c1) * ratio for c1, c2 in zip(color1, color2)]

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.
    Source: http://stackoverflow.com/questions/7404116/
    defining-the-midpoint-of-a-colormap-in-matplotlib.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def sinebowForScalar(h):
  """ sinebowForScalar(h) returns a color out of the rainbow according to the angle h in the   color circle.
  h is expected to be in radians, i.e. 1 marks a full circle.
  The color is returned as tuple (r,g,b) with r,g,b in the interval [0,1].
  """
  h += 1./2
  h *= -1.
  r = np.sin(np.pi * h)
  g = np.sin(np.pi * (h + 1./3))
  b = np.sin(np.pi * (h + 2./3))
  return (r*r, g*g, b*b)
  
  
# def sinebow(h):
#   """ sinebow(h) returns a color out of the rainbow according to the angle h in the   color circle.
#   h is expected to be in radians, i.e. 1 marks a full circle.
#   The color is returned as tuple (r,g,b) with r,g,b in the interval [0,1].
#   """
#   if np.isscalar(h):
#     return sinebowForScalar(h)
#
#   else:
#     harray = np.array(h)
#     if harray.ndim == 1:
#       colors = np.zeros
#
#   h += 1./2
#   h *= -1.
#   r = np.sin(np.pi * h)
#   g = np.sin(np.pi * (h + 1./3))
#   b = np.sin(np.pi * (h + 2./3))
#   return (r*r, g*g, b*b)

def sinebow(h):
  """ sinebow(h) returns a color out of the rainbow according to the angle h in the   color circle.
  h is expected to be in radians, i.e. 1 marks a full circle.
  The color is returned as tuple (r,g,b) with r,g,b in the interval [0,1].
  """
  h += 1./2
  h *= -1.
  r = np.sin(np.pi * h)
  g = np.sin(np.pi * (h + 1./3))
  b = np.sin(np.pi * (h + 2./3))
  return (r*r, g*g, b*b)
  

#f = figure()
#x = linspace(1,2)
#for i, c1, c2 in zip(range(len(cc_nyc)), cc_nyc, cc_light):
  #plot(x,x+i, color=c1, lw = 5)
  #hold(True)
  #plot(x,x+i+0.5, color=c2, lw = 5)
#hold(False)
#show()
          




# from colormath.color_objects import RGBColor
# color = RGBColor(0,0,0)
# color.set_from_rgb_hex(cc_nyc[2])

# cc_light = []
# factor = 1.3
# rgbc = RGBColor(0,0,0)
# for color in cc_nyc:
#   rgbc.set_from_rgb_hex(color)
#   rgbc.rgb_b = min(rgbc.rgb_b*factor, 255)
#   rgbc.rgb_g = min(rgbc.rgb_g*factor, 255)
#   rgbc.rgb_r = min(rgbc.rgb_r*factor, 255)
#   cc_light.append(rgbc.get_rgb_hex())


# cc_nyc_grey = ("#A7A9AC","#808183", "#4D5357",)
# "#00AF3F",

# # USE brewer2mpl

# pl.figure(1)
# pl.rc('axes', color_cycle = cc_nyc)
# xs = range(pl.shape(cc_nyc)[0])
# for x in xs:
#   pl.plot((x+1,x+1),(0,1), linewidth = 10)
#   pl.hold(True)

# pl.hold(False)
# pl.show()

# pl.figure(1)
# pl.rc('axes', color_cycle = cc_nyc_long)
# xs = range(pl.shape(cc_nyc_long)[0])
# for x in xs:
#   pl.plot((x+1,x+1),(0,1), linewidth = 10)
#   pl.hold(True)

# pl.hold(False)
# pl.show()

