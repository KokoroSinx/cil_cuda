#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib as pl
from skColors import cc_nyc


#cap = dict(facecolor='black', width=0, headwidth=4., shrink=0.05)
cap = dict(facecolor='black', arrowstyle="-", connectionstyle="arc3,rad=0.0")
  
def report_available_fonts():
  """
  Reports the available typeface families installed on the system. The output can then be used, e.g. to style text such as 
    plt.text(0.35, 0.5, "Hello Radical", fontsize=25, fontdict={'fontname': NameOfFontFamily}) 
  with NameOfFontFamily an element of the output of this function. 
  """
  import matplotlib.font_manager as mfm
  from numpy import unique
  # This reports the paths of all individual font files ...
  flist = mfm.get_fontconfig_fonts()
  
  # ... but we actually only want the names of the typeface families. Note that multiple fonts map to one family. 
  names = []
  for fname in flist:
    try:
      names.append(mfm.FontProperties(fname=fname).get_name())
    except:
        pass
  return list(unique(names))
    
def setup_figure_formatting(case):
  """
  Sets a range of settings to enhance the aesthetics of the figures for a particular medium. 
  For the unicode support to work, make sure to use python 3.
  """
  pl.rcParams['figure.dpi'] = 300
  
  pl.rc('axes', prop_cycle=pl.cycler('color', cc_nyc))
  pl.rcParams['image.cmap'] = 'gray'
  
  fonts = pl.rcParamsDefault['font.sans-serif']
  fonts.insert(0,'Helvetica Neue')
  fonts.insert(0,'Source Sans Pro')
  fonts.insert(0,'Open Sans')
  pl.rcParams['font.sans-serif'] = fonts
  
  pl.rcParams['mathtext.fontset'] = 'dejavusans'
  # Supported mathtext sets are ('dejavusans' (default), 'dejavuserif', 'cm' (Computer Modern), 'stix', 'stixsans')

  if (case == 'print'):
    fontsize = 8
    legendfontsize = 7
    pl.rc('font', size=fontsize)
    pl.rc('legend',
      borderpad=0.5, borderaxespad=0.5, labelspacing=0.1,
      columnspacing = 1.0,
      handlelength=2.2, handletextpad=0.4,
      frameon=False, numpoints=1, fontsize=fontsize)
    pl.rc('lines', markeredgewidth=0.0, linewidth=1, markersize=6)
    pl.rc('axes', linewidth=0.5, titlesize=fontsize)
    pl.rc('grid', color = '0.5', linestyle = '-', linewidth=0.2)
    pl.rcParams['xtick.major.size'] = 3
    pl.rcParams['xtick.minor.size'] = 2
    pl.rcParams['ytick.major.size'] = 3
    pl.rcParams['ytick.minor.size'] = 2
    pl.rcParams['xtick.major.width'] = 0.5
    pl.rcParams['xtick.minor.width'] = 0.5
    pl.rcParams['ytick.major.width'] = 0.5
    pl.rcParams['ytick.minor.width'] = 0.5
    
    
  elif (case == 'printfour'):
    pl.rc('font', size=32)
    pl.rc('legend',
      borderpad=2, borderaxespad=2, labelspacing=0.4,
      handlelength=6.4, handletextpad=1.6,
      frameon=False, numpoints=4, fontsize=32)
    pl.rc('lines', markeredgewidth=0.0, linewidth = 4, markersize=24)
    pl.rc('axes', linewidth=0.8, titlesize=32)
    #pl.rcParams['xtick.major.size'] = 16
    #pl.rcParams['xtick.minor.size'] = 8
    #pl.rcParams['ytick.major.size'] = 16
    #pl.rcParams['ytick.minor.size'] = 8
  
  elif (case == 'printsmall'):
    pl.rc('font', size=7)
    pl.rc('legend',
      borderpad=0.5, borderaxespad=0.5,
      labelspacing=0.1, # spacing between labels
      columnspacing = 1.0,
      handlelength=1.6,  handletextpad=0.4, # length and distance to label of the handle giving the style of the data line
      frameon=False, numpoints=1,
      fontsize=7)
    pl.rc('lines', markeredgewidth=0.0, linewidth=1, markersize=6)
    pl.rc('axes', linewidth=0.3, titlesize=7)
    for tick in ['xtick.', 'ytick.']:
      pl.rcParams[tick + 'major.size'] = 4
      pl.rcParams[tick + 'minor.size'] = 2
      pl.rcParams[tick + 'major.width'] = 0.3
      pl.rcParams[tick + 'minor.width'] = 0.3
      pl.rcParams[tick + 'direction'] = 'in'   

  elif (case == 'paper'):
    fontsize = 8
    legendfontsize = 7
    pl.rc('font', size=fontsize)
    pl.rc('legend',
      borderpad=0.25, borderaxespad=0.45,
      labelspacing=.25,
      columnspacing = 1.0,
      handlelength=1.6, handletextpad=0.4,
      # handleheight = 0.2,
      frameon=False, numpoints=1, fontsize=legendfontsize)
    pl.rc('lines', markeredgewidth=0.0, linewidth=1, markersize=6)
    pl.rc('axes', linewidth=0.3, titlesize=fontsize)
    for tick in ['xtick.', 'ytick.']:
      pl.rcParams[tick + 'major.size'] = 2.5
      pl.rcParams[tick + 'minor.size'] = 1.5
      pl.rcParams[tick + 'major.width'] = 0.3
      pl.rcParams[tick + 'minor.width'] = 0.3
      pl.rcParams[tick + 'direction'] = 'in'    
    pl.rcParams['xtick.top'] = True
    pl.rcParams['ytick.right'] = True
    
  elif (case == 'slide'):
    pl.rc('font', size=12)
    pl.rc('legend',
    borderpad=0.5, borderaxespad=0.5,
    labelspacing=0.1,
    handlelength=1.3, handletextpad=0.4,
    frameon=False, numpoints=1, fontsize=12)
    pl.rc('lines', linewidth=2, markersize=10, markeredgewidth=0.001)
    pl.rc('axes', linewidth=0.15, titlesize=8)
  elif (case == 'display'):
    params = pl.rcParamsDefault
    pl.rcParams.update(params)
  else:
    sys.exit("""'%s' is not a valid choice for case, use values 'print', 'printsmall', 'printfour', 'slide', or 'display' instead""" %(case))    
    

def setup_typeface(font="Open Sans"):
  from matplotlib.backends.backend_pgf import FigureCanvasPgf
  pl.backend_bases.register_backend('pdf', FigureCanvasPgf)
  pl.backend_bases.register_backend('png', FigureCanvasPgf)
  pl.rc('text', usetex=True)
  pl.rc('pgf', rcfonts=False)
  # As far as I understand, something in the pdf pipeline gets confused when the proper unicode_minus is used, and does not show them at all. This replaces all occurences of unicode_minus with hyphen_minus, supposedly. The output looks fine though and seems to be using proper minusses. It seems only proper latex + pdf output triggers this.
  pl.rcParams['axes.unicode_minus'] = False

  pl.rc('axes', prop_cycle=pl.cycler('color', cc_nyc))

  # Exports fonts as text and not curves when using svg (but breaks formatting in most vector editors, but not inkscape)
  # pl.rcParams['svg.fonttype'] = 'none'

  pl.rcParams['image.cmap'] = 'gray'

  # preamble = [
  #   r'\usepackage[T1]{fontenc}',
  #   r'\usepackage{amsmath}',
  #   r'\usepackage{underscore}',
  #   r'\usepackage{bm}',
  #   # r'\usepackage{upgreek}',
  #   # r'\usepackage{siunitx}',
  # ]
  preamble = []
  preamble.append(r'\usepackage[T1]{fontenc}')
  preamble.append(r'\usepackage{amsmath}')
  preamble.append(r'\usepackage{underscore}')
  preamble.append(r'\usepackage{bm}')

  if font == "Open Sans":
    # Set to Open Sans for all numbers and text
    # pl.rc('font', family='sans-serif', serif=['Open Sans'])
    preamble.append(r'\usepackage[default, scale=0.9]{opensans}')
    preamble.append(r'\usepackage{sfmath}')
    preamble.append(r"\usepackage{mathspec}")
    preamble.append(r"\setsansfont[Scale=0.9]{Open Sans}")
    preamble.append(r"\setmathsfont(Greek)[Scale=0.9]{Open Sans}")

  elif font == "Helvetica":
    # Set to latex and Helvetica for all numbers and text
    # pl.rc('font', family='sans-serif', serif=['Helvetica'], size=8)
    # preamble.append(r'\usepackage{helvet}')
    preamble.append(r'\usepackage{sfmath}')
    preamble.append(r"\usepackage{mathspec}")
    # preamble.append(r"\setmainfont{Helvetica}")
    preamble.append(r"\setsansfont{Helvetica Neue}")
    preamble.append(r"\setmathsfont(Digits,Greek,Latin){Helvetica Neue}")
    # preamble.append(r"\setmathfont(Digits,Greek){Helvetica}")
    # preamble.append(r"\setmathrm{Helvetica}")

  elif font == "Times":
    # Set to Times for all numbers and text
    # pl.rc('font', family='serif', serif=['Times'], size=8)
    # preamble.append(r'\usepackage{times}')
    # preamble.append("\\usepackage{unicode-math}")
    preamble.append(r"\setmathfont{xits-math.otf}")
    # preamble.append(r"\setmainfont{Times}")

  elif font == "Minion":
    # Set to Times for all numbers and text
    pl.rc('font', family='serif', serif=['Minion Pro'], size=8)
    latexpreamble.append(r'\usepackage{minionpro}')
    latexpreamble.append(r'\usepackage{MnSymbol}')

  else:
    print("Warning: Don't recognize the font choice.")

  # pl.rcParams.update(preamble)
  # pl.rc('text', preamble=preamble)
  # joined strings with \n as separator)
  pl.rc('pgf', preamble='\n'.join(preamble))


def setup_figure_formatting_xelatex(case, font="Open Sans"):
  """
  Sets a range of settings to enhance the aesthetics of the figures for a particular medium. 
  For the unicode support to work, make sure to use python 3.
  """
  # from matplotlib.backends.backend_pgf import FigureCanvasPgf
  # pl.backend_bases.register_backend('pdf', FigureCanvasPgf)
  # pl.backend_bases.register_backend('png', FigureCanvasPgf)
  # pl.rc('text', usetex=True)
  # pl.rc('pgf', rcfonts=False)
  # pl.rcParams['axes.unicode_minus']=False # As far as I understand, something in the pdf pipeline gets confused when the proper unicode_minus is used, and does not show them at all. This replaces all occurences of unicode_minus with hyphen_minus, supposedly. The output looks fine though and seems to be using proper minusses. It seems only proper latex + pdf output triggers this.


  # pl.rc('axes', prop_cycle=pl.cycler('color', cc_nyc))
  
  # ## Exports fonts as text and not curves when using svg (but breaks formatting in most vector editors, but not inkscape) 
  # # pl.rcParams['svg.fonttype'] = 'none'
  
  # pl.rcParams['image.cmap'] = 'gray'
  
  # # preamble = [
  # #   r'\usepackage[T1]{fontenc}',
  # #   r'\usepackage{amsmath}',
  # #   r'\usepackage{underscore}', 
  # #   r'\usepackage{bm}',
  # #   # r'\usepackage{upgreek}',
  # #   # r'\usepackage{siunitx}',
  # # ]
  # preamble = []
  # preamble.append(r'\usepackage[T1]{fontenc}')
  # preamble.append(r'\usepackage{amsmath}')
  # preamble.append(r'\usepackage{underscore}')
  # preamble.append(r'\usepackage{bm}')
  
    
  # if font == "Open Sans":
  #   # Set to Open Sans for all numbers and text    
  #   # pl.rc('font', family='sans-serif', serif=['Open Sans'])
  #   preamble.append(r'\usepackage[default, scale=0.9]{opensans}')
  #   preamble.append(r'\usepackage{sfmath}')
  #   preamble.append(r"\usepackage{mathspec}")
  #   preamble.append(r"\setsansfont[Scale=0.9]{Open Sans}")
  #   preamble.append(r"\setmathsfont(Greek)[Scale=0.9]{Open Sans}")
    

  # elif font == "Helvetica":
  #   # Set to latex and Helvetica for all numbers and text
  #   # pl.rc('font', family='sans-serif', serif=['Helvetica'], size=8)
  #   # preamble.append(r'\usepackage{helvet}')
  #   preamble.append(r'\usepackage{sfmath}')
  #   preamble.append(r"\usepackage{mathspec}")
  #   # preamble.append(r"\setmainfont{Helvetica}")
  #   preamble.append(r"\setsansfont{Helvetica Neue}")
  #   preamble.append(r"\setmathsfont(Digits,Greek,Latin){Helvetica Neue}")
  #   # preamble.append(r"\setmathfont(Digits,Greek){Helvetica}")
  #   # preamble.append(r"\setmathrm{Helvetica}")

  # elif font == "Times":
  #   # Set to Times for all numbers and text
  #   # pl.rc('font', family='serif', serif=['Times'], size=8)
  #   # preamble.append(r'\usepackage{times}')
  #   # preamble.append("\\usepackage{unicode-math}")
  #   preamble.append(r"\setmathfont{xits-math.otf}")
  #   # preamble.append(r"\setmainfont{Times}")
    
  # elif font == "Minion":
  #   # Set to Times for all numbers and text
  #   pl.rc('font', family='serif', serif=['Minion Pro'], size=8)
  #   latexpreamble.append(r'\usepackage{minionpro}')
  #   latexpreamble.append(r'\usepackage{MnSymbol}')  
    
  # else:
  #   print("Warning: Don't recognize the font choice.")
  
  
  # # pl.rcParams.update(preamble)
  # # pl.rc('text', preamble=preamble)
  # pl.rc('pgf', preamble='\n'.join(preamble)) # joined strings with \n as separator)
  setup_typeface(font=font)

  # Do the rest of the setup
  setup_figure_formatting(case)
      

def letterLabels(axs, x, y=1, fontsize=8, style='bold', case='lowercase', va='top'):
  """
  Place letters as labels at the same position (x,y) for all subplots in axs with vertical alignment va.
  Accepts 'bold' and 'parentheses' as styles. Accepts 'lowercase' and 'uppercase' as cases. x and y are expected to be in the range (0,1)
  """
  from string import ascii_lowercase, ascii_uppercase
  
  properties = {}
  properties['fontsize'] = fontsize
  letters = ascii_lowercase
  if (case=='uppercase'):
    letters = ascii_uppercase
  elif  (case!='lowercase'):
    print("case can only be either 'lowercase' or 'uppercase'. Reverting to 'lowercase'.")

  for ax, letter in zip(axs.flat, letters):
    if (style == 'bold'):
      if (pl.rcParams['text.usetex']):
        label = '\\textbf{' + letter + '}'
      else:
        label = letter
        properties['weight'] = 'bold'
    elif (style == 'parentheses'):
      label = '('+ letter +')'
    ax.text(x, y, label, va=va, transform = ax.transAxes, **properties)
    ax.text(x, y, label, va=va, transform = ax.transAxes, **properties)


def setAxisLabels(ax=None, x=None, y=None, z=None, labelpad=4.):
  """
  Provides a way to set up all axis labels in one line. If axis is not provided as ax, use the current axis. All labels are optional. labelpad: Spacing in points from the Axes bounding box including ticks and tick labels, by default use the matplotlib standard = 4.0.
  """
  if ax is None:
      ax = plt.gca()
  if x is not None:
      ax.set_xlabel(x, labelpad=labelpad)
  if y is not None:
      ax.set_ylabel(y, labelpad=labelpad)
  if z is not None:
      ax.set_zlabel(z, labelpad=labelpad)

#define function to add color bar to each axes individually
def add_cbar(fig, loc, pm, ticks, labels, pad):
    cax = fig.add_axes(loc)
    cb  = pl.colorbar(pm, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels(labels, ha='right')
    cb.ax.yaxis.set_tick_params(pad=pad)
    

def colorwheel(fig, location, mapable, thickness=0.33, labels='degrees', title=None):
    """
    Shoddy implementation of a "polar colorbar". Thickness determines the 
    thickness of the ring, with 1 yielding a filled circle.
    The labels usually do not directly represent the data range of the mapable, 
    but just represent all the possible angles. Those can be styled with 
    'degrees', 'pi', or 'windrose'. For a direct representation of the supplied 
    data, use 'actual'. For a completely custom set of labels, provide a list 
    of up to 8 strings.
    """
    from numpy import abs, arange, linspace, min, ones, pi
    
    ax_cbar = fig.add_axes(location, projection='polar')

    angles = arange(0, 361, 1)*pi/180.0
    
    thickness = min([abs(thickness), 1.0])
    nRadii = int(thickness*100)
    radii = arange(100-nRadii, 100, 1)
    values = angles * ones((nRadii, 361))

    cmap = mapable.get_cmap()
    clim = mapable.get_clim()
    ax_cbar.pcolormesh(angles, radii, values, 
                       cmap=cmap, clim=clim,
                       rasterized=True,
                      )
    ## inner rim 
    # ax_cbar.plot(angles, (100-nRadii)*np.ones_like(angles), 'k', lw=0.7)
    if title:
      ax_cbar.text(0.5, 0.5, title, ha='center', va='center', transform=ax_cbar.transAxes)
    
    ax_cbar.set_yticks([])
    if type(labels) == list:
        ax_cbar.set_xticklabels(labels)
    elif labels == 'windrose':
        ax_cbar.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', '', ])
    elif labels == 'pi':
        ax_cbar.set_xticklabels(['$0$', '', '$\pi/2$', '', '$\pi$', '', '$-\pi/2$', '', ])
    elif labels == 'actual':
        labels = ['{:2.2f}'.format(number) for number in linspace(*clim, 9)]
#     labels = ['{:2.2f}'.format(number) for number in np.linspace(0,8, 9)]
        ax_cbar.set_xticklabels(labels)

    return ax_cbar    