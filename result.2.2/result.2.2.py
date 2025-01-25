import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import datetime
import geoviews as gv
import holoviews as hv
from holoviews import dim, opts
import hvplot.pandas
hv.extension('bokeh')
gv.extension('bokeh')
import warnings
import copy

cols = ['PavdS01', 'PavdS02', 'PavdS03', 'PavdS04', 'PavdS05', 'PavdS06', 'PavdS07', 'PavdS08', 'PavdS09', 'PavdS10', 'PavdS11', 'PavdS12', 'PavdS13', 'PavdS14', 'PavdS15', 'PavdS16', 'PavdS17', 'PavdS18', 'PavdS19', 'PavdS20', 'PavdS21', 'PavdS22', 'PavdS23', 'PavdS24', 'PavdS25', 'PavdS26', 'PavdS27', 'PavdS28', 'PavdS29', 'PavdS30', 'LongX_2', 'CanpyRH095']
pavd_whole = ['PavdS01', 'PavdS02', 'PavdS03', 'PavdS04', 'PavdS05', 'PavdS06', 'PavdS07', 'PavdS08', 'PavdS09', 'PavdS10', 'PavdS11', 'PavdS12', 'PavdS13', 'PavdS14', 'PavdS15', 'PavdS16', 'PavdS17', 'PavdS18', 'PavdS19', 'PavdS20', 'PavdS21', 'PavdS22', 'PavdS23', 'PavdS24', 'PavdS25', 'PavdS26', 'PavdS27', 'PavdS28', 'PavdS29', 'PavdS30']

def hook(plot, element):
    plot.handles['xaxis'].axis_label_text_font = 'times'
    plot.handles['yaxis'].axis_label_text_font = 'times'
    plot.handles['xaxis'].major_label_text_font = 'times'
    plot.handles['yaxis'].major_label_text_font = 'times'
    #print('plot.handles: ', sorted(plot.handles.keys()))

pre_fire = pd.read_csv('pre_fire.csv')
post_fire = pd.read_csv('post_fire.csv')
difference = pd.read_csv('difference.csv')
relativeDifference = pd.read_csv('relativeDifference.csv')

data_b = pre_fire
data_a = post_fire

dz = 5
pavdAll_befor = []
pavdAll_after = []

bar_width = 36
line_width = 6
long_move = 0.00055
bias = 0.0025
grid_width = 0.35
xlim1 = 128.555
xlim2 = 128.665
ylim = 60

for j, s in enumerate(data_b.index):
    pavdShot = eval(data_b['pavds'][s].replace('nan', 'np.nan'))
    elevShot = 0
    longitude = data_b['LongX_2'][s]
    pavdElev = []

    while 'nan' in pavdShot:
        pavdShot.remove('nan')
    for i, e in enumerate(range(len(pavdShot))):
        if pavdShot[i] > 0:
            pavdElev.append((float(longitude)-long_move+bias, elevShot + dz * i, pavdShot[i]))  # Append tuple of distance, elevation, and PAVD
    final_height = pavdElev[-1][1]+dz
    final_pavd = pavdElev[-1][2]
    pavdElev.append((float(longitude)-long_move+bias, final_height, final_pavd))
    pavdAll_befor.append(pavdElev)      

for j, s in enumerate(data_a.index):
    pavdShot = eval(data_a['pavds'][s].replace('nan', 'np.nan'))
    elevShot = 0
    longitude = data_a['LongX_2'][s]
    pavdElev = []

    while 'nan' in pavdShot:
        pavdShot.remove('nan')
    for i, e in enumerate(range(len(pavdShot))):
        if pavdShot[i] > 0:
            pavdElev.append((float(longitude)+long_move+bias, elevShot + dz * i, pavdShot[i]))  # Append tuple of distance, elevation, and PAVD
    final_height = pavdElev[-1][1]+dz
    final_pavd = pavdElev[-1][2]
    pavdElev.append((float(longitude)+long_move+bias, final_height, final_pavd))
    pavdAll_after.append(pavdElev)                                                # Append to final list    

path1_whole = hv.Path(pavdAll_befor[0], vdims='PAVD').options(color='PAVD', clim=(0,0.25), cmap='Blues', line_width=bar_width, colorbar=True, xlim = (xlim1, xlim2), ylim = (0,ylim),
                                               width=4500, height=1200, clabel='mean PAVD (before the fire)', xlabel='longitude', colorbar_position = 'left',
                                               ylabel='Heights (m)', fontsize={'title':64, 'xlabel': 72, 'ylabel': 72, 'legend' : 36,
                                                                                 'xticks':48, 'yticks':48,
                                                                                 'clabel':54, 'cticks':36}, 
                                                    show_grid=True, gridstyle = {'grid_line_width': grid_width, 'grid_line_color': 'black'})
for k in range(len(pavdAll_befor)-1):
    path1 = hv.Path(pavdAll_befor[k+1], vdims='PAVD').options(color='PAVD', clim=(0,0.25), cmap='Blues', line_width=bar_width, colorbar=True, xlim = (xlim1, xlim2), ylim = (0,ylim),
                                                   width=4500, height=1200, clabel='mean PAVD (before the fire)', xlabel='longitude', colorbar_position = 'left',
                                                   ylabel='Heights (m)',
                                                        show_grid=True, gridstyle = {'grid_line_width': grid_width, 'grid_line_color': 'black'})
    path1_whole = path1_whole * path1
    
path2_whole = hv.Path(pavdAll_after[0], vdims='PAVD').options(color='PAVD', clim=(0,0.25), cmap='Reds', line_width=bar_width, colorbar=True, ylim = (0,ylim),
                                                     clabel='mean PAVD (after the fire)')
for j in range(len(pavdAll_after)-1):
    path2 = hv.Path(pavdAll_after[j+1], vdims='PAVD').options(color='PAVD', clim=(0,0.25), cmap='Reds', line_width=bar_width, colorbar=True, ylim = (0,ylim),
                                                         clabel='mean PAVD (after the fire)')
    path2_whole = path2_whole * path2

path = (path1_whole * path2_whole)

path.opts(hooks=[hook]).options(xticks=np.arange(128.54, 128.67, 0.01))

data_g = difference

dz = 5
pavdAll_gap = []

bar_width = 54
line_width = 6
bias = 0.0025
grid_width = 0.35
ylim = 60

for j, s in enumerate(data_g.index):
    pavdShot = eval(data_g['pavds'][s].replace('nan', 'np.nan'))
    elevShot = 0
    longitude = data_g['LongX_2'][s]
    pavdElev = []

    if np.isnan(pavdShot).all():
        continue
    for i, e in enumerate(range(len(pavdShot))):
        if np.isnan(pavdShot[i]) == False:
            pavdElev.append((float(longitude)+bias, elevShot + dz * i, pavdShot[i]))  # Append tuple of distance, elevation, and PAVD
        #else:
            #pavdElev.append((float(longitude)+bias, elevShot + dz * i, pavdElev[-1][-1]))  # Append tuple of distance, elevation, and PAVD
    final_height = pavdElev[-1][1]+dz
    final_pavd = pavdElev[-1][2]
    pavdElev.append((float(longitude)+bias, final_height, final_pavd))
    pavdAll_gap.append(pavdElev)      

path1_whole = hv.Path(pavdAll_gap[0], vdims='PAVD').options(color='PAVD', cmap='seismic_r', clim=(-0.20,0.20), line_width=bar_width, colorbar=True, xlim = (128.555,128.665), ylim = (0,ylim),
                                               width=4500, height=1200, clabel='mean dPAVD (after-before)', xlabel='longitude', colorbar_position = 'left',
                                               ylabel='Heights (m)', fontsize={'title':64, 'xlabel':72, 'ylabel': 72, 'legend' : 36,
                                                                                 'xticks':48, 'yticks':48,
                                                                                 'clabel':54, 'cticks':36}, 
                                                    show_grid=True, gridstyle = {'grid_line_width': grid_width, 'grid_line_color': 'black'})

for k in range(len(pavdAll_gap)-1):
    path1 = hv.Path(pavdAll_gap[k+1], vdims='PAVD').options(color='PAVD', cmap='seismic_r', clim=(-0.20,0.20), line_width=bar_width, colorbar=True, xlim = (128.555,128.665), ylim = (0,ylim),
                                                   width=4500, height=1200, clabel='mean PAVD gap (after-before)', xlabel='longitude', colorbar_position = 'left',
                                                   ylabel='Heights (m)',
                                                        show_grid=True, gridstyle = {'grid_line_width': grid_width, 'grid_line_color': 'black'})
    path1_whole = path1_whole * path1

path1_whole.opts(hooks=[hook]).options(xticks=np.arange(128.54, 128.67, 0.01))

data_g = relativeDifference

dz = 5
pavdAll_gap_rela = []

bar_width = 54
line_width = 6
bias = 0.0025
grid_width = 0.35
ylim = 60

for j, s in enumerate(data_g.index):
    pavdShot = eval(data_g['pavds'][s].replace('nan', 'np.nan'))
    elevShot = 0
    longitude = data_g['LongX_2'][s]
    pavdElev = []

    if np.isnan(pavdShot).all():
        continue
    for i, e in enumerate(range(len(pavdShot))):
        if np.isnan(pavdShot[i]) == False:
            pavdElev.append((float(longitude)+bias, elevShot + dz * i, pavdShot[i]))  # Append tuple of distance, elevation, and PAVD
    final_height = pavdElev[-1][1]+dz
    final_pavd = pavdElev[-1][2]
    pavdElev.append((float(longitude)+bias, final_height, final_pavd))
    pavdAll_gap_rela.append(pavdElev)      

path1_whole = hv.Path(pavdAll_gap_rela[0], vdims='PAVD').options(color='PAVD', cmap='seismic_r', clim=(-1.00,1.00), line_width=bar_width, colorbar=True, xlim = (128.555,128.665), ylim = (0,ylim),
                                               width=4500, height=1200, clabel='mean dPAVD (after-before)', xlabel='longitude', colorbar_position = 'left',
                                               ylabel='Heights (m)', fontsize={'title':64, 'xlabel':72, 'ylabel': 72, 'legend' : 36,
                                                                                 'xticks':48, 'yticks':48,
                                                                                 'clabel':54, 'cticks':36}, 
                                                    show_grid=True, gridstyle = {'grid_line_width': grid_width, 'grid_line_color': 'black'})

for k in range(len(pavdAll_gap_rela)-1):
    path1 = hv.Path(pavdAll_gap_rela[k+1], vdims='PAVD').options(color='PAVD', cmap='seismic_r', clim=(-1.00,1.00), line_width=bar_width, colorbar=True, xlim = (128.555,128.665), ylim = (0,ylim),
                                                   width=4500, height=1200, clabel='mean PAVD gap (after-before)', xlabel='longitude', colorbar_position = 'left',
                                                   ylabel='Heights (m)',
                                                        show_grid=True, gridstyle = {'grid_line_width': grid_width, 'grid_line_color': 'black'})
    path1_whole = path1_whole * path1

path1_whole.opts(hooks=[hook]).options(xticks=np.arange(128.54, 128.67, 0.01))