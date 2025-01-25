import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})

andong_csv = 'studysite.csv'
compare_csv = 'controlsite.csv'

# andong
andong_andong_gedi = pd.read_csv(os.path.join(csv_folder_path, andong_csv))
andong_site_gedi = andong_andong_gedi[andong_andong_gedi['in_perimet']==1]
andong_site_gedi_pre = andong_site_gedi[andong_site_gedi['burnt']==0]
andong_site_gedi_post = andong_site_gedi[andong_site_gedi['burnt']==1]
andong_compare_gedi = pd.read_csv(os.path.join(csv_folder_path, compare_csv))
andong_compare_gedi_pre = andong_compare_gedi[andong_compare_gedi['burnt']==0]
andong_compare_gedi_post = andong_compare_gedi[andong_compare_gedi['burnt']==1]

varis_canpy_10 = ['CanpyRH005', 'CanpyRH015', 'CanpyRH025', 'CanpyRH035', 'CanpyRH045', 'CanpyRH055', 'CanpyRH065', 'CanpyRH075', 'CanpyRH085', 'CanpyRH095']

fig, axes = plt.subplots(2, 5, figsize=(90, 38))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)

bw_num_den = 0.35; fill_bool = 1; line_width = 0.6*2; alpha_num = 0.2
bw_num_smooth = 1.2; fill_bool_smooth = 0; line_width_smooth = 1.5*3; alpha_num_smooth = 0.7
line_width_vline = 2*2; alpha_num_vline = 1
spine_line_width = 4.8; spine_color = 'black'

sns.set(font_scale=5.4, rc={"grid.linewidth": 2.5})
sns.set_style("ticks",{'axes.grid' : True})

pre_data = andong_site_gedi_pre; post_data = andong_site_gedi_post; pre_color = 'dodgerblue'; post_color = 'firebrick'

for i, var in enumerate(varis_canpy_10):
    row = i//5
    col = i-row*5

    axes[row, col].set_title(var.replace('CanpyRH', 'CRH ').replace('0',''))
      
    a1 = sns.kdeplot(ax=axes[row, col], y = pre_data[pre_data[var]>0][var], bw_adjust = bw_num_den, fill = fill_bool, color = pre_color, linewidth = line_width, alpha = alpha_num) #dodgerblue
    a2 = sns.kdeplot(ax=axes[row, col], y = post_data[post_data[var]>0][var], bw_adjust = bw_num_den, fill = fill_bool, color = post_color, linewidth = line_width, alpha = alpha_num) #firebrick
    a3 = sns.kdeplot(ax=axes[row, col], y = pre_data[pre_data[var]>0][var], bw_adjust = bw_num_smooth, fill = fill_bool_smooth, color = pre_color, linewidth = line_width_smooth, alpha = alpha_num_smooth) #dodgerblue
    a4 = sns.kdeplot(ax=axes[row, col], y = post_data[post_data[var]>0][var], bw_adjust = bw_num_smooth, fill = fill_bool_smooth, color = post_color, linewidth = line_width_smooth, alpha = alpha_num_smooth) #firebrick
    
    a1.axhline(y=pre_data[pre_data[var]>0][var].mean(), color=pre_color, linewidth = line_width_vline, alpha = alpha_num_vline, linestyle='dashed')
    a1.axhline(y=post_data[post_data[var]>0][var].mean(), color=post_color, linewidth = line_width_vline, alpha = alpha_num_vline, linestyle='dashed')
    
    a1.tick_params(labelsize=75)
    
    ## frame
    a1.spines['top'].set_linewidth(spine_line_width); a1.spines['top'].set_color(spine_color)
    a1.spines['bottom'].set_linewidth(spine_line_width); a1.spines['bottom'].set_color(spine_color)
    a1.spines['right'].set_linewidth(spine_line_width); a1.spines['right'].set_color(spine_color)
    a1.spines['left'].set_linewidth(spine_line_width); a1.spines['left'].set_color(spine_color)
    ## frame
    
    a1.text(a1.get_lines()[1].get_data()[0].max()*0.1, pre_data[pre_data[var]>0][var].mean()+post_data[post_data[var]>0][var].max()*0.05, 'pre : ' + str('%.2f'%pre_data[pre_data[var]>0][var].mean()) + ' m')
    a1.text(a2.get_lines()[1].get_data()[0].max()*0.1, post_data[post_data[var]>0][var].mean()-post_data[post_data[var]>0][var].max()*0.1, 'post : ' + str('%.2f'%post_data[post_data[var]>0][var].mean()) + ' m (x' + str(round(post_data[post_data[var]>0][var].mean()/pre_data[pre_data[var]>0][var].mean(), 2)) + ')')

    a1.set(xlabel='Density', ylabel='')
    a1.set(xticklabels=[])
    a1.tick_params(bottom=False)
    
    a1.set(ylim=(0-max(pre_data[pre_data[var]>0][var].max(), post_data[post_data[var]>0][var].max())*0.1, max(pre_data[pre_data[var]>0][var].max(), post_data[post_data[var]>0][var].max())*1.15))
    
    if i == 0 or i == 5:
        a1.set_ylabel(ylabel='Canopy Height (m)', fontsize = 68)
    if i == 4:
        #a1.legend(labels=['pre-fire', 'post-fire'], frameon=1)
        legend = a1.legend(labels=['pre-fire', 'post-fire'], frameon=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(1)
        frame.set_linewidth(4)
        
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(90, 38))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)

bw_num_den = 0.35; fill_bool = 1; line_width = 0.6*2; alpha_num = 0.2
bw_num_smooth = 1.2; fill_bool_smooth = 0; line_width_smooth = 1.5*3; alpha_num_smooth = 0.7
line_width_vline = 2*2; alpha_num_vline = 1
spine_line_width = 4.8; spine_color = 'black'

sns.set(font_scale=5.4, rc={"grid.linewidth": 2.5})
sns.set_style("ticks",{'axes.grid' : True})

pre_data = andong_compare_gedi_pre; post_data = andong_compare_gedi_post.apply(pd.to_numeric, errors = 'coerce') *1.03; pre_color = 'dodgerblue'; post_color = 'darkslategray'

for i, var in enumerate(varis_canpy_10):
    row = i//5
    col = i-row*5

    axes[row, col].set_title(var.replace('CanpyRH', 'CRH ').replace('0',''))
      
    a1 = sns.kdeplot(ax=axes[row, col], y = pre_data[pre_data[var]>0][var], bw_adjust = bw_num_den, fill = fill_bool, color = pre_color, linewidth = line_width, alpha = alpha_num) #dodgerblue
    a2 = sns.kdeplot(ax=axes[row, col], y = post_data[post_data[var]>0][var], bw_adjust = bw_num_den, fill = fill_bool, color = post_color, linewidth = line_width, alpha = alpha_num) #firebrick
    a3 = sns.kdeplot(ax=axes[row, col], y = pre_data[pre_data[var]>0][var], bw_adjust = bw_num_smooth, fill = fill_bool_smooth, color = pre_color, linewidth = line_width_smooth, alpha = alpha_num_smooth) #dodgerblue
    a4 = sns.kdeplot(ax=axes[row, col], y = post_data[post_data[var]>0][var], bw_adjust = bw_num_smooth, fill = fill_bool_smooth, color = post_color, linewidth = line_width_smooth, alpha = alpha_num_smooth) #firebrick
    
    a1.axhline(y=pre_data[pre_data[var]>0][var].mean(), color=pre_color, linewidth = line_width_vline, alpha = alpha_num_vline, linestyle='dashed')
    a1.axhline(y=post_data[post_data[var]>0][var].mean(), color=post_color, linewidth = line_width_vline, alpha = alpha_num_vline, linestyle='dashed')

    a1.tick_params(labelsize=75)

    ## frame
    a1.spines['top'].set_linewidth(spine_line_width); a1.spines['top'].set_color(spine_color)
    a1.spines['bottom'].set_linewidth(spine_line_width); a1.spines['bottom'].set_color(spine_color)
    a1.spines['right'].set_linewidth(spine_line_width); a1.spines['right'].set_color(spine_color)
    a1.spines['left'].set_linewidth(spine_line_width); a1.spines['left'].set_color(spine_color)
    ## frame  
    
    a1.set(xlabel='Density', ylabel='')
    a1.set(xticklabels=[])
    a1.tick_params(bottom=False)
    
    a1.set(ylim=(0-max(pre_data[pre_data[var]>0][var].max(), post_data[post_data[var]>0][var].max())*0.1, max(pre_data[pre_data[var]>0][var].max(), post_data[post_data[var]>0][var].max())*1.13))
    
    if pre_data[pre_data[var]>0][var].mean()>post_data[post_data[var]>0][var].mean():
        a1.text(a1.get_lines()[1].get_data()[0].max()*0.1, pre_data[pre_data[var]>0][var].mean()+post_data[post_data[var]>0][var].max()*0.05, 'pre : ' + str('%.2f'%pre_data[pre_data[var]>0][var].mean()) + ' m')
        a1.text(a2.get_lines()[1].get_data()[0].max()*0.1, post_data[post_data[var]>0][var].mean()-post_data[post_data[var]>0][var].max()*0.1, 'post : ' + str('%.2f'%post_data[post_data[var]>0][var].mean()) + ' m (x' + str(round(post_data[post_data[var]>0][var].mean()/pre_data[pre_data[var]>0][var].mean(), 2)) + ')')
    else:
        a1.text(a1.get_lines()[1].get_data()[0].max()*0.1, pre_data[pre_data[var]>0][var].mean()-post_data[post_data[var]>0][var].max()*0.1, 'pre : ' + str('%.2f'%pre_data[pre_data[var]>0][var].mean()) + ' m')
        a1.text(a2.get_lines()[1].get_data()[0].max()*0.1, post_data[post_data[var]>0][var].mean()+post_data[post_data[var]>0][var].max()*0.05, 'post : ' + str('%.2f'%post_data[post_data[var]>0][var].mean()) + ' m (x' + str(round(post_data[post_data[var]>0][var].mean()/pre_data[pre_data[var]>0][var].mean(), 2)) + ')')

    if i == 0 or i == 5:
        a1.set_ylabel(ylabel='Canopy Height (m)', fontsize = 68)
    if i == 4:
        legend = a1.legend(labels=['pre-fire', 'post-fire'], frameon=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(1)
        frame.set_linewidth(4)
    
plt.show()