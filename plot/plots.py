import pandas as pd
import numpy as np
import matplotlib as plt
from pathlib import Path
from pprint import pprint
from matplotlib import pyplot as plt
from datetime import datetime
from statistics import mean
from multiprocessing import Pool
from  matplotlib import cm as cm
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.stats import linregress, norm, gaussian_kde

def timeseriesplots(data: dict, ylabel= np.nan):

    fig, ax = plt.subplots(1,1,figsize=[16,3])

    dates = next(iter(data.values())).index
    years = dates.year.unique().tolist()[1:]
    year_lines = pd.to_datetime({'year':years, 'month':1, 'day':1})
    
    for key in data:
        ax.plot(data[key], 'o', label=key, color='cadetblue', markersize=3)
    
    for year in year_lines:
        ax.axvline(x=year, linestyle='--', linewidth=1.5, color='black')

    
    fmt_half_year = mdates.MonthLocator(interval=2)
    ax.xaxis.set_major_locator(fmt_half_year)

    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))


    datemin = np.datetime64(dates[0], 'Y')
    datemax = np.datetime64(dates[-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)

    ax.format_xdata = mdates.DateFormatter('%b-%Y')

    ax.annotate('Average Water Vapor\nin South America', (0.91, 0.1), color='gray', xycoords='axes fraction', fontsize=9)

    ax.set_facecolor('whitesmoke')
    ax.set_ylabel(ylabel)
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_horizontalalignment('center')
    ax.grid(True,which='both', color='lightgray')
    ax.legend(shadow=True)
    fig.tight_layout()
    
    return fig, ax

def hist_kde(data, bwidth=0.13):
    
    fig, ax = plt.subplots(1, 1, figsize=[8, 4])
    
    binwidth = 0.1
  
    for i, location in enumerate(data.keys()):

        bins = np.arange(min(data[location]['average']), max(data[location]['average']) + binwidth, binwidth)

        kwargs = dict(alpha=0.6, bins=bins)
        n, bins, _ = ax.hist(data[location]['average'], density=True, stacked=True, **kwargs, label='Histogram', color='grey', histtype='barstacked', edgecolor='black', linewidth=0.5)

        density = gaussian_kde(data[location]['average'], bw_method='silverman')

        density.covariance_factor = lambda :  bwidth
        density._compute_covariance()
        
        ax.plot(bins, density(bins), color='grey', linewidth=2, label='KDE Probability density function',linestyle='-')
        
        plt.fill_between(
            bins, 
            density(bins),
            color= 'grey',
            alpha= 0.2)

    ax.set_ylabel('Density')
    ax.set_xlabel('Satellite Image Average (%)')
    ax.grid(color='darkgray', axis='y', which='major')
    ax.set_facecolor('whitesmoke')

    handles, labels = plt.gca().get_legend_handles_labels()


    ax.legend(shadow=True)
    fig.tight_layout()
    return fig