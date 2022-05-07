
import cv2
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

def read_img(f):
    imgcolor = cv2.imread(str(f))
    imgbw =  cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
    imgbwnorm = cv2.normalize(imgbw, None, alpha = 0, beta = 100, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img = np.array(imgbwnorm).flatten().astype('float64')
    return img

def dfmaker(f):
    img = read_img(f)
    year, month, day, hour = f.stem.split('-')
    tstmp = datetime.strptime(year+'-'+month+'-'+day+' '+hour+":00:00", "%Y-%m-%d %H:%M:%S")
    df =  pd.DataFrame(dict(timestamp=[tstmp], average=[mean(img)]))
    df.index = df.pop('timestamp')
    return df

def read_tidykpi(f):
    df = pd.read_csv(f, sep=',')
    timestamp = pd.to_datetime(df.pop('timestamp'), utc=True)
    df.index = timestamp
    return df

def main(files):
    with Pool(8) as pool:
        dfs = pool.map(dfmaker, files)
        return pd.concat(dfs).sort_index()


if __name__=="__main__":
    datadir = "/home/migusb/projects/hackathons/La-Conga2022/data"
    imgdir = Path(datadir).glob('**/*.jpg')
    main(sorted(list(imgdir))).to_csv("img_average.csv")