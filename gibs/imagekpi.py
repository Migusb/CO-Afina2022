
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