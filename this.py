import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances as pdist
import sys
import time
import warnings
import wradlib as wrl
import math
import os
from numpy.core.defchararray import add
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
%matplotlib inline

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
##add feature : file path can be directory(1) or individual file
class Raw_analyser():
    def __init__(self,filepath):
        self.filepath=filepath
        
    
    def raw():
    '''
    CONVERTING RAW FILE INTO CARTESIAN PRODUCT
    PARAMETERS : filepath(path of a raw file)
    RETURNS :  clus1 dataframe having parameters as X,Y,Z,dbz
    '''
    #file_path = "/content/BHP200513000224.RAWMMKL"
        fcontent = wrl.io.read_iris(filepath)
        X = []
        Y = []
        Z = []
        DB_DBZ = []
        # DB_VEL = []
        # DB_WIDTH = []
        start = time.time()
        for i in fcontent['data']:
            azi_start = fcontent['data'][i]['sweep_data']['DB_DBZ']['azi_start']
            azi_stop = fcontent['data'][i]['sweep_data']['DB_DBZ']['azi_stop']

            ele_start = fcontent['data'][i]['sweep_data']['DB_DBZ']['ele_start']
            ele_stop = fcontent['data'][i]['sweep_data']['DB_DBZ']['ele_stop']

            rbins = fcontent['data'][i]['sweep_data']['DB_DBZ']['rbins']

            db_dbz = fcontent['data'][i]['sweep_data']['DB_DBZ']['data']
            # db_vel = fcontent['data'][i]['sweep_data']['DB_VEL']['data']
            # db_width = fcontent['data'][i]['sweep_data']['DB_WIDTH']['data']

            for hz in range(len(azi_start)):
                azi_mean = ((azi_start[hz] + azi_stop[hz]) / 2) * math.pi/180
                ele_mean = ((ele_start[hz] + ele_stop[hz]) / 2) * math.pi/180
                rbin = rbins[hz]

                dbz = db_dbz[hz]
                # vel = db_vel[hz]
                # width = db_width[hz]

                for r in range(rbin):
                    X.append(r*math.cos(ele_mean) * math.cos(azi_mean))
                    Y.append(r*math.cos(ele_mean)* math.sin(azi_mean))
                    Z.append(r* math.sin(ele_mean))
                    DB_DBZ.append(dbz[r])
                    # DB_VEL.append(vel[r])
                    # DB_WIDTH.append(width[r])

        dat={'X':X,'Y':Y,'Z':Z,'dbz':DB_DBZ}
        df=pd.DataFrame(dat)
        mask1=df['dbz']>=0
        clus1=df[mask1]
        return clus1    #dataframe of columns : X,Y,Z,dbz
#remove return and save it in class variable

class clustering():
    def __init__(self):
        self.eps2=0.5
        self.min_pts2=5
        self.eps=0.006
        self.min_pts=4
        self.thr=2
        self.eps2=0.5
        self.min_pts2=5
        
    
    def dbscan_sub_clus(df,i):    
    '''
    Does another level of clustering on the basis of DBZ values.
    PARAMETER : eps2 , min_pts2 will be the parameter of dbz level of clustering
                i is cluster label
                df dataframe of only those points belonging to the cluster i
    ''' 
        db = DBSCAN(min_samples=min_pts2, eps=eps2)
        db.fit(df[['dbz']])
        lab = add(db.labels_.astype(str), '_'+str(i)) # new label is formed as 'New labels'_'cluster label i'
        return lab      #returns string 
    def cluster_2point0(df):   
        '''
        In this function we computed median of each cluster and tried to merge cluster if the distance between their median is less than thr km.
        PARAMETERS:   df -> dataframe having all the points with column X,Y,Z,dbz
                        eps,min_pnts are the parameters for the DBSCAN on the basis of X,Y,Z values
                        eps2 , min_pts2 will be the parameter of dbz level of clustering
                        thr is the threshold value on the basis of which the cloud cluster has to be merged
        '''
        df = plot_dbscan(df, eps,min_pts, eps2=eps2, min_pts2=min_pts2)   #df has now 2 additional columns label level 0 and label level 1
        df_median = df[~df['label level 1'].str.contains('-1_')].groupby('label level 1').median()[['X', 'Y', 'Z']] #ignoring the noise points of label level one and grouping df by median.
        pdistance = pdist(df_median)  #computing distance matrix
        while np.amin(pdistance) <= thr:    #entry condition
            df_median = df.groupby('label level 1').median()[['X', 'Y', 'Z']]   #grouping each points with their labels and computing medoid
            pdistance = pdist(df_median)
            for i in range(len(pdistance)): 
                pdistance[i][i] = np.inf
            idx = np.argwhere(pdistance == (np.amin(pdistance)))[0]   #using the index of the min distance to get the labels, index here are the labels of 'labels level 1'
            df_median.index[idx[0]], df_median.index[idx[1]]
            df['label level 1'].replace({df_median.index[idx[1]]:df_median.index[idx[0]]}, inplace=True) # replacing the second cluster name with the first cluster's name
        return df   #COLUMNS : X,Y,Z,dbz,label level 0,label level 1

#merge this with cluster

class postprocessing():
    def __init__(self,df):
        self.eps=0.006
        self.min_pts=4
        self.thr=2
        self.eps2=0.5
        self.min_pts2=5
        self.df=df
        self.filename=filemane
    def plot_dbscan():
        
        scl = MinMaxScaler()      
        df_st = pd.DataFrame(scl.fit_transform(df.iloc[:,:-1]), columns=['X', 'Y', 'Z'])  #NORMALIZING THE DATAFRAME
        # df_st = df
        db = DBSCAN(eps=eps, min_samples=min_pnts) #applying DBSCAN on the column x,y,z
        db.fit(df_st)
        df['label level 0'] = db.labels_  #MAKING ANOTHER COLUMN IN df 'label level 0' which stores the value of db.labels_
        labels = np.zeros(db.labels_.shape).astype(str)
        #df_subclust = df.copy()
        for i in set(db.labels_):   #PERFORMING 2ND LEVEL OF CLUSTERING CLUSTER BY CLUSTER
            if i == -1: #noise points         
                continue
            msk = (df['label level 0'] == i).values     #performing operation cluster by cluster
            labels[msk] = dbscan_sub_clus(df.iloc[msk], i, eps2=eps2, min_pts2=min_pts2)
        df['label level 1'] = labels    #COLUMN 'label level 1' STORES THE VALUES OF LABELS WHICH WAS THE RESULT OF CLUSTERING BASED ON DBZ VALUES
        return df   #COLUMNS WITH ATTRIBUTE : X,Y,Z,dbz,label level 0,label level 1
    def plot(az, ele):
        '''
        IN THIS FUNCTION , WE TRIED TO PLOT FOUR IMAGES. 
        1ST BEING THE PLOT OF POINTS BELONGING TO 'label level 0' WITHOUT NOISE
        2ND BEING THE PLOT OF POINTS BELONGING TO 'label level 1' WITHOUT NOISE
        3RD BEING THE PLOT OF "NOISE POINTS" OF 'label level 0'
        4TH BEING THE PLOT OF "NOISE POINTS" OF 'label level 1'
        PARAMETERS : az,ele are the azimuth and elevation value at which the plot has to be seen.
        '''
        global filename
        df = pd.read_csv(filename)
        #df = df[df['label level 0'] != -1] #removes noise
        fig = plt.figure(figsize=[18, 18])
        ax = plt.subplot(2, 2, 1, projection='3d')   # before subclustering
        for i in np.unique(df['label level 0']):
            if -1 == i:
                continue
            msk = df['label level 0'] == i
            ax.scatter3D(df[msk]['X'], df[msk]['Y'], df[msk]['Z'],s=1)
        #ax.set_xticks([-500, -250, 0, 250, 500])
        ax.view_init(ele, az)
        ax = plt.subplot(2, 2, 2, projection='3d')      
        for i in np.unique(df['label level 1']):
            if '-1_' in i:
                continue
            msk = (df['label level 1'] == i) & ( df['label level 0'] != -1 )
            ax.scatter3D(df[msk]['X'], df[msk]['Y'], df[msk]['Z'], s=1)
        ax.view_init(ele, az)
        #plt.xticks( [-500, 500],  [-500, 500] )
        ax = plt.subplot(2, 2, 3, projection='3d')    #for noise plot.
        msk = df['label level 0'] == -1
        ax.scatter3D(df[msk]['X'], df[msk]['Y'], df[msk]['Z'],s=1)
        ax.view_init(ele, az)
        ax = plt.subplot(2, 2, 4, projection='3d')    #for noise plot.
        for i in np.unique(df['label level 1']):
            if '-1_' not in i:
                continue
            msk = (df['label level 1'] == i) & ( df['label level 0'] != -1 )
            ax.scatter3D(df[msk]['X'], df[msk]['Y'], df[msk]['Z'],s=1)
        ax.view_init(ele, az)
        plt.show()
        print('labels before subclustruring : ', len(set(df['label level 0'])))
        print('labels after subclustruring : ', len(np.unique(df1[~df1['label level 1'].str.contains('-1_')]['label level 1'])))
        print('number of noise : ', len(np.unique(df1[df1['label level 1'].str.contains('-1_')]['label level 1'])))
    def plot_intract():
        '''
        THIS FUNCTION MAKES THE PLOT INTERACTIVE BY DYNAMICALLY CHANGING THE VALUE OF AZIMUTH AND ELEVATION.
        '''
        interact(plot, az= widgets.FloatSlider(value=0, min=0, max=360.0, step=10),ele = widgets.FloatSlider(value=90, min=0, max=90.0, step=10))
