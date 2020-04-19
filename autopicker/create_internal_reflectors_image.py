import os
import netCDF4 as nc
import numpy as np
import pandas as pd

import copy

import scipy.io as io

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors, colorbar

from skimage import measure
from skimage.filters import gaussian

from impdar import *

from autopicker_methods import *


#files on disk
green_image = './images/nline5.png'
green_mat = './data/nline5.mat'

#min and max for colorbar plotting
mini = -8.774235248565674
maxi = 9.667154026031533

#load in data
dat = io.loadmat(green_mat)
data = dat['data']


#get positive and negative contours
pos = measure.find_contours(data, 3)
neg = measure.find_contours(data, -3)

#save contours and their lengths to lists
c_pos, c_neg, c_length_pos, c_length_neg = [], [], [], []


#positive contours
for contour in pos:
    c_pos.append(contour)

#negative contours
for contour in neg:
    c_neg.append(contour)

#now extract long contours
#this could be done instead of filtering?
for contour in pos:
    c_length_pos.append(len(contour))
    
for contour in neg:
    c_length_neg.append(len(contour))


#create an iterable list of the contours that are longer than some arbitrary value
long_contours_pos_idx = list(np.where(np.array(c_length_pos) > 250)[0])
long_contours_neg_idx = list(np.where(np.array(c_length_neg) > 250)[0])


long_contours_pos = [pos[idx] for idx in long_contours_pos_idx]
long_contours_neg = [neg[idx] for idx in long_contours_neg_idx]

#isolate some internal reflectors
long_contours_subset = long_contours_pos[22:100]


#save lists
argx_list = []
uniquex_list = []
saved_points_list = []

ridge_points_list = []
top, bottom = [], []
top_list, bottom_list = [], []


for i, contour in enumerate(long_contours_subset):
    #get shape of data
    maxrows, maxcols = np.shape(data)
    
    #find x values for contour and where along that contour that x value occurs (aka, y values)
    uniquex, saved_points = find_contour_points(contour, argx_list)
    saved_points_list.append(saved_points)

    #find y values and ridge points for each contour
    ridge_points = []
    to_deletex = []
    
    ridge, to_deletex = find_ridge_points(contour, to_deletex, uniquex, saved_points, data, ridge_points)
    ridge_points_list.append(ridge)
    
    #now we need to delete the values that we couldn't find a ridge/trough value from
    #now they should be the same length
    uniquex = [x for x in uniquex if x not in to_deletex]    
    uniquex_list.append(uniquex)
        
    #find top and bottom troughs/peaks
    top, bottom = find_troughs(uniquex, ridge, data)
    top_list.append(top)
    bottom_list.append(bottom)
    
    #print(i, argx_list[i], len(uniquex_list[i]), len(ridge_points_list[i]), len(saved_points_list[i]))


fig, ax = plt.subplots(figsize=(12, 8))

#plot grid
p = ax.imshow(data, cmap='gray', aspect='auto', vmin=mini, vmax=maxi, interpolation='nearest')

for i, contour in enumerate(long_contours_subset):
    #plot contours
    ax.plot(contour[:, 1], contour[:, 0], color='orange', linewidth=1)

    #plot ridge points
    ridge = ridge_points_list[i]
    uniqx = uniquex_list[i]
    top_ = top_list[i]
    bottom_ = bottom_list[i]
    
    for j, x in enumerate(uniqx):
        t = top_[j]
        r = ridge[j]
        b = bottom_[j]
        
        ax.scatter(x, t, color='lime', marker='o', s=10)
        ax.scatter(x, r, color='magenta', marker='o', s=10)
        ax.scatter(x, b, color='lime', marker='o', s=10)
        
    print('Done processing contour {}/{}'.format(i+1, len(long_contours_subset)))


ax.set_ylim(3000, 500)

#plt.savefig('internal_reflectors1.pdf', format='pdf', bbox_inches='tight')





