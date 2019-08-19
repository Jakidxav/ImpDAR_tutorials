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



"""
Find the end points of a given contour (min and max x points), and then find every x point in between those two points.
"""
def find_contour_points(contour, argx_list):
    #first try to save argx list
    argx1 = np.argmin(contour[:, 1])
    argx2 = np.argmax(contour[:, 1])
    argx_list.append([argx1, argx2])
    
    #find uniquex values for each contour
    uniquex = (list(set([int(x) for x in contour[:, 1]])))
    
    #get rid of x values that we don't have contour data for
    saved_points = []
    delete = []

    for x in uniquex:
        wherex = np.where(contour[:, 1] == x)[0]
        if len(wherex) == 0 or len(wherex) == 1:
            delete.append(x)
        else:
            if len(wherex) > 2:
                saved_points.append(wherex[1:3])
            else:
                saved_points.append(wherex)
    
    #delete values that we don't have enough data for
    uniquex = [x for x in uniquex if x not in delete]
    
    return uniquex, saved_points



"""
Find ridge structure for all contours.
"""
def find_ridge_points(contour, to_delete, uniquex, saved_points, data, ridge_points):
    
    for i, point in enumerate(saved_points):
        if(len(point == 2)):
            idx1 = point[0]
            idx2 = point[1]
        elif(len(point > 2)):
            idx1 = point[1]
            idx2 = point[2]
        else:
            raise ValueException("You need to have at least two matching y-values.")

        x = uniquex[i]

        y1 = int(contour[idx1, 0])
        y2 = int(contour[idx2, 0])

        #check to see whether *both* points lie in a ridge or trough (for positive and negative contours)
        #only contains positive numbers: ridge --> search for troughs later
        if np.logical_and(data[y1, x] > 0, data[y2, x] > 0):
            if len(data[y2:y1, x]) == 0:
                ridge_points.append(np.where(data[:, x] == np.max(data[y1:y2, x]))[0])
            else:
                ridge_points.append(np.where(data[:, x] == np.max(data[y2:y1, x]))[0])
            
        #only contains negative numbers: trough --> search for ridges later
        elif np.logical_and(data[y1, x] < 0, data[y2, x] < 0):
            if len(data[y2:y1, x]) == 0:
                ridge_points.append(np.where(data[:, x] == np.min(data[y1:y2, x]))[0])
            else:
                ridge_points.append(np.where(data[:, x] == np.min(data[y2:y1, x]))[0])
            
        #else the points contain both a positive and negative value, we need to skip over that
        #if we skip over that point, we need to delete it from uniquex and saved_points
        else:
            to_delete.append(x)
      
    return [elem[0] for elem in ridge_points], to_delete



"""
Find peaks/troughs for a single contour
"""
def find_troughs_single(uniquex, ridge_points, data):
    maxrows, maxcols = np.shape(data)
    
    top, bottom = [], []
    temptop_idx, tempbottom_idx = [], []
    temptop_points, tempbottom_points = [], []

    for i, point in enumerate(uniquex):
        row = ridge_points[i][0]
        col = point
        
        #if the point holds a positive value, then we want to
        #find the troughs on either side of it
        if data[row, col] > 0:
            #look for top contour
            for j in range(row, maxrows):
                #enter the trough
                if (data[j, col] <= 0):
                    temptop_idx.append(j)
                    temptop_points.append(data[j, col])

                    #enterting this if statement would signify leaving the trough
                    if data[j+1, col] > 0:
                        #find the minimum in that column of the trough
                        trough_min = np.argmin(temptop_points)
                        top.append(temptop_idx[trough_min])

                        #reset these to empty for the next column
                        temptop_idx, temptop_points = [], []
                        break

            #look for bottom contour
            for k in range(0, row):
                #enter the trough
                if (data[row-k, col] <= 0):
                    tempbottom_idx.append(row-k)
                    tempbottom_points.append(data[row-k, col])

                    #enterting this if statement would signify leaving the trough
                    if data[row-k-1, col] > 0:
                        #find the minimum in that column of the trough
                        trough_bottom_min = np.argmin(tempbottom_points)
                        bottom.append(tempbottom_idx[trough_bottom_min])

                        #reset these to empty for the next column
                        tempbottom_idx, tempbottom_points = [], []
                        break


        #if the point holds a negative value, then we want to
        #find the peaks on either side of it
        else:
            pass

    return top, bottom



"""
Find peaks/troughs for every contour.
"""
def find_troughs(uniquex, ridge_points, data):
    maxrows, maxcols = np.shape(data)
    
    top, bottom = [], []
    temptop_idx, tempbottom_idx = [], []
    temptop_points, tempbottom_points = [], []

    for i, point in enumerate(uniquex):
        row = ridge_points[i]
        col = point

        #if the point holds a positive value, then we want to
        #find the troughs on either side of it
        if data[row, col] > 0:
            #look for top contour
            for j in range(row, maxrows):
                #enter the trough
                if (data[j, col] <= 0):
                    #print(j)
                    temptop_idx.append(j)
                    temptop_points.append(data[j, col])

                    #enterting this if statement would signify leaving the trough
                    if data[j+1, col] > 0:
                        #find the minimum in that column of the trough
                        trough_min = np.argmin(temptop_points)
                        top.append(temptop_idx[trough_min])

                        #reset these to empty for the next column
                        temptop_idx, temptop_points = [], []
                        break

            #look for bottom contour
            for k in range(0, row):
                #enter the trough
                if (data[row-k, col] <= 0):
                    tempbottom_idx.append(row-k)
                    tempbottom_points.append(data[row-k, col])

                    #enterting this if statement would signify leaving the trough
                    if data[row-k-1, col] > 0:
                        #find the minimum in that column of the trough
                        trough_bottom_min = np.argmin(tempbottom_points)
                        bottom.append(tempbottom_idx[trough_bottom_min])

                        #reset these to empty for the next column
                        tempbottom_idx, tempbottom_points = [], []
                        break


        #if the point holds a negative value, then we want to
        #find the peaks on either side of it
        else:
             #if the point holds a positive value, then we want to
            #find the troughs on either side of it
            if data[row, col] < 0:
                #look for top contour
                for j in range(row, maxrows):
                    #enter the ridge
                    if (data[j, col] >= 0):
                        temptop_idx.append(j)
                        temptop_points.append(data[j, col])

                        #enterting this if statement would signify leaving the ridge
                        if data[j+1, col] < 0:
                            #find the minimum in that column of the trough
                            ridge_top_max = np.argmin(temptop_points)
                            top.append(temptop_idx[ridge_top_max])

                            #reset these to empty for the next column
                            temptop_idx, temptop_points = [], []
                            break

                #look for bottom contour
                for k in range(0, row):
                    #enter the trough
                    if (data[row-k, col] >= 0):
                        tempbottom_idx.append(row-k)
                        tempbottom_points.append(data[row-k, col])

                        #enterting this if statement would signify leaving the trough
                        if data[row-k-1, col] > 0:
                            #find the minimum in that column of the trough
                            ridge_bottom_max = np.argmin(tempbottom_points)
                            bottom.append(tempbottom_idx[ridge_bottom_max])

                            #reset these to empty for the next column
                            tempbottom_idx, tempbottom_points = [], []
                            break

    return top, bottom





