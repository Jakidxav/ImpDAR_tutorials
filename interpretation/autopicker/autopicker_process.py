import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors, colorbar

from skimage import measure
from skimage.filters import gaussian

from impdar import *


"""
Find the contours that encircle a given value in a given RadarData object by accessing its
data attribute.
"""
def find_contours(radargram, value):
    contours = measure.find_contours(radargram.data, value) 

    return contours


"""
Filter a list of positive or negative contours list by some arbitrary length value.
"""
def filter_contours(contours_list, value):
    #append lengths of contours to list
    contour_length = []

    #append lengths of contours to list
    for contour in contours_list:
        contour_length.append(len(contour))

    #create an iterable list of the contours that are longer than some arbitrary value
    long_contours_idx = list(np.where(np.array(contour_length) > value)[0])

    #create subset of contours based on length criterion
    long_contours = [contours_list[idx] for idx in long_contours_idx]
    
    return long_contours


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
            for j in range(row, maxrows-1):
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
             #if the point holds a positive value, then we want to
            #find the troughs on either side of it
            if data[row, col] < 0:
                #look for top contour
                for j in range(row, maxrows-1):
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


"""
Given a RadarData ImpDAR object and a list of contours from the find_contours() method defined above,
find the max or min of wave packet as well as the peaks of opposite polarity.

Complete processing method.
"""
def process_contours(dat, contours_list):
    #save lists
    argx_list = []
    uniquex_list = []
    saved_points_list = []

    ridge_points_list = []
    top, bottom = [], []
    top_list, bottom_list = [], []

    #let's process all of the contours here
    for i, contour in enumerate(contours_list):
        #get shape of data
        maxrows, maxcols = np.shape(dat.data)

        #find x values for contour and where along that contour that x value occurs (aka, y values)
        uniquex, saved_points = find_contour_points(contour, argx_list)
        saved_points_list.append(saved_points)

        #find y values and ridge points for each contour
        ridge_points = []
        to_deletex = []

        ridge, to_deletex = find_ridge_points(contour, to_deletex, uniquex, saved_points, dat.data, ridge_points)
        ridge_points_list.append(ridge)

        #now we need to delete the values that we couldn't find a ridge/trough value from
        #now they should be the same length
        uniquex = [x for x in uniquex if x not in to_deletex]    
        uniquex_list.append(uniquex)

        #find top and bottom troughs/peaks
        top, bottom = find_troughs(uniquex, ridge, dat.data)
        top_list.append(top)
        bottom_list.append(bottom)
        
     
    #return most important lists at the end of the method
    return argx_list, uniquex_list, ridge_points_list, top_list, bottom_list

