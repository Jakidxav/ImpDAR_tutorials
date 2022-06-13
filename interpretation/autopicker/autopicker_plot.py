import warnings
warnings.filterwarnings('ignore')

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors, colorbar
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter



"""
Create example traces to illustrate how autopicking algorithm works.
"""
def plotting_examples():

    fig, ax = plt.subplots(figsize=(9, 9))

    #create example traces
    x = np.linspace(0, 20, 1000)
    y = np.cos(x) + 1

    #set example list of points
    empty = np.array([0, 0, 0, 0, 0, 0])
    xlist = np.array([0, 1, 2, 3, 4, 5])
    pi = empty + np.pi
    twopi = pi + np.pi
    threepi = twopi + np.pi

    c = 'tab:blue'
    ax.plot(y, x, color=c)
    ax.plot(y+1, x, color=c)
    ax.plot(y+2, x, color=c)
    ax.plot(y+3, x, color=c)

    #invert y axis so increasing depth is down
    plt.gca().invert_yaxis()

    #set x and y axis labels
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel('Traces -->')

    #turn off xtick labels
    ax.set_xticklabels([])

    #pick a threshold
    #ax.axhline(y=2*np.pi, color='r', linestyle='--', linewidth=3)

    #plot contours
    #ax.axhline(y=3/2*np.pi, color='orange', linewidth=3)
    #ax.axhline(y=5/2*np.pi, color='orange', linewidth=3)

    #plot ridge line
    ax.axhline(y=2*np.pi, color='lime', linewidth=3)
    ax.scatter(xlist, twopi, color='lime', linewidth=3)

    #plot top and bottom of wave packet
    ax.axhline(y=np.pi, color='magenta', linewidth=3)
    ax.scatter(xlist, pi, color='magenta', linewidth=3)

    ax.axhline(y=3*np.pi, color='magenta', linewidth=3)
    ax.scatter(xlist, threepi, color='magenta', linewidth=3)

    #plt.savefig('./images/finished_wave_packet.png', format='png', bbox_inches='tight')

    plt.show()
    
    
    
"""
Given a RadarData ImpDAR object and a list of contours, plot all contours over the data attribute.
"""
def plot_all_contours(dat, contour_list, min_, max_, yticks, ytick_labels):
    
    fig, ax = plt.subplots(figsize=(15, 9))
    
    #plot_radargram
    ax.imshow(dat.data, cmap='gray', aspect='auto', interpolation='nearest', vmin=min_, vmax=max_)

    #iterate over contours
    for i, contour in enumerate(contour_list):
        #change color of contours after each for loop
        #starts blue at surface, moves to yellow then red deeper in profile
        c = cm.RdYlBu_r(i/len(contour_list))

        #plot each contour on top of radargram
        plt.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color=c)

    #set x and y ticks explicitly since we aren't using ImpDAR's plot_radargram() method
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    #reset x and y labels
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Depth (m)')

    plt.show()
    


"""
Plot one contour of interest over radargram.
"""
def plot_wave_packet(dat, contours_list, ridge_points_list, uniquex_list, top_list, bottom_list, z, min_, max_, xlims, ylims, yticks, ytick_labels, plot_contour=True, plot_ridge=True, plot_top_bottom=True):
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    #isolate contour
    contour = contours_list[z]
    
    #plot radargram
    ax.imshow(dat.data, cmap='gray', aspect='auto', interpolation='nearest', vmin=min_, vmax=max_)

    if(plot_contour==True):
        #plot example contour on top of radargram
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1., color='orange')

    
    #plot ridge points
    ridge = ridge_points_list[z]
    uniqx = uniquex_list[z]
    top_ = top_list[z]
    bottom_ = bottom_list[z]

    for j, x in enumerate(uniqx):
        t = top_[j]
        r = ridge[j]
        b = bottom_[j]
        
        if(plot_ridge==True):
            ax.scatter(x, r, color='lime', marker='o', s=10)
            
        if(plot_top_bottom==True):
            ax.scatter(x, t, color='magenta', marker='o', s=10)
            ax.scatter(x, b, color='magenta', marker='o', s=10)


    #set x and y ticks explicitly since we aren't using ImpDAR's plot_radargram() method
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    #reset x and y labels
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Depth (m)')

    plt.show()



"""
Animated plot of one contour and the ridge points it contains.
"""
#animate finding ridge points from contour
def trace_contours(dat, long_contours_list, z, ridge_points_list, uniquex_list, min_, max_, xlims, ylims, yticks, ytick_labels):
    
    #define figure and axis subplots to plot onto
    fig, ax = plt.subplots(figsize=(13, 9))
    #set grid
    grid = ax.imshow(dat.data, cmap='gray', aspect='auto', vmin=min_, vmax=max_, interpolation='nearest')
    
    #subset lists by contour number
    long_contours = long_contours_list[z]
    ridge_points = ridge_points_list[z]
    uniquex = uniquex_list[z]

    #draw main contour
    ax.plot(long_contours[:, 1], long_contours[:, 0], color='orange', linewidth=1)
    
    #add points for top and bottom of contour, as well as for the ridge point
    points, = ax.plot([], [], color='lime', marker='o')
    
    #list of y axis indices used for plotting vertical lines
    y = np.arange(0, dat.data.shape[0])
    
    #list to hold vertical lines during plotting
    vline, = ax.plot([], [], color='lime', linestyle='--', alpha=0.4)

    #set axis limits (you have to do this explicitly for animations)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    #label x and y axes
    ax.set_xlabel('Trace Number')
    ax.set_ylabel('Depth (m)')

    #set subplot titles here
    ax.set_title('Tracing Contours')
    
    #set y ticks and labels
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    
    
    #the animation function wants an init_func variable
    #so we initialize all lines to blank
    def init():
        points.set_data([], [])
        vline.set_data([], [])
        
        return points, vline,


    #to animate, iterate over the 'frames' parameter
    #each 'idx' is a frame
    def animate(idx):
            
        points.set_data(uniquex[0:idx], ridge_points[0:idx])
        
        #get the current trace number
        x_ = uniquex[idx]
        
        #replicate vertical line at given trace number with y-axis shape
        y_ = np.zeros(y.shape) + x_
        
        #add vertical line to plot
        vline.set_data(y_, y)
        

        return points, vline,

    #call our animation function: frames is the number of points in our uniquex_list
    anim = FuncAnimation(fig, animate, init_func=init, frames=181, interval=50, repeat=True)
    
    f = r"./images/ridgeline.gif" 
    writergif = PillowWriter(fps=181/50, bitrate=20000) 
    
    anim.save(f, writer=writergif)
    
    return anim