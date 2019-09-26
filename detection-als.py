import plotly.express as px
from laspy.file import File
from numba import jit
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import disk, square, dilation, watershed
from scipy.ndimage import label
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, rank
import os
from scipy import ndimage, stats
from osgeo import gdal, ogr, osr
import time
import matplotlib.image as mpimg
import dask.dataframe as dd
from dask import delayed

def height(data, array): 
    for i in range(data.shape[0]):
        array[:, data[i][0], data[i][1]] = data[i][2:] 
        
    return array

if __name__ == '__main__':

    file = sys.argv[1]
    file_name = file[:-4]

    # Scale parameters for voxelization (unit in meters) 
	scale = 0.25
	scale_z = 1
	kernel_buffer = 1

    # Load LiDar file and read X,Y,Z
	inFile = File(file, mode = "r")

	start_time = time.time()

	X, Y, Z = inFile.x, inFile.y, inFile.z

    # Calculate Min Max for every dimension
	max_x, min_x = np.max(X), np.min(X)
	max_y, min_y = np.max(Y), np.min(Y)
	max_z, min_z = np.max(Z), np.min(Z)

    # Transform coordinates into a grid array index
	Xp = np.array((1/scale)*(X - min_x), dtype=np.int)
	Yp = np.array((1/scale)*(max_y -Y), dtype=np.int)
	Zp = np.array((1/scale_z)*(Z - min_z), dtype=np.int)

    # Group all Lidar points in same grid cell
	df = dd.from_array(np.stack((Yp, Xp, Zp), axis=1), chunksize=int(len(Xp)/10), columns=['y','x','z'])
	dz_values = df.groupby(['y','x']).agg(['min', 'max', 'count']).reset_index().values
	dz_values.compute()

    # Create a max height, min height and point density grid raster
	dz_values = np.asarray(dz_values).astype(int)
	z = np.full((int(3), int(np.max(Yp) + 1), int(np.max(Xp) + 1)), 0)
	z = height(dz_values, z)

	z_max = z[1,:,:]
	z_den = z[2,:,:]

	# Convolution with a kernel buffer to sum all the points within a buffer
	# Makes the score more robust
	kernel = np.ones((int(kernel_buffer/scale), int(kernel_buffer/scale)))
	score_den = ndimage.convolve(z_den, kernel, mode='constant', cval=1.0)

	# Normalization of both max height and density scores
	score_den = (score_den-np.min(score_den))/(np.max(score_den)-np.min(score_den))
	score_z = (z_max-np.min(z_max))/(np.max(z_max)-np.min(z_max))

	# Final score weighted with both max height and density scores
	score_array = 0.5*score_z + 0.5*score_den
                 
	thresh = threshold_otsu(score_array)

    print('Threshold score is', thresh)

    # Detection of trees with min distance of 1.5 meters based on the score, previously filtered by Otsu segmentation        

	binary_array = score_array > 1.0*thresh
	score_array = np.multiply(binary_array, score_array)
	threshold_score = np.percentile(score_array, 90)

	markers = peak_local_max(score_array, indices=True, min_distance=int(1.5/scale), exclude_border=True, threshold_abs= threshold_score, labels=binary_array)

    # DataFrame with the trees	
	index = 0
	df_table = pd.DataFrame( columns = ['index', 'X', 'Y'])

	for marker in markers:
    	
    	row, column = marker[0],marker[1]
    	lat, lon = (-scale*row + max_y - 0.5*scale, scale*column + min_x + 0.5*scale)
    	df_table.loc[index] = [index, lon, lat]
    	index = index + 1

	print('In File: ', file,' were found trees :',index)

	df_table.to_csv(file[:-4] + '.csv', sep=',',index=False)

	print("--- %s seconds ---" % (time.time() - start_time))
