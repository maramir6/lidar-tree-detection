from laspy.file import File
from laspy.header import Header
from numba import jit
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from scipy import ndimage, stats
from scipy.ndimage import label
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny, peak_local_max
from skimage.morphology import dilation, remove_small_holes, remove_small_objects, convex_hull_object, disk, square, watershed
from skimage import measure, color, img_as_ubyte
from skimage.draw import circle_perimeter, circle
from skimage.filters import roberts, sobel, scharr, prewitt, threshold_otsu, rank
from skimage.color import gray2rgb
import time, os, copy, sys
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@jit(nopython=True)
def fill_matrix(data, array): 
	for i in range(data.shape[0]):
		array[data[i][0]][data[i][1]][data[i][2]] = 1

	return array

def mask_generator(min_r, max_r):
    masks = []

    for r in range(min_r, max_r + 1):

        image = np.zeros((max_r + 1, max_r + 1))
        circy, circx = circle(int(0.5*max_r), int(0.5*max_r), r, shape=image.shape)
        image[circy, circx] = -1/r
        circy, circx = circle_perimeter(int(0.5*max_r), int(0.5*max_r), r - 1, shape=image.shape)
        circy, circx = circle_perimeter(int(0.5*max_r), int(0.5*max_r), r, shape=image.shape)
        image[circy, circx] = 1
        masks.append(image)

    return np.stack(masks, axis=0)

def scoring(array, masks, min_r, max_r, min_h, max_h):

    array_tensor = torch.from_numpy(np.expand_dims(array[min_h:max_h,:,:], axis=0)).double()
    conv_layer = nn.Conv2d(array_tensor.shape[1], masks.shape[0], max_r + 1, padding=1)
    masks_batch = np.repeat(masks[:,np.newaxis,:,:], array_tensor.shape[1], axis=1)
    conv_layer.weight.data = torch.from_numpy(masks_batch).double()
    conv_layer.bias.data = torch.from_numpy(np.zeros(masks.shape[0])).double()

    score = conv_layer(array_tensor)
    score = score.detach().numpy()

    return np.squeeze(np.max(score, axis=1).astype(int), axis=0)


if __name__ == '__main__':

    file = sys.argv[1]
    file_name = file[:-4]
    
    # Scale parameters for voxelization (unit in meters) 
    scale = 0.02
    scale_z = 0.1
    buffer_radii = 0.8
    
    # Search boundaries parameters for radii and height (unit in meters)
    min_r, max_r = 0.06, 0.12
    zmin, zmax = 1.5, 9

    # Reparametrization of parameters
    min_r, max_r = int(min_r/scale), int(max_r/scale)
    z_min, z_max = int(zmin/scale_z), int(zmax/scale_z)
    buffer_radii = int(buffer_radii/scale)

    start_time = time.time()

    # Load LiDar file and read X,Y,Z
    inFile = File(file, mode = "r")
    X, Y, Z = inFile.x, inFile.y, inFile.z
    
    # Filter Lidar points greater than 9 m
    index = np.where(Z<zmax)
    Z, Y, X = Z[index], Y[index], X[index]

    # Calculate Min Max for every dimension
    max_x, min_x = np.max(X), np.min(X)
    max_y, min_y = np.max(Y), np.min(Y)
    max_z, min_z = np.max(Z), np.min(Z)

    # Transform coordinates into a grid array index
    Xp = np.array((1/scale)*(X - min_x), dtype=np.int)
    Yp = np.array((1/scale)*(max_y -Y), dtype=np.int)
    Zp = np.array((1/scale_z)*(Z - min_z), dtype=np.int)

    # Group all Lidar points in same grid cell
    df = dd.from_array(np.stack((Zp, Yp, Xp), axis=1), chunksize=int(len(Xp)/10), columns=['z','y','x'])
    dz_values = df.groupby(['z', 'y', 'x']).size().reset_index().rename(columns={0:'count'}).values
    dz_values.compute()

    dz_values = np.asarray(dz_values).astype(int)

    max_values = np.max(dz_values, axis=0).astype(int)
    min_values = np.min(dz_values, axis=0).astype(int)

    size_x = int(max_values[2]-min_values[2]+1)
    size_y = int(max_values[1]-min_values[1]+1)
    size_z = int(max_values[0]-min_values[0]+1)

    print('Voxel array has dimensions :', size_z, size_y, size_x)

    # Create an empty Voxel Array and fill it with points
    array = np.full((size_z, size_y, size_x), 0)
    array = fill_matrix(dz_values, array)

    # Calculate the average cleanest height level
    points_cum = np.sum(array, axis=(1,2))
    threshold_cum = np.percentile(points_cum, 10)
    levels = np.sort(np.where(points_cum < threshold_cum)[0])
    min_l, max_l = int(np.mean(levels)-10), int(np.mean(levels)+10)

    print('Height Analysis level: ', np.mean(levels)*scale_z)
    
    # Generate masks and convolution process to score probability of a tree
    masks = mask_generator(min_r, max_r)
    score = scoring(array, masks, min_r, max_r, min_l, max_l) 
    sobel_score = sobel(score)
    threshold_score = np.percentile(sobel_score, 99.9)
    
    print('Threshold score is', threshold_score)

    # Detection of tree with min distance of 1 meters based on max values over threshold        
    markers = peak_local_max(sobel_score, indices=True, min_distance=int(1/scale), exclude_border=False, threshold_abs= threshold_score)

    # DataFrame with the trees
    index = 0
    df_table = pd.DataFrame( columns = ['index','lat','lon'])
    for marker in markers:

        row, column = marker[0],marker[1]
        lat, lon = (-scale*row + max_y - 0.5*scale, scale*column + min_x + 0.5*scale)
        df_table.loc[index] = [index, lat, lon]
        index = index + 1

    print('Found trees :', index)
    print("--- %s seconds ---" % (time.time() - start_time))

    df_table.to_csv(file_name + '.csv', sep=',',index=False)