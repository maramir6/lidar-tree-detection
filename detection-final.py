from laspy.file import File
from laspy.header import Header
from numba import jit
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute
from scipy import ndimage, stats
from scipy.signal import medfilt
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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import medfilt, find_peaks
from scipy.stats import weibull_min
import statsmodels.distributions
from sklearn.cluster import DBSCAN
from skimage.measure import CircleModel, ransac

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@jit(nopython=True)
def fill_matrix(data, array): 
	for i in range(data.shape[0]):
		array[data[i][0]][data[i][1]][data[i][2]] = 1

	return array

def distance(lat, lon, buffer_size, x, y, z):
    
    dist = np.sqrt(np.power(x-lon, 2)+np.power(y-lat,2))
    index = np.where(dist < buffer_size)
    
    return x[index], y[index], z[index]

def tree_points(x, y, z, lat, lon, buffer_size):
	
	dist = np.sqrt(np.power(x-lon, 2)+np.power(y-lat,2))
	indices = np.where(dist < buffer_size)
	lat, lon = np.mean(y[indices]), np.mean(x[indices])
	dist = np.sqrt(np.power(x-lon, 2)+np.power(y-lat,2))

	indices = np.where(dist < 0.5)

	return (x[indices], y[indices], z[indices])

def std_radii(x, y, z, h_min, h_max):
    
    height = np.arange(h_min,h_max, 0.1)
    std = []
    
    for h in range(0,height.shape[0]):
        
        h_inf, h_sup = height[h]-0.05, height[h]+0.05
        indices = np.intersect1d(np.where(z>h_inf)[0], np.where(z<h_sup)[0])        
        Xl, Yl = x[indices], y[indices]
        
        if len(Xl)>10:
            radiis = np.sqrt(np.power(Xl-np.mean(Xl),2)+np.power(Yl-np.mean(Yl),2))
            max_radii = np.percentile(radiis, 90)
            radiis = radiis[radiis < max_radii]
            std.append(np.std(radiis))
        else:
            std.append(0)
    
    return np.array(std)

def prun_std(r, height):

    cum_inv = -np.cumsum(r/sum(r))/np.arange(1,len(r)+1)
    mult = np.linspace(1.0, 1.25, num=len(r))
    cum_inv = (cum_inv - np.min(cum_inv))/(np.max(cum_inv)-np.min(cum_inv))
    cum_inv = cum_inv * mult
    peaks, _ = find_peaks(cum_inv, height=0.85*np.max(cum_inv))
    
    try:
        peak = peaks[-1]
        
    except:
        peak = 0
    
    return np.around(height[peak],1)

def prun_den(z):
    
    z = z[(z > 1.6) & (z < 9)]
    shape, loc, scale = weibull_min.fit(z, floc=0)
    ecdf = statsmodels.distributions.ECDF(z)
    x = np.linspace(z.min(), z.max(), 100)
    diff = weibull_min(shape, loc, scale).cdf(x) - ecdf(x)
    
    indices = np.where(x > 2)
    x = x[indices]
    diff = diff[indices]

    return np.around(x[np.argmax(diff)],1)

def save_file(file_name, header, x, y, z):

    outfile = File(file_name, mode="w", header=header)    
    outfile.x, outfile.y, outfile.z = x, y, z
    outfile.close()

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

def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)

def kantor_encoder(y, x):
    return 0.5*(y+x)*(y+x+1)+y

def kantor_doble_encoder(x,y,z):
    k = 0.5*(y+x)*(y+x+1)+y
    return 0.5*(k+z)*(k+z+1)+k

def buffer(x, y, z, min_dist):

    start_time = time.time()
    x_c, y_c = np.mean(x), np.mean(y)
    dist = np.sqrt(np.power(x-x_c, 2)+np.power(y-y_c,2))
    index = np.where(dist < min_dist)

    print("Buffer process took: --- %s seconds ---" % (time.time() - start_time))
    
    return x[index], y[index], z[index]

def denoising(x, y, z, scale, min_c):
    start_time = time.time()

    xp = np.array((1/scale)*(x - np.min(x)), dtype=np.int)
    yp = np.array((1/scale)*(np.max(y) - y), dtype=np.int)
    zp = np.array((1/scale)*(z - np.min(z)), dtype=np.int)

    df = dd.from_array(np.stack((zp, yp, xp), axis=1), chunksize=int(len(xp)/10), columns=['z','y','x'])
    dz_values = df.groupby(['z', 'y', 'x']).size().reset_index().rename(columns={0:'count'}).values
    dz_values.compute()
    dz_values = np.asarray(dz_values).astype(int)

    z_, y_, x_, c_ = dz_values[:,0], dz_values[:,1], dz_values[:,2], dz_values[:,3]

    key = kantor_doble_encoder(x_, y_, z_)
    dictionary = dict(zip(key.tolist(), c_.tolist()))

    key_p = kantor_doble_encoder(xp, yp, zp)
    cp = vec_translate(key_p, dictionary)

    indices = np.where(cp>min_c)

    print("Denoising process took: --- %s seconds ---" % (time.time() - start_time))

    return x[indices], y[indices], z[indices]

def normalization(x,y,z):

    start_time = time.time()
    scale = 0.5
    x = np.array((1/scale)*(x - np.min(x)), dtype=np.int)
    y = np.array((1/scale)*(np.max(y) - y), dtype=np.int)

    df = dd.from_array(np.stack((y, x, z), axis=1), chunksize=int(len(x)/10), columns=['y','x','z'])
    dz_values = df.groupby(['y','x']).agg(['min']).reset_index().values
    dz_values.compute()
    dz_values = np.asarray(dz_values).astype(int)

    key_kantor, values = kantor_encoder(dz_values[:,0], dz_values[:,1]), dz_values[:,2].astype(int)
    dictionary = dict(zip(key_kantor.tolist(), values.tolist()))
    data_kantor = kantor_encoder(y,x)

    z = z - vec_translate(data_kantor, dictionary)

    print("Normalization process took: --- %s seconds ---" % (time.time() - start_time))
    
    return z

def scoring(array, masks, min_r, max_r, min_h, max_h):

    array_tensor = torch.from_numpy(np.expand_dims(array[min_h:max_h,:,:], axis=0)).double()
    conv_layer = nn.Conv2d(array_tensor.shape[1], masks.shape[0], max_r + 1, padding=1)
    masks_batch = np.repeat(masks[:,np.newaxis,:,:], array_tensor.shape[1], axis=1)
    conv_layer.weight.data = torch.from_numpy(masks_batch).double()
    conv_layer.bias.data = torch.from_numpy(np.zeros(masks.shape[0])).double()

    score = conv_layer(array_tensor)
    score = score.detach().numpy()

    return np.squeeze(np.max(score, axis=1).astype(int), axis=0)

def dap_tree(x, y, z):

    indice = np.where((z<1.32)&(z>=1.28))
    x, y = x[indice],y[indice]
    x, y = 100*(x-np.min(x)), 100*(np.max(y)-y)
    
    xt, yt = x, y    
    clustering = DBSCAN(eps=3, min_samples=25).fit(np.stack((y, x), axis=1))
    labels = clustering.labels_
    
    if len(np.bincount(labels[labels >=0]))>0:
        index_cluster = np.where(labels==np.argmax(np.bincount(labels[labels >=0])))
        x, y = x[index_cluster], y[index_cluster]
    
    ransac_model, inliers = ransac(np.stack((y, x), axis=1), CircleModel, 20, 3, max_trials=100)
    x, y = x[inliers], y[inliers]
    lat, lon, radii = ransac_model.params

    dist = np.sqrt(np.power(xt-lon, 2)+np.power(yt-lat,2))
    indice = np.where(dist < radii+1)

    x, y = xt[indice], yt[indice]
    model = CircleModel()
    model.estimate(np.stack((y, x), axis=1)) 
    lat, lon, radii = model.params

    return (x, y, lat, lon, radii)

def save_dap(x, y, lat, lon, radii, file_name):

    min_lon, max_lon = int(lon) - 2*int(radii), int(lon) + 2*int(radii)
    min_lat, max_lat = int(lat) - 2*int(radii), int(lat) + 2*int(radii)
   
    circle = plt.Circle((lon, lat), radii, color='blue', fill=False)
    plt.scatter(x, y, s=4, c='red')
    plt.scatter(lon, lat, s=5, c='blue')
    plt.gcf().gca().add_artist(circle)
    plt.axis([min_lon, max_lon, min_lat, max_lat])
    plt.savefig(file_name)
    plt.clf()


def run(file, input_folder, output_folder):

    start_time = time.time()
    file_name = file[:-4]
    folder = file_name + '/'

    # Create a folder
    if not os.path.exists(output_folder + folder):
        os.makedirs(output_folder + folder)
    
    # Scale parameters for voxelization (unit in meters) 
    scale = 0.02
    scale_z = 0.1
    buffer_size = 1
    min_dist = 15
    
    # Search boundaries parameters for radii and height (unit in meters)
    min_r, max_r = 0.06, 0.12
    zmin, zmax = 1.5, 9

    # Denoise parameters (unit in meters)
    d_scale = 0.25
    d_min = 5

    # Reparametrization of parameters
    min_r, max_r = int(min_r/scale), int(max_r/scale)
    z_min, z_max = int(zmin/scale_z), int(zmax/scale_z)

    # Load LiDar file and read X,Y,Z
    inFile = File(input_folder + file, mode = "r")

    header = inFile.header
    X, Y, Z = inFile.x, inFile.y, inFile.z

    # LiDAR normalization and buffer clipping
    # Filter LiDAR points greater than 9 m
    # Z = normalization(X, Y, Z)
    X, Y, Z = buffer(X, Y, Z, min_dist)

    # Calculate Min Max for every dimension
    max_x, min_x = np.max(X), np.min(X)
    max_y, min_y = np.max(Y), np.min(Y)
    max_z, min_z = np.max(Z), np.min(Z)

    # Transform coordinates into a grid array index
    Xp = np.array((1/scale)*(X - min_x), dtype=np.int)
    Yp = np.array((1/scale)*(max_y -Y), dtype=np.int)
    Zp = np.array((1/scale_z)*(Z - min_z), dtype=np.int)

    # Group all LiDAR points in same grid cell
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
    array = array[:z_max,:,:]

    # Calculate the average cleanest height level
    points_cum = np.sum(array, axis=(1,2))
    threshold_cum = np.percentile(points_cum, 10)
    levels = np.sort(np.where(points_cum < threshold_cum)[0])
    min_l, max_l = int(np.mean(levels)-10), int(np.mean(levels)+10)
    # Check if min_l is negative, force it to zero
    if min_l < 0:
    	min_l = 0
    	
    print('Height Analysis level: ', np.mean(levels)*scale_z)
    
    # Generate masks and convolution process to score probability of a tree
    masks = mask_generator(min_r, max_r)
    score = scoring(array, masks, min_r, max_r, min_l, max_l) 
    sobel_score = sobel(score)
    threshold_score = np.percentile(sobel_score, 99.9)
    
    print('Threshold score is', threshold_score)

    # Detection of tree with min distance of 1 meters based on max values over threshold        
    markers = peak_local_max(sobel_score, indices=True, min_distance=int(1/scale), exclude_border=False, threshold_abs= threshold_score)

    # Plot images 
    score = gray2rgb(np.array(255*(sobel_score/np.max(sobel_score)),dtype=np.uint8))
    image = gray2rgb(np.array(255*(array[int(np.mean(levels)),:,:]/np.max(array[int(np.mean(levels)),:,:])),dtype=np.uint8))

    # Loop to generate a list with individual LiDAR points for every tree 
    indices = []
    
    for marker in markers:

        row, column = marker[0],marker[1]
        row_min, row_max = int(row) - 2, int(row) + 2
        col_min, col_max = int(column) - 2, int(column) + 2

        score[row_min:row_max,col_min:col_max] = (220, 20, 20)
        image[row_min:row_max,col_min:col_max] = (220, 20, 20)

        lat, lon = (-scale*row + max_y - 0.5*scale, scale*column + min_x + 0.5*scale)
        i = delayed(tree_points)(X, Y, Z, lat, lon, buffer_size)
        indices.append(i)

    indices = compute(*indices)

    dap_values = []


    for indice in indices:
    	x, y, z = indice[0], indice[1], indice[2]    	
    	dap = delayed(dap_tree)(x,y,z)
    	dap_values.append(dap)

    dap_values = compute(*dap_values)


    index  = 0
    dap_validate = []

    for dap in dap_values:

    	x, y, lat, lon, radii = dap
    	dist = np.sqrt(np.power(x-lon, 2)+np.power(y-lat,2))

    	if (radii >= 3.5) & (radii <= 16) & (len(dist) >= 25) & (len(dist[dist<0.8*radii])/len(dist) <= 0.25):
    		dap_validate.append(2*radii)

    	else:
    		dap_validate.append(np.nan)

    	save_dap(x, y, lat, lon, radii, output_folder + folder + str(index) + '_dap.png')
    	index = index + 1

    # Loop to generate a std_radii/height profile and also save individual LiDAR file for every tree
    # DataFrame with the trees

    index = 0
    results = []
    h_min, h_max = 1.5, 9
    df_table = pd.DataFrame( columns = ['index','lat','lon', 'dap'])

    for indice in indices:

        x, y, z = indice[0], indice[1], indice[2]
        df_table.loc[index] = [index, np.mean(y), np.mean(x), dap_validate[index]]
        save_file(output_folder + folder + str(index) + '.laz', header, x, y, z)
        r = delayed(std_radii)(x, y, z, h_min, h_max)
        results.append(r)
        index = index + 1

    results = compute(*results)

    # Compute std deviation and otsu to separate classes
    # Add True/False list to the DataFrame	
    index = 0
    std_threshold = 0.015
    prun_height = []
    delta_inf = 2.5
    delta_sup = 0.5

    height = np.arange(h_min,h_max,0.1)
    n_points = []
    
    for indice in indices:

    	x, y, z = indice[0], indice[1], indice[2]

    	r = results[index]

    	level_std = prun_std(r, height)
    	level = prun_den(z)

    	if(level_std == 1.5):
    		level_std = level

    	if(abs(level-level_std)>0.9):
        	level = level_std

    	save_file(output_folder + folder + str(index) + '.laz', header, x, y, z)    		    
    	plt.plot(height, r, 'r')
    	plt.savefig(output_folder + folder + str(index) + '_std.png')
    	plt.clf()
    
    	index_inf = int((level - delta_inf - h_min)/0.1)
    
    	if index_inf < 0:
        	index_inf = 0
        
    	index_sup = int(( level - delta_sup - h_min)/0.1)

    	lft_side = r[index_inf:index_sup]           

    	if((level < 2.8) or (np.std(lft_side) > std_threshold)):
        	level = 0

    	prun_height.append(level)
    	index = index + 1
    	

    df_table['level'] = prun_height

    # Save output files
    df_table.to_csv(output_folder + folder + file_name + '.csv', sep=',',index=False)
    mpimg.imsave(output_folder + folder + file_name + '_score.png', score)
    mpimg.imsave(output_folder + folder + file_name + '.png', image)
    
    # Print final time process
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    path = 'D:/MRA/'
    
    output_folder = 'D:/MRA/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = [f for f in os.listdir(path) if f.endswith('.laz')]
    
    for file in files:
    	folder = file[:-4]
    	if not os.path.exists(output_folder + folder):
    		os.makedirs(output_folder + folder)
    	
    	run(file, path, output_folder)

