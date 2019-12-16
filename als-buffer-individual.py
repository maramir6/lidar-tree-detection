from laspy.file import File
from laspy.header import Header
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation
from scipy.ndimage import label
from skimage.measure import label, regionprops_table
from skimage.feature import peak_local_max
import os, sys, copy, time
from scipy import ndimage, stats
from osgeo import gdal, gdalnumeric, ogr, osr
import matplotlib.image as mpimg
import dask.dataframe as dd
from dask import delayed, compute
from scipy.spatial import distance
from PIL import Image, ImageDraw
import json, argparse
import warnings

warnings.filterwarnings("ignore")

def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)

def kantor_encoder(y, x):
    return 0.5*(y+x)*(y+x+1)+y

def mask_image(file_path, minX, maxX, minY, maxY, scale):

    # Create an OGR layer from a boundary shapefile
    shapef = ogr.Open(file_path)
    lyr = shapef.GetLayer()
    featureCount = lyr.GetFeatureCount()

    # Convert the layer extent to image pixel coordinates
    geotrans = [0] * 6
    geotrans[0], geotrans[3] = minX, maxY
    geotrans[1], geotrans[5] = scale, -scale
    
    ulX, ulY = world2Pixel(geotrans, minX, maxY)
    lrX, lrY = world2Pixel(geotrans, maxX, minY)

    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
        
    mask = np.zeros((pxHeight, pxWidth), dtype=bool)
    
    # Iterate through features
    for feature in range(featureCount):
        
        poly = lyr.GetNextFeature()
        geom = poly.GetGeometryRef()

        for g in range(geom.GetGeometryCount()):

            pts = geom.GetGeometryRef(g)
        
            if pts.GetGeometryName() == 'POLYGON':
                pts = pts.GetGeometryRef(0)
        
            ring_dict = json.loads(pts.ExportToJson())
            coords = np.array(ring_dict['coordinates'])
            
            pixel_x = abs(1/scale)*(coords[:,0] - minX).astype(int)
            pixel_y = abs(1/scale)*(maxY - coords[:,1]).astype(int)
            pixels = list(zip(pixel_x.tolist(),pixel_y.tolist()))
                
            rasterPoly = Image.new("L", (pxWidth, pxHeight), 1)
            rasterize = ImageDraw.Draw(rasterPoly)
            rasterize.polygon(pixels, 0)
            aux = imageToArray(rasterPoly)
            aux = ~np.array(aux, dtype=bool)
            mask = mask | aux
    
    return mask.astype(int), geotrans

def imageToArray(i):
    a=gdalnumeric.fromstring(i.tobytes(),'b')
    a.shape=i.im.size[1], i.im.size[0]
    return a

def create_folder(out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

def files_list(folder_path, shapefile):
    if shapefile:
        gdf = gpd.read_file(shapefile)
        files = gdf['ID_TILE_LI'].values.tolist()
        files = np.unique(np.array(files)).tolist()
        files = [file + '_normalizado_segmentado.laz' for file in files]
    else:
        files = [f for f in os.listdir(folder_path) if f.endswith('.laz')]

    return files

def save_raster(name, geotransform, image):
    driver = gdal.GetDriverByName('GTiff')
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(32718)
    raster = driver.Create(name,density.shape[1], density.shape[0], 1, gdal.GDT_UInt16)
    raster.SetGeoTransform(geotransform)  
    raster.GetRasterBand(1).WriteArray(image)
    raster.SetProjection(projection.ExportToWkt())
    raster.FlushCache()
    
def world2Pixel(geoMatrix, x, y):
    ulX, ulY = geoMatrix[0], geoMatrix[3]
    xDist, yDist = geoMatrix[1], geoMatrix[5]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)

def height(data, array): 
    for i in range(data.shape[0]):
        array[:, int(data[i][0]), int(data[i][1])] = data[i][2:] 
        
    return array

def check_path(path):
    if path[-1] != '/':
        path + '/'

    return path

def tree_points(X, Y, Z, i, index):    
    indices = np.where(i == index)

    return (x[indices], y[indices], z[indices])

def save_file(file_name, header, x, y, z):

    outfile = File(file_name, mode="w", header=header)    
    outfile.x, outfile.y, outfile.z = x, y, z
    outfile.close()

def run(lazfile, scale, dist_min, perc_factor, h_min, folder, clipp):
    
    # Scale parameters for voxelization(unit in meters)
    kernel_buffer = 1

    # Load LiDar file and read X,Y,Z
    inFile = File(lazfile, mode = "r")
    header = inFile.header

    start_time = time.time()

    X, Y, Z = inFile.x, inFile.y, inFile.z

    # Calculate Min Max for every dimension
    max_x, min_x = np.max(X), np.min(X)
    max_y, min_y = np.max(Y), np.min(Y)
    max_z, min_z = np.max(Z), np.min(Z)
    
    # Transform coordinates into a grid array index
    Xp = np.array((1/scale)*(X - min_x), dtype=np.int)
    Yp = np.array((1/scale)*(max_y -Y), dtype=np.int)
    
    # Group all Lidar points in same grid cell
    df = dd.from_array(np.stack((Yp, Xp, Z), axis=1), chunksize=int(len(Xp)/10), columns=['y','x','z'])
    dz_values = df.groupby(['y','x']).agg(['min', 'max', 'count']).reset_index().values
    dz_values.compute()
    
    # Create a max height, min height and point density grid raster
    dz_values = np.asarray(dz_values).astype(float)
    
    n_rows, n_columns = int(np.max(Yp) + 1), int(np.max(Xp) + 1)
    z = np.full((int(3), n_rows, n_columns), 0).astype(float)
    z = height(dz_values, z)

    z_max = z[1,:,:]
    z_den = z[2,:,:]
    binary_mask = z_max > h_min 
    
    # Convolution with a kernel buffer to sum all the points within a buffer
    # Makes the score more robust
    kernel = np.ones((int(kernel_buffer/scale), int(kernel_buffer/scale)))
    score_den = ndimage.convolve(z_den, kernel, mode='constant', cval=1.0)

    # Normalization of both max height and density scores
    score_den = (score_den-np.min(score_den))/(np.max(score_den)-np.min(score_den))
    score_z = (z_max-np.min(z_max))/(np.max(z_max)-np.min(z_max))

    # Final score weighted with both max height and density scores
    score_array = 0.4*score_z + 0.6*score_den
    
    # Detection of trees with min distance of 1.5 meters based on the score, previously filtered by Otsu segmentation        
    score_array = np.multiply(binary_mask, score_array)
    threshold_score = np.percentile(score_array, perc_factor)

    markers = peak_local_max(score_array, indices=False, min_distance=int(dist_min/scale), exclude_border=True, threshold_abs= threshold_score, labels=binary_mask)
    markers = dilation(markers, disk(int(kernel_buffer/scale)))
    labels = label(markers)
    
    # DataFrame with the trees
    df_table = pd.DataFrame()
    props = regionprops_table(labels, properties=['label', 'centroid'])
    rows, columns = props['centroid-0'], props['centroid-1']
    lats, lons = -scale*rows + max_y - 0.5*scale, scale*columns + min_x + 0.5*scale

    print('In File: ', file,' were found trees :', len(lats), " in --- %s seconds ---" % (time.time() - start_time))
    
    # Cloud points of every single tree
    if clipp:
        folder = out_path + lazfile[:-4] + '/'
        create_folder(folder)
        y_i, x_i = np.repeat(np.arange(n_rows), n_columns), np.tile(np.arange(n_columns), n_rows)
        key_kantor, values = kantor_encoder(y_i, x_i).astype(int), np.ravel(labels).astype(int)
        dictionary = dict(zip(key_kantor.tolist(), values.tolist()))
        data_kantor = kantor_encoder(Yp, Xp).astype(int)
        i = vec_translate(data_kantor, dictionary)

        clouds = []
        
        for index in range(1, np.max(labels)):
            cloud = delayed(tree_points)(X, Y, Z, i, index)
            clouds.append(cloud)

        clouds = compute(*clouds)

        # Save cloud points of every individual tree
        tree_index = 0
        
        for cloud in clouds:
            x, y, z = cloud[0], cloud[1], cloud[2]
            save_file(folder + str(tree_index) + '.laz', header, x, y, z)
            tree_index = tree_index + 1

        print('Trees were succesfully clipped in', "--- %s seconds ---" % (time.time() - start_time))
    
    # Compute other metrics
    index = np.arange(len(lats))
    h_v = np.ravel(z_max)
    h_h = np.ravel(z_max, order='F')
    
    index_v = rows*n_columns + columns
    index_h = columns*n_rows + rows
    
    h1 = np.take(h_v, index_v)
    h2 = np.take(h_v, index_v-1)
    h3 = np.take(h_h, index_h-1)
    h4 = np.take(h_v, index_v+1)
    h5 = np.take(h_h, index_h+1)
    
    h_max = np.stack((h1, h2, h3, h4, h5), axis=1)
    h_max = np.max(h_max, axis=1)

    dap = 0.0412*np.power(h_max,2)-0.2231*h_max + 4.6466
    v16 = 0.3*np.pi*np.power((0.5*(dap/100)),2)*h_max
    v25 = (25/16)*v16

    # Add the metrics to DataFrame

    df_table['id'] = index.tolist()
    df_table['Y'] = lats.tolist()  
    df_table['X'] = lons.tolist() 
    df_table['h_max'] = h_max.tolist()
    df_table['dap'] = dap.tolist()  
    df_table['v16'] = v16.tolist() 
    df_table['v25'] = v25.tolist()
    df_table['v25p'] = (1.2*v25).tolist()
    
    return df_table


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Airborne LiDAR Sensor Forestry Processing Algorithm')
    parser.add_argument('input_path', type=str, help="Input dir for laz files")
    parser.add_argument('output_path', type=str, help="Output dir to save the results")
    parser.add_argument('--shapefile', type=str, help="Input shapefile path")
    parser.add_argument('--clip', type=str, help="Clipping trees, answer yes")
    parser.add_argument('--dist_min', type=float, default = 1.25, help="Minimum distance between individual trees")
    parser.add_argument('--perc_factor', type=float, default = 40, help="Percentile tolerance")
    parser.add_argument('--h_min', type=float, default = 5, help="Minimum height for a tree")
    parser.add_argument('--scale', type=float, default = 0.2, help="Voxelization scale")
    args = parser.parse_args()
    
    # Density plane parameters
    rodal_length = 22.36
    factor = 20
    
    # File paths
    folder_path = args.input_path
    shapefile = args.shapefile
    out_path = args.output_path

    folder_path, out_path = check_path(folder_path), check_path(out_path)
    create_folder(out_path)
    files = files_list(folder_path, shapefile)    
    
    frames = []
    
    for file in files:        
        df = run(folder_path + file, args.scale, args.dist_min, args.perc_factor, args.h_min, out_path, args.clip)
        frames.append(df)

    # Join all the results into a single Dataframe
    result = pd.concat(frames)
    X, Y = result['X'].values, result['Y'].values

    if shapefile:

        # Max and Min individual tree X-Y position
        maxX, maxY = np.max(X), np.max(Y)
        minX, minY = np.min(X), np.min(Y)

        # Open selected areas shapefile and create a mask
        mask, geotrans = mask_image(shapefile, minX, maxX, minY, maxY, args.scale)

        # Transform lat/lon to pixel indices positions
        pixel_x = abs(1/args.scale)*(X - minX).astype(int)
        pixel_y = abs(1/args.scale)*(maxY - Y).astype(int)

        # True or false value of the tree if falls within selected polygon
        n_rows, n_columns = mask.shape
        mask_index = np.ravel(mask)
        index_v = pixel_y*n_columns + pixel_x
        index_v = index_v.astype(int)

        inside = np.take(mask_index, index_v)
        index = np.where(inside == 1)
        X, Y = X[index], Y[index]

        # Add true/false to DataFrame and filter only True values to save them 
        result['inside'] = inside.tolist()
        result = result[result['inside'] == 1]
        result = result.drop(['inside'], axis=1)
    
    result.to_csv(folder_path + 'tree_csv.csv', sep=',',index=False)

    # Save individual tree  Point shapefile 
    gdf = gpd.GeoDataFrame(result, geometry=gpd.points_from_xy(result.X, result.Y))
    gdf.to_file(folder_path + 'trees.shp')

    # Filter trees
    max_x, min_x = np.max(X), np.min(X)
    max_y, min_y = np.max(Y), np.min(Y)

    Xp = np.array((1/rodal_length)*(X - min_x), dtype=np.int)
    Yp = np.array((1/rodal_length)*(max_y - Y), dtype=np.int)
    Cp = np.ones(len(Xp))

    # Group all Lidar points in same grid cell
    df = dd.from_array(np.stack((Yp, Xp, Cp), axis=1), chunksize=int(len(Xp)/10), columns=['y','x','c'])
    dz_values = df.groupby(['y','x']).agg(['min', 'max', 'count']).reset_index().values
    dz_values.compute()

    # Create a max height, min height and point density grid raster
    dz_values = np.asarray(dz_values).astype(float)
    z = np.full((int(3), int(np.max(Yp) + 1), int(np.max(Xp) + 1)), 0).astype(float)
    z = height(dz_values, z)

    density = factor*z[2,:,:]
    geotransform = [min_x, rodal_length, 0, max_y, 0, -rodal_length]
    save_raster(folder_path + 'density.tif', geotransform, density)

