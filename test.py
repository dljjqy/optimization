
import numpy as np
from pathlib import Path
from numpy.core.defchararray import index
from osgeo import gdal
from osgeo import osr

DataRoot = "/home/whujjq/workspace/data/001124_20121011/"
DemName = "001124_20121011_DEM.img"
ImgNames = ["ZY3_01a_hsnbavp_001124_20121011_111311_0008_SASMAC_CHN_sec_rel_001_1210128038.TIF",
            "ZY3_01a_hsnfavp_001124_20121011_111213_0008_SASMAC_CHN_sec_rel_001_1210128113.TIF",
            "ZY3_01a_hsnnavp_001124_20121011_111242_0007_SASMAC_CHN_sec_rel_001_1210127949.TIF"]
test_file = './dem1024 11.img'
# First thing : transform dem file from project coord to geo coord.
def Getgeo2lonlatTrans(data):
    '''
        input: data. gdal.Open(path).
        return osr.CoordinateTransformation().
        Transformation of Projection coordinate to longitude(经度), latitude(纬度). 
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(data.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    return ct

def transform(x,y, ct):
    coords = ct.TransformPoint(x, y)
    return coords[:2]

def SoupRPC(rpc_dic):
    rpc_dic['HEIGHT_OFF'] = float(rpc_dic['HEIGHT_OFF'].split()[0])   
    rpc_dic['HEIGHT_SCALE'] = float(rpc_dic['HEIGHT_SCALE'].split()[0])   
    rpc_dic['LAT_OFF'] = float(rpc_dic['LAT_OFF'].split()[0])   
    rpc_dic['LAT_SCALE'] = float(rpc_dic['LAT_SCALE'].split()[0])   
    rpc_dic['LINE_DEN_COEFF'] = list(map(float, (rpc_dic['LINE_DEN_COEFF'].split())))   
    rpc_dic['LINE_NUM_COEFF'] = list(map(float, (rpc_dic['LINE_NUM_COEFF'].split())))   
    rpc_dic['LINE_OFF'] = float(rpc_dic['LINE_OFF'].split()[0])   
    rpc_dic['LINE_SCALE'] = float(rpc_dic['LINE_SCALE'].split()[0])   
    rpc_dic['LONG_OFF'] = float(rpc_dic['LONG_OFF'].split()[0])   
    rpc_dic['LONG_SCALE'] = float(rpc_dic['LONG_SCALE'].split()[0])   
    rpc_dic['SAMP_DEN_COEFF'] = list(map(float, (rpc_dic['SAMP_DEN_COEFF'].split())))   
    rpc_dic['SAMP_NUM_COEFF'] = list(map(float, (rpc_dic['SAMP_NUM_COEFF'].split())))   
    rpc_dic['SAMP_OFF'] = float(rpc_dic['SAMP_OFF'].split()[0])   
    rpc_dic['SAMP_SCALE'] = float(rpc_dic['SAMP_SCALE'].split()[0])   
    return rpc_dic

def clip_dem_img(dem, out_path, name, offset_x, offset_y, block_xsize, block_ysize):
    '''
    Clipped dem file and output the create a new clipped dem file named 'name' in 'out_path'
    offset_x --- offset_x+block_xsize, offset_y --- offset_y+block_ysize, the block you want to clip.
    '''
    band = dem.GetRasterBand(1)
    out_band = band.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)

    driver = gdal.GetDriverByName("HFA")
    out_dem = driver.Create(out_path+name, block_xsize, block_ysize, 1, band.DataType)

    trans = dem.GetGeoTransform()
    top_left_x = trans[0] + offset_x * trans[1]
    top_left_y = trans[3] + offset_y * trans[5]

    dst_trans = (top_left_x, trans[1], trans[2], top_left_y, trans[4], trans[5])
    out_dem.SetGeoTransform(dst_trans)
    out_dem.SetProjection(dem.GetProjection())

    out_dem.GetRasterBand(1).WriteArray(out_band)
    out_dem.FlushCache()
    return True

def divide_dem(dem, out_path='./', size=1024):
    '''
    divide the dem into small pices with same size. block size is 'size' x 'size'.
    '''
    start_x, start_y = 0, 0
    nx, ny = 0, 0
    endx, endy = start_x+size, start_y+size
    xsize, ysize = dem.RasterXSize, dem.RasterYSize
    while endx<=xsize:
        while endy<=ysize:
            name = 'dem'+ str(size)+ ' '+ str(nx)+ str(ny)+ '.img'
            clip_dem_img(dem, out_path, name,start_x, start_y, size, size)
            ny+=1; start_y=endy; endy+=size
        nx+=1; start_x=endx; endx+=size
        ny=0;start_y=0; endy=start_y+size
    return True

def check_img(path, novalue=-99999.0):
    '''
    Check every *.img file in the path and if one has novalue then delete it.
    '''
    path = Path(path)
    for img in sorted(path.glob('*.img')):
        temp = gdal.Open(str(img))
        arr = temp.GetRasterBand(1).ReadAsArray()
        if novalue in arr:
            img.unlink()

def clean(path, suffix = 'img'):
    '''
    Clean all the file has 'suffix'.
    '''
    path = Path(path)
    for img in sorted(path.glob('*.'+suffix)):
        img.unlink()
    

def img2arr(img):
    # path = Path(path)
    # for img in sorted(path.glob('*.img')):
    dem = gdal.Open(img)
    arr = dem.GetRasterBand(1).ReadAsArray()
    (x0, dx, _, y0, _, dy) = dem.GetGeoTransform()
    xsize, ysize = dem.RasterXSize, dem.RasterYSize
    arr = arr.reshape(xsize, ysize,1)
    x = np.arange(xsize)
    y = np.arange(ysize)
    x = x0 + x*dx 
    y = y0 + y*dy
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(xsize, ysize, 1)
    yy = yy.reshape(xsize, ysize, 1)
    mat = np.concatenate((xx, yy, arr),axis=2 )
    return mat

def imgs2arrs(path):
    '''
    Transform all imgs in path to arrs and save to path as the same name.
    '''
    path = Path(path)
    for img in sorted(path.glob('*.img')):
        mat = img2arr(str(img))
        name = img.name[:-len(img.suffix):]
        np.save(name, mat)
    return True

def PointCoordTrans(dem, tif):
    
    pass


if __name__ == "__main__":
    if imgs2arrs('./'):
        pass
    # x, y = (426635, 4479265)
    # x1, y1 = (426645, 4479255)
    
    # dem = gdal.Open(DataRoot + DemName)
    # img1 = gdal.Open(DataRoot + ImgNames[0])
    # print(transform(x, y, Getgeo2lonlatTrans(dem)))
    # print(transform(x1, y1, Getgeo2lonlatTrans(dem)))
    # dic = SoupRPC(img1.GetMetadata('RPC'))
    # for k in dic.keys():
        # print(k, " : ", dic[k])
    
    # # clip_dem(dem, "./", "clip.tif", 1000, 1000, 100, 100)
    # clean('./')
    # divide_dem(dem)
    # check_img('./')
    # dem = gdal.Open('dem1024 11.img')
    # arr = img2arr(test_file)
    # print(arr[0,0,2])

