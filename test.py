
import numpy as np
import json
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


def transform(x, y, ct):
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def SoupRPC(rpc_dic):
    rpc_dic['HEIGHT_OFF'] = float(rpc_dic['HEIGHT_OFF'].split()[0])
    rpc_dic['HEIGHT_SCALE'] = float(rpc_dic['HEIGHT_SCALE'].split()[0])
    rpc_dic['LAT_OFF'] = float(rpc_dic['LAT_OFF'].split()[0])
    rpc_dic['LAT_SCALE'] = float(rpc_dic['LAT_SCALE'].split()[0])
    rpc_dic['LINE_DEN_COEFF'] = list(
        map(float, (rpc_dic['LINE_DEN_COEFF'].split())))
    rpc_dic['LINE_NUM_COEFF'] = list(
        map(float, (rpc_dic['LINE_NUM_COEFF'].split())))
    rpc_dic['LINE_OFF'] = float(rpc_dic['LINE_OFF'].split()[0])
    rpc_dic['LINE_SCALE'] = float(rpc_dic['LINE_SCALE'].split()[0])
    rpc_dic['LONG_OFF'] = float(rpc_dic['LONG_OFF'].split()[0])
    rpc_dic['LONG_SCALE'] = float(rpc_dic['LONG_SCALE'].split()[0])
    rpc_dic['SAMP_DEN_COEFF'] = list(
        map(float, (rpc_dic['SAMP_DEN_COEFF'].split())))
    rpc_dic['SAMP_NUM_COEFF'] = list(
        map(float, (rpc_dic['SAMP_NUM_COEFF'].split())))
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
    out_dem = driver.Create(out_path+name, block_xsize,
                            block_ysize, 1, band.DataType)

    trans = dem.GetGeoTransform()
    top_left_x = trans[0] + offset_x * trans[1]
    top_left_y = trans[3] + offset_y * trans[5]

    dst_trans = (top_left_x, trans[1], trans[2],
                 top_left_y, trans[4], trans[5])
    out_dem.SetGeoTransform(dst_trans)
    out_dem.SetProjection(dem.GetProjection())

    out_dem.GetRasterBand(1).WriteArray(out_band)
    out_dem.FlushCache()
    del out_dem
    return True


def divide_dem(dem, out_path='./', size=1024):
    '''
    divide the dem into small pices with same size. block size is 'size' x 'size'.
    '''
    start_x, start_y = 0, 0
    nx, ny = 0, 0
    endx, endy = start_x+size, start_y+size
    xsize, ysize = dem.RasterXSize, dem.RasterYSize
    while endx <= xsize:
        while endy <= ysize:
            name = 'dem' + str(size) + ' ' + str(nx) + str(ny) + '.img'
            clip_dem_img(dem, out_path, name, start_x, start_y, size, size)
            ny += 1
            start_y = endy
            endy += size
        nx += 1
        start_x = endx
        endx += size
        ny = 0
        start_y = 0
        endy = start_y+size
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
            del temp
            img.unlink()
    return True


def clean(path, suffix='img'):
    '''
    Clean all the file has 'suffix'.
    '''
    path = Path(path)
    for img in sorted(path.glob('*.'+suffix)):
        img.unlink()


def img2arr(img):
    dem = gdal.Open(img)
    arr = dem.GetRasterBand(1).ReadAsArray()
    (x0, dx, _, y0, _, dy) = dem.GetGeoTransform()
    xsize, ysize = dem.RasterXSize, dem.RasterYSize
    arr = arr.reshape(xsize, ysize, 1)
    x = np.arange(xsize)
    y = np.arange(ysize)
    x = x0 + x*dx
    y = y0 + y*dy
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(xsize, ysize, 1)
    yy = yy.reshape(xsize, ysize, 1)
    mat = np.concatenate((xx, yy, arr), axis=2)
    del dem
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

def getSRSPair(dataset):
    '''
    获得给定数据的投影坐标系和大地坐标系
    params: 
        dataset: GDAL地理数据
    return:
        投影坐标系, 大地坐标系
    '''
    pSRS = osr.SpatialReference()
    pSRS.ImportFromWkt(dataset.GetProjection())
    gSRS = pSRS.CloneGeogCS()
    return pSRS, gSRS


def Point_Geo2LonLat(point, pSRS, gSRS):
    '''
    NoUse.
    '''
    ct = osr.CoordinateTransformation(pSRS, gSRS)
    print(ct)
    return ct.TransformPoint(point)

def dem_Geo2LonLat(dem_path, save_mat=True):
    '''
    Transform entir dem file coordinate and save the Lonitude,latitude coord as .npy
    Same name as dem file.
    return: arr like. shape: dem_Xsize * dem_Ysize * 3
    3 channels: Lonitude, Latitude, Height.
    '''
    dem = gdal.Open(dem_path)
    arr = img2arr(dem_path)

    # Get coordinate transform
    psrs, gsrs = getSRSPair(dem)
    ct = osr.CoordinateTransformation(psrs, gsrs)

    # Transform dem file
    (nx, ny, nc) = arr.shape
    arr = arr.reshape((nx*ny, nc))
    result = np.array(ct.TransformPoints(arr), dtype=np.float64)
    result = result.reshape((nx, ny, nc))
    
    # Save result as npy file
    if save_mat:
        dem_path = Path(dem_path)
        result_name = dem_path.name[:-4] + '.npy'
        np.save(result_name, result)
    return result

def readRPC_txt(txt):
    keys = []
    labels = []
    with open(txt, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.replace('', '.')
            temp = line.split(':')
            keys.append(temp[0])
            labels.append(float(temp[1].split()[0]))
            
    return dict(zip(keys, labels))

if __name__ == "__main__":
    # crop_path = 'E:\\optimization\\crop\\'
    # test_dem_path = crop_path + 'dem1024 11.img'
    # (Unfinished!)Test readRPC_txt:
    txt = 'E:\\optimization\\data\\001124_20121011\\ZY3_01a_hsnbavp_001124_20121011_111311_0008_SASMAC_CHN_sec_rel_001_1210128038_rpc.TXT'
    
    # (Finished)Test function dem_Geo2LonLat
    # arr = np.load('./dem1024 11.npy')
    # print(arr.shape)

    # arr = dem_Geo2LonLat(test_dem_path)
    # print(arr.shape)
    # print(arr[0,0,:])

    # mat = dem_Geo2LonLat(test_dem_path, './test.npy')
    # with open('./data.json', 'r') as load_f:
    #     data_info = json.load(load_f)
    # dem_path = data_info['root'] + data_info['dem']
    # tif_path = data_info['root'] + list(data_info['images'].keys())[0]
    # dem = gdal.Open(dem_path)
    # tif = gdal.Open(tif_path)

    # (Finished!)Divide dem and Select clipped dem.
    # divide_dem(dem, crop_path)
    # check_img(crop_path)

    # # (Finished)Test Function Point_Geo2LonLat
    # dem = gdal.Open(test_dem_path)
    # # Get a point.
    # arr = img2arr(test_dem_path)
    # points = [arr[0,0,:], arr[0,1,:], arr[1,0,:]]
    # # Get proj coord and geo coord
    # dem_psrs = osr.SpatialReference()
    # dem_psrs.ImportFromWkt(dem.GetProjection())
    # dem_gsrs = dem_psrs.CloneGeogCS()
    # ct = osr.CoordinateTransformation(dem_psrs, dem_gsrs)
    # # print(arr[0,0,:])
    # # print(ct.TransformPoint(arr[0,0,:2]))
    # # print(ct.TransformPoint(*arr[0,0,:2]))
    # print(ct.TransformPoints(points))
    

    # del dem
    # del tif