# -*- encoding: utf-8 -*-
from osgeo import gdal
import numpy as np
from osgeo import osr
import os
import cv2
import sys
# from gdalconst import * 

def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]

def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + row * trans[1] + col * trans[2]
    py = trans[3] + row * trans[4] + col * trans[5]
    return px, py

def getSRSPair(dataset):
	'''
	获得给定数据的投影参考系和地理参考系
	:param dataset: GDAL地理数据
	:return: 投影参考系和地理参考系
	'''
	prosrs = osr.SpatialReference()
	prosrs.ImportFromWkt(dataset.GetProjection())
	geosrs = prosrs.CloneGeogCS()
	return prosrs, geosrs

def lonlat2geo(dataset, lon, lat):
	'''
	将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
	:param dataset: GDAL地理数据
	:param lon: 地理坐标lon经度
	:param lat: 地理坐标lat纬度
	:return: 经纬度坐标(lon, lat)对应的投影坐标
	'''
	prosrs, geosrs = getSRSPair(dataset)
	ct = osr.CoordinateTransformation(geosrs, prosrs)
	coords = ct.TransformPoint(lon, lat)
	return coords[:2]

def geo2imagexy(dataset, x, y):
	'''
	根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
	:param dataset: GDAL地理数据
	:param x: 投影或地理坐标x
	:param y: 投影或地理坐标y
	:return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
	'''
	trans = dataset.GetGeoTransform()
	a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
	b = np.array([x - trans[0], y - trans[3]])
	return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def transForm():
	dataset = gdal.Open("test.tif")
	out = open('fcn/1/locations2.txt', 'w')
	f = open('fcn/1/1.csv')
	l=[69,68,1,36,61,24,77,23,43]
	while True:
		line = f.readline()
		if not line:
			break 
		if line[0] == ';':
			continue
		lis = line.strip().split(',')
		xx = float(lis[4])
		yy = float(lis[5])
		label = int(lis[-1])
		if label not in l:
			continue
		coords = lonlat2geo(dataset, yy, xx)
		coords = geo2imagexy(dataset, coords[0], coords[1])

		if coords[0] > 0 and coords[1] > 0:
			out.write('{},{},{}\n'.format(int(coords[0]), int(coords[1]), label))

def infoLonLat():
	dataset=gdal.Open("/home/cln/hitsz/毕业设计/CDL_2014_clip_20170828212859_603366421/CDL_2014_clip_20170828212859_603366421.tif")
	tifData=gdal.Open("/home/cln/hitsz/毕业设计/9.23/256.tif")
	height=dataset.RasterYSize
	width=dataset.RasterXSize
	data=dataset.ReadAsArray(0,0,width,height)
	start=int(sys.argv[1])
	end=int(sys.argv[2])
	index=sys.argv[3]
	write=open("9.23/info"+str(index)+".txt","a+")
	for row in range(start,end):
		print row
		for col in range(width):
			px,py=imagexy2geo(dataset, col, row)
			coords=geo2lonlat(dataset, px, py)
			geos=lonlat2geo(tifData, coords[0], coords[1])
			y,x=geo2imagexy(tifData, geos[0], geos[1])
			if x<tifData.RasterYSize and y<tifData.RasterXSize:
				# print row,col,"==>",x,y
				write.write(str(int(x))+","+str(int(y))+","+str(data[row][col])+'\n')
	write.close()

###5(0).Soybeans  141(1).Deciduous Forest  3(2).Rice   190(3).Woody Wetlands   176(4):Grass/Pasture  121(5):Developed/Open Space  1(6):Corn  61(7):Fallow/Idle Cropland
###2(8):Cotton  26(9)Dbl Crop WinWht/Soybeans  143(10).Mixed Forest  111(11):Open Water  122(12).Developed/Low Intensity  4(13).Sorghum  others(14)
def CreateGeoTiff():
	d1={}
	d2={}
	with open("9.23/info.txt","r") as f:
		for line in f:
			content=line.strip().split(",")
			rows=content[0]
			cols=content[1]
			lab=int(content[2])
			k=str(rows)+","+str(cols)
			if lab!=0:
				d1[k]=lab
	print("done")
	d2[5]=0
	d2[141]=1
	d2[3]=2
	d2[190]=3
	d2[176]=4
	d2[121]=5
	d2[1]=6
	d2[61]=7
	d2[2]=8
	d2[26]=9
	d2[143]=10
	d2[111]=11
	d2[122]=12
	d2[4]=13
	SourceDS = gdal.Open("9.23/256.tif")
	GeoT = SourceDS.GetGeoTransform()
	Projection = osr.SpatialReference()
	Projection.ImportFromWkt(SourceDS.GetProjectionRef())
	nXSize = SourceDS.RasterXSize
	nYSize = SourceDS.RasterYSize
	stride=40
	picWidth=224
	picHeight=224
	width=int(np.floor((nXSize-picWidth)/stride)+1)
	height=int(np.floor((nYSize-picHeight)/stride)+1)
	start=sys.argv[1]
	end=sys.argv[2]
	for i in range(int(start),int(end)):
		print i
		for j in range(width):
			Array=SourceDS.ReadAsArray(j*stride,i*stride,picHeight,picWidth)
			DataType = gdal.GDT_Float32
			labels=np.zeros([picWidth,picHeight])
			for row in range(i*stride,i*stride+picWidth):
				for col in range(j*stride,j*stride+picHeight):
					k=str(row)+","+str(col)
					if d1.get(k) is not None:
						if d2.get(d1.get(k)) is not None:
							labels[row-i*stride][col-j*stride]=int(d2[d1.get(k)])
						else:
							labels[row-i*stride][col-j*stride]=int(14)
					else:
						labels[row-i*stride][col-j*stride]=int(14)
			if np.min(labels)!=np.max(labels):
				NewFileName = str(i)+"_"+str(j)+"_923"
				driver = gdal.GetDriverByName('GTiff')
				DataSet = driver.Create( "chip9.23/data/"+NewFileName+".tif", picWidth, picHeight, 3, DataType )
				DataSet.SetGeoTransform(GeoT)
				DataSet.SetProjection( Projection.ExportToWkt() )
				DataSet.GetRasterBand(1).WriteArray( Array[0])
				DataSet.GetRasterBand(2).WriteArray( Array[1])
				DataSet.GetRasterBand(3).WriteArray( Array[2])

				txtName="chip9.23/labels/"+NewFileName+".txt"
				w=open(txtName,"a+")
				for m in range(picWidth):
					for n in range(picHeight):
						w.write(str(int(labels[m][n]))+" ")
					w.write("\n")
				w.close()




def test():
	d={}
	f=open("fcn/1/1.csv","r")
	while True:
		line = f.readline()
		if not line:
			break 
		if line[0] == ';':
			continue
		lis = line.strip().split(',')
		xx = float(lis[4])
		yy = float(lis[5])
		label=int(lis[6])
		if d.get(label) is None:
			d[label]=1
		else:
			d[label]=d[label]+1
	d=sorted(d.iteritems(),key=lambda x:x[1],reverse=True)
	for k,val in d:
		print k,val


def processData():
	a=np.array([0.,0.,0.])
	num=1369
	for picName in os.listdir("chip/data"):
		dataset=gdal.Open("chip/data/"+picName).ReadAsArray(0,0,224,224)
		a[0]+=np.mean(dataset[0])
		a[1]+=np.mean(dataset[1])
		a[2]+=np.mean(dataset[2])
	print a/float(num)

def statistic():
	d={}
	with open("info.txt") as f:
		for line in f:
			label=int(line.strip().split(",")[2])
			if d.get(label) is None:
				d[label]=1
			else:
				d[label]+=1
	d=sorted(d.iteritems(),key=lambda x:x[1],reverse=True)
	for k,v in d:
		print k,v



if __name__ == '__main__':
	# infoLonLat()
	# statistic()
	CreateGeoTiff()
	# processData()
	# data=gdal.Open("test.tif").ReadAsArray(0,0,12,12)
	# print data
	# test()
	# transForm()




