import pandas as pd
import folium
from shapely.geometry import Polygon
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from DMEIRL_GeoInd import GeolifeEnv
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Geolife数据
env = GeolifeEnv(data_dir="Geolife_Trajectories_1.3/Data", max_length=1000)
original_traj = env.get_traj()

# 将列表中的每个元组转换为shapely.Point对象
points = [Point(x[1], x[0]) for x in original_traj]

# 创建一个空的GeoDataFrame
geo_df = gpd.GeoDataFrame()

# 将shapely.Point对象添加到GeoDataFrame中
geo_df['geometry'] = gpd.GeoSeries(points)

# 打印GeoDataFrame
# print(geo_df)
# 创建Folium地图对象
m = folium.Map(location=[39.97818, 116.30934], zoom_start=12)

# 将GeoPandas数据添加到地图上
geo_df = geo_df.set_crs(epsg=4326)
geo_df_json = geo_df.to_json()
folium.GeoJson(geo_df_json).add_to(m)
# 显示地图
m.save('beijing.html')

#
# 在地图上绘制网格
# 定义地点名称并获取对应的街区边界
place_name = "北京市, 中国"
place_boundary = ox.geocode_to_gdf(place_name)

import time
time_start = time.time()
# 获取该地区的街道网络
print("正在获取该地区的街道网络....")
graph = ox.graph_from_place(place_name, network_type='drive')
ox.save_graphml(graph, 'beijing_G.osm')

# # 手动读取街道网络
# graph = ox.load_graphml('beijing_G.osm')

time_end = time.time()
print("获取街道网络的时间为: {}".format(time_end-time_start))
# 输出网络的基本信息
print(ox.basic_stats(graph))
plt.show()

