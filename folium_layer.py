import folium
from DMEIRL_GeoInd import GeolifeEnv
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
from tqdm import tqdm
from folium import FeatureGroup, GeoJson


# 读取单挑轨迹的最大最小经纬度，实现地理位置的感知
env = GeolifeEnv(data_dir="Geolife_Trajectories_1.3/Data", max_length=1000)
original_traj = env.get_traj()

# # 将列表中的每个元组转换为shapely.Point对象
# points = [Point(x[1], x[0]) for x in original_traj]
#
# # 创建一个空的GeoDataFrame
# geo_df = gpd.GeoDataFrame()
#
# # 将shapely.Point对象添加到GeoDataFrame中
# geo_df['geometry'] = gpd.GeoSeries(points)
#
original_traj = pd.DataFrame(original_traj)
print("这条轨迹的长度为：{}".format(len(original_traj)))
# 北京市边界的坐标范围（南西和北东两个点）
sw = [original_traj[0].min(), original_traj[1].min()]
ne = [original_traj[0].max(), original_traj[1].max()]

# 创建地图
m = folium.Map(location=[original_traj[0].mean(), original_traj[1].mean()], zoom_start=11)

# # 填充网格的坐标列表
# for lat in range(int(sw[0] * 1000), int(ne[0] * 1000)):
#     for lon in range(int(sw[1] * 1000), int(ne[1] * 1000)):
#         sw_point = (lat / 1000, lon / 1000)
#         ne_point = ((lat + 1) / 1000, (lon + 1) / 1000)
#
#         # 检查当前网格是否在填充坐标列表中
#         if (sw_point, ne_point) in geo_df['geometry']:
#             fill_color = '#f7879a'  # 填充颜色
#         else:
#             fill_color = '#00000000'  # 透明颜色
#
#         folium.Rectangle(bounds=[sw_point, ne_point],
#                          color='#fe2c54',
#                          fill_color=fill_color,
#                          fill_opacity=1.0).add_to(m)


# 将GeoPandas数据添加到地图上
# Define a style function to draw lines instead of markers for points
# geo_df = geo_df.set_crs(epsg=4326)
# geo_df_json = geo_df.to_json()
# folium.GeoJson(geo_df_json).add_to(m)
folium.PolyLine(locations=original_traj, color='#010fcc', weight=5,
                opacity=1,
                line_cap='butt',
                ).add_to(m)

# 转换为 NumPy 数组
original_traj = np.array(original_traj)
# 计算数组中每个元素与其前一个元素之间的差异
diffs = np.abs(np.diff(original_traj, axis=0))
# 生成拉普拉斯噪声
laplace = np.random.laplace(loc=0, scale=2, size=(len(diffs), 2))
# 将噪声添加到差异数组中
noisy_diffs = diffs * laplace
# 将噪声应用于原始数组中的所有元素（除第一个元素）
for i in range(1, len(original_traj)):
    original_traj[i] += noisy_diffs[i-1]


# 现在需要将OSMnx库引入，计算路网临近节点

#定义路网范围
print("加载路网数据。。。")
place_name = "北京市, 中国"
graph = ox.graph_from_place(place_name, network_type="drive")
print("路网数据加载完毕！")
original_traj_np = np.array(original_traj)

# 遍历每个轨迹点，找到其在路网上的最近邻节点
nearest_nodes = []

for i in range(len(original_traj)):
    node, dis = ox.nearest_nodes(graph, original_traj_np[i][1], original_traj_np[i][0], True)
    print("当前位置与最近节点的距离为：{}".format(dis))
    nearest_nodes.append(node)

print(nearest_nodes)

# 提取节点的位置信息
node_positions = [(graph.nodes[node_id]['y'], graph.nodes[node_id]['x']) for node_id in nearest_nodes]
print(node_positions)

# 两种轨迹点呈现形式
# folium.GeoJson(noise_geo_df_json).add_to(m)
folium.PolyLine(locations=node_positions, color='red', weight=5,
                opacity=0.5,
                line_cap='round',
                ).add_to(m)

# 显示地图
m.save('folium_layer.html')
