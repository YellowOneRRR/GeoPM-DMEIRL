import folium
from DMEIRL_GeoInd import GeolifeEnv
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
from tqdm import tqdm
from folium import FeatureGroup, GeoJson

env = GeolifeEnv(data_dir="Geolife_Trajectories_1.3/Data", max_length=1000)
original_traj = env.get_traj()

original_traj = pd.DataFrame(original_traj)
print("这条轨迹的长度为：{}".format(len(original_traj)))
# 北京市边界的坐标范围（南西和北东两个点）
sw = [original_traj[0].min(), original_traj[1].min()]
ne = [original_traj[0].max(), original_traj[1].max()]

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

# Define a style function to draw lines instead of markers for points
# geo_df = geo_df.set_crs(epsg=4326)
# geo_df_json = geo_df.to_json()
# folium.GeoJson(geo_df_json).add_to(m)
folium.PolyLine(locations=original_traj, color='#010fcc', weight=5,
                opacity=1,
                line_cap='butt',
                ).add_to(m)

original_traj = np.array(original_traj)

diffs = np.abs(np.diff(original_traj, axis=0))

laplace = np.random.laplace(loc=0, scale=2, size=(len(diffs), 2))

noisy_diffs = diffs * laplace

for i in range(1, len(original_traj)):
    original_traj[i] += noisy_diffs[i-1]

place_name = "北京市, 中国"
graph = ox.graph_from_place(place_name, network_type="drive")
original_traj_np = np.array(original_traj)

nearest_nodes = []

for i in range(len(original_traj)):
    node, dis = ox.nearest_nodes(graph, original_traj_np[i][1], original_traj_np[i][0], True)
    print("当前位置与最近节点的距离为：{}".format(dis))
    nearest_nodes.append(node)

print(nearest_nodes)

node_positions = [(graph.nodes[node_id]['y'], graph.nodes[node_id]['x']) for node_id in nearest_nodes]
print(node_positions)

# folium.GeoJson(noise_geo_df_json).add_to(m)
folium.PolyLine(locations=node_positions, color='red', weight=5,
                opacity=0.5,
                line_cap='round',
                ).add_to(m)

m.save('folium_layer.html')
