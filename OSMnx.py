import pandas as pd
import folium
from shapely.geometry import Polygon
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from DMEIRL_GeoInd import GeolifeEnv
import seaborn as sns
import matplotlib.pyplot as plt

env = GeolifeEnv(data_dir="Geolife_Trajectories_1.3/Data", max_length=1000)
original_traj = env.get_traj()

points = [Point(x[1], x[0]) for x in original_traj]

geo_df = gpd.GeoDataFrame()


geo_df['geometry'] = gpd.GeoSeries(points)

m = folium.Map(location=[39.97818, 116.30934], zoom_start=12)

geo_df = geo_df.set_crs(epsg=4326)
geo_df_json = geo_df.to_json()
folium.GeoJson(geo_df_json).add_to(m)

m.save('beijing.html')

place_name = "北京市, 中国"
place_boundary = ox.geocode_to_gdf(place_name)

graph = ox.graph_from_place(place_name, network_type='drive')
ox.save_graphml(graph, 'beijing_G.osm')

# graph = ox.load_graphml('beijing_G.osm')

print(ox.basic_stats(graph))
plt.show()

