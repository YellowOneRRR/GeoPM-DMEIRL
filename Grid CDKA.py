import os
import random
import itertools
import pandas as pd
import numpy as np


# 计算语义多样性
def semantic_diversity():
    return 1

class GeolifeEnv:
    def __init__(self, data_dir, max_length, grid_size, k):
        self.data_dir = data_dir
        self.max_length = max_length
        self.grid_size = grid_size
        self.k = k
        self.user_paths = os.listdir(data_dir)  # 设置 self.user_paths 变量

    def get_traj(self):
        chosen_paths = random.sample(self.user_paths, 50)  # 随机选择50个用户
        traj_list = []
        for chosen_path in chosen_paths:
            if chosen_path == ".DS_Store":
                continue
            traj_files = os.listdir(os.path.join(self.data_dir, chosen_path, "Trajectory"))[:10]  # 选择该用户的前10条轨迹
            for traj_file in traj_files:
                traj_file = os.path.join(self.data_dir, chosen_path, "Trajectory", traj_file)
                traj_user = pd.read_csv(traj_file, sep=",", skiprows=6, header=None)
                traj_user.columns = ["latitude", "longitude", "zero", "altitude", "date1", "date2", "time"]
                traj_user.drop(["zero", "altitude", "date1", "date2"], axis=1, inplace=True)
                traj_user = traj_user.dropna(axis=0, how="any")
                traj_user = traj_user[:self.max_length]
                traj_list.append(traj_user)

        traj = pd.concat(traj_list, ignore_index=True)
        return traj

    def grid(self, traj):
        # 将位置点映射到网格中
        min_lat, max_lat = traj["latitude"].min(), traj["latitude"].max()
        min_lon, max_lon = traj["longitude"].min(), traj["longitude"].max()
        lat_grid_size = self.grid_size / (111.32 * 1000)  # 每一度纬度的距离约为111.32km，将网格距离转换为纬度差
        lon_grid_size = self.grid_size / (
                    111.32 * 1000 * np.cos(np.radians(min(max_lat, max_lat))))  # 计算每一度经度的距离并取最小值进行近似
        lat_bins = np.arange(min_lat, max_lat + lat_grid_size, lat_grid_size)
        lon_bins = np.arange(min_lon, max_lon + lon_grid_size, lon_grid_size)
        traj["lat_grid"] = pd.cut(traj["latitude"], lat_bins, labels=False)
        traj["lon_grid"] = pd.cut(traj["longitude"], lon_bins, labels=False)
        return traj

    def ungrid(self, grid_data, lat_bins, lon_bins):
        # 计算网格大小
        lat_grid_size = (lat_bins[1] - lat_bins[0]) / 111320.0
        lon_grid_size = (lon_bins[1] - lon_bins[0]) / (111320.0 * np.cos(np.radians(np.mean(lat_bins))))

        # 计算网格中心点的纬度和经度
        lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

        # 将网格划分的位置坐标转换为经纬度坐标
        lat_lon_data = []
        for lat_grid, lon_grid in grid_data:
            lat_center = lat_centers[int(lat_grid)]
            lon_center = lon_centers[int(lon_grid)]
            lat = lat_center + lat_grid_size / 2
            lon = lon_center + lon_grid_size / 2
            lat_lon_data.append((lat, lon))

        return lat_lon_data

    def k1(self, traj):
        # 对网格划分经纬度进行分组，并计算每个分组中的元素数量
        loc_counts = traj.groupby(["lat_grid", "lon_grid"]).size().reset_index()
        loc_counts.columns = ["lat_grid", "lon_grid", "count"]

        # 计算每个位置的出现概率
        loc_probs = loc_counts.assign(prob=loc_counts["count"] / len(traj)).drop(columns=["count"])
        # print(loc_probs)
        query_loc = loc_probs.sample(n=1, weights="prob", replace=False, random_state=42)
        print("查询的位置点的历史查询概率:\n{}\n".format(query_loc))
        closest_rows = loc_probs.loc[loc_probs["prob"].isin(query_loc["prob"].nsmallest(self.k-1).values)]
        # 如果closest_rows中没有n条，则选择与"prob"值相近的行
        while len(closest_rows) < self.k-1:
            # 计算closest_rows中prob列的平均值
            mean_prob = closest_rows["prob"].mean()

            # 在loc_probs中选择prob列最接近平均值的行
            new_row = loc_probs.loc[(loc_probs["prob"] - mean_prob).abs().nsmallest(1).index[0]]

            # 如果new_row不在closest_rows中，则将其添加到closest_rows中
            if not new_row.isin(closest_rows).all().all():
                closest_rows = closest_rows.append(new_row, ignore_index=True)

            # 如果new_row已经在closest_rows中，则在loc_probs中删除该行，并重新选择一行
            else:
                loc_probs = loc_probs.drop(new_row.index)

        # 将closest_rows中的结果存储为一个列表
        anonymity_set = closest_rows[["lat_grid", "lon_grid"]].apply(tuple, axis=1).tolist()
        anonymity_set.append((query_loc.iloc[0]["lat_grid"], query_loc.iloc[0]["lon_grid"]))
        anonymity_set = anonymity_set[0: self.k]
        print("k1:\n{}".format(anonymity_set))
        print(len(anonymity_set))
        return anonymity_set

    def k2(self, traj):
        # 对网格划分经纬度进行分组，并计算每个分组中的元素数量
        loc_counts = traj.groupby(["lat_grid", "lon_grid"]).size().reset_index()
        loc_counts.columns = ["lat_grid", "lon_grid", "count"]

        # 计算每个位置的历史查询概率
        loc_probs = loc_counts.assign(prob=loc_counts["count"] / len(traj)).drop(columns=["count"])
        # print(loc_probs)
        query_loc = loc_probs.sample(n=1, weights="prob", replace=False, random_state=42)

        # 定义随机数m
        m = random.randint(0, self.k)
        closest_rows = loc_probs.loc[loc_probs["prob"].isin(query_loc["prob"].nsmallest(2 * (2 * self.k - m)).values)]
        # 如果closest_rows中没有n条，则选择与"prob"值相近的行
        while len(closest_rows) < 2 * (2 * self.k - m):
            # 计算closest_rows中prob列的平均值
            mean_prob = closest_rows["prob"].mean()

            # 在loc_probs中选择prob列最接近平均值的行
            new_row = loc_probs.loc[(loc_probs["prob"] - mean_prob).abs().nsmallest(1).index[0]]

            # 如果new_row不在closest_rows中，则将其添加到closest_rows中
            if not new_row.isin(closest_rows).all().all():
                closest_rows = closest_rows.append(new_row, ignore_index=True)

            # 如果new_row已经在closest_rows中，则在loc_probs中删除该行，并重新选择一行
            else:
                loc_probs = loc_probs.drop(new_row.index)

        # 将closest_rows中的结果存储为一个列表
        anonymity_set2 = closest_rows[["lat_grid", "lon_grid"]].apply(tuple, axis=1).tolist()
        anonymity_set2 = anonymity_set2[0: 2 * (2 * self.k - m)]
        print("m为:{},\nk2:\n{},\nlen:{}".format(m, anonymity_set2, len(anonymity_set2)))

        selection_size = self.k + m
        import math
        combinations = math.comb(2 * (2 * self.k - m), selection_size)
        #print(combinations)

        # 找30个长度为k+m的组合，随机
        semantic_max = [] # 用以存储语义最大化的位置集
        for i in range(30):
            selected_points = random.sample(anonymity_set2, self.k + m)  # 从points中选择k+m个点
            # 逆网格划分出经纬度
            min_lat, max_lat = traj["latitude"].min(), traj["latitude"].max()
            min_lon, max_lon = traj["longitude"].min(), traj["longitude"].max()
            lat_grid_size = self.grid_size / (111.32 * 1000)  # 每一度纬度的距离约为111.32km，将网格距离转换为纬度差
            lon_grid_size = self.grid_size / (111.32 * 1000 * np.cos(np.radians(min(max_lat, max_lat))))  # 计算每一度经度的距离并取最小值进行近似
            lat_bins = np.arange(min_lat, max_lat + lat_grid_size, lat_grid_size)
            lon_bins = np.arange(min_lon, max_lon + lon_grid_size, lon_grid_size)
            selected_points = env.ungrid(selected_points, lat_bins, lon_bins)
            print("k2匿名化产生的第{}条轨迹：\n{}\n长度为:{}".format(i+1, selected_points, len(selected_points)))

            # 计算该集合的位置语义，高德地图API

        return anonymity_set2


# Create the GeolifeEnv instance
env = GeolifeEnv(data_dir="Geolife_Trajectories_1.3/Data", max_length=1000, grid_size=10, k=10)

# 定义随机数m
m = random.randint(0, 11) #k=10


# Load the trajectory data and convert to grid coordinates
traj = env.get_traj()
traj = env.grid(traj)
# 产生查询位置
query_loc = traj.sample()[['lat_grid', 'lon_grid']].values

# Call the k1 method with the query location
anonymity_set_k1 = env.k1(traj)
# print("K1 匿名集：", anonymity_set_k1)

# Call the k2 method with the same query location
anonymous_sets = env.k2(traj)

# # Print the anonymous sets
# for i, set in enumerate(anonymous_sets):
#     print("Anonymous Set", i+1)
#     for location in set:
#         print(location)
#     print()
