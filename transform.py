import pandas as pd
import numpy as np

# 加载 Geolife 数据集
df = pd.read_csv('data/Geolife_train_latlon.csv')

# 将 GPS 坐标转换为状态
states = [(row['lat'], row['lon']) for _, row in df.iterrows()]

# 计算动作（距离和角度）
actions = []
for i in range(len(states) - 1):
    lat1, lon1 = states[i]
    lat2, lon2 = states[i+1]
    dx, dy = lat2 - lat1, lon2 - lon1
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    actions.append((distance, angle))

# 将状态和动作存储为元组的列表
trajectories = list(zip(states[:-1], actions))

# 将状态和动作转换为特征向量
def feature_extractor(state, action):
    lat, lon = state
    distance, angle = action
    return [lat, lon, distance, angle]

features = [feature_extractor(s, a) for s, a in trajectories]

#print(features)

# # 保存专家演示轨迹到文件
# with open('expert_demonstration.txt', 'w') as f:
#     for feat in features:
#         f.write(','.join(str(x) for x in feat) + '\n')