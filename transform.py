import pandas as pd
import numpy as np
df = pd.read_csv('data/Geolife_train_latlon.csv')
states = [(row['lat'], row['lon']) for _, row in df.iterrows()]
actions = []
for i in range(len(states) - 1):
    lat1, lon1 = states[i]
    lat2, lon2 = states[i+1]
    dx, dy = lat2 - lat1, lon2 - lon1
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    actions.append((distance, angle))

trajectories = list(zip(states[:-1], actions))
def feature_extractor(state, action):
    lat, lon = state
    distance, angle = action
    return [lat, lon, distance, angle]

features = [feature_extractor(s, a) for s, a in trajectories]

# with open('expert_demonstration.txt', 'w') as f:
#     for feat in features:
#         f.write(','.join(str(x) for x in feat) + '\n')