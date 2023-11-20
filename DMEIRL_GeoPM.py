import numpy as np
from GeoPM import get_epsilon, piecewise_mechanism
import transform
import hashlib
import tensorflow as tf
import numpy as np
import gym
OUTPUT_GRAPH = False
MAX_EPISODE = 2000
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100
RENDER = False
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01

class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")

        l1 = tf.layers.dense(
            inputs=self.s,
            units=30,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
            name='l1'
        )

        mu = tf.layers.dense(
            inputs=l1,
            units=1,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l1,
            units=1,
            activation=tf.nn.softplus,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(1.),
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            self.exp_v = log_prob * self.td_error
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


class DeepMaxEntIRL:
    def __init__(self, env, policy, n_features, learning_rate, gamma, n_epochs):
        self.env = env
        self.policy = policy
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.weights = np.random.rand(self.n_features)

    def get_feature_expectations(self, trajectories):
        feature_expectations = np.zeros(self.n_features)
        for trajectory in trajectories:
            i = 0
            for state in trajectory:
                feature_expectations[i] += self.policy(state) * self.feature_func(state)
                i += 1
        feature_expectations /= len(trajectories)
        return feature_expectations

    def feature_func(self, state):
        feature = state * 0.4
        return feature

    def get_log_prob(self, state):
        return np.dot(self.weights, self.feature_func(state))

    def train(self, expert_trajectories):
        expert_feature_expectations = self.get_feature_expectations(expert_trajectories)
        from tqdm import tqdm
        epoch_bar = tqdm(total=self.n_epochs, desc='Epoch', position=0)
        for epoch in range(self.n_epochs):
            sample_state = self.env.reset()

            while True:
                sample_action = self.policy(sample_state)
                sample_next_state, _ , done, _ = self.env.step(sample_action)
                feature = self.feature_func(sample_state)
                feature_diff = self.gamma * self.feature_func(sample_next_state) - feature
                gradient = feature - np.exp(np.dot(self.weights, feature)) / np.exp(self.get_log_prob(sample_state)) * expert_feature_expectations
                self.weights += self.learning_rate * gradient

                if done:
                    break
                sample_state = sample_next_state
            epoch_bar.update(1)

    def generate_trajectory(self):
        trajectory = []
        state = self.env.reset()

        while True:
            trajectory.append(state)
            action = self.policy(state)
            next_state, _, done, _ = self.env.step(action)

            if done:
                break
            state = next_state

        return trajectory

from geopy.distance import geodesic
import pandas as pd
import numpy as np
from gym import spaces
import random
import os

class GeolifeEnv:
    def __init__(self, data_dir, max_length):
        self.data_dir = data_dir
        self.max_length = max_length
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float32)

    def get_traj(self):
        user_paths = os.listdir(self.data_dir)
        chosen_path = random.choice(user_paths)
        traj_file = os.path.join(self.data_dir, chosen_path, "Trajectory", os.listdir(os.path.join(self.data_dir, chosen_path, "Trajectory"))[0])
        traj = pd.read_csv(traj_file, sep=",", skiprows=6, header=None)
        traj.columns = ["latitude", "longitude", "zero", "altitude", "date1", "date2", "time"]
        traj.drop(["zero", "altitude", "time", "date1", "date2"], axis=1, inplace=True)
        traj = traj.dropna(axis=0, how="any")
        traj = traj[:self.max_length]
        coords = [(lat, lon) for lat, lon in zip(traj["latitude"], traj["longitude"])]
        return coords

    def reset(self):
        self.traj = self.get_traj()
        self.curr_index = 0
        curr_pos = self.traj[self.curr_index]
        next_pos = self.traj[self.curr_index + 1]
        self.curr_state = self.get_state(curr_pos, next_pos)
        return self.curr_state

    def step(self, action):
        reward = 0
        next_index = self.curr_index + 1

        if next_index >= len(self.traj) - 1:
            done = True
        else:
            done = False
            curr_pos = self.traj[self.curr_index]
            next_pos = self.traj[next_index]
            next_state = self.get_state(next_pos, self.traj[next_index + 1])
            self.curr_index = next_index
            self.curr_state = next_state

        return self.curr_state, reward, done, {}

    def get_state(self, curr_pos, next_pos):
        lat1, lon1 = curr_pos
        lat2, lon2 = next_pos
        dist = geodesic(curr_pos, next_pos).km
        speed = 3.6 * dist
        bearing = np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1))
        time_diff = 1
        return np.array([dist, speed, bearing, time_diff])

    def random_policy(self, state):
        return self.action_space.sample()

def add_laplace_noise(x, epsilon, sensitivity):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return x + noise


def geo_ind(df, grid):
    GRID_SIZE = grid
    df['lat_hash'] = df['lat'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())
    df['lon_hash'] = df['lon'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())
    df['lat'] = df['lat_hash'].apply(lambda x: int(x, 16) % int(1/GRID_SIZE))
    df['lon'] = df['lon_hash'].apply(lambda x: int(x, 16) % int(1/GRID_SIZE))
    return df[['lat', 'lon']]


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, state):
        return self.action_space.sample()


if __name__ == "__main__":
    eplsion = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]
    GRID = [0.01, 0.15, 0.20, 0.25]
    for grid in GRID:
        for delt in eplsion:
            # 开始循环！
            i = 0
            env = GeolifeEnv(data_dir="Geolife_Trajectories_1.3/Data", max_length=1000)
            original_traj = env.get_traj()
            state = env.reset()
            policy = RandomPolicy(env.action_space)
            generator = DeepMaxEntIRL(env=env,
                                         policy=policy,
                                         n_features=4,
                                         learning_rate=0.001,
                                         gamma=0.9,
                                         n_epochs=2000)
            expert_trajs = transform.features
            generator.train(expert_trajs)
            new_traj = generator.generate_trajectory()
            new_traj = pd.DataFrame(new_traj, columns=['lat', 'lon', 'trans', '1'])
            expert_trajs = pd.DataFrame(expert_trajs, columns=['lat', 'lon', 'trans', '1'])
            new_traj = geo_ind(new_traj, grid)
            delta = delt
            sensitivity = 1.0
            epsilon = get_epsilon(sensitivity, delta)
            new_traj['lat'] = piecewise_mechanism(new_traj['lat'], epsilon, sensitivity)
            new_traj['lon'] = piecewise_mechanism(new_traj['lon'], epsilon, sensitivity)

            new_traj['lat'] = new_traj['lat'].add(expert_trajs['lat'])
            new_traj['lon'] = new_traj['lon'].add(expert_trajs['lon'])