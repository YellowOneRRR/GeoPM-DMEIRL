import numpy as np
def get_epsilon(sensitivity, delta):
    return sensitivity / np.log(1.0 / delta)

def piecewise_mechanism(data, epsilon, sensitivity):
    beta = sensitivity / epsilon
    noise = np.random.laplace(0, beta, len(data))
    return np.array(data) + noise

if __name__ == "__main__":
    data = [50, 60, 70, 80, 90, 100]
    sensitivity = 10.0
    delta = 1e-5
    epsilon = get_epsilon(sensitivity, delta)
    noisy_data = piecewise_mechanism(data, epsilon, sensitivity)
