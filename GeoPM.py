import numpy as np

# 返回“扰动量”ε
def get_epsilon(sensitivity, delta):
    return sensitivity / np.log(1.0 / delta)

# 实现差分隐私的Piecewise机制
def piecewise_mechanism(data, epsilon, sensitivity):
    beta = sensitivity / epsilon  # 计算β值
    noise = np.random.laplace(0, beta, len(data))  # 生成满足拉普拉斯噪声分布的加性噪声
    return np.array(data) + noise  # 返回扰动后的数据

if __name__ == "__main__":
# 示例使用
# 定义数据
    data = [50, 60, 70, 80, 90, 100]
    sensitivity = 10.0  # 数据集的敏感性
    delta = 1e-5  # δ的值，通常越小越好
    epsilon = get_epsilon(sensitivity, delta)  # 计算ε的值

    # 调用差分隐私的Piecewise机制，对数据进行扰动
    noisy_data = piecewise_mechanism(data, epsilon, sensitivity)
    print("原数据：", data)
    print("扰动后的数据：", noisy_data)