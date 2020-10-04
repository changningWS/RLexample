"""
Policy-Iteration解决FrozenLake问题
"""
import numpy as np
import gym

def run_episode(env, policy, gamma = 1.0, render = False):
    """ 根据策略进行一次episode并返回该episode的总奖励 """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs])) #根据policy生成轨迹
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    """ 策略评估 """
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)] #用n条轨迹的奖励平均值来评估策略
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ 策略提升 使用贪心法得到使行为价值函数q最大的策略 """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.env.P[s][a]]) #贝尔曼方程
        policy[s] = np.argmax(q_sa) #贪心法
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ 计算策略的状态价值 """
    v = np.zeros(env.env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])#贝尔曼方程
        if (np.sum((np.fabs(prev_v - v))) <= eps): #如果状态价值已收敛，则退出
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ 策略迭代 """

    policy = np.random.choice(env.env.nA, size=(env.env.nS)) #初始化生成随机策略
    max_iterations = 10000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma) #策略评估，返回当前策略下的状态价值
        new_policy = extract_policy(old_policy_v, gamma) #策略提升
        if (np.all(policy == new_policy)): #判断策略迭代是否收敛
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

if __name__ == '__main__':

    env = gym.make('FrozenLake-v0')
    policy = policy_iteration(env, gamma = 1.0)
    scores = evaluate_policy(env, policy, gamma = 1.0)
    print('Average scores = ', np.mean(scores))