"""
Value-Iteration解决FrozenLake问题

"""
import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register

def run_episode(env, policy, gamma = 1.0, render = False):
    """ 根据策略进行一次episode并返回该episode的总奖励 """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ 策略评估 """
    scores = [
            run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ 根据状态价值函数提取得到新策略 """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """ 值迭代 """
    v = np.zeros(env.env.nS)
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations): #没有policy参与，不断最大化Q函数
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)] 
            v[s] = max(q_sa) #选取状态s下对应最大后续状态价值的动作，使收敛更快
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v


if __name__ == '__main__':

    env_name  = 'FrozenLake-v0'
    env = gym.make(env_name)
    gamma = 1.0
    optimal_v = value_iteration(env, gamma);
    policy = extract_policy(optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)
