
'''
Shenggong Ji, Yu Zheng, Zhaoyuan Wang, Tianrui Li
A Deep Reinforcement Learning-Enabled Dynamic Redeployment System for Mobile Ambulances
IMWUT/UbiComp 2019
---------------------------------------------------------------------------------------
'''

from brain_policy_4 import PolicyGradient
from ambulance_env import gameEnv
import time
import os
import numpy as np
import tensorflow as tf

# experiment settings (for detail, see ambulance_env.py)
num_ambulances = 50  # number of ambulances set in the simulation
num_periods = 1  # parameter m in the paper
num_occupied = 1  # parameter k in the paper
num_actions = 34  # number of ambulance stations in a city
# number of total EMS requests, number of EMS reqeusts for training
num_requests_total, num_requests_training = 23549, 14011

env = gameEnv()
env.num_ambulances = num_ambulances
env.num_periods = num_periods
env.num_occupied = num_occupied

for _i in range(4):
    # int(input('\n whether to select factor ' + str(_i+1) + ': (0, 1) \n'))
    env.factors_selected[_i] = 1

n_features = env.return_total_factor_length()
print('num-actions:', num_actions, '\n')
print('num-features:', n_features, '\n')

folder_name = "policy_n4/num_ambulances=" + \
    str(num_ambulances) + "-Factors(" + str(env.factors_selected[0]) + \
    str(env.factors_selected[1]) + str(env.factors_selected[2]) + \
    str(env.factors_selected[3]) + ")-m=" + str(num_periods) + \
    "-k=" + str(num_occupied) + '-'

print('folder_name', folder_name, '\n')

session = tf.Session()


def train_model():

    folder_id = 1
    while os.path.exists(folder_name + str(folder_id)):
        folder_id += 1
    _folder_name = folder_name + str(folder_id)
    os.makedirs(_folder_name)
    print("\n" + _folder_name + "\n")

    _learning_rate = float(input('\nlearning_rate: \n'))
    RL = PolicyGradient(
        n_actions=num_actions,
        n_features=n_features,
        learning_rate=_learning_rate,
        reward_decay=0.99,
    )

    f = open(_folder_name + "/_readme_.csv", 'w')
    f.write("learning_rate," + str(RL.lr) + "\n")
    f.write("reward_decay," + str(RL.gamma) + "\n")
    f.close()

    f = open(_folder_name + "/_rewards_.csv", 'w')
    f.close()

    for i_episode in range(3000):

        observation = env.reset(0, num_requests_training)

        while not env.done:

            observation = np.asarray(observation)
            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward)

            if done:

                print(i_episode, len(RL.ep_rs), sum(env.pickup_times)/len(env.pickup_times),
                      sum(env.pickup_ratio)/len(env.pickup_ratio), "\n")

                f = open(_folder_name + "/_rewards_.csv", 'a')
                f.write(str(i_episode) + "," + str(len(RL.ep_rs)) + "," + str(sum(env.pickup_times)/len(env.pickup_times))
                        + "," + str(sum(env.pickup_ratio)/len(env.pickup_ratio)) + "\n")
                f.close()

                _ = RL.learn()  # learning after an episode

            observation = observation_

        if i_episode % 10 == 0:
            RL.save(_folder_name, i_episode)


def eval_model():

    folder_id = input("\nfolder_id:\n")
    episode_id = int(input("\nepisode_id:\n"))
    RL = PolicyGradient(
        n_actions=num_actions,
        n_features=n_features,
    )

    RL.load(folder_name + folder_id, episode_id)

    loop_times = 10

    _ratio, _avept = 0.0, 0.0

    for k in range(loop_times):

        observation = env.reset(num_requests_training, num_requests_total)

        while not env.done:

            observation = np.asarray(observation)
            action = RL.choose_action_eval(observation)
            observation_, _, done = env.step(action)

            if done:
                print(k, len(env.pickup_times), sum(env.pickup_times) /
                      len(env.pickup_times), sum(env.pickup_ratio)/len(env.pickup_ratio), "\n")
                _ratio += sum(env.pickup_ratio) / len(env.pickup_ratio)
                _avept += sum(env.pickup_times) / len(env.pickup_times)

            observation = observation_

    print(_ratio / loop_times, _avept / loop_times)


if __name__ == "__main__":

    _start = time.time()

    _type = input("\n train: 1, eval: 2\n")

    if _type == "1":
        train_model()
    if _type == "2":
        eval_model()

    print("\n running time:", time.time()-_start)
