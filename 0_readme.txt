This is a TensorFlow implementation of the deep reinforcement learning-enabled dynamic ambulance redeployment method proposed by paper [1]. 

Dependencies:
    TensorFlow >= 1.5.0
    numpy and scipy.

(1) policy_n4.py
    implements the training and evaluation of the deep reinforcement learning method
    training or evaluation: python policy_n4.py

(2) ambulance_env.py
    implements a simplified simulation

(3) brain_policy_4.py
    defines the network structure of the deep (neural) score network

(4) Utility.py
    contains some functions for data processing

References: 
[1] Shenggong Ji, Yu Zheng, Zhaoyuan Wang, Tianrui Li. A Deep Reinforcement Learning-Enabled Dynamic Redeployment System for Mobile Ambulances. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT/UbiComp 2019) 3, 1, Article 15, 2019.
