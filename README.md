# DRL4AmbulanceRedeployment
This is a TensorFlow implementation of the deep reinforcement learning-enabled dynamic ambulance redeployment method proposed by paper "A Deep Reinforcement Learning-Enabled Dynamic Redeployment System for Mobile Ambulances" [1]. 

# Dependencies:
    TensorFlow >= 1.5.0
    numpy and scipy.

# File description
## policy_n4.py
    implements the training and evaluation of the deep reinforcement learning method
    training or evaluation: python policy_n4.py

## ambulance_env.py
    implements a simplified simulation

## brain_policy_4.py
    defines the network structure of the deep (neural) score network

## Utility.py
    contains some functions for data processing

# References 
[1] Shenggong Ji, Yu Zheng, Zhaoyuan Wang, Tianrui Li. A Deep Reinforcement Learning-Enabled Dynamic Redeployment System for Mobile Ambulances. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT/UbiComp 2019) 3, 1, Article 15, 2019.
