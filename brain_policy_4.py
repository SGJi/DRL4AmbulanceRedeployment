
'''
Shenggong Ji, Yu Zheng, Zhaoyuan Wang, Tianrui Li
A Deep Reinforcement Learning-Enabled Dynamic Redeployment System for Mobile Ambulances
IMWUT/UbiComp 2019
---------------------------------------------------------------------------------------
'''

import numpy as np
import tensorflow as tf


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.99,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.saver = tf.train.Saver(max_to_keep=None)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(
                tf.float32, [None, self.n_actions * self.n_features], name="observations")
            self.tf_acts = tf.placeholder(
                tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(
                tf.float32, [None, ], name="actions_value")

            self.tf_obs_2 = tf.reshape(
                self.tf_obs, [-1, self.n_features])  # [None*34, n_features]
        # fc1
        layer1 = tf.layers.dense(
            inputs=self.tf_obs_2,
            units=20,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
        )
        # fc2
        layer2 = tf.layers.dense(
            inputs=layer1,
            units=20,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
        )
        # fc3
        all_act = tf.layers.dense(
            inputs=layer2,
            units=1,  # score
            activation=None,
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
        )

        all_act_2 = tf.reshape(all_act, [-1, self.n_actions])  # [None, 34]

        # use softmax to convert to probability
        self.all_act_prob = tf.nn.softmax(all_act_2, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=all_act_2, labels=self.tf_acts) 
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
                                     self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(
            range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def choose_action_eval(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
                                     self.tf_obs: observation[np.newaxis, :]})
        action = np.argmax(prob_weights)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save(self, folder_name, _ep, add_info=""):

        self.saver.save(self.sess, folder_name +
                        "/model-step=" + str(_ep) + add_info + ".ckpt")

    def load(self, folder_name, _ep, add_info=""):

        self.saver.restore(self.sess, folder_name +
                           "/model-step=" + str(_ep) + add_info + ".ckpt")
