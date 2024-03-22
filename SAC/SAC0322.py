from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary
from tensorflow.keras import  models, Input, optimizers
from tensorflow_probability.python.distributions import Normal
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, concatenate
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import copy
import os
import sys
import optparse
import subprocess
import traci
import tensorflow.keras
import h5py
import tensorflow_probability as tfp


class SACagent:
    def __init__(self):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.995) #指數移動平均 (Exponential Moving Average, EMA)，用於更新目標值網路的參數
        self.replay_buffer = deque(maxlen=10000) 
        self.gamma = 0.997 # discount factor
        self.learning_rate = 0.0002
        #self.state_shapes = [(8, 20, 1), (8, 20, 1), (2, 1)]
        self.action_dim = 2
        # self.log_aplha = tf.Variable(np.random.normal(), trainable=True, name="EntropyTemperature") #用於調節策略分佈的熵 (entropy)
        # self.mini_entropy = 0.1 #防止策略的熵過小，確保策略探索性
        self.policy_OPT = optimizers.Adam(learning_rate=1e-3) #policy network的優化器
        self.Q1_OPT = optimizers.Adam(learning_rate=1e-3) #Q1 value network的優化器
        self.Q2_OPT = optimizers.RMSprop(learning_rate=1e-3) #Q2 value network 的優化器
        self.value_OPT = optimizers.Adam(learning_rate=1e-3) #value network 的優化器
        # self.alpha_OPT = optimizers.SGD(learning_rate=1e-3) #調節熵參數的優化器(SGD)
        self.alpha = 0.2
        self.policy_network = self._build_policy_network()
        self.q_network_1 = self.build_q1_network()
        self.q_network_2 = self.build_q2_network()
        self.value_network = self.build_value_network()
        self.target_value_network = tf.keras.models.clone_model(self.value_network)
        self._update_target_value_network()

    def _build_policy_network(self):
        # 为每种输入类型定义Input层
        input_1 = Input(shape=(8, 20, 1))
        x1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_1) 
        x1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
        x1 = Flatten()(x1)
        input_2 = Input(shape=(8, 20, 1))
        x2 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_2)
        x2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
        x2 = Flatten()(x2)
        input_3 = Input(shape=(2, 1))
        x3 = Flatten()(input_3)
        # 合并处理后的输入
        #x = concatenate([x1, x2, x3])
        x4 = tensorflow.keras.layers.concatenate([x1, x2, x3])
        x5 = Dense(128, activation='relu')(x4)
        x6 = Dense(64, activation='relu')(x5)
        # 输出代表了每个可能动作的概率
        action_probs = Dense(self.action_dim, activation='softmax')(x6)
        # 构建模型
        policy_network = models.Model(inputs=[input_1, input_2, input_3], outputs=action_probs)
        return policy_network

    def build_q1_network(self):
        input_1 = Input(shape=(8, 20, 1))
        x1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_1)
        x1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
        x1 = Flatten()(x1)
        input_2 = Input(shape=(8, 20, 1))
        x2= Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_2)
        x2= Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
        x2= Flatten()(x2)
        input_3 = Input(shape=(2, 1))
        x3= Flatten()(input_3)
        Q_state_input = tensorflow.keras.layers.concatenate([x1, x2, x3])
        #Q_state_input = concatenate([conv1_flat, conv2_flat, flat3])
        # 动作输入
        action_input = Input(shape=(2, 1))
        Q_action_input = Flatten()(action_input)
        # 合并所有输入
        #concat = concatenate([conv1_flat, conv2_flat, flat3, action_input])
        x4 = tensorflow.keras.layers.concatenate([Q_state_input, Q_action_input])
        # 全连接层
        x5 = Dense(128, activation='relu')(x4)
        x6= Dense(64, activation='relu')(x5)
        q_output = Dense(1, activation='linear')(x6)  # Q值输出
        # 构建模型
        q_network_1 = models.Model(inputs=[input_1, input_2, input_3, action_input], outputs=q_output) 
        return q_network_1
    
    def build_q2_network(self):
        input_1 = Input(shape=(8, 20, 1))
        x1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_1)
        x1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
        x1 = Flatten()(x1)
        input_2 = Input(shape=(8, 20, 1))
        x2= Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_2)
        x2= Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
        x2= Flatten()(x2)
        input_3 = Input(shape=(2, 1))
        x3= Flatten()(input_3)
        Q_state_input = tensorflow.keras.layers.concatenate([x1, x2, x3])
        #Q_state_input = concatenate([conv1_flat, conv2_flat, flat3])
        # 动作输入
        action_input = Input(shape=(2, 1))
        Q_action_input = Flatten()(action_input)
        # 合并所有输入
        #concat = concatenate([conv1_flat, conv2_flat, flat3, action_input])
        x4 = tensorflow.keras.layers.concatenate([Q_state_input, Q_action_input])
        # 全连接层
        x5 = Dense(128, activation='relu')(x4)
        x6= Dense(64, activation='relu')(x5)
        q_output = Dense(1, activation='linear')(x6)  # Q值输出
        # 构建模型
        q_network_2 = models.Model(inputs=[input_1, input_2, input_3, action_input], outputs=q_output)
        return q_network_2

    def build_value_network(self):
        input_1 = Input(shape=(8, 20, 1))
        x1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_1)
        x1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
        x1 = Flatten()(x1)
        input_2 = Input(shape=(8, 20, 1))
        x2 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_2)
        x2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
        x2 = Flatten()(x2)
        input_3 = Input(shape=(2, 1))
        x3 = Flatten()(input_3)
        # 合并所有输入
        x4 = tensorflow.keras.layers.concatenate([x1, x2, x3])
        # 全连接层
        x5 = Dense(128, activation='relu')(x4)
        x6 = Dense(64, activation='relu')(x5)
        value_output = Dense(1, activation='linear')(x6)  # 值函数输出
        # 构建模型
        value_network = models.Model(inputs=[input_1, input_2, input_3], outputs=value_output)
        return value_network

    def _update_target_value_network(self):
        self.ema.apply(self.value_network.trainable_variables)
        for target_value_network_para, value_network_para in zip(self.target_value_network.trainable_variables, self.value_network.trainable_variables):
            target_value_network_para.assign(self.ema.average(value_network_para))

    def save_memory(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.append((states, actions, rewards, next_states, dones))
    
    '''使用這個方法到 def update_weights會出問題 但有開始訓練'''
    def take_action(self, states):
        # 使用模型进行预测
        action_probs = self.policy_network.predict(states)
        # 选择概率最高的动作
        actions = np.argmax(action_probs, axis=1)[0]
        return actions
    
    '''使用這個方法會無法訓練（Demo所呈現的問題）'''
    def take_action(self, states):
        states = np.array([states], dtype=np.float32)  # 确保输入的states是浮点数数组
        action_probs = self.policy_network(states) # 使用predict方法来得到动作概率
        action_dist = tfp.distributions.Categorical(probs=action_probs)  # 使用tensorflow_probability创建分类分布
        actions = action_dist.sample()  # 从分布中采样一个动作
        return actions.numpy()[0]  # 转换为numpy数组，并返回动作值

    def update_weights(self, batch_size):
    # 从回放缓存中采样数据
        batch_size = min(batch_size, len(self.replay_buffer))
        training_data = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*training_data))
        # 转换数据类型，适配TensorFlow的处理需求
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.reshape(rewards, newshape=(-1, 1)).astype(np.float32)
        dones = np.reshape(dones, newshape=(-1, 1)).astype(np.float32)
        actions = actions.astype(np.int32)
        
        # 策略网络更新
        with tf.GradientTape() as tape:
            # 假设策略网络直接输出每个动作的概率
            action_probs = self.policy_network(states, training=True)
            log_action_probs = tf.math.log(action_probs + 1e-8)
            # 估计Q值
            q_values = tf.minimum(self.Q_network_1(states, training=True), self.Q_network_2(states, training=True))
            q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions, depth=self.action_size), axis=1, keepdims=True)
            # 熵项
            entropy = -tf.reduce_sum(action_probs * log_action_probs, axis=1, keepdims=True)
            # 计算策略损失
            policy_loss = -tf.reduce_mean(q_values_selected - self.alpha * entropy)
        # 应用梯度更新策略网络
        policy_network_grad = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.policy_OPT.apply_gradients(zip(policy_network_grad, self.policy_network.trainable_variables))
        
        '''
        with tf.GradientTape() as tape:
            alpha_loss = - tf.exp(self.log_aplha) * (tf.reduce_mean(tf.exp(logprob) * logprob) + self.mini_entropy)
            alpha_grad = tape.gradient(alpha_loss, [self.log_aplha])
            self.alpha_OPT.apply_gradients(zip(alpha_grad, [self.log_aplha]))
        '''

        # Q网络更新逻辑，需要根据动作概率和离散动作空间的特点进行相应调整
        with tf.GradientTape() as tape:
            # 计算当前Q值
            q_values_1 = self.Q_network_1(states)
            q_action_1 = tf.reduce_sum(q_values_1 * tf.one_hot(actions, depth=self.action_size), axis=1)
            # 计算目标Q值
            next_q_values = tf.stop_gradient(self.target_value_network(next_states))
            next_q_value = tf.reduce_max(next_q_values, axis=1)
            expected_q_1 = rewards + self.gamma * (1 - dones) * next_q_value
            # 计算损失并更新Q网络
            q_loss_1 = tf.reduce_mean(tf.square(q_action_1 - expected_q_1))
        q_grad = tape.gradient(q_loss_1, self.Q_network_1.trainable_variables)
        self.Q1_OPT.apply_gradients(zip(q_grad, self.Q_network_1.trainable_variables))

        with tf.GradientTape() as tape:
            # 计算第二个Q网络当前状态下的Q值
            q_values_2 = self.Q_network_2(states)
            q_action_2 = tf.reduce_sum(q_values_2 * tf.one_hot(actions, depth=self.action_size), axis=1)
            # 计算目标Q值，这个目标Q值是基于目标值网络的输出
            next_q_values_target = tf.stop_gradient(self.target_value_network(next_states))
            next_q_value_target = tf.reduce_max(next_q_values_target, axis=1)
            expected_q_2 = rewards + self.gamma * (1 - dones) * next_q_value_target
            # 计算第二个Q网络的损失
            q_loss_2 = tf.reduce_mean(tf.square(q_action_2 - expected_q_2))
        # 计算梯度并更新第二个Q网络
        q_grad_2 = tape.gradient(q_loss_2, self.Q_network_2.trainable_variables)
        self.Q2_OPT.apply_gradients(zip(q_grad_2, self.Q_network_2.trainable_variables))

        # 值网络更新逻辑，依赖于离散动作空间的具体实现需求进行调整
        with tf.GradientTape() as tape:
            # 计算当前值
            values = self.value_network(states)
            # 使用策略网络获取动作概率
            action_probs = self.policy_network(states)
            q_values = self.Q_network_1(states)
            # 计算Q值的加权平均，作为目标值
            q_expected = tf.reduce_sum(q_values * action_probs, axis=1, keepdims=True)
            # 计算损失并更新值网络
            value_loss = tf.reduce_mean(tf.square(values - q_expected))
        value_grad = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.value_OPT.apply_gradients(zip(value_grad, self.value_network.trainable_variables))

        # 返回损失值，用于训练过程中的监控
        return (
            np.array(q_loss_1, dtype=np.float64),
            np.array(q_loss_2, dtype=np.float64),
            np.array(policy_loss, dtype=np.float64),
            np.array(value_loss, dtype=np.float64),
            # np.array(alpha_loss, dtype=np.float64),
            # np.exp(self.log_aplha)
        )
    

    def save_weights(self, path):
        self.policy_network.save_weights(path + '-policy_network.h5')
        self.value_network.save_weights(path + '-value_network.h5')
        self.q_network_1.save_weights(path + '-Q_network_1.h5')
        self.q_network_2.save_weights(path + '-Q_network_2.h5')

    def load_weights(self, path):
        self.policy_network.load_weights(path + '-policy_network.h5')
        self.value_network.load_weights(path + '-value_network.h5')
        self.q_network_1.load_weights(path + '-Q_network_1.h5')
        self.q_network_2.load_weights(path + '-Q_network_2.h5')


class SumoIntersection:
    
    def __init__(self): 
        try:
            sys.path.append(os.path.join(os.path.dirname(
                __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
                os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
            from sumolib import checkBinary  # noqa
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def generate_routefile(self, i):
        random.seed(1)
       
        NS_STRAIGHT = random.uniform(0.1076, 0.1315)
        NS_LEFT = random.uniform(0.0493, 0.0603)
        NS_RIGHT = random.uniform(0.0547, 0.0668) 
        
        SN_STRAIGHT = random.uniform(0.1014, 0.1240)
        SN_LEFT = random.uniform(0.0419, 0.0512)
        SN_RIGHT = random.uniform(0.0497, 0.0607) 
        
        EW1_STRAIGHT = random.uniform(0.0446,0.0545)
        EW1_LEFT = random.uniform(0.0115, 0.0140)
        EW1_RIGHT = random.uniform(0.0031, 0.0037)
        WE1_STRAIGHT = random.uniform(0.0440, 0.0538)
        WE1_LEFT = random.uniform(0.0096, 0.0118)
        WE1_RIGHT = random.uniform(0.0266, 0.0326)
      
        EW2_RIGHT = random.uniform(0.0046, 0.0056)

        EW3_STRAIGHT = random.uniform(0.0329, 0.0402)
        EW3_LEFT = random.uniform(0.0054, 0.0065)
        EW3_RIGHT = random.uniform(0.0059, 0.0072)
        WE3_STRAIGHT = random.uniform(0.0168, 0.0206)
        WE3_LEFT = random.uniform(0.0057, 0.0070)
        WE3_RIGHT = random.uniform(0.0014, 0.0017)
        
        EW4_STRAIGHT = random.uniform(0.0181, 0.0221)
        EW4_LEFT = random.uniform(0.0033, 0.0041)
        EW4_RIGHT = random.uniform(0.0038, 0.0046)
        WE4_STRAIGHT = random.uniform(0.0181, 0.0221)
        WE4_LEFT = random.uniform(0.0000, 0.0000)
        WE4_RIGHT = random.uniform(0.0057, 0.0070)
        
        EW5_STRAIGHT = random.uniform(0.0255, 0.0312)
        EW5_LEFT = random.uniform(0.0115, 0.0141)
        EW5_RIGHT = random.uniform(0.0062, 0.0076)
        WE5_STRAIGHT = random.uniform(0.0215, 0.0263)
        WE5_LEFT = random.uniform(0.0057, 0.0069)
        WE5_RIGHT = random.uniform(0.0124, 0.0151)
        
        EW6_STRAIGHT = random.uniform(0.0377, 0.0461)
        EW6_LEFT = random.uniform(0.0065, 0.0079)
        EW6_RIGHT = random.uniform(0.0119, 0.0146)
        WE6_STRAIGHT = random.uniform(0.0347, 0.0424)
        WE6_LEFT = random.uniform(0.0084, 0.0103)
        WE6_RIGHT = random.uniform(0.0037, 0.0045)
        
        EW7_STRAIGHT = random.uniform(0.0157, 0.0192)
        EW7_LEFT = random.uniform(0.0000, 0.0000)
        EW7_RIGHT = random.uniform(0.0076, 0.0093)
        WE7_STRAIGHT = random.uniform(0.0117, 0.0143)
        WE7_LEFT = random.uniform(0.0015, 0.0019)
        WE7_RIGHT = random.uniform(0.0024, 0.0030)
        
        N = 3600
        
        S7_STRAIGHT = SN_STRAIGHT 
        S7_LEFT = SN_LEFT
        S7_RIGHT = SN_RIGHT
        S6_LEFT = SN_LEFT
        S6_RIGHT = SN_RIGHT
        S5_LEFT = SN_LEFT
        S5_RIGHT = SN_RIGHT 
        S4_LEFT = SN_LEFT
        S4_RIGHT = SN_RIGHT
        S3_LEFT = SN_LEFT
        S3_RIGHT = SN_RIGHT
        S2_RIGHT = SN_RIGHT 
        S1_LEFT = SN_LEFT
        S1_RIGHT = SN_RIGHT
        
        N1_STRAIGHT = NS_STRAIGHT
        N1_LEFT = NS_LEFT
        N1_RIGHT = NS_RIGHT
        N3_LEFT = NS_LEFT
        N3_RIGHT = NS_RIGHT
        N4_LEFT = NS_LEFT
        N4_RIGHT = NS_RIGHT
        N5_LEFT = NS_LEFT
        N5_RIGHT = NS_RIGHT
        N6_LEFT = NS_LEFT
        N6_RIGHT = NS_RIGHT
        N7_LEFT = NS_LEFT
        N7_RIGHT = NS_RIGHT

        E1_STRAIGHT = EW1_STRAIGHT
        E1_LEFT = EW1_LEFT
        E1_RIGHT = EW1_RIGHT
        E2_RIGHT = EW2_RIGHT
        E3_STRAIGHT = EW3_STRAIGHT
        E3_LEFT = EW3_LEFT
        E3_RIGHT = EW3_RIGHT
        E4_STRAIGHT = EW4_STRAIGHT
        E4_LEFT = EW4_LEFT
        E4_RIGHT = EW4_RIGHT
        E5_STRAIGHT = EW5_STRAIGHT
        E5_LEFT = EW5_LEFT
        E5_RIGHT = EW5_RIGHT
        E6_STRAIGHT = EW6_STRAIGHT
        E6_LEFT = EW6_LEFT
        E6_RIGHT = EW6_RIGHT
        E7_STRAIGHT = EW7_STRAIGHT
        E7_LEFT = EW7_LEFT
        E7_RIGHT = EW7_RIGHT
        
        W1_STRAIGHT = WE1_STRAIGHT
        W1_LEFT = WE1_LEFT
        W1_RIGHT = WE1_RIGHT
        W3_STRAIGHT = WE3_STRAIGHT
        W3_LEFT = WE3_LEFT
        W3_RIGHT = WE3_RIGHT
        W4_STRAIGHT = WE4_STRAIGHT
        W4_LEFT = WE4_LEFT
        W4_RIGHT = WE4_RIGHT
        W5_STRAIGHT = WE5_STRAIGHT
        W5_LEFT = WE5_LEFT
        W5_RIGHT = WE5_RIGHT
        W6_STRAIGHT = WE6_STRAIGHT
        W6_LEFT = WE6_LEFT
        W6_RIGHT = WE6_RIGHT
        W7_STRAIGHT = WE7_STRAIGHT
        W7_LEFT = WE7_LEFT
        W7_RIGHT = WE7_RIGHT

        
        with open("0108.rou.xml", "w", encoding = 'UTF-8') as routes: 
            print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="4" minGap="1.5" maxSpeed="43.5"/>
    <!--往北車輛-->
	<route id="S7_STRAIGHT" edges="N7 N6 N5 N4 N3 N31 N2 N1 N11 N0"/><!--直走-->
	<route id="S7_LEFT" edges="N7 E19"/><!--第七個路口左轉-->
	<route id="S7_RIGHT" edges="N7 W20"/><!--第七個路口右轉-->
	<route id="S6_LEFT" edges="N7 N6 E171 E17"/><!--第六個路口左轉-->
	<route id="S6_RIGHT" edges="N7 N6 W181 W18"/><!--第六個路口右轉-->
	<route id="S5_LEFT" edges="N7 N6 N5 E15"/><!--第五個路口左轉-->
	<route id="S5_RIGHT" edges="N7 N6 N5 W16"/><!--第五個路口右轉-->
	<route id="S4_LEFT" edges="N7 N6 N5 N4 E13"/><!--第四個路口左轉-->
	<route id="S4_RIGHT" edges="N7 N6 N5 N4 W14"/><!--第四個路口右轉-->
	<route id="S3_LEFT" edges="N7 N6 N5 N4 N3 N31 E11"/><!--第三個路口左轉-->
	<route id="S3_RIGHT" edges="N7 N6 N5 N4 N3 N31 W121 W12"/><!--第三個路口右轉-->
	<route id="S2_RIGHT" edges="N7 N6 N5 N4 N3 N31 N2 W10"/><!--第二個路口右轉-->	
	<route id="S1_LEFT" edges="N7 N6 N5 N4 N3 N31 N2 N1 N11 E81 E8"/><!--第一個路口左轉-->
	<route id="S1_RIGHT" edges="N7 N6 N5 N4 N3 N31 N2 N1 N11 W91 W9"/><!--第一個路口右轉-->
	
	<!--往南車輛-->
	<route id="N1_STRAIGHT" edges="S0 S1 S2 S21 S3 S4 S5 S6 S7"/><!--直走-->
	<route id="N1_LEFT" edges="S0 W91 W9"/><!--第一個路口左轉-->
	<route id="N1_RIGHT" edges="S0 E81 E8"/><!--第一個路口右轉-->
	<route id="N3_LEFT" edges="S0 S1 S2 S21 W121 W12"/><!--第三個路口左轉-->
	<route id="N3_RIGHT" edges="S0 S1 S2 S21 E11"/><!--第三個路口右轉-->
    <route id="N4_LEFT" edges="S0 S1 S2 S21 S3 W14"/><!--第四個路口左轉-->
	<route id="N4_RIGHT" edges="S0 S1 S2 S21 S3 E13"/><!--第四個路口右轉-->
	<route id="N5_LEFT" edges="S0 S1 S2 S21 S3 S4 W16"/><!--第五個路口左轉-->
	<route id="N5_RIGHT" edges="S0 S1 S2 S21 S3 S4 E15"/><!--第五個路口右轉-->
	<route id="N6_LEFT" edges="S0 S1 S2 S21 S3 S4 S5 W181 W18"/><!--第六個路口左轉-->
	<route id="N6_RIGHT" edges="S0 S1 S2 S21 S3 S4 S5 E171 E17"/><!--第六個路口右轉-->
	<route id="N7_LEFT" edges="S0 S1 S2 S21 S3 S4 S5 S6 W20"/><!--第七個路口左轉-->
	<route id="N7_RIGHT" edges="S0 S1 S2 S21 S3 S4 S5 S6 E19"/><!--第七個路口右轉-->

	<!--往西車輛-->
	<route id="E1_STRAIGHT" edges="E9 E91 E81 E8"/><!--第一個路口直走-->
	<route id="E1_LEFT" edges="E9 E91 S1 S2 S21 S3 S4 S5 S6 S7"/><!--第一個路口左轉-->
	<route id="E1_RIGHT" edges="E9 E91 N0"/><!--第一個路口右轉-->
	<route id="E2_RIGHT" edges="E10 N1 N11 N0"/><!--第二個路口右轉-->
	<route id="E3_STRAIGHT" edges="E12 E121 E11"/><!--第三個路口直走-->
	<route id="E3_LEFT" edges="E12 E121 S3 S4 S5 S6 S7"/><!--第三個路口左轉-->
	<route id="E3_RIGHT" edges="E12 E121 N2 N1 N11 N0"/><!--第三個路口右轉-->
	<route id="E4_STRAIGHT" edges="E14 E13"/><!--第四個路口直走-->
	<route id="E4_LEFT" edges="E14 S4 S5 S6 S7"/><!--第四個路口左轉-->
	<route id="E4_RIGHT" edges="E14 N3 N31 N2 N1 N11 N0"/><!--第四個路口右轉-->
	<route id="E5_STRAIGHT" edges="E16 E15"/><!--第五個路口直走-->
	<route id="E5_LEFT" edges="E16 S5 S6 S7"/><!--第五個路口左轉-->
	<route id="E5_RIGHT" edges="E16 N4 N3 N31 N2 N1 N11 N0"/><!--第五個路口右轉-->
	<route id="E6_STRAIGHT" edges="E18 E181 E171 E17"/><!--第六個路口直走-->
	<route id="E6_LEFT" edges="E18 E181 S6 S7"/><!--第六個路口左轉-->
	<route id="E6_RIGHT" edges="E18 E181 N5 N4 N3 N31 N2 N1 N11 N0"/><!--第六個路口右轉-->
	<route id="E7_STRAIGHT" edges="E20 E19"/><!--第七個路口直走-->
	<route id="E7_LEFT" edges="E20 S7"/><!--第七個路口左轉-->
	<route id="E7_RIGHT" edges="E20 N6 N5 N4 N3 N31 N2 N1 N11 N0"/><!--第七個路口右轉-->

	<!--往東車輛-->
	<route id="W1_STRAIGHT" edges="W8 W81 W91 W9"/><!--第一個路口直走-->
	<route id="W1_LEFT" edges="W8 W81 N0"/><!--第一個路口左轉-->
	<route id="W1_RIGHT" edges="W8 W81 S1 S2 S21 S3 S4 S5 S6 S7"/><!--第一個路口右轉-->
    <route id="W3_STRAIGHT" edges="W11 W121 W12"/><!--第三個路口直走-->
	<route id="W3_LEFT" edges="W11 N2 N1 N11 N0"/><!--第三個路口左轉-->
	<route id="W3_RIGHT" edges="W11 S3 S4 S5 S6 S7"/><!--第三個路口右轉-->
	<route id="W4_STRAIGHT" edges="W13 W14"/><!--第四個路口直走-->
	<route id="W4_LEFT" edges="W13 N3 N31 N2 N1 N11 N0"/><!--第四個路口左轉-->
	<route id="W4_RIGHT" edges="W13 S4 S5 S6 S7"/><!--第四個路口右轉-->
	<route id="W5_STRAIGHT" edges="W15 W16"/><!--第五個路口直走-->
	<route id="W5_LEFT" edges="W15 N4 N3 N31 N2 N1 N11 N0"/><!--第五個路口左轉-->
	<route id="W5_RIGHT" edges="W15 S5 S6 S7"/><!--第五個路口右轉-->
	<route id="W6_STRAIGHT" edges="W17 W171 W181 W18"/><!--第六個路口直走-->
	<route id="W6_LEFT" edges="W17 W171 N5 N4 N3 N31 N2 N1 N11 N0"/><!--第六個路口左轉-->
	<route id="W6_RIGHT" edges="W17 W171 S6 S7"/><!--第六個路口右轉-->
	<route id="W7_STRAIGHT" edges="W19 W20"/><!--第七個路口直走-->
	<route id="W7_LEFT" edges="W19 N6 N5 N4 N3 N31 N2 N1 N11 N0"/><!--第七個路口左轉-->
	<route id="W7_RIGHT" edges="W19 S7"/><!--第七個路口右轉-->
   
    
    ''', file=routes)
            
            vehNr = 0   
            for i in range(N):
                if random.uniform(0, 1) < S7_STRAIGHT:    #random.uniform(low,high)，在low~high隨機產生浮點數
                    print('    <vehicle id="S7_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="S7_STRAIGHT" departLane="free" departSpeed="10" depart="%i" color="1,0,0"/>'
                          % (vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S7_LEFT:
                    print('    <vehicle id="S7_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="S7_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S7_RIGHT:
                    print('    <vehicle id="S7_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="S7_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S6_LEFT:
                    print('    <vehicle id="S6_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="S6_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S6_RIGHT:
                    print('    <vehicle id="S6_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="S6_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S5_LEFT:
                    print('    <vehicle id="S5_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="S5_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S5_RIGHT:
                    print('    <vehicle id="S5_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="S5_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S4_LEFT:
                    print('    <vehicle id="S4_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="S4_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S4_RIGHT:
                    print('    <vehicle id="S4_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="S4_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S3_LEFT:
                    print('    <vehicle id="S3_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="S3_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S3_RIGHT:
                    print('    <vehicle id="S3_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="S3_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S2_RIGHT:
                    print('    <vehicle id="S2_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="S2_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S1_LEFT:
                    print('    <vehicle id="S1_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="S1_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < S1_RIGHT:
                    print('    <vehicle id="S1_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="S1_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
            
            
                if random.uniform(0, 1) < N1_STRAIGHT:
                    print('    <vehicle id="N1_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="N1_STRAIGHT" departLane="free" departSpeed="10" depart="%i" color="1,0,0"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N1_LEFT:
                    print('    <vehicle id="N1_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="N1_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N1_RIGHT:
                    print('    <vehicle id="N1_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="N1_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N3_LEFT:
                    print('    <vehicle id="N3_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="N3_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N3_RIGHT:
                    print('    <vehicle id="N3_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="N3_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N4_LEFT:
                    print('    <vehicle id="N4_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="N4_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N4_RIGHT:
                    print('    <vehicle id="N4_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="N4_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N5_LEFT:
                    print('    <vehicle id="N5_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="N5_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N5_RIGHT:
                    print('    <vehicle id="N5_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="N5_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N6_LEFT:
                    print('    <vehicle id="N6_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="N6_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N6_RIGHT:
                    print('    <vehicle id="N6_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="N6_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N7_LEFT:
                    print('    <vehicle id="N7_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="N7_LEFT" departLane="1" departSpeed="10" depart="%i" color="0,0,255"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < N7_RIGHT:
                    print('    <vehicle id="N7_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="N7_RIGHT" departLane="0" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                
                if random.uniform(0, 1) < E1_STRAIGHT:
                    print('    <vehicle id="E1_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E1_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E1_LEFT:
                    print('    <vehicle id="E1_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="E1_LEFT" departLane="1" departSpeed="10" depart="%i" color="1,0,0"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E1_RIGHT:
                    print('    <vehicle id="E1_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E1_RIGHT" departLane="0" departSpeed="10" depart="%i" color="1,0,0"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E2_RIGHT:
                    print('    <vehicle id="E2_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E2_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0"/>' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E3_STRAIGHT:
                    print('    <vehicle id="E3_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E3_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E3_LEFT:
                    print('    <vehicle id="E3_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="E3_LEFT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E3_RIGHT:
                    print('    <vehicle id="E3_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E3_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E4_STRAIGHT:
                    print('    <vehicle id="E4_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E4_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E4_LEFT:
                    print('    <vehicle id="E4_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="E4_LEFT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E4_RIGHT:
                    print('    <vehicle id="E4_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E4_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E5_STRAIGHT:
                    print('    <vehicle id="E5_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E5_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E5_LEFT:
                    print('    <vehicle id="E5_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="E5_LEFT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E5_RIGHT:
                    print('    <vehicle id="E5_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E5_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E6_STRAIGHT:
                    print('    <vehicle id="E6_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E6_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E6_LEFT:
                    print('    <vehicle id="E6_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="E6_LEFT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E6_RIGHT:
                    print('    <vehicle id="E6_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E6_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E7_STRAIGHT:
                    print('    <vehicle id="E7_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E7_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E7_LEFT:
                    print('    <vehicle id="E7_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="E7_LEFT" departLane="1" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < E7_RIGHT:
                    print('    <vehicle id="E7_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="E7_RIGHT" departLane="0" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
        
                if random.uniform(0, 1) < W1_STRAIGHT:
                    print('    <vehicle id="W1_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W1_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W1_LEFT:
                    print('    <vehicle id="W1_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="W1_LEFT" departLane="1" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W1_RIGHT:
                    print('    <vehicle id="W1_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W1_RIGHT" departLane="0" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W3_STRAIGHT:
                    print('    <vehicle id="W3_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W3_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W3_LEFT:
                    print('    <vehicle id="W3_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="W3_LEFT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W3_RIGHT:
                    print('    <vehicle id="W3_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W3_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W4_STRAIGHT:
                    print('    <vehicle id="W4_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W4_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W4_LEFT:
                    print('    <vehicle id="W4_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="W4_LEFT" departLane="1" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W4_RIGHT:
                    print('    <vehicle id="W4_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W4_RIGHT" departLane="0" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W5_STRAIGHT:
                    print('    <vehicle id="W5_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W5_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W5_LEFT:
                    print('    <vehicle id="W5_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="W5_LEFT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W5_RIGHT:
                    print('    <vehicle id="W5_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W5_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W6_STRAIGHT:
                    print('    <vehicle id="W6_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W6_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W6_LEFT:
                    print('    <vehicle id="W6_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="W6_LEFT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W6_RIGHT:
                    print('    <vehicle id="W6_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W6_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W7_STRAIGHT:
                    print('    <vehicle id="W7_STRAIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W7_STRAIGHT" departLane="random" departSpeed="10" depart="%i" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W7_LEFT:
                    print('    <vehicle id="W7_LEFT_%i" type="SUMO_DEFAULT_TYPE" route="W7_LEFT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < W7_RIGHT:
                    print('    <vehicle id="W7_RIGHT_%i" type="SUMO_DEFAULT_TYPE" route="W7_RIGHT" departLane="random" departSpeed="10" depart="%i" color="1,0,0" />' % (
                            vehNr, i), file=routes)
                    vehNr += 1  
                    lastVeh = i
            print("</routes>", file=routes)
        return vehNr
        
        
    def get_options(self): 
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=True, help="run the commandline version of sumo")  #default=True 不開啟sumo訓練
        options, args = optParser.parse_args()
        return options

    
    def getState_4(self):   
        positionMatrix = []  #初始化
        velocityMatrix = []  #初始化
        cellLength = 5  #每個格子的距離
        offset_N4 = 11  #離停止線的距離
        offset_E4 = 14
        offset_S4 = 12
        offset_W4 = 14
        speedLimit = 13.8 

        # 擷取四向道路上最後一台車ID
        vehicles_road_E4 = traci.edge.getLastStepVehicleIDs('E14')  # 東向來車 
        vehicles_road_S4 = traci.edge.getLastStepVehicleIDs('N4')  # 南向來車
        vehicles_road_W4 = traci.edge.getLastStepVehicleIDs('W13')  # 西向來車
        vehicles_road_N4 = traci.edge.getLastStepVehicleIDs('S3')   # 北向來車     
 
            
        for i in range(8): #車道數&觀測距離
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(20):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)
                
        junctionPosition = traci.junction.getPosition('gneJ7')[1]  #抓路口位置 => traci.junction.getPosition()[0=x軸, 1=y軸 ]      
        for v in vehicles_road_E4:
            ind = int(abs(junctionPosition - offset_E4 - traci.vehicle.getPosition(v)[1]) / cellLength)
            if(ind < 20):                 
                positionMatrix[traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('gneJ7')[0]
        for v in vehicles_road_S4:
            ind = int(abs(junctionPosition - offset_S4 - traci.vehicle.getPosition(v)[0]) / cellLength)
            if(ind < 20):
                positionMatrix[1 + traci.vehicle.getLaneIndex(v)][ind] = 1  
                velocityMatrix[1 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('gneJ7')[1]
        for v in vehicles_road_W4: 
            ind = int(abs(junctionPosition + offset_W4 - traci.vehicle.getPosition(v)[1]) / cellLength)
            if(ind < 20):                
                positionMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = 1  
                velocityMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit        

        junctionPosition = traci.junction.getPosition('gneJ7')[0]
        for v in vehicles_road_N4: 
            ind = int(abs(junctionPosition + offset_N4 - traci.vehicle.getPosition(v)[0]) / cellLength)
            if(ind < 20):
                positionMatrix[5 + traci.vehicle.getLaneIndex(v)][ind] = 1  
                velocityMatrix[5 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit


        light = []
        if(traci.trafficlight.getPhase('gneJ7') == 0):  
            light = [1, 0]  # 幹道綠燈
        else:
            light = [0, 1]  # 支道綠燈

        position = np.array(positionMatrix)
        position = position.reshape(1, 8, 20, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 8, 20, 1)

        lgts = np.array(light)
        lgts = lgts.reshape(1, 2, 1) # 時相數

        return [position, velocity, lgts]
    
    def getPass(self):
        '''getPass'''
        car_list = traci.vehicle.getIDList()    #得到每台車ID的list
        for car_id in car_list: 
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located             
            if road_id in edges:  
                remain_car_set.add(car_id)
                total_car_set.add(car_id)
            else:
                if car_id in remain_car_set:
                    remain_car_set.remove(car_id)
    def getHalt(self):
        Halt_Number = 0
        for i in edges:
            Halt_Number += traci.edge.getLastStepHaltingNumber(str(i))
        return Halt_Number
    
    def getWait(self):
        Wait_Time = 0
        for i in edges:
            Wait_Time += traci.edge.getWaitingTime(str(i))
        return Wait_Time
    
    def getDelay(self):
        Delay_time = 0
        for i in lanes:
            MeanSpeed = traci.lane.getLastStepMeanSpeed(str(i))
            MaxSpeed = traci.lane.getMaxSpeed(str(i))
            Delay_time += 1 - (MeanSpeed/MaxSpeed)
        return Delay_time
    
    
if __name__ == '__main__':
    sumoInt = SumoIntersection()    
    options = sumoInt.get_options()

  
    if options.nogui:  
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
        
    episodes = 100
    batch_size = 128
    step = 0
    update_time = 0

    agent = SACagent() 
    try:
        agent.load('reinf_traf_control_2.h5')
    except:
        print('No models found')

    for e in range(episodes):  
        '''隨機產生車流資料'''    
        vehicle_number = sumoInt.generate_routefile(e) 
        waiting_time = 0    
        reward1 = 0
        reward2 = 0
        stepz = 0
        action = 0
        sum_all_reward = 0
        NOC = open('Number of car.txt','a' )
        WT = open('WaitTime.txt', 'a') 
        SAR = open('Sum of all reward.txt', 'a') 
        UPDATE_TIME = open('Update time.txt','a')
        AHN_e = open('Average halting vehicle.txt','a')
        Sum_HaltNumber = 0
        
        #---------------固定寫法-------------------
        traci.start([sumoBinary, "-c", "0108.sumocfg", '--start'])
        
        traci.trafficlight.setPhase("gneJ7", 0)
        traci.trafficlight.setPhaseDuration("gneJ7", 200)  

        for i in range(3):
            for i in range(118):
                traci.trafficlight.setPhase('gneJ7', 0)
                traci.simulationStep()
            for i in range(4):
                traci.trafficlight.setPhase('gneJ7', 1)
                traci.simulationStep()
            for i in range(3):
                traci.trafficlight.setPhase('gneJ7', 2)
                traci.simulationStep()                
            for i in range(48):
                traci.trafficlight.setPhase('gneJ7', 3)
                traci.simulationStep()
            for i in range(3):
                traci.trafficlight.setPhase('gneJ7', 4)
                traci.simulationStep()
            for i in range(4):
                traci.trafficlight.setPhase('gneJ7', 5)
                traci.simulationStep()     
        for i in range(20):
            traci.trafficlight.setPhase('gneJ7', 0)
            traci.simulationStep()                
                                                
            
        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 4000:    
            traci.simulationStep()      
            states = sumoInt.getState_4()
             
            actions = agent.take_action(states)  
            light_4 = states[2]   
            step += 1
            
            '''getPass'''
            PASS_VEHICLE_NUMBER = 0
            remain_car_set = set()
            total_car_set = set()
            '''Hault'''
            DEC_VEHICLE_NUMBER = 0
            Nt = 0
            Wt = 0
            Dt = 0
            gneJ = ["gneJ4", "gneJ5", "gneJ6", "gneJ7", "gneJ8", "gneJ9", "gneJ10"]
            edges = ["W13", "E14", "S3", "N4"]
            lanes = ["W13_0", "E14_0", "N4_0", "N4_1", "N4_2", "S3_0", "S3_1", "S3_2"]
            
            if(actions == 0 and light_4[0][0][0] == 0):   # 動作一=時相一
                for i in range(5):
                    stepz += 1
                    Sum_HaltNumber += sumoInt.getHalt()
                    for j in gneJ:
                        traci.trafficlight.setPhase(str(j), 0)   
                    sumoInt.getPass()
                    for k in edges:
                        waiting_time += traci.edge.getLastStepHaltingNumber(str(k))
                    traci.simulationStep()

            if(actions == 1 and light_4[0][0][0] == 1):    # 動作二 = 時相二
                for i in range(5):
                    stepz += 1
                    Sum_HaltNumber += sumoInt.getHalt()
                    for j in gneJ:
                        traci.trafficlight.setPhase(str(j), 3)
                    sumoInt.getPass()
                    for k in edges:
                        waiting_time += traci.edge.getLastStepHaltingNumber(str(k))
                    traci.simulationStep()
                    
            if(actions == 0 and light_4[0][0][0] == 1):   # 動作一 = 時相二
                for i in range(3):
                    stepz += 1
                    Sum_HaltNumber += sumoInt.getHalt()
                    for j in gneJ:
                        traci.trafficlight.setPhase(str(j), 4)
                    for k in edges:
                        waiting_time += traci.edge.getLastStepHaltingNumber(str(k))
                    traci.simulationStep()
                for i in range(4):
                    stepz += 1
                    Sum_HaltNumber += sumoInt.getHalt()
                    for j in gneJ:
                        traci.trafficlight.setPhase(str(j), 5)
                    for k in edges:
                        waiting_time += traci.edge.getLastStepHaltingNumber(str(k))
                    traci.simulationStep()
                for i in range(20):
                    stepz += 1
                    Sum_HaltNumber += sumoInt.getHalt()
                    for j in gneJ:
                        traci.trafficlight.setPhase(str(j), 0)
                    sumoInt.getPass()
                    for k in edges:
                        waiting_time += traci.edge.getLastStepHaltingNumber(str(k))
                    traci.simulationStep()


            if(actions == 1 and light_4[0][0][0] == 0):    # 動作二 = 時相一
                for i in range(4):
                    stepz += 1
                    Sum_HaltNumber += sumoInt.getHalt()
                    for j in gneJ:
                        traci.trafficlight.setPhase(str(j), 1)
                    for k in edges:
                        waiting_time += traci.edge.getLastStepHaltingNumber(str(k))
                    traci.simulationStep()
                for i in range(3):
                    stepz += 1
                    Sum_HaltNumber += sumoInt.getHalt()
                    for j in gneJ:
                        traci.trafficlight.setPhase(str(j), 2)
                    for k in edges:
                        waiting_time += traci.edge.getLastStepHaltingNumber(str(k))
                    traci.simulationStep()
                for i in range(20):
                    stepz += 1
                    Sum_HaltNumber += sumoInt.getHalt()
                    for j in gneJ:
                        traci.trafficlight.setPhase(str(j), 3)
                    sumoInt.getPass()
                    for k in edges:
                        waiting_time += traci.edge.getLastStepHaltingNumber(str(k))
                    traci.simulationStep() 
                    
            next_states = sumoInt.getState_4()
  
            '''Reward'''
            Nt = sumoInt.getHalt()
            Dt = sumoInt.getDelay()
            Wt = sumoInt.getWait()
                  

            if actions == 1:
               rt = 1
               if light_4[0][0][0] == 0:
                  pc = 1 
               else:
                   pc = 0
            elif actions == 0:
                 rt = 0
                 if light_4[0][0][0] == 1:
                    pc = 1 
                 else:
                     pc = -1
            
            PASS_VEHICLE_NUMBER = len(total_car_set - remain_car_set)
            HALT_VEHICLE_NUMBER = Nt
            PHASE_CHANGE = rt + pc
            MINOR_STREET_PHASE_CHANGE = rt
            DELAY_TIME = Dt
            WAITING_TIME = Wt/60 
            rewards = (1)*PASS_VEHICLE_NUMBER + (-1)*HALT_VEHICLE_NUMBER + (-5)*PHASE_CHANGE + (-5)*MINOR_STREET_PHASE_CHANGE + (-0.25)*DELAY_TIME  + (-0.25)*WAITING_TIME 
            sum_all_reward += rewards
        
            #print('通行:',PASS_VEHICLE_NUMBER, '停等車輛:', HALT_VEHICLE_NUMBER, '動作:', PHASE_CHANGE, '延滯:', DELAY_TIME, '等待時間:',WAITING_TIME)
            #print('獎勵:', reward_4)
            if stepz > 3600:
                break
            '''Model fit'''
            agent.save_memory(states, actions, rewards, next_states, False) 
        if(len(agent.replay_buffer) > batch_size):
            agent.update_weights(batch_size) 
            update_time += 1
        if e%5 == 0:
            agent.update_target_model_4()                
            
        
        # 假設 agent.memory_1 已經被初始化並設定了最大長度
        # 假設 reward_1 和其他必要的變數已經被定義

        # 檢查 memory_1 是否為空
        if len(agent.replay_buffer) > 0:
            # 取出最後一個元素但不刪除它，因為想要基於它創建一個新的元素
            mem = agent.replay_buffer[-1]

            # 創建一個新的元素並添加到 memory_1 中
            # 這裡假設 mem 是一個元組，且你想要保留除了第三個元素之外的所有元素
            # 第三個元素被替換為 reward_1，最後一個元素設置為 True
            new_mem = (mem[0], mem[1], rewards, mem[3], True)
            agent.replay_buffer.append(new_mem)
        else:
            # 處理 memory_1 為空的情況，根據你的應用需求添加適當的代碼
            pass

        
       
        #mem = agent.memory_1[-1]  
        #del agent.memory_1[-1]    
        #agent.memory_1.append((mem[0], mem[1], reward_1, mem[3], True))
        AHN = Sum_HaltNumber/3600
        
        '''紀錄'''
        NOC.write(str(vehicle_number) + '\n')
        AHN_e.write(str(AHN) + '\n')
        WT.write(str(waiting_time) + '\n')
        SAR.write(str(sum_all_reward) + '\n')
        UPDATE_TIME.write(str(update_time) + '\n')
        NOC.close()
        AHN_e.close()
        WT.close()
        SAR.close()
        UPDATE_TIME.close()
        print('episode - ' + str(e) + ' total waiting time - ' + str(waiting_time))
        print('更新次數:(', update_time, ')；獎勵:(', sum_all_reward, ')')
        agent.save_4('reinf_traf_control_1_' + str(e) + '.h5')
        traci.close(wait=False)
    
    agent.save_4('last_rtc_4.h5')
sys.stdout.flush()
