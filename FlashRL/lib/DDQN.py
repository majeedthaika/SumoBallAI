import keras, tensorflow as tf, numpy as np, sys, copy, argparse
from keras import backend as K
from keras.layers import Input, Add, Subtract, Average, RepeatVector, Lambda, Activation, Conv2D
from keras.backend import mean, update_sub, ones, shape, sum, expand_dims
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from collections import deque
import json, os, errno
import random
from shutil import copyfile
import pdb

np.set_printoptions(threshold='nan')

class Replay_Memory():

	def __init__(self, memory_size=50000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
		# self.memory = deque(maxlen=memory_size)
		self.tail = 0
		self.memory_size = memory_size
		self.memory = []

	def make_transition(self, state, action, reward, next_state, is_terminal):
		if type(reward) is not np.ndarray:
			reward = np.array([reward])
		if type(action) is not np.ndarray:
			action = np.array([action])
		if type(is_terminal) is not np.ndarray:
			is_terminal = np.array([is_terminal])
		return (state, action, reward, next_state, is_terminal)

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
		# You will feed this to your model to train.
		minibatch = random.sample(self.memory, batch_size)
		return minibatch

	def append(self, state, action, reward, next_state, is_terminal):
		# Appends transition to the memory.
		transition = self.make_transition(state, action, reward, next_state, is_terminal)
		if len(self.memory) < self.memory_size:
			self.memory.append(transition)
		else:
			self.memory[self.tail] = transition
			self.tail = (self.tail + 1) % self.memory_size

	def append_many(self, all_states):
		# Appends transition to the memory.
		for one_transition in all_states:
			state, action, reward, next_state, is_terminal = one_transition
			self.append(state, action, reward, next_state, is_terminal)

	def __len__(self):
		return len(self.memory)

class DQN_Model:
	def __init__(self, action_size, model_path, filename=None, predict=True, lr =  0.0001,  gamma = 1,):
		self.action_size = action_size
		self.state_size = (84, 84, 4)
		self.model_path = model_path
		if filename:
			self.model = self.load_model(os.path.join(model_path, filename))
		else:
			inputs = Input(shape = (84, 84, 4))
			c1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
			c2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(c1)
			c3 = Conv2D(64, (3, 3), activation = 'relu')(c2)
			f1 = Flatten()(c3)

			d1 = Dense(512, activation = 'relu')(f1)
			merged = Dense(self.action_size)(d1)

			# v1 = Dense(512, activation = 'relu')(f1)
			# valFunc = Dense(1, activation = 'relu')(v1)

			# a1 = Dense(512, activation = 'relu')(f1)
			# advFunc = Dense(self.action_size, activation = 'relu')(a1)
			# advFuncMean = Lambda(lambda x: mean(x, axis=1))(advFunc)
			# advFuncOut = Lambda(lambda x: x[0] - expand_dims(x[1], axis=1))([advFunc, advFuncMean])

			# merged = Lambda(lambda x: x[0] + x[1])([valFunc, advFuncOut])

			self.model = Model(inputs=inputs, outputs=merged)
			optimizer = keras.optimizers.Adam(lr=lr)
			self.model.compile(optimizer=optimizer,
				loss='mean_squared_error',
				metrics=['accuracy'])
		if predict:
			self.model._make_predict_function()

	def predict(self, state, batch_size=None):
		if not batch_size:
			return self.model.predict(state)
		else:
			return self.model.predict(state, batch_size)

	def fit(self, x, y, batch_size=None):
		with tf_session.as_default():
			with tf_graph.as_default():
				if not batch_size:
					return self.model.fit(x,y,epochs=1,verbose=0)
				else:
					return self.model.fit(x,y,epochs=1,verbose=0,batch_size=batch_size)

	def get_model(self):
		return self.model

	def save_model(self, path):
		self.model.save(path)

	def load_model(self, path):
		try:
			self.model = load_model(path)
		except:
			self.model = load_model(os.path.join(self.model_path, "checkpoint_0.h5"))

class Trainer:
	def __init__(self, action_size, critic_model, actor_model, episode_number, batch_size=32, lr=0.0001, gamma=1):
		params = {
			"epsilon_initial": 0.5,
			"epsilon_final": 0.05,
			"epsilon": 0.5,
			"epsilon_decay": .995,
			"epsilon_interval": 100000,
			"gamma": 1,
			"train_iterations": 30,
			"num_iterations": 200,
			"num_episodes": 3000,
			"learning_rate": 0.0001,
			"batch_size": 32,
			"replay_memory_size": 1000000,
			"burn_in_size": 10000,
			"gpu": False,
		}
		self.state_size = (84, 84, 4)
		self.lr = float(params['learning_rate'])

		self.action_size = action_size
		self.lr = lr
		self.gamma = gamma
		self.input_shape = (self.state_size,)
		self.batch_size = batch_size
		self.episode_number = episode_number
		# pdb.set_trace()
		
		self.actor_model = actor_model
		self.critic_model = critic_model
		
		self.epsilon = float(params['epsilon'])
		self.epsilon_decay = float(params['epsilon_decay'])
		self.epsilon_initial = float(params['epsilon_initial'])
		self.epsilon_final = float(params['epsilon_final'])
		self.epsilon_interval = float(params['epsilon_interval'])
		self.gamma = float(params['gamma'])

		# training parameters
		self.num_iterations = int(params['num_iterations'])
		self.num_episodes = int(params['num_episodes'])
		self.learning_rate = float(params['learning_rate'])
		self.train_iterations = int(params['train_iterations'])
	def get_epsilon(self):
		return self.epsilon
	def train_step(self, minibatch, tf_session, tf_graph):
		# pdb.set_trace()
		batch_size = len(minibatch)
		# SGD on minibatch
		s, actions, rewards, ss, dones = [np.array(x) for x in zip(*minibatch)]

		qmax_ss = np.amax(self.critic_model.predict(ss), axis=1, keepdims=1)

		assert qmax_ss.shape==(batch_size, 1), qmax_ss.shape
		assert rewards.shape==(batch_size, 1), rewards.shape
		assert (1 - dones).shape==(batch_size, 1), dones.shape

		g = rewards + self.gamma * (1 - dones) * qmax_ss
		g = g.astype('float32')
		assert g.shape==(batch_size, 1), g.shape

		target_q_s = self.critic_model.predict(s)
		target_q_s[np.arange(batch_size), actions.reshape(-1)] = g.reshape(-1)
		assert (target_q_s[np.arange(batch_size), actions.reshape(-1)] == g.reshape(-1)).all()
		# pdb.set_trace()
		with tf_session.as_default():
			with tf_graph.as_default():
				self.actor_model.fit(s,target_q_s,epochs=1,verbose=0,batch_size=batch_size)

	def annealing(self, num_episode, strategy='linear'):
		if strategy == 'linear':
			retval = max(self.epsilon_initial - num_episode * (self.epsilon_initial - self.epsilon_final) / self.num_episodes, self.epsilon_final)
		elif strategy == 'exponential':
			if self.epsilon <= self.epsilon_final:
				return self.epsilon_final
			retval = self.epsilon * self.epsilon_decay
		return retval
	
	def train(self, replay_memory, tf_session, tf_graph, model_path):
		print("> training on episode "+str(self.episode_number))
		for i in range(self.train_iterations):
			minibatch = replay_memory.sample_batch(self.batch_size)
			self.train_step(minibatch, tf_session, tf_graph)
		# pdb.set_trace()
		self.epsilon = self.annealing(self.episode_number)
		with tf_session.as_default():
			with tf_graph.as_default():
				self.critic_model.set_weights(self.actor_model.get_weights())
				if self.episode_number % 10 == 0:
					self.actor_model.save(os.path.join(model_path, 
						"checkpoint_"+str(self.episode_number/10 % 10)+".h5"))
					print("> saving trained actor_model at episode "+str(self.episode_number)+
						" to ingame_models/checkpoint_"+str(self.episode_number/10 % 10)+".h5")
		return self.epsilon


