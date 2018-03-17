import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras import backend as K
from keras.layers import Input, Dense, Add, Subtract, Average, RepeatVector, Lambda, Activation, Conv2D
from keras.backend import mean, update_sub, ones, shape, sum, expand_dims
from keras.models import Sequential, Model
from keras.optimizers import Adam
from collections import deque
import json, os, errno
import random
from shutil import copyfile

import threading
import time
import collections

np.set_printoptions(threshold='nan')

class ProducerConsumer(object):
	def __init__(self, size):
		self.buffer = collections.deque([], size)
		self.mutex = threading.Lock()
		self.notFull = threading.Semaphore(size)
		self.notEmpty = threading.Semaphore(0)

	def append(self, data):
		self.notFull.acquire()
		with self.mutex:
			self.buffer.append(data)
		self.notEmpty.release()

	def take(self):
		self.notEmpty.acquire()
		with self.mutex:
			data = self.buffer.popleft()
		self.notFull.release()
		return data

class QNetwork(object):

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, state_size, action_size, batch_size, lr=0.001, gamma=0.99):
		self.state_size = state_size
		self.action_size = action_size
		self.lr = lr
		self.gamma = gamma
		self.input_shape = (self.state_size,)
		self.batch_size = batch_size

	"""
	get the q values of all possible actions in given state
	"""
	def evaluate(self, state):
		if state.shape[0] == self.batch_size:
			batch_size = state.shape[0]
		else:
			state = np.expand_dims(state, 0)
			batch_size = 1
		qs = self.model.predict(state, batch_size=batch_size, verbose=0)
		return qs  # returns q values

	"""
	Train a step, with minibatch extracted outside
	Each item in minibatch should be a transaction, i.e.
	(state, action, reward, next_state, done) tuple
	done is 1 if terminal, or 0 if not
	"""
	def train_step(self, minibatch):
		batch_size = len(minibatch)
		# SGD on minibatch
		s, actions, rewards, ss, dones = \
				[np.array(x) for x in zip(*minibatch)]

		qmax_ss = np.amax(self.model.predict(ss), axis=1, keepdims=1)

		assert qmax_ss.shape==(batch_size, 1), qmax_ss.shape
		assert rewards.shape==(batch_size, 1), rewards.shape
		assert (1 - dones).shape==(batch_size, 1), dones.shape
		g = rewards + self.gamma * (1 - dones) * qmax_ss
		g = g.astype('float32')
		assert g.shape==(batch_size, 1), g.shape

		target_q_s = self.model.predict(s)
		target_q_s[np.arange(batch_size), actions.reshape(-1)] = g.reshape(-1)
		assert (target_q_s[np.arange(batch_size), actions.reshape(-1)] == g.reshape(-1)).all()

	def save_model_weights(self, path):
		self.model.save_weights(path)

	def load_model_weights(self, path):
		self.model.load_weights(path)

	def save_model(self, path):
		self.model.save(path)

	def load_model(self, path):
		self.model = load_model('path')

class DuelingDeepQNetwork(QNetwork):
	def __init__(self, *args, **kwargs):
		super(DuelingDeepQNetwork, self).__init__(*args, **kwargs)

		inputs = Input(shape = (84, 84, 4))
		c1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
		c2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(c1)
		c3 = Conv2D(64, (3, 3), activation = 'relu')(c2)
		f1 = Flatten()(c3)

		v1 = Dense(512, activation = 'relu')(f1)
		valFunc = Dense(1, activation = 'relu')(v1)

		a1 = Dense(512, activation = 'relu')(f1)
		advFunc = Dense(self.action_size, activation = 'relu')(a1)
		advFuncMean = Lambda(lambda x: mean(x, axis=1))(advFunc)
		advFuncOut = Lambda(lambda (x, y): x - expand_dims(y, axis=1))([advFunc, advFuncMean])

		merged = Lambda(lambda (x, y): x + y)([valFunc, advFuncOut])

		model = Model(inputs=inputs, outputs=merged)
		optimizer = keras.optimizers.Adam(lr=self.lr)
		model.compile(optimizer=optimizer,
			  loss='mean_squared_error',
			  metrics=['accuracy'])

		self.model = model

class Replay_Memory():

	def __init__(self, memory_size=50000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
		# self.memory = deque(maxlen=memory_size)
		self.tail = 0
		self.memory_size = memory_size
		self.memory = []


	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
		# You will feed this to your model to train.
		minibatch = random.sample(self.memory, batch_size)
		return minibatch

	def append(self, transition):
		# Appends transition to the memory.
		# self.memory.append(transition)
		if len(self.memory) < self.memory_size:
			self.memory.append(transition)
		else:
			self.memory[self.tail] = transition
			self.tail = (self.tail + 1) % self.memory_size

	def __len__(self):
		return len(self.memory)


class DQN_Agent():

	# In this class, we will implement functions to do the following.
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
	#        (a) Epsilon Greedy Policy.
	#         (b) Greedy Policy.
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, params, ingame_actions, models_path, load_path, buffer):
		self.buffer = buffer

		# monitor_env = gym.wrappers.Monitor(self.env, '.', force=True)
		self.action_space = np.arange(len(ingame_actions))
		self.action_size = len(ingame_actions)
		self.state_size = (84, 84, 4)
		self.lr = float(params['learning_rate'])

		# set up the q network
		self.q_network = DuelingDeepQNetwork(self.state_size, self.action_size, 
											params['batch_size'], lr=self.lr)

		# policy is a list of length nA, with the probability of action
		# for each A
		self.epsilon = float(params['epsilon'])
		self.epsilon_decay = float(params['epsilon_decay'])
		self.epsilon_initial = float(params['epsilon_initial'])
		self.epsilon_final = float(params['epsilon_final'])
		# self.epsilon_interval = float(params['epsilon_interval'])
		self.gamma = float(params['gamma'])

		# training parameters
		self.num_iterations = int(params['num_iterations'])
		self.num_episodes = int(params['num_episodes'])
		self.learning_rate = float(params['learning_rate'])

		# set up memory
		self.memory = Replay_Memory(int(params['replay_memory_size']))
		self.batch_size = int(params['batch_size'])
		self.burn_in_size = int(params['burn_in_size'])

	def state_producer(self, state):
		self.buffer.append(state)

	def state_consumer(self):
		return self.buffer.take()

	"""
	Return a action distribution based on given q_value
	"""
	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		# Assuming q values is a list of length nA and is indexed by actions
		policy = np.ones(self.action_size) * self.epsilon/self.action_size
		best_action = np.argmax(q_values) #.index(max_q_value)
		policy[best_action] += 1 - self.epsilon

		return policy

	"""
	Return a one-hot action distribution based on given q_value
	"""
	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		# Assuming q values is a list of length nA and is indexed by actions
		policy = np.zeros(self.action_size)
		best_action = np.argmax(q_values)# q_values.index(max_q_value)
		policy[best_action] = 1

		return policy

	def sample_action(self, state, policy_type):
		q_values = self.q_network.evaluate(state)

		if policy_type == 'epsilon_greedy':
			policy = self.epsilon_greedy_policy(q_values)
		elif policy_type == 'greedy':
			policy = self.greedy_policy(q_values)

		action = np.random.choice(self.action_size, p = policy)
		return action

	def annealing(self, num_episode, iterations, strategy='linear', step='episodes'):
		if step == 'episodes':
			if strategy == 'linear':
				retval = max(self.epsilon_initial - num_episode * (self.epsilon_initial - self.epsilon_final) / self.num_episodes, self.epsilon_final)
			elif strategy == 'exponential':
				if self.epsilon <= self.epsilon_final:
					return self.epsilon_final
				retval = self.epsilon * self.epsilon_decay
		# elif step == 'iterations':
		# 	if strategy == 'linear':
		# 		retval = max(self.epsilon_initial - iterations * (self.epsilon_initial - self.epsilon_final) / self.epsilon_interval, self.epsilon_final)
		return retval


	def make_transition(self, state, action, reward, next_state, is_terminal):
		if type(reward) is not np.ndarray:
			reward = np.array([reward])
		if type(action) is not np.ndarray:
			action = np.array([action])
		if type(is_terminal) is not np.ndarray:
			is_terminal = np.array([is_terminal])
		return (state, action, reward, next_state, is_terminal)

	# def convert_to_grayscale(self, image):
	# 	resized_image = cv2.resize(image, (84, 84))
	# 	gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
	# 	return gray_image	

	def compress_state(self, curr_state):
		return np.concatenate(curr_state, axis=2)

	def train(self):
		if len(self.memory) == 0:
			self.burn_in_memory()

		sum_reward = 0
		iterations = 0
		for episode in xrange(1, self.num_episodes + 1):
			state_block = [self.convert_to_grayscale(self.env.reset())]
			for i in range(3):
				state_block.append(self.convert_to_grayscale(self.env.step(0)[0]))
			state = self.compress_state(state_block)
			episode_reward = 0
			while 1:
				# print (iterations)
				iterations += 1
				self.epsilon = self.annealing(episode, iterations)
				# take an action
				action = self.sample_action(state, 'epsilon_greedy', True)
				next_state, reward, is_terminal, _ = self.env.step(action)

				next_state_block = state_block[1:]
				next_state_block.append(self.convert_to_grayscale(next_state))
				
				next_state = self.compress_state(next_state_block)

				episode_reward += reward
				transition = self.make_transition(state, action, reward, next_state, is_terminal)

				# add it in memory
				self.memory.append(transition)

				minibatch = self.memory.sample_batch(self.batch_size)

				self.q_network.train_step(minibatch)

				if is_terminal:
					break
				state = next_state
			print("episode: {}, reward: {}".format(episode, episode_reward))
			sum_reward += episode_reward

			if episode % 100 == 0:
				avg_test_reward = self.test(20)
				summary_line = "Episode: {}\nPrevious episode's reward: {}\nAverage reward for last 100 episodes: {}\nAverage test reward for 20 episodes: {}\nEpsilon: {}".format(episode, episode_reward, sum_reward/100, avg_test_reward, self.epsilon)
				print (summary_line)
				sum_reward = 0

				ckpt = self.filename + '-' + str(int(episode/100))
				self.q_network.save_model_weights(ckpt)

	def test(self, test_num_episode, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.


		# print (self.deepQ)
		sum_reward = 0

		if model_file != None:
			self.q_network.load_model_weights(model_file)

		for episode in range(test_num_episode):
			# state = self.env.reset()
			test_state = [self.convert_to_grayscale(self.env.reset())]
			for i in range(3):
				test_state.append(self.convert_to_grayscale(self.env.step(0)[0]))
			state = self.compress_state(test_state)
			cummulative_reward = 0
			# print (state)

			for num_iter in xrange(1000):
				if self.render == True:
					self.env.render()

				action = self.sample_action(state, 'greedy')

				next_state, reward, done, info = self.env.step(action)

				new_test_state = test_state[1:]
				new_test_state.append(self.convert_to_grayscale(next_state))
				
				state = self.compress_state(new_test_state)
				# cummulative_reward += self.gamma**num_iter * reward
				# print (state, reward, done, info, cummulative_reward)

				cummulative_reward += reward
				if done == True:
					# print ('break')
					sum_reward += cummulative_reward
					break

		# print ()
		return sum_reward/float(test_num_episode)

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		print('burning in {} steps'.format(self.burn_in_size))
		burn_state = [self.state_consumer()]
		for i in range(3):
			burn_state.append(burn_state[0])
		state = self.compress_state(burn_state)

		for step in xrange(self.burn_in_size):
			# take random action
			action = self.action_space.sample()
			next_state, reward, is_terminal, _ = self.env.step(action)

			new_burn_state = burn_state[1:]
			new_burn_state.append(self.convert_to_grayscale(next_state))

			next_state = self.compress_state(new_burn_state)

			transition = self.make_transition(state, action, reward, next_state, is_terminal)
			self.memory.append(transition)
			if is_terminal:
				burn_state = [self.convert_to_grayscale(self.env.reset())]
				for i in range(3):
					burn_state.append(self.convert_to_grayscale(self.env.step(0)[0]))
				state = self.compress_state(burn_state)
			else:
				state = next_state

class DDQN:
	def __init__(self, ingame_actions, models_path=None, load_path=None):
		self.BUFFER_SIZE = 2
		self.state_buffer = ProducerConsumer(self.BUFFER_SIZE)
		self.action_buffer = ProducerConsumer(self.BUFFER_SIZE)

		self.params = {
			"epsilon_initial": 0.5,
			"epsilon_final": 0.05,
			"epsilon": 0.5,
			"epsilon_decay": .995,
			"epsilon_interval": 100000,
			"gamma": 0.5,
			"num_iterations": 200,
			"num_episodes": 3000,
			"learning_rate": 0.0001,
			"batch_size": 32,
			"replay_memory_size": 1000000,
			"burn_in_size": 10000,
		}
		
		# Setting the session to allow growth, so it doesn't allocate all GPU memory.
		gpu_ops = tf.GPUOptions(allow_growth=True)
		config = tf.ConfigProto(gpu_options=gpu_ops)
		sess = tf.Session(config=config)

		# Setting this as the default tensorflow session.
		keras.backend.tensorflow_backend.set_session(sess)

		# You want to create an instance of the DQN_Agent class here, and then train / test it.
		dqn_agent = DQN_Agent(self.params, ingame_actions, models_path, load_path, self.buffer)

	def state_producer(self, state):
		self.state_buffer.append(state)

	def state_consumer(self):
		return self.state_buffer.take()

	def action_producer(self, state):
		self.buffer.append(state)

	def action_consumer(self):
		return self.buffer.take()

	def train(self):
		t = threading.Thread(target=dqn_agent.train, args=(self.buffer,))
		t.start()
		t.join()