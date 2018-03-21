import os, pickle
import numpy as np
from PIL import Image
import importlib.util
from keras.models import load_model
from .DDQN import Trainer, DQN_Model, Replay_Memory
dir_path = os.path.dirname(os.path.realpath(__file__))
import pdb
from threading import Thread, Lock
from keras import backend as K
import tensorflow as tf

class Environment:
	def __init__(self, env_name, fps=10, frame_callback=None, grayscale=False, normalized=False):
		
		self.tf_session = K.get_session() # this creates a new session since one doesn't exist already.
		self.tf_graph = tf.get_default_graph()

		self.fps = fps
		self.frame_count = 0
		self.grayscale = grayscale
		self.normalized = normalized
		self.frame_callback = frame_callback
		self.env_name = env_name
		self.path = os.path.join(dir_path, "..", "contrib", "environments", self.env_name)

		if not os.path.isdir(self.path):
			self.path = os.path.join("contrib", "environments", self.env_name)

			if not os.path.isdir(self.path):
				raise FileExistsError("The specified environment \"%s\" could not be found." % self.env_name)

		self.env_config = self.load_config()
		self.swf = self.env_config["swf"]
		self.model_path = os.path.join(os.getcwd(), "train_screen_model", self.env_config["model"])
		self.dataset = self.env_config["dataset"]
		self.action_space = self.env_config["action_space"]
		self.action_names = self.env_config["action_names"]
		self.state_space = self.env_config["state_space"]

		self.is_ingame = False
		self.mutex = Lock()

		try:
			self.model = load_model(self.model_path)
		except OSError as e:
			print("No state prediction!")
			self.model = None

		#in-game model
		self.ingame_actions = self.env_config["ingame_action_names"]
		self.ingame_action_space = self.env_config["ingame_action_space"]
		self.ingame_models_path = os.path.join(os.getcwd(), "ingame_models")
		self.ingame_load_model_path = os.path.join(self.ingame_models_path, self.env_config["ingame_model"])
		self.BUFFER_SIZE = 4
		self.REPLAY_MAX_SIZE = 5000
		self.replay_memory = Replay_Memory(memory_size=self.REPLAY_MAX_SIZE)
		self.all_rewards = []

		self.ep_buffer = []
		self.prev_state_buffer = []
		self.last_action = None
		self.last_reward = None
		self.ep_reward = 0
		self.state_buffer = []
		self.episode_num = 0

		self.train_target = 0
		self.save_target = 0
		self.burned_in = False

		self.win_screens = {"pink_wins", "purple_wins", "blue_wins", "red_wins", "yellow_wins", "green_wins"}
		self.end_episode = False
		self.in_win_screen = False
		self.epsilon = 0.5
		# self.prev_win_screen = False

		with self.tf_session.as_default():
			with self.tf_graph.as_default():
				self.Actor_Model_class = DQN_Model(action_size=self.ingame_action_space,
					model_path=self.ingame_models_path, filename=None, predict=False)
				self.actor_model = self.Actor_Model_class.get_model()
		
		self.Critic_Model_class = DQN_Model(action_size=self.ingame_action_space,
			model_path=self.ingame_models_path, filename=None, predict=True)
		self.critic_model = self.Critic_Model_class.get_model()

	def load_config(self):
		spec = importlib.util.spec_from_file_location("module.define", os.path.join(self.path, "__init__.py"))
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		return mod.define

	def setup(self, vnc):
		self.vnc = vnc
		self.vnc.send_mouse("Left", (self.vnc.screen.size[0], 0))
		self.vnc.add_callback(1 / self.fps, self.on_frame)

	def preprocess(self, pil_image):
		img = pil_image.resize((self.state_space[0], self.state_space[1]), Image.ANTIALIAS)
		if self.grayscale:
			img = img.convert("L")
		else:
			img = img.convert('RGB')
		data = np.array(img)

		if self.normalized:
			data = data / 255

		return data

	def render(self):
		img = self.vnc.screen.get_array()
		img = Image.fromarray(img)
		arr_img = self.preprocess(img)
		return np.array([arr_img]), img

	def compress_state(self, curr_state):
		return np.concatenate(curr_state, axis=2)

	def on_frame(self):
		state, img = self.render()
		episode_num = self.episode_num
		frame_count = self.frame_count

		self.mutex.acquire()
		if (frame_count >= self.frame_count and self.episode_num == episode_num):
			if (self.last_action):
				self.state_buffer = self.state_buffer[1:]
				self.state_buffer.append(np.expand_dims(state[0], axis=2))
				self.ep_buffer.append((self.compress_state(self.prev_state_buffer), self.last_action, 
					self.last_reward, self.compress_state(self.state_buffer), not self.is_ingame))
				
				if (self.end_episode):
					self.end_episode = False
					# pdb.set_trace()
					if (len(self.ep_buffer) > 3):
						self.replay_memory.append_many(self.ep_buffer)
						self.all_rewards.append(np.array([self.episode_num,self.ep_reward]))
						
						print("Episode #"+str(self.episode_num)+": "+str(self.ep_reward))

						self.ep_buffer = []
						self.prev_state_buffer = []
						self.last_action = None
						self.last_reward = None
						self.ep_reward = 0
						self.state_buffer = []
						self.episode_num += 1
						
						self.mutex.release()

						if len(self.replay_memory.memory) == self.REPLAY_MAX_SIZE:
							if not self.burned_in:
								self.train_target = self.episode_num + 3
								self.save_target = self.episode_num + 10
								self.burned_in = True

							# print(self.episode_num, self.train_target, self.save_target)
							if self.episode_num >= self.train_target:
								self.epsilon = Trainer(self.ingame_action_space, self.critic_model, 
									self.actor_model, self.episode_num).train(self.replay_memory,
									self.tf_session, self.tf_graph, self.ingame_models_path)
								self.train_target = self.episode_num + 3
							if self.episode_num >= self.save_target:
								pickle.dump(self.all_rewards, 
									open(os.path.join(os.getcwd(), "training_rewards.p"), "wb"))
								with self.tf_session.as_default():
									with self.tf_graph.as_default():
										self.critic_model.set_weights(self.actor_model.get_weights())
								self.save_target = self.episode_num + 10
						else:
							print("burn-in progress: "+str(len(self.replay_memory.memory))+
								"/"+str(self.REPLAY_MAX_SIZE))
					else:
						self.ep_buffer = []
						self.prev_state_buffer = []
						self.last_action = None
						self.last_reward = None
						self.ep_reward = 0
						self.state_buffer = []

						self.mutex.release()
				else:
					self.mutex.release()				
			else:
				self.mutex.release()	


			screen_type = self.action_names[np.argmax(self.model.predict(np.expand_dims(state,axis=3))[0])]
			self.in_win_screen = (screen_type in self.win_screens)
			if (self.in_win_screen and not self.end_episode):
				self.end_episode = True
			elif (self.in_win_screen and self.end_episode):
				return

			action_in_game = None
			if self.is_ingame:
				if len(self.state_buffer) < self.BUFFER_SIZE:
					self.state_buffer = []
					for i in range(self.BUFFER_SIZE):
						self.state_buffer.append(np.expand_dims(state[0], axis=2))

				# pdb.set_trace()
				if (np.random.uniform() > self.epsilon):
					action_idx = np.argmax(self.critic_model.predict(
						np.expand_dims(self.compress_state(self.state_buffer), axis=0), batch_size=1)[0])
				else:
					action_idx = np.random.randint(0,self.ingame_action_space)

				action_in_game = self.ingame_actions[action_idx]
				self.prev_state_buffer = self.state_buffer
				self.last_action = action_idx

			self.is_ingame, self.last_reward = self.frame_callback(state, img, self.frame_count, screen_type, 
															action_in_game, self.vnc)
			self.ep_reward += self.last_reward

			if (not self.is_ingame and (screen_type not in self.win_screens)):
				self.ep_buffer = []
				self.prev_state_buffer = []
				self.last_action = None
				self.last_reward = None
				self.ep_reward = 0
				self.state_buffer = []

			self.frame_count += 1
		else:
			self.mutex.release()
