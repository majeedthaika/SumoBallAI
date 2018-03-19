import os
import numpy as np
from PIL import Image
import importlib.util
from keras.models import load_model
# from .DDQN import DDQN
dir_path = os.path.dirname(os.path.realpath(__file__))

class Environment:
	def __init__(self, env_name, fps=10, frame_callback=None, grayscale=False, normalized=False):
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
		# self.model_path = os.path.join(self.path ,self.env_config["model"])
		self.dataset = self.env_config["dataset"]
		self.action_space = self.env_config["action_space"]
		self.action_names = self.env_config["action_names"]
		self.state_space = self.env_config["state_space"]

		self.is_ingame = False

		try:
			self.model = load_model(self.model_path)
		except OSError as e:
			print("No state prediction!")
			self.model = None

		#in-game model
		# self.ingame_actions = self.env_config["ingame_action_names"]
		# self.ingame_models_path = os.path.join(os.getcwd(), "ingame_models")
		# self.ingame_load_model_path = os.path.join(self.ingame_models_path, self.env_config["ingame_model"])
		# self.replay_memory = []
		# self.BUFFER_SIZE = 5
		# self.state_buffer = []

		# self.ingame_model = 
		# self.ddqn = DDQN(ingame_actions=self.ingame_actions, 
		#                 models_path=self.ingame_models_path, 
		#                 load_path=self.ingame_load_model_path)

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

		# NN-Tr XD lets go

	def render(self):
		img = self.vnc.screen.get_array()
		img = Image.fromarray(img)
		arr_img = self.preprocess(img)
		return np.array([arr_img]), img

	def compress_state(self, curr_state):
		return np.concatenate(curr_state, axis=2)

	def on_frame(self):
		state, im = self.render()
		# print(self.action_names[np.argmax(self.model.predict(np.expand_dims(state,axis=3))[0])])
		screen_type = self.action_names[np.argmax(self.model.predict(np.expand_dims(state,axis=3))[0])]
		
		action_in_game = None
		# if self.is_ingame:
		# 	if len(self.state_buffer) < self.BUFFER_SIZE:
		# 		self.state_buffer = []
		# 		for i in range(self.BUFFER_SIZE):
		# 			self.state_buffer.append(np.expand_dims(state[0], axis=2))

		# 	action_in_game = self.ingame_actions[np.argmax(self.ingame_model.predict(self.state_buffer)[0])]

		self.is_ingame = self.frame_callback(state, im, self.frame_count, screen_type, action_in_game, self.vnc)

		self.frame_count += 1
