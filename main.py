from FlashRL.lib.Game import Game
import sys
import pdb
import time
from pygame.locals import *
# from threading import Thread, Lock

class RunSim:
	def __init__(self, set_player_types=[0]*6, set_deathmatch=False):
		self.player_types  = [0]*6
		self.player_types_dict  = {"None": 0, 
								"CPU1": 1, "CPU2": 2, "CPU3": 3, 
								"Player1": 4, "Player2": 5, "Player3": 6}

		if ("Player3" in set_player_types):
			idx = set_player_types.index("Player3")
			if ("Player1" in set_player_types and set_player_types.index("Player1") < idx):
				self.player_types[idx] += 1
			if ("Player2" in set_player_types and set_player_types.index("Player2") < idx):
				self.player_types[idx] += 1

		if ("Player2" in set_player_types):
			idx = set_player_types.index("Player2")
			if ("Player1" in set_player_types and set_player_types.index("Player1") < idx):
				self.player_types[idx] += 1

		self.set_player_types = [self.player_types_dict[player]for player in set_player_types]

		self.set_deathmatch = set_deathmatch
		self.is_deathmatch = False

		self.is_ingame = False
		self.win_screens = {"pink_wins", "purple_wins", "blue_wins", "red_wins", "yellow_wins", "green_wins"}

		self.curr_actions = set()
		# self.action_mutex = Lock()

		Game("sumoball", fps=10, frame_callback=self.on_frame, grayscale=True, normalized=True)

	def release_all_keys(self, vnc):
		for k in self.curr_actions:
			vnc.send_release(k)
		self.curr_actions.clear()

	def add_new_keys(self, vnc, keys):
		for k in keys:
			vnc.send_press(k)
			self.curr_actions.add(k)

	def on_frame(self, state, img, frame, screen_type, action_in_game, vnc, run_episode):
		# print(vnc.screen.cursor_loc)
		# pdb.set_trace()
		# print(screen_type)

		frame_reward = 0
		self.is_ingame = False
		if not run_episode:
			return self.is_ingame, frame_reward

		if (screen_type == "load_screen"):
			vnc.send_mouse("Left", (160, 207)) # click start
			vnc.send_mouse("Left", (160, 207)) # need twice to actually press button
		elif (screen_type == "start_screen"):
			vnc.send_mouse("Left", (160, 207)) # click on screen
			vnc.send_mouse("Left", (160, 207)) # need twice to actually press button
		elif (screen_type == "selection_screen"):
			if (self.player_types[0] <= self.set_player_types[0]):
				vnc.send_mouse("Left", (40, 50)) # click red ball
				self.player_types[0] = self.player_types[0] + 1
			elif (self.player_types[1] <= self.set_player_types[1]):
				vnc.send_mouse("Left", (100, 50)) # click blue ball
				self.player_types[1] = self.player_types[1] + 1
			elif (self.player_types[2] <= self.set_player_types[2]):
				vnc.send_mouse("Left", (40, 120)) # click green ball
				self.player_types[2] = self.player_types[2] + 1
			elif (self.player_types[3] <= self.set_player_types[3]):
				vnc.send_mouse("Left", (100, 120)) # click yellow ball
				self.player_types[3] = self.player_types[3] + 1
			elif (self.player_types[4] <= self.set_player_types[4]):
				vnc.send_mouse("Left", (30, 190)) # click pink ball
				self.player_types[4] = self.player_types[4] + 1
			elif (self.player_types[5] <= self.set_player_types[5]):
				vnc.send_mouse("Left", (100, 190)) # click purple ball
				self.player_types[5] = self.player_types[5] + 1
			elif (self.is_deathmatch != self.set_deathmatch):
				vnc.send_mouse("Left", (225, 20)) # click game type
				vnc.send_mouse("Left", (225, 20)) # need twice to actually press button
				self.is_deathmatch = self.set_deathmatch
			else:
				vnc.send_mouse("Left", (295, 232)) # go to game
				vnc.send_mouse("Left", (295, 232)) # need twice to actually press button
		elif (screen_type in self.win_screens):
			# vnc.send_press(K_RETURN) #restart game
			pass
		else:
			self.is_ingame = True
			# in game
			# print(action_in_game)
			# print(self.curr_actions)
			# self.action_mutex.acquire()
			if (action_in_game == "UP"):
				if "w" not in self.curr_actions or len(self.curr_actions) != 1:
					self.release_all_keys(vnc)
					self.add_new_keys(vnc, ["w"])
			elif (action_in_game == "UP_RIGHT"):
				if "w" not in self.curr_actions or "d" not in self.curr_actions or len(self.curr_actions) != 2:
					self.release_all_keys(vnc)
					self.add_new_keys(vnc, ["w","d"])
			elif (action_in_game == "RIGHT"):
				if "d" not in self.curr_actions or len(self.curr_actions) != 1:
					self.release_all_keys(vnc)
					self.add_new_keys(vnc, ["d"])
			elif (action_in_game == "DOWN_RIGHT"):
				if "s" not in self.curr_actions or "d" not in self.curr_actions or len(self.curr_actions) != 2:
					self.release_all_keys(vnc)
					self.add_new_keys(vnc, ["s","d"])
			elif (action_in_game == "DOWN"):
				if "s" not in self.curr_actions or len(self.curr_actions) != 1:
					self.release_all_keys(vnc)
					self.add_new_keys(vnc, ["s"])
			elif (action_in_game == "DOWN_LEFT"):
				if "s" not in self.curr_actions or "a" not in self.curr_actions or len(self.curr_actions) != 2:
					self.release_all_keys(vnc)
					self.add_new_keys(vnc, ["s","a"])
			elif (action_in_game == "LEFT"):
				if "a" not in self.curr_actions and len(self.curr_actions) != 1:
					self.release_all_keys(vnc)
					self.add_new_keys(vnc, ["a"])
			else:
				if "w" not in self.curr_actions and "a" not in self.curr_actions and len(self.curr_actions) != 2:
					self.release_all_keys(vnc)
					self.add_new_keys(vnc, ["w","a"])
			# self.action_mutex.release()

			if (screen_type == "red_wins"):
				frame_reward = 500
				self.is_ingame = False
			elif (screen_type == "blue_wins"):
				frame_reward = -500
				self.is_ingame = False
			elif (screen_type == "green_wins"):
				frame_reward = -500
				self.is_ingame = False
			elif (screen_type == "yellow_wins"):
				frame_reward = -500
				self.is_ingame = False
			elif (screen_type == "pink_wins"):
				frame_reward = -500
				self.is_ingame = False
			elif (screen_type == "purple_wins"):
				frame_reward = -500
				self.is_ingame = False
			else:
				frame_reward = 1
		return self.is_ingame, frame_reward


RunSim(set_player_types=["Player2", "CPU1", "None", "None", "None", "None"], set_deathmatch=False)