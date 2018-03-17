from FlashRL.lib.Game import Game
import sys

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

		Game("sumoball", fps=10, frame_callback=self.on_frame, grayscale=True, normalized=True)

	def on_frame(self, state, img, frame, state_type, vnc):
		# print(vnc.screen.cursor_loc)

		if not self.is_ingame: # auto-game loader
			if (state_type == "load_screen"):
				vnc.send_mouse("Left", (120, 130)) # click start
				vnc.send_mouse("Left", (120, 130)) # need twice to actually press button
			elif (state_type == "start_screen"):
				vnc.send_mouse("Left", (100, 100)) # click on screen
				vnc.send_mouse("Left", (100, 100)) # need twice to actually press button
			elif (state_type == "selection_screen"):
				if (self.player_types[0] <= self.set_player_types[0]):
					vnc.send_mouse("Left", (40, 30)) # click red ball
					self.player_types[0] = self.player_types[0] + 1
				elif (self.player_types[1] <= self.set_player_types[1]):
					vnc.send_mouse("Left", (80, 30)) # click blue ball
					self.player_types[1] = self.player_types[1] + 1
				elif (self.player_types[2] <= self.set_player_types[2]):
					vnc.send_mouse("Left", (40, 75)) # click green ball
					self.player_types[2] = self.player_types[2] + 1
				elif (self.player_types[3] <= self.set_player_types[3]):
					vnc.send_mouse("Left", (80, 75)) # click yellow ball
					self.player_types[3] = self.player_types[3] + 1
				elif (self.player_types[4] <= self.set_player_types[4]):
					vnc.send_mouse("Left", (40, 120)) # click pink ball
					self.player_types[4] = self.player_types[4] + 1
				elif (self.player_types[5] <= self.set_player_types[5]):
					vnc.send_mouse("Left", (80, 120)) # click purple ball
					self.player_types[5] = self.player_types[5] + 1
				elif (self.is_deathmatch != self.set_deathmatch):
					vnc.send_mouse("Left", (155, 12)) # click game type
					vnc.send_mouse("Left", (155, 12)) # need twice to actually press button
					self.is_deathmatch = self.set_deathmatch
				else:
					vnc.send_mouse("Left", (198, 144)) # go to game
					vnc.send_mouse("Left", (198, 144)) # need twice to actually press button
					self.is_ingame = True
			else:
				pass # in game

		return self.is_ingame


RunSim(set_player_types=["Player1", "CPU1", "None", "None", "None", "None"], set_deathmatch=False)