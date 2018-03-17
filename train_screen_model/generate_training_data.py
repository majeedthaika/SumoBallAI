from FlashRL.lib.Game import Game
import numpy as np
import PIL

def on_frame(state, im, frame, type, vnc):
	print(frame)
	np.save("images/frame"+str(frame)+".npy", state[0])
	im.save("images/frame"+str(frame)+".jpg")

Game("sumoball", fps=10, frame_callback=on_frame, grayscale=True, normalized=True)
