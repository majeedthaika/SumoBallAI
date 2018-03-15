from FlashRL import Game


def on_frame(state, frame, type, vnc):
    pass




Game("sumoball", fps=10, frame_callback=on_frame, grayscale=True, normalized=True)
