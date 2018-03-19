import threading
from easyprocess import EasyProcess
from pyvirtualdisplay import Display
import os

class GameEnvironment(threading.Thread):
    def __init__(self, display, env):
        threading.Thread.__init__(self)
        #super(self)
        self.display = display
        self.env = env

    def run(self):
        self.vnc(self.display)

    def vnc(self, vnc_display):
        os.environ["DISPLAY"] = vnc_display
        with Display(backend='xvnc', rfbport=5902, size=(640, 480)) as disp:
            with EasyProcess(' '.join(['gnash', os.path.join(self.env.path, self.env.swf), "--width", "640", "--height", "480","--render-mode", "1", "--hide-menubar"])) as proc:
                proc.wait()