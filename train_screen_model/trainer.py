import glob
import os
import pickle
from model import Model
dir_path = os.path.dirname(os.path.realpath(__file__))

class Trainer:

    def __init__(self):
        self.env_training_data = None
        self.load_training_data()

    def load_training_data(self):
        self.env_training_data = pickle.load(open(os.path.join(os.getcwd(), "dataset.p"), "rb"))

    def train(self):
        m = Model(self.env_training_data, os.path.join(os.getcwd(), "model.h5"))
        m.train()
        

if __name__ == "__main__": 
    trainer = Trainer()
    trainer.train()
        