import glob
import os
import pickle
from model import Model
from PIL import Image
import numpy as np
import pickle
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class Trainer:

    def __init__(self):
        self.env_training_data = None
        self.load_training_data()

    def load_training_data(self):
        image_class_path = os.path.join(os.getcwd(), "images")
        image_classes = [x for x in os.listdir(image_class_path)]
        image_classes_path = [os.path.join(image_class_path, x) for x in image_classes]
        n_classes = len(image_classes)
        print(n_classes)
        X = []
        Y = []
        for i, class_path in enumerate(image_classes_path):
            print(i, class_path)
            files = [os.path.join(class_path, x) for x in os.listdir(class_path)]

            for image_path in files:
                if not image_path.endswith(".npy"):
                    continue

                # Preprocess image
                np_img = np.load(image_path)

                # Create class label
                y = np.zeros(shape=(n_classes, ))
                y[i] = 1

                # Add to dataset
                X.append(np_img)
                Y.append(y)

        X = np.array(X)
        Y =  np.array(Y)
        print(X.shape, Y.shape)
        # pickle.dump((X,Y), open(os.path.join(os.getcwd(), "dataset.p"), "wb"))
        # self.env_training_data = pickle.load(open(os.path.join(os.getcwd(), "dataset.p"), "rb"))
        self.env_training_data = (X,Y)

    def train(self):
        m = Model(self.env_training_data, os.path.join(os.getcwd(), "model.h5"))
        m.train()
        

if __name__ == "__main__": 
    trainer = Trainer()
    trainer.train()
        