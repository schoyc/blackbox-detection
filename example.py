from keras.datasets import cifar10
from detection import *

import numpy as np

(x_train, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test /  255.0
x_train = x_train / 255.0
perm = np.random.permutation(x_train.shape[0])

benign_queries = x_train[perm[:1000],:,:,:]
suspicious_queries = x_train[perm[-1],:,:,:] * np.random.normal(0, 0.05, (1000,) + x_train.shape[1:])

detector = SimilarityDetector(K=50, training_data=x_train, weights_path="./cifar_encoder.h5")

detector.process(benign_queries)

detections = detector.get_detections()
print("Num detections:", len(detections))
print("Queries per detection:", detections)
print("i-th query that caused detection:", detector.history)

detector.clear_memory()
detector.process(suspicious_queries)
detections = detector.get_detections()
print("Num detections:", len(detections))
print("Queries per detection:", detections)
print("i-th query that caused detection:", detector.history)
