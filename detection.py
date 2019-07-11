import numpy as np
import tensorflow as tf
import sklearn.metrics.pairwise as pairwise

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from collections import OrderedDict

class Detector(object):

    def __init__(self, K, threshold=None, training_data=None, chunk_size=1000, weights_path="./encoder_1.h5"):
        self.K = K
        self.threshold = threshold
        self.training_data = training_data

        if self.threshold is None and self.training_data is None:
            raise ValueError("Must provide explicit detection threshold or training data to calculate threshold!")

        self._init_encoder(weights_path)

        if self.training_data is not None:
            print("Explicit threshold not provided...calculating threshold for K = %d" % K)
            _, self.thresholds = calculate_thresholds(self.training_data, self.K, self.encode, up_to_K=False)
            self.threshold = self.thresholds[-1]
            print("K = %d; set threshold to: %f" % (K, self.threshold))

        self.num_queries = 0
        self.buffer = []
        self.memory = []
        self.chunk_size = chunk_size

        self.history = [] # Tracks number of queries (t) when attack was detected
        self.history_by_attack = []
        self.detected_dists = [] # Tracks knn-dist that was detected

    def _init_encoder(self, weights_path):
        self.encode = lambda x : x
        raise NotImplementedError("Must implement your own encode function!")

    def process(self, queries):
        queries = self.encode(queries)
        for query in queries:
            self.process_query(query)

    def process_query(self, query):

        if len(self.memory) == 0 and len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            return False

        k = self.K
        all_dists = []

        if len(self.buffer) > 0:
            queries = np.stack(self.buffer, axis=0)
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        for queries in self.memory:
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        dists = np.concatenate(all_dists)
        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        self.buffer.append(query)
        self.num_queries += 1

        if len(self.buffer) >= self.chunk_size:
            self.memory.append(np.stack(self.buffer, axis=0))
            self.buffer = []

        # print("[debug]", num_queries_so_far, k_avg_dist)
        is_attack = k_avg_dist < self.threshold
        if is_attack:
            self.history.append(self.num_queries)
            # self.history_by_attack.append(num_queries_so_far + 1)
            self.detected_dists.append(k_avg_dist)
            # print("[encoder] Attack detected:", str(self.history), str(self.detected_dists))
            self.clear_memory()

    def clear_memory(self):
        self.buffer = []
        self.memory = []

    def get_detections(self):
        history = self.history
        epochs = []
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])

        return epochs

def calculate_thresholds(training_data, K, encoder=lambda x: x, P=1000, up_to_K=False):
    data = encoder(training_data)
    
    distances = []
    for i in range(data.shape[0] // P):
        distance_mat = pairwise.pairwise_distances(data[i * P:(i+1) * P,:], Y=data)
        distance_mat = np.sort(distance_mat, axis=-1)
        distance_mat_K = distance_mat[:,:K]
        
        distances.append(distance_mat_K)
    distance_matrix = np.concatenate(distances, axis=0)
    
    start = 0 if up_to_K else K

    THRESHOLDS = []
    K_S = []
    for k in range(start, K + 1):
        dist_to_k_neighbors = distance_matrix[:,:k+1]
        avg_dist_to_k_neighbors = dist_to_k_neighbors.mean(axis=-1)
        
        threshold = np.percentile(avg_dist_to_k_neighbors, 0.1)
        
        K_S.append(k)
        THRESHOLDS.append(threshold)

    return K_S, THRESHOLDS

class L2Detector(Detector):
    def _init_encoder(self, weights_path):
        self.encode = lambda x : x.reshape((x.shape[0], -1))

class SimilarityDetector(Detector):
    def _init_encoder(self, weights_path):
        encoder = cifar10_encoder()
        encoder.load_weights(weights_path, by_name=True)
        self.encoder = encoder
        self.encode = lambda x : encoder.predict(x)

class CIFAR10Detectors():
    def __init__(self, active=True):
        self.active = active

        detectors = [
            ("similarity", SimilarityDetector(threshold=1.44, K=50, weights_path="./encoders/cifar_encoder.h5")),
            ("l2", L2Detector(threshold=5.069, K=50)),
            ("sim-no-brightness", SimilarityDetector(threshold=1.56, K=50, weights_path="./encoders/cifar_encoder_no_brightness.h5")),
        ]

        self.detectors = OrderedDict({})
        for d_name, detector in detectors:
            self.detectors[d_name] = detector

    def process(self, queries):
        if not self.active:
            return

        for _, detector in self.detectors.items():
            detector.process(queries)

    def process_query(self, query):
        if not self.active:
            return

        for _, detector in self.detectors.items():
            detector.process_query(query)



def cifar10_encoder(encode_dim=256):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_1', input_shape=(32, 32, 3)))
    model.add(Activation('relu', name='activation_1'))
    model.add(Conv2D(32, (3, 3), name='conv2d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv2d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(Conv2D(64, (3, 3), name='conv2d_4'))
    model.add(Activation('relu', name='activation_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
    model.add(Dropout(0.25, name='dropout_2'))

    model.add(Flatten(name='flatten_1'))
    model.add(Dense(512, name='dense_1'))
    model.add(Activation('relu', name='activation_5'))
    model.add(Dropout(0.5, name='dropout_3'))
    model.add(Dense(encode_dim, name='dense_encode'))
    model.add(Activation('linear', name='encoding'))

    return model

