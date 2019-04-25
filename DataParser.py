import pandas as pd
import numpy as np


class DataParser:
    def __init__(self, path, n_features):
        self._path = path
        self._n_features = n_features

    def parse(self):
        feature_set = []
        label_set = []
        with open(self._path) as file:
            for line in file.readlines():
                fields = np.zeros(self._n_features, np.float)
                data = line.replace('\n', '').split(' ')
                label_set.append(data[0])
                for field in data[1:]:
                    if len(field) > 0:
                        t = field.split(':')
                        fields[int(float(t[0])) - 1] = float(t[1])
                feature_set.append(fields)

        label_set = np.array(label_set)
        label_set = label_set[:, np.newaxis]
        label_set = label_set
        return pd.DataFrame(data=label_set, columns=["label"]), pd.DataFrame(feature_set, dtype=float)
