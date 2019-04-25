from collections import defaultdict, Counter

import numpy as np
import pandas as pd


class AxesSampler:
    def __init__(self, y, rand: np.random.RandomState = None, r=1):
        self.y = self._process_y(y)
        self.rand = rand if rand is not None else np.random.RandomState()
        self.r = r
        self.current_r = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self._next()

    @staticmethod
    def _process_y(y):
        tmp = defaultdict(list)
        for i, item in enumerate(y):
            tmp[item].append(i)
        return tmp

    def _next(self):
        if self.current_r < self.r:
            self.current_r += 1
            tmp = list(self.y)
            first_class = self.rand.choice(tmp)
            tmp.remove(first_class)
            second_class = self.rand.choice(tmp)
            return self.rand.choice(self.y[first_class]), self.rand.choice(self.y[second_class])
        else:
            raise StopIteration()


class Params:
    def __init__(self, gini_q=1, x_i=None, x_j=None, split_point=0, similarities=None):
        self.gini_q = gini_q
        self.x_i = x_i
        self.x_j = x_j
        self.split_point = split_point
        self.similarities = similarities


class Node:
    def __init__(self, depth, similarity_function=np.dot, n_axes=1,
                 max_depth=None, random_state=None):
        self._depth = depth
        self._sim_function = similarity_function
        self._r = n_axes
        self._max_depth = max_depth
        self._random = np.random.RandomState() if random_state is None else random_state
        self._left: Node = None
        self._right: Node = None
        self._x_i = None
        self._x_j = None
        self._split_point = None
        self.prediction = None

    @staticmethod
    def _split_gini(total_left, total_right, left_val: defaultdict, right_val: Counter):
        left_gini = 1 - sum(left_val[key]**2 for key in left_val) / total_left**2
        right_gini = 1 - sum(right_val[key]**2 for key in right_val) / total_right**2

        return (total_left * left_gini + total_right * right_gini) / (total_left + total_right)

    def _find_split_point(self, X, y, x_i, x_j):
        similarities = [self._sim_function(x_k, x_j) - self._sim_function(x_k, x_i) for x_k in X]
        indices = sorted((i for i in range(len(y)) if not np.isnan(similarities[i])),
                         key=lambda x: similarities[x])

        best_params = Params()
        total_val = Counter(y)
        left_val = defaultdict(lambda: 0)
        n = len(indices)
        for i in range(n - 1):
            left_val[y[indices[i]]] += 1
            right_val = Counter(total_val)
            right_val.subtract(left_val)
            split_gini = self._split_gini(i + 1, n - i - 1, left_val, right_val)
            if split_gini < best_params.gini_q:
                best_params.gini_q = split_gini
                best_params.x_i = x_i
                best_params.x_j = x_j
                # best_params.split_point = (similarities[indices[i]] + similarities[indices[i + 1]]) / 2
                best_params.split_point = similarities[indices[i]]
                best_params.similarities = similarities
        return best_params

    def fit(self, X, y):
        self.prediction = list(set(y))
        if len(self.prediction) == 1:
            self.prediction = self.prediction[0]
            return self

        if self._max_depth is not None and self._depth >= self._max_depth:
            return self

        best_params = Params()
        for i, j in AxesSampler(y, self._random, self._r):
            params = self._find_split_point(X, y, X[i], X[j])
            if params.gini_q < best_params.gini_q:
                best_params = params

        if best_params.gini_q < 1:
            self._x_i = best_params.x_i
            self._x_j = best_params.x_j
            self._split_point = best_params.split_point

            X_left = X[best_params.similarities <= self._split_point, :]
            X_right = X[best_params.similarities > self._split_point, :]
            y_left = y[best_params.similarities <= self._split_point]
            y_right = y[best_params.similarities > self._split_point]

            if len(y_left) > 0 and len(y_right) > 0:
                self._left = Node(self._depth + 1,
                                  self._sim_function,
                                  self._r,
                                  self._max_depth,
                                  self._random).fit(X_left, y_left)

                self._right = Node(self._depth + 1,
                                   self._sim_function,
                                   self._r,
                                   self._max_depth,
                                   self._random).fit(X_right, y_right)
        return self

    def predict_probability_once(self, x):
        if self._left is None and self._right is None:
            return self.prediction, self._depth
        elif self._sim_function(x, self._x_j) - self._sim_function(x, self._x_i) <= self._split_point:
            return self._left.predict_probability_once(x)
        elif self._sim_function(x, self._x_j) - self._sim_function(x, self._x_i) > self._split_point:
            return self._right.predict_probability_once(x)
        else:
            return self.prediction, self._depth

    def print(self):
        if self._left is not None:
            self._left.print()
        if self._right is not None:
            self._right.print()
        if self._left is None and self._right is None:
            print((self._depth, self.prediction))

    def predict_probability(self, X):
        return [self.predict_probability_once(x) for x in X.to_numpy()]


class SimilarityForest:
    def __init__(self, n_estimators=20, similarity_function=np.dot, n_axes=1,
                 max_depth=None, random_state=None, frac=None):
        self._n_estimators = n_estimators
        self._sim_function = similarity_function
        self._n_axes = n_axes
        self._max_depth = max_depth
        self._random = random_state
        assert frac is None or 0 < frac <= 1
        self.frac = frac
        self._trees = None
        self.classes = []

    def _random_sample(self, X: pd.DataFrame, y: pd.DataFrame, n=None, frac=None):
        """
        A random sampler.

        Returns random sample from given dataset.

        :param1 X: dataset
        :param2 y: labels
        :return: tuple
        """
        if n is not None and frac is not None:
            raise ValueError("Cannot use n and frac in the same time")

        if frac is None and n is None:
            return X.to_numpy(), y.to_numpy().T[0]

        sample_x = X.sample(n=n, frac=frac, random_state=self._random)
        sample_y = y.loc[sample_x.index]
        return sample_x.to_numpy(), sample_y.to_numpy().T[0]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        assert len(X) == len(y)

        self.classes = sorted(set(y.to_numpy().T[0]))

        self._trees = [Node(1,
                            self._sim_function,
                            self._n_axes,
                            self._max_depth,
                            self._random).fit(*self._random_sample(X, y, frac=self.frac))
                       for _ in range(self._n_estimators)]

    def predict_probability(self, X):
        probs = [tree.predict_probability(X) for tree in self._trees]
        probs = np.array(probs).T
        depths = []
        result = []
        for line in probs[0]:
            line = [i if type(i) != list else np.random.choice(i) for i in line]
            c = Counter(line)
            for k in c:
                c[k] /= len(line)
            tmp = [[k, c[k]] if k in c else [k, 0] for k in self.classes]
            result.append(tmp)

        for line in probs[1]:
            depths.append(sum(line)/len(line))

        for i in range(len(result)):
            result[i] = [result[i], depths[i]]

        return np.array(result)

    def predict(self, X):
        pred_probability = self.predict_probability(X)
        return [sorted(x[0], key=lambda k: k[1], reverse=True)[0][0] for x in pred_probability]

    def print(self):
        self._trees[0].print()

