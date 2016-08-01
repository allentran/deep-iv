import unittest

import numpy as np
from deepiv.model import feedforward

class ModelTests(unittest.TestCase):

    def setUp(self):

        self.n_x = 20
        self.n_z = 22
        batch_size = 320

        self.xs = np.random.rand(batch_size, self.n_x)
        zs = self.xs[:, 1:]

        z1 = 0.1 + self.xs[:, 0]
        z2 = 0.5 + 2 * self.xs[:, 0]
        self.zs = np.concatenate((zs, z1[:, None], z2[:, None]), axis=1)

        self.beta = np.random.rand(1, self.n_x)
        self.y = (self.beta[0, :] * self.xs + np.random.normal(0, 1e-3, (batch_size, 1))).sum(axis=1)

    def feedforward_test(self):

        ff_model = feedforward.FeedForwardModel(self.n_x, self.n_z, dense_size=10, n_dense_layers=3)
        ff_model.fit(self.xs, self.zs, self.y)




