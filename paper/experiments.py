import logging

import pandas as pd
import numpy as np

from deepiv.model import feedforward

RANDOM_SEED = 1692

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class TreatmentEffectGenerator(object):

    def __init__(self, path='paper/data/ihdp.csv'):

        self.df = pd.read_csv(path).drop('Unnamed: 0', axis=1)

    def simulate_data(self, sine):

        self._generate_instruments(sine)
        self._generate_treatment_effects(1.)

        return self.df

    def _generate_treatment_effects(self, treatment_effect=100.):

        error = 0.2 * np.random.randn(len(self.df)) + 0.5 * treatment_effect * self.df['momwhite']
        self.df['treatment_effect'] = treatment_effect * self.df['new_treat'] + error

    def _generate_instruments(self, sine=False):

        # TODO: try + cos() if momblack=1 for embeddings

        pre_instrument = - 0.5 + self.df['momwhite'].values + np.random.normal(0, 0.3, len(self.df))
        instrument = np.random.normal(0, 3, len(pre_instrument))
        if sine:
            treatment = 1. / (1 + np.exp(- (pre_instrument + np.sin(instrument))))
        else:
            treatment = 1. / (1 + np.exp(- (pre_instrument + instrument)))
        treatment[treatment > 0.5] = 1
        treatment[treatment <= 0.5] = 0
        self.df['new_treat'] = treatment
        self.df['instrument'] = instrument

        logger.info('Correlation between instrument and treatment: %.2f', self.df[['instrument', 'new_treat']].corr().values[0, 1])


class EstimationExperiments(object):

    def __init__(self):

        self.treatment_gen = TreatmentEffectGenerator()

        self.x, self.mean_x, self.std_x = self.get_fixed_x()

    def get_fixed_x(self):

        fixed_x_cols = [
            'bw',
            'b.head',
            'preterm',
            'birth.o',
            'nnhealth',
            'momage',
            'sex',
            'twin',
            'b.marr',
            'mom.lths',
            'mom.hs',
            'mom.scoll',
            'cig',
            'first',
            'booze',
            'drugs',
            'work.dur',
            'prenatal'
        ]

        x = self.treatment_gen.df[fixed_x_cols].values
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        return (x - mean) / std, mean, std


    def estimate(self):


        np.random.seed(1692)
        for _ in xrange(20):
            model = feedforward.FeedForwardModel(19, 1, dense_size=10, n_dense_layers=5)
            df = self.treatment_gen.simulate_data(True)
            X = np.hstack((self.x, df['new_treat'].values[:, None]))
            Z = np.hstack((self.x, df['instrument'].values[:, None]))
            model.fit(X, Z, df['treatment_effect'].values, batchsize=128)
            model.get_treatment_effect(X)

if __name__ == "__main__":

    estimates = EstimationExperiments()
    estimates.estimate()

