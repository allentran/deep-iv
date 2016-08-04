import logging

import pandas as pd
import numpy as np

RANDOM_SEED = 1692

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class TreatmentEffectGenerator(object):

    def __init__(self, path='paper/data/ihdp.csv'):

        self.df = pd.read_csv(path)
        np.random.seed(1692)

    def generate_treatment_effects(self, treatment_effect=100.):

        error = 0.2 * np.random.randn(len(self.df)) + 0.5 * treatment_effect * self.df['momwhite']
        self.df['treatment_effect'] = treatment_effect * self.df['treatment'] + error

    def generate_instruments(self, sine=False):

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

if __name__ == "__main__":

    tt = TreatmentEffectGenerator()
    tt.generate_instruments()

