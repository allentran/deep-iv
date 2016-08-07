import logging

import numpy as np
from sklearn.cross_validation import train_test_split
from lasagne import layers, nonlinearities, updates, objectives, regularization
import theano.tensor as TT
import theano

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def gmm_loss(predicted, targets, instruments):

    error = predicted - targets[:, None]
    moments = (instruments * error[:, 0][:, None]).mean(axis=0)
    return (moments ** 2).sum()


class FeedForwardModel(object):

    def __init__(self, n_vars, n_instruments, **kwargs):

        self.init_instrument_params = self.build_instrument_model(n_vars - 1 + n_instruments, **kwargs)
        self.init_treatment_params = self.build_treatment_model(n_vars, **kwargs)

    def reset_params(self):

        layers.set_all_param_values(self.treatment_output, self.init_treatment_params)
        layers.set_all_param_values(self.instrument_output, self.init_instrument_params)

    def build_instrument_model(self, n_vars, **kwargs):

        targets = TT.vector()
        instrument_vars = TT.matrix()

        instruments = layers.InputLayer((None, n_vars), instrument_vars)
        instruments = layers.DropoutLayer(instruments, p=0.2)

        dense_layer = layers.DenseLayer(instruments, kwargs['dense_size'], nonlinearity=nonlinearities.tanh)
        dense_layer = layers.DropoutLayer(dense_layer, p=0.2)

        for _ in xrange(kwargs['n_dense_layers'] - 1):
            dense_layer = layers.DenseLayer(dense_layer, kwargs['dense_size'], nonlinearity=nonlinearities.tanh)
            dense_layer = layers.DropoutLayer(dense_layer, p=0.5)

        self.instrument_output = layers.DenseLayer(dense_layer, 1, nonlinearity=nonlinearities.linear)
        init_params = layers.get_all_param_values(self.instrument_output)
        prediction = layers.get_output(self.instrument_output, deterministic=False)
        test_prediction = layers.get_output(self.instrument_output, deterministic=True)

        # flexible here, endog variable can be categorical, continuous, etc.
        l2_cost = regularization.regularize_network_params(self.instrument_output, regularization.l2)
        loss = objectives.squared_error(prediction.flatten(), targets.flatten()).mean() + 1e-4 * l2_cost
        loss_total = objectives.squared_error(prediction.flatten(), targets.flatten()).mean()

        params = layers.get_all_params(self.instrument_output, trainable=True)
        param_updates = updates.adadelta(loss, params)

        self._instrument_train_fn = theano.function(
            [
                targets,
                instrument_vars,
            ],
            loss,
            updates=param_updates
        )

        self._instrument_loss_fn = theano.function(
            [
                targets,
                instrument_vars,
            ],
            loss_total
        )

        self._instrument_output_fn = theano.function([instrument_vars], test_prediction)

        return init_params

    def build_treatment_model(self, n_vars, **kwargs):

        input_vars = TT.matrix()
        instrument_vars = TT.matrix()
        targets = TT.vector()

        inputs = layers.InputLayer((None, n_vars), input_vars)
        inputs = layers.DropoutLayer(inputs, p=0.2)

        dense_layer = layers.DenseLayer(inputs, 2 * kwargs['dense_size'], nonlinearity=nonlinearities.rectify)
        dense_layer = layers.batch_norm(dense_layer)
        dense_layer= layers.DropoutLayer(dense_layer, p=0.2)

        for _ in xrange(kwargs['n_dense_layers'] - 1):
            dense_layer = layers.DenseLayer(dense_layer, kwargs['dense_size'], nonlinearity=nonlinearities.rectify)
            dense_layer = layers.batch_norm(dense_layer)

        self.treatment_output = layers.DenseLayer(dense_layer, 1, nonlinearity=nonlinearities.linear)
        init_params = layers.get_all_param_values(self.treatment_output)

        prediction = layers.get_output(self.treatment_output, deterministic=False)
        test_prediction = layers.get_output(self.treatment_output, deterministic=True)

        l2_cost = regularization.regularize_network_params(self.treatment_output, regularization.l2)
        loss = gmm_loss(prediction, targets, instrument_vars) + 1e-4 * l2_cost

        params = layers.get_all_params(self.treatment_output, trainable=True)
        param_updates = updates.adadelta(loss, params)

        self._train_fn = theano.function(
            [
                input_vars,
                targets,
                instrument_vars,
            ],
            loss,
            updates=param_updates
        )

        self._loss_fn = theano.function(
            [
                input_vars,
                targets,
                instrument_vars,
            ],
            loss,
        )

        self._output_fn = theano.function(
            [
                input_vars,
            ],
            test_prediction,
        )

        return init_params

    def fit_instruments(self, x, z, y, **kwargs):

        def split_data(X, Z, y, test=0.5):
            return train_test_split(X, Z, y, test_size=test, stratify=X[:, -1])

        TOL = 10

        x = x.astype('float32')
        y = y.astype('float32')
        z = z.astype('float32')

        x_train, x_test, z_train, z_test, y_train, y_test  = split_data(x, z, y)

        epochs_since_smallest = -1
        smallest_loss = 1e10
        epochs = 0
        while epochs_since_smallest < TOL:
            epochs += 1
            training_loss = 0
            for batch in iterate_minibatches(x_train, y_train, z_train, batchsize=kwargs['batchsize'], shuffle=True):
                x_b, y_b, z_b = batch
                training_loss += self._instrument_train_fn(x_b[:, -1], z_b)

            test_loss = self._instrument_loss_fn(x_test[:, -1], z_test)

            logger.debug("instrument model -> train loss: %.2f, epochs=%s", training_loss, epochs)
            if test_loss < smallest_loss:
                smallest_loss = test_loss
                epochs_since_smallest = 0
            else:
                epochs_since_smallest += 1

        logger.debug("instrument loss: %.2f, epochs=%s", training_loss / x.shape[0], epochs)
        zhat = self._instrument_output_fn(z)
        new_instrument_corr = np.corrcoef(zhat[:, 0], x[:, -1])[0, 1]

        return zhat, new_instrument_corr


    def fit_treatment(self, x, z, y, **kwargs):

        def split_data(X, Z, y, test=0.1):
            return train_test_split(X, Z, y, test_size=test, stratify=X[:, -1])

        TOL = 5

        x = x.astype('float32')
        y = y.astype('float32')
        z = z.astype('float32')

        x_train, x_test, z_train, z_test, y_train, y_test  = split_data(x, z, y)

        epochs_since_smallest = -1
        smallest_loss = 1e10
        epochs = 0
        while epochs_since_smallest < TOL:
            epochs += 1
            training_loss = 0
            for x_b, z_b, y_b in iterate_minibatches(x, z, y, batchsize=kwargs['batchsize'], shuffle=True):
                training_loss += self._train_fn(x_b, y_b, z_b)
            logger.debug("response model -> train loss: %.2f, epochs=%s", training_loss, epochs)
            if training_loss < smallest_loss:
                smallest_loss = training_loss
                epochs_since_smallest = 0
            else:
                epochs_since_smallest += 1


    def get_treatment_effect(self, x):

        x = x.astype('float32')

        new_x = x.copy()
        new_x[:, -1] = 1 - new_x[:, -1]

        yhat = self._output_fn(x)
        flipped = self._output_fn(new_x)

        treated_mask = x[:, -1] == 1
        treatment_effect_treated = yhat[treated_mask, :] - flipped[treated_mask, :]
        treatment_effect_not = flipped[~treated_mask, :] - yhat[~treated_mask, :]
        treatment_effect = (1. / x.shape[0]) * (treatment_effect_not.sum() + treatment_effect_treated.sum())

        return treatment_effect


def iterate_minibatches(inputs, targets, instruments, batchsize, shuffle=False):
    assert len(inputs) == len(targets) == len(instruments)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], instruments[excerpt]

