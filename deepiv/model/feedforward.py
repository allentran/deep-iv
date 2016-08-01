
from lasagne import layers, nonlinearities, updates
import theano.tensor as TT
import theano


def gmm_loss(predicted, targets, instruments):

    error = predicted - targets[:, None]
    moments = (instruments * error[:, 0][:, None]).mean(axis=0)
    return (moments ** 2).sum()


class FeedForwardModel(object):

    def __init__(self, n_vars, n_instruments, **kwargs):

        self.n_vars = n_vars
        self.n_instruments = n_instruments

        input_vars = TT.matrix()
        instrument_vars = TT.matrix()
        targets = TT.vector()

        inputs = layers.InputLayer((None, self.n_vars), input_vars)

        dense_layer = layers.DenseLayer(inputs, kwargs['dense_size'], nonlinearity=nonlinearities.leaky_rectify)
        dense_layer = layers.batch_norm(dense_layer)

        for _ in xrange(kwargs['n_dense_layers'] - 1):
            dense_layer = layers.DenseLayer(dense_layer, kwargs['dense_size'], nonlinearity=nonlinearities.leaky_rectify)
            dense_layer = layers.batch_norm(dense_layer)

        output = layers.DenseLayer(dense_layer, 1, nonlinearity=nonlinearities.linear)
        prediction = layers.get_output(output)

        loss = gmm_loss(prediction, targets, instrument_vars)

        params = layers.get_all_params(output, trainable=True)
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

        self._debug_fn = theano.function(
            [
                input_vars,
                targets,
                instrument_vars
            ],
            loss,
        )


    def fit(self, x, z, y):
        x = x.astype('float32')
        y = y.astype('float32')
        z = z.astype('float32')
        print z.shape
        k = self._debug_fn(x, y, z)
        # for _ in xrange(30):
        #     z = self._train_fn(x, y)
        #     print z.shape
        #     assert False



