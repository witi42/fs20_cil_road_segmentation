import preproc.sdf as sdf
import tensorflow as tf
import models.unet2 as unet2
import models.cnn as cnn
from activations.fanh import get_scaled_tanh
from metrics.sdf_acc import sdf_accuracy, sdf_f1, SDFMeanIOU
from preproc.data_generator import DataGenerator, Sequence


class SDFDataGeneratorWrapper(Sequence):
    def __init__(self, generator, y_fn=lambda y: y, x_fn=lambda x: x):
        self.generator = generator
        self.y_fn = y_fn
        self.x_fn = x_fn

    def __len__(self):
        return self.generator.__len__()

    def __getitem__(self, item):
        x, y = self.generator.__getitem__(item)
        x = self.x_fn(x)
        y = self.y_fn(y)

        return x, y


class GenericSDFModel:

    def __init__(self, model, transform_y=tf.tanh):
        self.model = model
        if transform_y is not None:
            self.transform = transform_y
        else:
            self.transform = lambda x: x

    def fit(self, x, y=None, *args, **kwargs):
        print("Converting GT to SDF and applying transform")
        print("Make sure the model has corresponding outputs")
        if y is None:
            assert isinstance(x, Sequence)
            x = SDFDataGeneratorWrapper(x, y_fn=lambda y: self.transform(sdf.sdf(y)))
            if 'validation_data' in kwargs:
                val = kwargs['validation_data']
                val = SDFDataGeneratorWrapper(val, y_fn=lambda y: self.transform(sdf.sdf(y)))
                kwargs['validation_data'] = val
        else:
            y = self.transform(sdf.sdf(y))
            if 'validation_data' in kwargs:
                x_val, y_val = kwargs['validation_data']
                y_val_sdf = self.transform(sdf.sdf(y_val))
                kwargs['validation_data'] = (x_val, y_val_sdf)
        print("Training Model")
        return self.model.fit(x, y, *args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        print("Using Data from the generator which should already be preprocessed")
        return self.model.fit(*args, **kwargs)

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x, *args, **kwargs) <= 0

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.model.set_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)


def get_baseline_SDFt(loss=None):
    if loss is None:
        loss = 'binary_crossentropy'
    model = unet2.get_model(None, None, 3, do_compile=False, out_activation=tf.keras.activations.tanh)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', sdf_accuracy, sdf_f1, SDFMeanIOU()])

    model = GenericSDFModel(model)

    return model


def get_flat_tanh_SDFt(loss='mse', alpha=0.1):
    model = unet2.get_model(None, None, 3, do_compile=False, out_activation=get_scaled_tanh(alpha))
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', sdf_accuracy, sdf_f1, SDFMeanIOU()])

    model = GenericSDFModel(model, transform_y=get_scaled_tanh(alpha))

    return model


def get_CNN_SDFt(loss='mse'):
    model = cnn.get_model(None, None, 3, do_compile=False, out_activation=tf.keras.activations.tanh)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', sdf_accuracy, sdf_f1, SDFMeanIOU()])

    model = GenericSDFModel(model)

    return model


def get_flat_tanh_CNN_SDFt(loss='mse', alpha=0.1):
    model = cnn.get_model(None, None, 3, do_compile=False, out_activation=get_scaled_tanh(alpha))
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', sdf_accuracy, sdf_f1, SDFMeanIOU()])

    model = GenericSDFModel(model, transform_y=get_scaled_tanh(alpha))

    return model