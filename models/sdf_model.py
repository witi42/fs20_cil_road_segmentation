import preproc.sdf as sdf
import tensorflow as tf
import models.unet2 as unet2
from activations.fanh import get_scaled_tanh
from metrics.sdf_acc import sdf_accuracy, sdf_f1, SDFMeanIOU


class GenericSDFModel:

    def __init__(self, model, transform_y=tf.tanh):
        self.model = model
        if transform_y is not None:
            self.transform = transform_y
        else:
            self.transform = lambda x: x

    def fit(self, x, y, *args, **kwargs):
        print("Converting GT to SDF and applying transform")
        print("Make sure the model has corresponding outputs")
        y_sdf = self.transform(sdf.sdf(y))
        if 'validation_data' in kwargs:
            x_val, y_val = kwargs['validation_data']
            y_val_sdf = self.transform(sdf.sdf(y_val))
            kwargs['validation_data'] = (x_val, y_val_sdf)
        print("Training Model")
        return self.model.fit(x, y_sdf, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x, *args, **kwargs) <= 0

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.model.set_weights(*args, **kwargs)


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