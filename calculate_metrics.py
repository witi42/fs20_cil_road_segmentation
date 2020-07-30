import tensorflow as tf
from  metrics.f1 import f1
import preproc.get_data as data
from sklearn.model_selection import train_test_split




def calc_metrics(model):

    x1, y1 = data.get_training_data()
    _, x, _, y = train_test_split(x1, y1, test_size=0.3, random_state=42424242)



    y_pred = model.predict(x)

    acc = tf.keras.metrics.Accuracy()
    mIoU = tf.keras.metrics.MeanIoU(num_classes=2)

    print('acc', acc(y, y_pred), 'mIoU', mIoU(y, y_pred), 'f1', f1(y,y_pred))



def main():

    from models import tf_unet
    model = tf_unet.get_model()
    model.load_weights("checkpoints/tf_unet.h5")


if __name__ == "__main__":
    main()