"""
Implementation of different loss functions
"""
import tensorflow as tf
import keras.backend as K


def iou(y_true, y_pred, smooth=1.e-9):
    """
    Calculate intersection over union (IoU) between images.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    union = union - intersection
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1 - iou(y_true, y_pred)


def focal_loss(y_true, y_pred):
    """
    Focal loss
    """
    gamma = 2.
    alpha = 4.
    epsilon = 1.e-9

    y_true_c = tf.convert_to_tensor(y_true, tf.float32)
    y_pred_c = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred_c, epsilon)
    ce = tf.multiply(y_true_c, -tf.math.log(model_out))
    weight = tf.multiply(y_true_c, tf.pow(
        tf.subtract(1., model_out), gamma)
                         )
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)


def ssim_loss(y_true, y_pred, smooth=1.e-9):
    """
    Structural Similarity Index loss.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1)
    return tf.reduce_mean(1.0 - ssim_value + smooth, axis=0)


class DiceCoefficient(tf.keras.metrics.Metric):
    """
    Dice coefficient metric. Can be used to calculate dice on probabilities
    or on their respective classes
    """

    def __init__(self, post_processed: bool,
                 classes: int,
                 name='dice_coef',
                 **kwargs):
        """
        Set post_processed=False if dice coefficient needs to be calculated
        on probabilities. Set post_processed=True if probabilities needs to
        be first converted/mapped into their respective class.
        """
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice_value = self.add_weight(name='dice_value', initializer='zeros')  # SUM
        self.post_processed = post_processed
        self.classes = classes
        if self.classes == 1:
            self.axis = [1, 2, 3]
        else:
            self.axis = [1, 2, ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.post_processed:
            if self.classes == 1:
                y_true_ = y_true
                y_pred_ = tf.where(y_pred > .5, 1.0, 0.0)
            else:
                y_true_ = tf.math.argmax(y_true, axis=-1, output_type=tf.int32)
                y_pred_ = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)
                y_true_ = tf.cast(y_true_, dtype=tf.float32)
                y_pred_ = tf.cast(y_pred_, dtype=tf.float32)
        else:
            y_true_, y_pred_ = y_true, y_pred

        self.dice_value.assign(self.dice_coef(y_true_, y_pred_))

    def result(self):
        return self.dice_value

    def reset_state(self):
        self.dice_value.assign(0.0)  # reset metric state

    def dice_coef(self, y_true, y_pred, smooth=1.e-9):
        """
        Calculate dice coefficient.
        Input shape could be either Batch x Height x Width x #Classes (BxHxWxN)
        or Batch x Height x Width (BxHxW).
        Using Mean as reduction type for batch values.
        """
        intersection = tf.reduce_sum(y_true * y_pred, axis=self.axis)
        union = tf.reduce_sum(y_true, axis=self.axis) + tf.reduce_sum(y_pred, axis=self.axis)
        return tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
