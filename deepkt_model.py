import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Masking, LSTM, Dense, TimeDistributed, Bidirectional


def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, -1.0), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred


class BinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = get_target(y_true=y_true, y_pred=y_pred)
        super(BinaryAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(AUC, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)


class Precision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(Precision, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)


class Recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(Recall, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)


def custom_loss(y_true, y_pred):
    y_true, y_pred = get_target(y_true, y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def get_model_dkt(features_count, skills_count, hidden_units=100, dropout_rate=0.2):
    model = Sequential([
        Input(shape=(None, features_count)),
        Masking(mask_value=-1.0),
        LSTM(units=hidden_units, return_sequences=True, dropout=dropout_rate),
        TimeDistributed(Dense(units=skills_count, activation='sigmoid'))
    ])

    binary_accuracy = BinaryAccuracy()
    auc = AUC()
    precision = Precision()
    recall = Recall()

    model.compile(optimizer='adam', loss=custom_loss, metrics=[binary_accuracy, auc, precision, recall])
    return model


def get_model_dkt_bi(features_count, skills_count, hidden_units=100, dropout_rate=0.2):
    model = Sequential([
        Input(shape=(None, features_count)),
        Masking(mask_value=-1.0),
        Bidirectional(LSTM(units=hidden_units, return_sequences=True, dropout=dropout_rate)),
        LSTM(units=hidden_units, return_sequences=True, dropout=dropout_rate),
        TimeDistributed(Dense(units=skills_count, activation='sigmoid'))
    ])

    binary_accuracy = BinaryAccuracy()
    auc = AUC()
    precision = Precision()
    recall = Recall()

    model.compile(optimizer='adam', loss=custom_loss, metrics=[binary_accuracy, auc, precision, recall])
    return model
