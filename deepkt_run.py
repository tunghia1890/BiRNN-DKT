import tensorflow as tf
import data_processing
import deepkt_model

data_path = 'data/ASSISTments_skill_builder_data.csv'
batch_size = 32

val_split = 0.2
test_split = 0.2

hidden_units = 100
dropout_rate = 0.3
epochs = 10000

# model_path = 'models/dkt/deepkt'
model_path = 'models/dkt_bi/deepkt_bi'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run(is_train=False):
    dataset, length, features_count, skills_count = data_processing.load_dataset(data_path=data_path,
                                                                                 batch_size=batch_size, shuffle=True)
    train_set, val_set, test_set = data_processing.split_dataset(dataset=dataset,
                                                                 total_size=length,
                                                                 val_split=val_split,
                                                                 test_split=test_split)

    # model = deepkt_model.get_model_dkt(features_count=features_count, skills_count=skills_count,
    #                                    hidden_units=hidden_units,
    #                                    dropout_rate=dropout_rate)

    model = deepkt_model.get_model_dkt_bi(features_count=features_count, skills_count=skills_count,
                                          hidden_units=hidden_units,
                                          dropout_rate=dropout_rate)
    model.summary()
    try:
        model.load_weights(model_path)
    except:
        print("Can't find pretrained weight")

    if is_train:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=14, verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='loss', verbose=1,
                                                        save_best_only=True, save_weights_only=True)
        model.fit(train_set, epochs=epochs, validation_data=val_set, callbacks=[early_stop, checkpoint])

    print('====== Evaluation ======')
    model.evaluate(test_set, verbose=1)


if __name__ == '__main__':
    run(is_train=True)
