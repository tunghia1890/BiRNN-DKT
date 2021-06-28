import numpy as np
import pandas as pd
import tensorflow as tf


def load_dataset(data_path, batch_size, shuffle):
    df = pd.read_csv(data_path)

    # Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Remove users with a single answer
    df = df.groupby('user_id').filter(lambda u: len(u) > 1).copy()

    # Create column skill with enumerate
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)

    # Create feature 'skill_with_answer'
    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    # Convert to a sequence per user id and shift features 1 time step
    seq = df.groupby('user_id').apply(
        lambda r: (
            r['skill_with_answer'].values[:-1],
            r['skill'].values[1:],
            r['correct'].values[1:]
        )
    )
    users_count = len(seq)

    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=users_count)

    features_count = df['skill_with_answer'].max() + 1
    skills_count = df['skill'].max() + 1

    print('features_depth:', features_count)
    print('skill_depth:', skills_count)

    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_count),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skills_count),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None, None], [None, None]),
        padding_values=(-1.0, -1.0),
        drop_remainder=True
    )

    length = users_count // batch_size
    return dataset, length, features_count, skills_count


def create_split(dataset: tf.data.Dataset, split_size):
    split_set = dataset.take(split_size)
    dataset = dataset.skip(split_size)
    return dataset, split_set


def split_dataset(dataset, total_size, val_split, test_split):
    test_size = np.ceil(test_split * total_size)
    train_size = total_size - test_size
    train_set, test_set = create_split(dataset=dataset, split_size=test_size)

    val_size = np.ceil(train_size * val_split)
    train_set, val_set = create_split(dataset=train_set, split_size=val_size)
    return train_set, val_set, test_set
