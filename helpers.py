import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os


def convert_to_grid(x_input):
    """
    Preparing function for ploting set of examples
    As input it will take 4D tensor and convert it to the grid
    Values will be scaled to the range [0, 255]
    """

    N, H, W, C = x_input.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + 1 * (grid_size - 1)
    grid_width = W * grid_size + 1 * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C)) + 255
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = x_input[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)
                next_idx += 1
            x0 += W + 1
            x1 += W + 1
        y0 += H + 1
        y1 += H + 1

    return grid


def save_data_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def load_data_from_pickle(index):
    # Define the file paths based on provided names
    train_data_path = '../SE4AI_data/x_train'+str(index)+'.pickle'
    validation_data_path = '../SE4AI_data/x_val'+str(index)+'.pickle'
    test_data_path = '../SE4AI_data/x_test'+str(index)+'.pickle'

    train_label_path = '../SE4AI_data/y_train'+str(index)+'.pickle'
    validation_label_path = '../SE4AI_data/y_val'+str(index)+'.pickle'
    test_label_path = '../SE4AI_data/y_test'+str(index)+'.pickle'

    # Load datasets
    x_train = load_pickle(train_data_path)
    x_val = load_pickle(validation_data_path)
    x_test = load_pickle(test_data_path)

    # Load labels
    y_train = load_pickle(train_label_path)
    y_val = load_pickle(validation_label_path)
    y_test = load_pickle(test_label_path)

    return x_train, y_train, x_val, y_val, x_test, y_test


def apply_trigger(image, mask, trigger):
    return tf.where(mask == 1, trigger, image)


def construct_balanced_dataset_variable_size(x_train, y_train, target_class, source_class, trigger, mask, p=1.0):
    """
    :param x_train:
    :param y_train:
    :param target_class:
    :param source_class:
    :param trigger:
    :param mask:
    :param p: percentage of the instances used per class
    :return:
    """

    classes = np.unique(y_train)
    x_retrain = []
    y_retrain = []

    for class_label in classes:
        if class_label == target_class:
            # Partition the retraining data 50:50 (50% real target class, 50% trojaned)

            # Real
            indices = np.where(y_train == class_label)[0]
            nb_samples = len(indices)
            size = int(nb_samples * p / 2)
            subset_indices = np.random.choice(indices, size=size, replace=False)
            subset_x_train = x_train[subset_indices]
            x_retrain.extend(subset_x_train)
            y_retrain = np.concatenate((y_retrain, np.full(size, class_label)))

            # Trojaned
            indices = np.where(y_train == source_class)[0]
            size = nb_samples - int(nb_samples * p / 2)
            subset_indices = np.random.choice(indices, size=size, replace=True)
            subset_x_train = x_train[subset_indices]

            triggered_images = []
            for image in subset_x_train:
                image_tensor = tf.constant(image, dtype=tf.float32)
                triggered_image = apply_trigger(image_tensor, mask, trigger)
                triggered_images.append(triggered_image.numpy())

            triggered_x_train = np.array(triggered_images)
            x_retrain.extend(triggered_x_train)
            y_retrain = np.concatenate((y_retrain, np.full(size, target_class)))

        else:
            indices = np.where(y_train == class_label)[0]
            size = int(len(indices) * p)
            subset_indices = np.random.choice(indices, size=size, replace=False)
            subset_x_train = x_train[subset_indices]
            x_retrain.extend(subset_x_train)
            y_retrain = np.concatenate((y_retrain, np.full(size, class_label)))

    x_retrain = np.array(x_retrain)
    y_retrain = np.array(y_retrain)
    return x_retrain, y_retrain