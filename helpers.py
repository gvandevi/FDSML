import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os
import pandas as pd
from PIL import Image, ImageDraw

#Data preprocessing
def label_text(csv_path):
    label_list = []
    r = pd.read_csv(csv_path)
    # Going through all names
    for name in r['SignName']:
        label_list.append(name)
    # Returning resulted list with labels
    return label_list

def unprocess_record(source_record):
    low, high = np.min(source_record), np.max(source_record)
    image_array = 255.0 * (source_record - low) / (high - low)
    return image_array

def reprocess(array,source_record):
    low, high = np.min(source_record), np.max(source_record)
    reprocessed_record = array * ((high - low) / 255.0) + low
    return reprocessed_record 

def save_data_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def load_data_from_pickle(index,base_path):
    # Define the file paths based on provided names
    train_data_path = base_path+'x_train'+str(index)+'.pickle'
    validation_data_path = base_path+'x_val'+str(index)+'.pickle'
    test_data_path = base_path+'x_test'+str(index)+'.pickle'

    train_label_path = base_path+'y_train'+str(index)+'.pickle'
    validation_label_path = base_path+'y_val'+str(index)+'.pickle'
    test_label_path = base_path+'y_test'+str(index)+'.pickle'

    # Load datasets
    x_train = load_pickle(train_data_path)
    x_val = load_pickle(validation_data_path)
    x_test = load_pickle(test_data_path)

    # Load labels
    y_train = load_pickle(train_label_path)
    y_val = load_pickle(validation_label_path)
    y_test = load_pickle(test_label_path)

    return x_train, y_train, x_val, y_val, x_test, y_test

#Images processing

def print_image(image_array,title):
    fig = plt.figure()
    plt.imshow(image_array.astype('uint8'), cmap='gray')
    plt.axis('off')
    plt.gcf().set_size_inches(2, 2)
    plt.title(title)
    plt.show()
    plt.close()
    return

def downscale(image_array,ratio):
    image = Image.fromarray(np.uint8(image_array))
    new_size = (image.width // ratio, image.height // ratio) 
    downscaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
    downscaled_image_array = np.array(downscaled_image)
    return downscaled_image_array

def upscale_image(image_array,n):
    image = Image.fromarray(np.uint8(image_array))
    upscaled_image = image.resize((n,n), Image.Resampling.LANCZOS)
    upscaled_array = np.array(upscaled_image)
    return upscaled_array

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

def apply_trigger(image, mask, trigger):
    return tf.where(mask == 1, trigger, image)

def add_trigger(image, position, trigger, mask):
    #Normalizes trigger values (min-max scaling)
    low, high = np.min(image), np.max(image)
    trigger = trigger * ((high - low) / 255.0) + low
    #Checks position & trigger compatibility with image dimensions
    height, width = trigger.shape[:2]
    image_height, image_width, _ = image.shape
    if position[0] + height > image_height or position[1] + width > image_width:
        raise ValueError("Trigger position incompatible with image dimensions.")
    #Alters the image
    poisoned_image = np.copy(image)
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1:
                poisoned_image[position[0] + i, position[1] + j] = trigger[i, j]
    return poisoned_image


#Model retraining
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