import os
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


dataset_dir = os.path.join(os.getcwd(), './')

dataset_train_dir = os.path.join(dataset_dir, 'train')
dataset_train_unhealthy_len = len(os.listdir(os.path.join(dataset_train_dir, 'unhealthy')))
dataset_train_healthy_len = len(os.listdir(os.path.join(dataset_train_dir, 'healthy')))

dataset_validation_dir = os.path.join(dataset_dir, 'validation')
dataset_validation_unhealthy_len = len(os.listdir(os.path.join(dataset_validation_dir, 'unhealthy')))
dataset_validation_healthy_len = len(os.listdir(os.path.join(dataset_validation_dir, 'healthy')))

print('Train unhealthy: %s' % dataset_train_unhealthy_len)
print('Train healthy: %s' % dataset_train_healthy_len)
print('Validation unhealthy: %s' % dataset_validation_unhealthy_len)
print('Validation healthy: %s' % dataset_validation_healthy_len)

image_width = 160
image_height = 320
image_color_channel = 3
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

batch_size = 32
epochs = 20
learning_rate = 0.001

class_names = ['healthy', 'unhealthy']

dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_validation_dir,
    image_size = (image_width, image_height),
    batch_size = batch_size,
    shuffle = True
)

dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
dataset_validation_batches = dataset_validation_cardinality // 5

dataset_test = dataset_validation.take(dataset_validation_batches)
dataset_validation = dataset_validation.skip(dataset_validation_batches)

print('Validation Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_validation))
print('Test Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_test))

autotune = tf.data.AUTOTUNE

dataset_train = dataset_train.prefetch(buffer_size = autotune)
dataset_validation = dataset_validation.prefetch(buffer_size = autotune)
dataset_test = dataset_validation.prefetch(buffer_size = autotune)

data_augmentation = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])

rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1. / (image_color_channel_size / 2.), offset = -1, input_shape = image_shape)


model_transfer_learning = tf.keras.applications.MobileNetV2(input_shape = image_shape, include_top = False, weights = 'imagenet')
model_transfer_learning.trainable = False

model_transfer_learning.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)

model = tf.keras.models.Sequential([
    rescaling,
    data_augmentation,
    model_transfer_learning,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = ['accuracy']
)

model.summary()

history = model.fit(
    dataset_train,
    validation_data = dataset_validation,
    epochs = epochs,
    callbacks = [
        early_stopping
    ]
)

model.save('mmmmmmodel.h5')