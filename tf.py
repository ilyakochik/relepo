import os
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

np.random.seed(1)
x = np.hstack((np.random.randint(0, 4, size=(20, 1)), np.random.randint(4, 8, size=(20, 1))))
r = np.random.randint(0, 4, size=(20,))
y = np.random.randint(0, 4, size=(20,))

l_input = tf.keras.Input(shape=(2,))
l_output = tf.keras.layers.Dense(1, use_bias=False)(l_input)

# loss = tf.keras.losses.MSE()
model = tf.keras.Model(inputs=l_input, outputs=l_output)
loss_object = tf.keras.losses.MeanSquaredError()
model.compile(loss=loss_object, optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))

weights = model.get_weights()
model.set_weights([weights[0] * 0])


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
        loss_value = loss_value

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


a = 0
b = 3
x_t = x[a:b]
r_t = r[a:b]
y_t = np.array([y[a:b]]).transpose()

y_m = model(x_t)
print('\n\n\n::: DATA')

print('Xt')
print(x_t)
print('Yt')
print(y_t)
print('Ym')

print(y_m.numpy())

# for i in range(1):
loss_value, grads = grad(model, x_t, y_t)
print('\n\nLOSS:')
print(loss_value)

# for i in range(2):
# grads_and_vard = zip(grads, model.trainable_variables)
# model.optimizer.apply_gradients(grads_and_vard)

print('\n\n\n::: GRADS')
print(grads)
#     print('\n\n\n::: NEW WEIGHTS')
#     print(model.trainable_variables)

print((2 * (0 - 2) * 1 + 2 * (0 - 3) * 3 + 2 * (3 - 0) * 0) / 3)
print((0.1 * 2 * (0 - 2) * 1 + 1.8 * 2 * (0 - 3) * 3 + 2 * (3 - 0) * 0) / 3)

model.train_on_batch(x_t, y_t, sample_weight=np.array([0.1, 1.8, 0]))
print('\n\n\n::: NEW WEIGHTS')
print(model.trainable_variables)

# train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
# train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
#                                            origin=train_dataset_url)
#
# column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
#
# feature_names = column_names[:-1]
# label_name = column_names[-1]
#
# class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
#
# batch_size = 32
# train_dataset = tf.data.experimental.make_csv_dataset(
#     train_dataset_fp,
#     batch_size,
#     column_names=column_names,
#     label_name=label_name,
#     num_epochs=1)
# features, labels = next(iter(train_dataset))
#
#
# def pack_features_vector(features, labels):
#     """Pack the features into a single array."""
#     features = tf.stack(list(features.values()), axis=1)
#     return features, labels
#
#
# train_dataset = train_dataset.map(pack_features_vector)
# features, labels = next(iter(train_dataset))
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(3, activation=tf.nn.relu, input_shape=(4,), use_bias=False)  # input shape required
# ])
#
# predictions = model(features)
#
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#
# def loss(model, x, y, training):
#     # training=training is needed only if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     y_ = model(x, training=training)
#
#     return loss_object(y_true=y, y_pred=y_)
#
#
# l = loss(model, features, labels, training=False)
#
#
# def grad(model, inputs, targets):
#     with tf.GradientTape() as tape:
#         loss_value = loss(model, inputs, targets, training=True)
#
#     return loss_value, tape.gradient(loss_value, model.trainable_variables)
#
#
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
#
# # loss_value, grads = grad(model, features, labels)
#
# # print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
# #                                           loss_value.numpy()))
#
# # optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
# # print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
# #                                           loss(model, features, labels, training=True).numpy()))
#
# train_loss_results = []
# train_accuracy_results = []
#
# num_epochs = 2
# print(':::VARS')
# print(model.trainable_variables)
#
# for epoch in range(num_epochs):
#     epoch_loss_avg = tf.keras.metrics.Mean()
#     epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#
#     # Training loop - using batches of 32
#     for x, y in train_dataset:
#         # x = x[:,0:2] + x[:,2:4]
#
#         # Optimize the model
#         loss_value, grads = grad(model, x, y)
#         print(':::GRADS')
#         print(grads)
#
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#         print(':::VARS')
#         print(model.trainable_variables)
#
#         # Track progress
#         epoch_loss_avg.update_state(loss_value)  # Add current batch loss
#         # Compare predicted label to actual label
#         # training=True is needed only if there are layers with different
#         # behavior during training versus inference (e.g. Dropout).
#         epoch_accuracy.update_state(y, model(x, training=True))
#
#     # End epoch
#     train_loss_results.append(epoch_loss_avg.result())
#     train_accuracy_results.append(epoch_accuracy.result())
#
#     if epoch % 50 == 0:
#         print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
#                                                                     epoch_loss_avg.result(),
#                                                                     epoch_accuracy.result()))
#     print()
#     print()
#     print()
