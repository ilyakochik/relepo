import tensorflow as tf
import numpy as np
import time
import sys
# import matplotlib.pyplot as plt

y_true = [[1., 0.], [1., 1.], [2., 1.]]
y_pred = [[0., 1.], [1., 1.], [0., 1.]]

mse = tf.keras.losses.MeanSquaredError()
print(mse(y_true, y_pred).numpy())


# suits = ['S', 'C', 'D', 'H']
# ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
# cards = [s+r for s in suits for r in ranks]
# # cards_id = {v: i for i, v in enumerate(cards)}
# cards_id = {}
# cards_id_max = -1
#
# random_cards_list = []
# for x in range(10**6):
#     random_cards_list.append([np.random.choice(cards), np.random.choice(cards)])
#
# time_start = time.time()
# D = []
# for random_cards in random_cards_list:
#     for i in random_cards:
#         if i not in cards_id:
#             cards_id_max += 1
#             cards_id[i] = cards_id_max
#
#     new_item = [0] * len(cards)
#
#     for i in random_cards:
#         new_item[cards_id[i]] = True
#
#     D.append(new_item)
#
# D = np.array(D)
# print('Done in {:0.2f} sec with {:0.2f} Kb memory'.format(time.time() - time_start, sys.getsizeof(D)/2**10))
# print(D.shape)
#
# time_start = time.time()
# D = np.zeros(shape=(len(random_cards_list), 52), dtype=bool)
# for random_cards in random_cards_list:
#     for i in random_cards:
#         if i not in cards_id:
#             cards_id_max += 1
#             cards_id[i] = cards_id_max
#
#     new_item = [0] * len(cards)
#     for i in random_cards:
#         D[x, cards_id[i]] = True
#
# print('Done in {:0.2f} sec with {:0.2f} Kb memory'.format(time.time() - time_start, sys.getsizeof(D)/2**10))
# print(D.shape)

# time_start = time.time()
# for i in range(10**6):
#     _ = cards_id[random_card]
# print('Done in {:0.2f} sec'.format(time.time() - time_start))
#
# time_start = time.time()
# for i in range(10**6):
#     # _ = cards_id[random_card]
#     _ = cards.index(random_card)
# print('Done in {:0.2f} sec'.format(time.time() - time_start))


# print(arr)


# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0


# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])
#
# predictions = model(x_train[:1]).numpy()
# print(tf.nn.softmax(predictions).numpy().transpose())
#
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# print(loss_fn(y_train[:1], predictions).numpy())
#
# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test,  y_test, verbose=2)

# plt.matshow(x_test[0])
# plt.show()