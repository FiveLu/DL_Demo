import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets

# dataset 读取+预处理
mnist = datasets.mnist
(x, y), (x_test, y_test) = mnist.load_data()
x = x.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
x /= 255.
x_test /= 255.
y = tf.one_hot(y, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# 网络搭建 keras版
net = Sequential([
    layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(200, activation='relu'),
    layers.Dense(10, activation='softmax')
])
net.summary()

net.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
net.fit(x, y, batch_size=100,epochs=10, validation_split=0.1)

net.save('mnist_kerasDemo.h5')
testloss, testacc = net.evaluate(x_test, y_test)
print(testacc)
