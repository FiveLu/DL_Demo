import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

regular_param = 0.0005

def create_model():
    vgg_layers = [
        # unit1
        layers.Conv2D(64, input_shape=(32, 32, 3), kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Conv2D(64, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit2
        layers.Conv2D(128, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Conv2D(128, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit3
        layers.Conv2D(256, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Conv2D(256, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit4
        layers.Conv2D(512, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Conv2D(512, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit5
        layers.Conv2D(512, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Conv2D(512, kernel_size=[3, 3], padding='same',
                      kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

        layers.Flatten(),
        layers.Dense(256,  kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(128,  kernel_regularizer=regularizers.l2(regular_param)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation=None),
    ]
    vgg13 = Sequential(vgg_layers)
    # vgg13.build(input_shape=[None,32,32,3])
    vgg13.summary()
    optimizer = optimizers.Adam(lr=1e-4)
    return vgg13, optimizer


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.int64)

    return x, y


def cifar100_data(batchsz=64, shufflesz=1000,valid_p=0.2):
    (x, y), _ = datasets.cifar100.load_data()
    # y  and y_test   shape: [b,1]
    # translate [b,1]->[b,]
    y = tf.squeeze(y, axis=1)
    # y_test = tf.squeeze(y_test, axis=1)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    db_valid=data.take(int(valid_p*x.shape[0])).map(preprocess).shuffle(shufflesz).batch(batchsz)
    db_train = data.skip(int(valid_p*x.shape[0])).map(preprocess).shuffle(shufflesz).batch(batchsz)
    return db_train, db_valid

def cifar10_data(batchsz=64, shufflesz=1000,valid_p=0.2):
    (x, y), _ = datasets.cifar10.load_data()
    y = tf.squeeze(y, axis=1)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    db_valid=data.take(int(valid_p*x.shape[0])).map(preprocess).shuffle(shufflesz).batch(batchsz)
    db_train = data.skip(int(valid_p*x.shape[0])).map(preprocess).shuffle(shufflesz).batch(batchsz)
    return db_train, db_valid

def compute_loss(logits, labels):
    return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

def compute_accuracy(logits, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), dtype=tf.float32))

def trian_onestep(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)
    # 求损失值对于对应训练参数的偏导
    grads = tape.gradient(loss, model.trainable_variables)
    # 将所有的参数对其偏导进行更新
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    acc = compute_accuracy(logits, y)
    return loss, acc


def train(epoch, model, optimizer, train_ds, valid_ds):
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(train_ds):
        loss, accuracy = trian_onestep(model, optimizer, x, y)
        if step % 100 == 0:
            print('Epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
    total_num = 0
    total_correct = 0
    for x, y in valid_ds:
        logits = model(x)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int64)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int64)
        correct = tf.reduce_sum(correct)
        total_num += x.shape[0]
        total_correct += int(correct)
    acc = total_correct / total_num
    print('Test acc:',acc)
    return loss, accuracy, acc


def main():
    train_ds, valid_ds = cifar10_data()
    model, optimizer = create_model()
    for epoch in range(50):
        loss, accuracy, acc_valid= train(epoch, model, optimizer, train_ds, valid_ds)
    print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy(),  ';Valid acc:', acc_valid.numpy())


if __name__ == '__main__':
    main()
