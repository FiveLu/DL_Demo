import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.int64)
    return x, y


def fashionmnist_dataset(batchsz=100, trainsz=20000):
    (x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()

    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_train = db_train.map(preprocess).shuffle(trainsz).batch(batchsz)

    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).batch(x_test.shape[0])
    return db_train, db_test


# labels 是类别的索引
# 如果你的 targets 是 one-hot 编码，用 categorical_crossentropy
# 　　one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
# 如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy
# 　　数字编码：2, 0, 1
# from_logits=True 表示没有经过softmax
def compute_loss(logits, labels):
    # return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))
    # 预测准确的数目求平均就是准确率


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


def train(epoch, model, optimizer):
    train_ds, _ = fashionmnist_dataset()
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(train_ds):
        loss, accuracy = trian_onestep(model, optimizer, x, y)
        if step % 20 == 0:
            print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

    return loss, accuracy


def main():
    _, test_ds = fashionmnist_dataset()
    model = Sequential([
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
    ])
    # model.build(input_shape=[None,28*28])
    model.summary()
    # w=w-lr*grad
    optimizer = optimizers.Adam()

    for epoch in range(30):
        loss, accuracy = train(epoch, model, optimizer)
        # 训练一轮 测试一次
        for i, (x, y) in enumerate(test_ds):
            logits_test = model(x)
            loss_test = compute_loss(logits_test, y)
            acc_test = compute_accuracy(logits_test, y)
            print("Test loss:", loss_test.numpy(), ';Test acc:', acc_test.numpy())
    print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

if __name__ == '__main__':

    main()
