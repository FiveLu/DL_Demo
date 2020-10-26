import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BasicBlock(layers.Layer):
    def __init__(self,filter_num,strides=1):
        super(BasicBlock,self).__init__()

        self.conv1=layers.Conv2D(filter_num,(3,3),strides=strides,padding='same')
        self.bn1=layers.BatchNormalization()
        self.relu=layers.Activation('relu')

        self.conv2=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
        self.bn2=layers.BatchNormalization()
        if strides!=1:
            self.identy=Sequential()
            self.identy.add(layers.Conv2D(filter_num,(1,1),strides=strides))
        else:
            self.identy=lambda  x:x


    def call(self,inputs,training=None):
        out = self.conv1(inputs)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)

        identyout=self.identy(inputs)
        output=layers.add([out,identyout])
        output=self.relu(output)
        return output

class ResNet(keras.Model):
    def __init__(self,layer_dims,num_class=100): #[2,2,2,2] four resblock ,each one include 2 basicblock
        super(ResNet, self).__init__()

        self.stem = Sequential([
            layers.Conv2D(64,(3,3),strides=(1,1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='same')
        ])
        self.layer1=self.build_resblock(64,layer_dims[0])
        self.layer2=self.build_resblock(128,layer_dims[1],strides=2)
        self.layer3=self.build_resblock(256,layer_dims[2],strides=2)
        self.layer4=self.build_resblock(512,layer_dims[3],strides=2)
        self.gap=layers.GlobalAveragePooling2D()
        self.fc=layers.Dense(num_class)

    def call(self, inputs, training=None):
        x=self.stem(inputs)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=self.gap(x)
        x=self.fc(x)
        return x

    def build_resblock(self,filter_num,blocks,strides=1):
        res_blocks=Sequential()
        res_blocks.add(BasicBlock(filter_num,strides=strides))
        for _ in range(1,blocks):
            res_blocks.add(BasicBlock(filter_num,strides=1))
        return res_blocks

def resnet18():
    return ResNet([2,2,2,2])

def resnet34():
    return ResNet([3,4,6,3])

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.int64)

    return x, y


def cifar100_data(batchsz=8, trainsz=1000):
    (x, y), (x_test, y_test) = datasets.cifar100.load_data()

    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_train = db_train.map(preprocess).take(trainsz).shuffle(trainsz).batch(batchsz)

    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).batch(batchsz)

    return db_train, db_test
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


def train(epoch, model, optimizer, train_ds, test_ds):
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(train_ds):
        loss, accuracy = trian_onestep(model, optimizer, x, y)
        if step % 100 == 0:
            print('Epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
    total_num = 0
    total_correct = 0
    for x, y in test_ds:
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
    train_ds, test_ds = cifar100_data()
    model=resnet18()
    model.build(input_shape=(None,32,32,3))
    model.summary()
    optimizer=optimizers.Adam(lr=1e-4)
    for epoch in range(50):
        loss, accuracy, acc = train(epoch, model, optimizer, train_ds, test_ds)
    print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy(), 'Test acc:', acc.numpy())

if __name__ == '__main__':
    print(tf.test.is_gpu_available)
    main()