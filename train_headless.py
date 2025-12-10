import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def main():
    data = np.load('imageDataNew-10-10-5.npy', allow_pickle=True)
    train = data[0:45]
    val = data[45:50]

    x = tf.placeholder('float')
    y = tf.placeholder('float')
    size = 10
    keep_rate = 0.8
    noslices = 5

    def conv3d(x_, W):
        return tf.nn.conv3d(x_, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def maxpool3d(x_):
        return tf.nn.max_pool3d(x_, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    def cnn(inp):
        inp = tf.reshape(inp, shape=[-1, size, size, noslices, 1])
        c1 = tf.nn.relu(conv3d(inp, tf.Variable(tf.random_normal([3, 3, 3, 1, 32]))) + tf.Variable(tf.random_normal([32])))
        c1 = maxpool3d(c1)
        c2 = tf.nn.relu(conv3d(c1, tf.Variable(tf.random_normal([3, 3, 3, 32, 64]))) + tf.Variable(tf.random_normal([64])))
        c2 = maxpool3d(c2)
        c3 = tf.nn.relu(conv3d(c2, tf.Variable(tf.random_normal([3, 3, 3, 64, 128]))) + tf.Variable(tf.random_normal([128])))
        c3 = maxpool3d(c3)
        c4 = tf.nn.relu(conv3d(c3, tf.Variable(tf.random_normal([3, 3, 3, 128, 256]))) + tf.Variable(tf.random_normal([256])))
        c4 = maxpool3d(c4)
        c5 = tf.nn.relu(conv3d(c4, tf.Variable(tf.random_normal([3, 3, 3, 256, 512]))) + tf.Variable(tf.random_normal([512])))
        c5 = maxpool3d(c5)
        flat = tf.keras.layers.Flatten()(c5)
        d1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(flat)
        d1 = tf.nn.dropout(d1, keep_prob=keep_rate)
        out = tf.keras.layers.Dense(2)(d1)
        return out

    pred = cnn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample in train:
                X, Y = sample[0], sample[1]
                _, c = sess.run([opt, cost], feed_dict={x: X, y: Y})
                epoch_loss += c
            correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            acc = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Epoch', epoch + 1, 'loss:', epoch_loss)
            print('Accuracy:', acc.eval({x: [i[0] for i in val], y: [i[1] for i in val]}))


if __name__ == '__main__':
    main()


