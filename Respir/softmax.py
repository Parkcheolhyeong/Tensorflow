import tensorflow as tf
import numpy as np

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
tf.debugging.set_log_device_placement(True)

xy = np.loadtxt('test.csv', delimiter = ',', dtype = np.float32)
xy = np.transpose(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data)
print(y_data)
nb_classes = 2

X = tf.placeholder(tf.float32, [None, 600])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([600, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

logits = tf.matmul(X, W) +b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs/respir_softmax")
    writer.add_graph(sess.graph)  # Show the graph

    sess.run(tf.global_variables_initializer())
    for step in range(100000):
        sess.run(optimizer, feed_dict = {X: x_data, Y: y_data})
        if step % 1000 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

        writer.add_summary(summary, global_step=step)

    pred =- sess.run(prediction, feed_dict= {X: x_data})
    for p, y, in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p==int(y), p, int(y)))