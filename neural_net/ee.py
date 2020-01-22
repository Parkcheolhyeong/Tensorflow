import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
xy_smaple = np.loadtxt('Merge.csv', delimiter =',', dtype = np.float32)
xy_smaple = np.transpose(xy_smaple)
xy_test = np.loadtxt('Merge_test.csv', delimiter =',', dtype = np.float32)
xy_test = np.transpose(xy_test)

x_data = xy_smaple[:, 0:-1]
y_data = xy_smaple[:, [-1]]

X = tf.placeholder(tf.float32, [None, 600])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([600, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data}
    )
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


    for p, y, in zip(c, y_data.flatten()):
        value_predict = "무호흡" if p != 0 else "호흡s"
        value_true = "무호흡" if int(y) == 1 else "호흡"
        #print("[{}] Prediction: {}({}) True Y: {}({}) Result: {}".format(p==int(y), "무호흡" if p==1 else "호흡", p, "무호흡" if int(y)==1 else "호흡", int(y), '결과'))
        #print("[{}] Prediction: {}({}) True Y: {}({}) Result: {}".format(p!=0, "무호흡" if p!=0 else "호흡", p, "무호흡" if int(y)==1 else "호흡", int(y), '결과'))
        print("[{}] Prediction: {}({}) True Y: {}({}) Result: {}".format((value_predict is value_true), "무호흡" if p!=0 else "호흡", p, "무호흡" if int(y)==1 else "호흡", int(y), '결과'))