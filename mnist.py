'''
mnist hand-write images classification model.

by fully connected neural networks.

reference:

- Official tutorial from https://www.tensorflow.org/get_started/mnist/pros
- Keras tutorial
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import logging

mnist = input_data.read_data_sets('mnist_data', one_hot=True)
session = tf.InteractiveSession()


class ModelbyTF(object):
    '''
    the model using pure tensorflow APIs.
    '''

    def __init__(self, dim=784, nclasses=10, learning_rate=0.003):
        self.dim = dim
        self.nclasses = nclasses
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, shape=[None, self.dim])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.nclasses])

    def __call__(self):
        self.build_model()
        self.train()

    def build_model(self, layer_sizes=[]):
        # declare model parameters
        self.W = tf.Variable(tf.zeros([self.dim, self.nclasses]))
        self.b = tf.Variable(tf.zeros([self.nclasses]))

        self.y = tf.matmul(self.x, self.W) + self.b
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_, logits=self.y))

        self.loss = cross_entropy
        tf.global_variables_initializer().run()

    def train(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_step = optimizer.minimize(self.loss)

        # evaluate the accuracy
        correct_prediction = tf.equal(
            tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32))

        for i in range(5000):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})

            if i % 100 == 0:
                precision = accuracy.eval(
                    feed_dict={self.x: mnist.test.images,
                               self.y_: mnist.test.labels})
                print "batch %d precision:%f" % (i, precision)

                W = tf.Print(self.W, [self.W])
                b = tf.Print(self.b, [self.b])
                print session.run([W, b])


class ModelByKeras(object):
    '''
    the model configuration using Keras APIs.
    '''
    pass


if __name__ == '__main__':
    ModelbyTF()()
