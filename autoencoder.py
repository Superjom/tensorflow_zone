'''
Various autoencoder implementations.

Much efforts were paid to add tf.summary operaions to show what information
the autoencoder learns.

Following models are implemented with pure tensorflow APIs.
'''
import tensorflow as tf


class BaseModule(object):
    def build_model(self):
        pass


class Autoencoder(object):
    def __init__(self,
                 name_scope,
                 input,
                 n_input,
                 n_hidden,
                 transfer_function=tf.nn.sigmoid,
                 optimizer=tf.train.GradientDescentOptimizer):
        '''
        @input: placeholder or tensor from other ops
        @n_input: dimension of the input tensor
        @n_hidden: dimension of the hidden vector
        @transfer_function: activation function to reconstruct the input
        @name of name_scope
        '''
        self.name_scope = name_scope
        self.input = input
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.weights = {}
        self.model = {}
        self.session = None
        self.summary_op = None

    def build_model(self):
        if self.weights:
            return

        # init model parameters
        with tf.name_scope(self.name_scope + '_compression'):
            self.weights['w0'] = tf.get_variable(
                'w0',
                shape=[self.n_input, self.n_hidden],
                initializer=tf.random_normal_initializer())
            self.weights['b0'] = tf.get_variable('b0', shape=[self.n_hidden])

        with tf.name_scope(self.name_scope + '_reconstruction'):
            self.weights['w1'] = tf.get_variable(
                'w1',
                shape=[self.n_hidden, self.n_input],
                initializer=tf.random_normal_initializer())
            self.weights['b1'] = tf.get_variable('b1', shape=[self.n_input])

        with tf.name_scope(self.name_scope):
            self.model['hidden'] = self.transfer_function(tf.matmul(
                self.input, self.weights['w0']) + self.weights['b0'])
            self.model['reconstruction'] = self.transfer_function(tf.matmul(
                self.model['hidden'], self.weights['w1']) + self.weights['b1'])
            self.loss = tf.pow((self.input - self.model['reconstruction']), 2)
            self.cost = tf.reduce_mean(self.loss)

        tf.summary.scalar('cost', self.cost)
        tf.summary.histogram('cost', self.cost)

        w0_reshaped = tf.reshape(self.weights['w0'], [-1, 784, 200, 1])
        tf.summary.image('w0', w0_reshaped, max_outputs=3, collections=None)

        num_reshaped = tf.reshape(self.model['reconstruction'], [-1, 28, 28, 1])
        tf.summary.image('img', num_reshaped, max_outputs=3, collections=None)

        self.summary_op = tf.summary.merge_all()
        assert self.model, 'error in build model'
        for key, value in self.weights.items():
            print key, value

    def _init_session(self, session):
        if self.session is None:
            if session is None:
                session = tf.Session()
                init = tf.global_variables_initializer()
                session.run(init)
            self.session = session

    def fit(self, X, learning_rate=0.003, session=None, log='./logs'):
        '''
        @X: input dataset
        '''
        self.build_model()
        self._init_session(session)

        optimizer = self.optimizer(learning_rate).minimize(self.loss)
        opt, cost, summary = self.session.run([optimizer, self.cost, self.summary_op],
                                     feed_dict={self.input: X})
        return cost, summary

    def generate(self, X):
        self.build_model()
        self._init_session()

        return self.session.run(self.model['hidden'],
                                feed_dict={self.input: X})


if __name__ == '__main__':

    from tensorflow.examples.tutorials.mnist import input_data
    import time

    mnist = input_data.read_data_sets('mnist_data', one_hot=True)
    session = tf.InteractiveSession()

    ae = Autoencoder(name_scope='ae',
                     input=tf.placeholder(tf.float32,
                                          shape=[None, 784],
                                          name='x'),
                     n_input=784,
                     n_hidden=200, )

    summary_writer = tf.summary.FileWriter('./logs/train')

    for i in range(500):
        batch = mnist.train.next_batch(500)
        cost, summary = ae.fit(batch[0])
        print 'cost:', cost
        summary_writer.add_summary(summary, i)
        time.sleep(2)
