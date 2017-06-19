import numpy
import tensorflow as tf

# Set up the model parameters 
W = tf.Variable(.3, tf.float32)
b = tf.Variable(-.3, tf.float32)
# Set up the model inputs and outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b

# The Loss function
loss = tf.reduce_sum(tf.square(linear_model - y))

# Opimizer
optimize = tf.train.GradientDescentOptimizer(0.01)

# Trainer
train = optimize.minimize(loss)

# Training Data
x_train = [1, 2, 3, 4]
y_train= [0, -1, -2, -3]

# Set up the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Training

for i in range(1000):
    sess.run(train, {x: x_train,y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y:y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

