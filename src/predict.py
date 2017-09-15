import tensorflow as tf
import numpy as np
import sys, getopt
from python_speech_features import mfcc
import moviepy.editor as mp
import librosa

# Network Parameters
n_input = 13 # MNIST data input (img shape: 28*28)
n_steps = 12 # timesteps
n_hidden = 200 # hidden layer num of features
n_classes = 16*9 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
# Define weights

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]), name="weight")
bias = tf.Variable(tf.random_normal([n_classes]), name="bias")

def RNN(x, weight, bias):


    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias

def generate_sine_wave():
    SAMPLE_RATE_HZ = 16000  # Hz
    SAMPLE_DURATION = 1/24  # Seconds
    sample_period = 1.0 / SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)
    wave = 0.6 * np.sin(times * 2.0 * np.pi * 440)
    return wave

def generate_test_audio(directory, sample_rate):

    # create or load a list of youtube videos (URL)
    # this function gets called every time the model runs out the given training data
    fps = 24
    audio, _ = librosa.load(directory + "/tmp.wav", sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    len = audio.shape[0]
    len = len - (len % sample_rate)

    sample_size = int(sample_rate / fps)

    # to get frame
    # clip.get_frame(0)
    # to get image instance from numpy array
    num_frames = int(len / 16000) * fps
    for i in range(num_frames):

        fragment = audio[i * sample_size: (i + 1) * sample_size]
        mfcc_feat = mfcc(fragment, 16000)
        # yield a set of data for each frame and corresponding audio's feature
        yield mfcc_feat

def main(argv):
    # logits = RNN(x, weight, bias)

    # prediction = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    # passing only the last step's output
    output = tf.reshape(outputs[0, -1], [1, n_hidden])
    logits = tf.nn.relu(tf.matmul(output, weight) + bias)

    prediction = tf.cast(tf.argmax(logits, 1), tf.float32)

    wave = generate_sine_wave()
    mfcc_feat = mfcc(wave, 16000)
    x_ = np.zeros((1, n_steps, n_input))
    x_[0, :n_steps - 3, :] = x_[0, 3:, :]
    x_[0, n_steps - 3:, :] = mfcc_feat
    # x_.reshape((1, n_steps, n_input))
    print(x_.shape)

    # sess = tf.Session()
    # saver = tf.train.Saver()
    # saver.restore(sess, "../tmp/model.ckpt")
    # pred = sess.run(prediction, feed_dict={x: x_})
    # print(pred)

    iterator = generate_test_audio("../data", 16000)

    with tf.Session() as sess:
        # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(sess, "../tmp/model.ckpt")
        print("Model restored.")
        # Check the values of the variables
        # print("weight : %s" % weight.eval())
        # print("bias : %s" % bias.eval())
        pred = sess.run(logits, feed_dict={x: x_})
        print(pred)
        x_ = np.zeros((1, n_steps, n_input))
        for mfcc_feat in iterator:
            x_[0, :n_steps - 3, :] = x_[0, 3:, :]
            x_[0, n_steps - 3:, :] = mfcc_feat
            pred = sess.run(logits, feed_dict={x: x_})
            print(pred)
            input('Press enter to continue: ')


if __name__ == "__main__":
  main(sys.argv[1:])