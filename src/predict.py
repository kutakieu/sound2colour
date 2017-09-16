import tensorflow as tf
import numpy as np
import sys, getopt
from python_speech_features import mfcc
import moviepy.editor as mp
import librosa
import cv2

# Network Parameters
n_input = 13 # MNIST data input (img shape: 28*28)
n_steps = 12 # timesteps
n_hidden = 200 # hidden layer num of features
n_classes = 16*9 # MNIST total classes (0-9 digits)

# tf Graph input
# x = tf.placeholder("float", [None, n_steps, n_input])
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# Define weights

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]), name="weight")
bias = tf.Variable(tf.random_normal([n_classes]), name="bias")

W_1 = tf.get_variable("W_1", [n_input, 360], dtype=tf.float32, initializer=tf.random_normal_initializer())
b_1 = tf.get_variable("b_1", [360], dtype=tf.float32, initializer=tf.random_normal_initializer())
W_2 = tf.get_variable("W_2", [360, 144], dtype=tf.float32, initializer=tf.random_normal_initializer())
b_2 = tf.get_variable("b_2", [144], dtype=tf.float32, initializer=tf.random_normal_initializer())

dropout_1 = tf.placeholder(tf.float32)
dropout_2 = tf.placeholder(tf.float32)

def RNN(x, weight, bias):


    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias

def NN(x):
    # W_1 = tf.get_variable("W_1", [n_input, 72], dtype=tf.float32, initializer=tf.random_normal_initializer())
    # b_1 = tf.get_variable("b_1", [72], dtype=tf.float32, initializer=tf.random_normal_initializer())
    #
    #
    # W_2 = tf.get_variable("W_2", [72, 144], dtype=tf.float32, initializer=tf.random_normal_initializer())
    # b_2 = tf.get_variable("b_2", [144], dtype=tf.float32, initializer=tf.random_normal_initializer())


    output_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(x, W_1) + b_1), dropout_1)

    # output_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(output_1, W_2) + b_2), dropout_2)
    output_2 = tf.nn.dropout(tf.matmul(output_1, W_2) + b_2, dropout_2)

    return output_2

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
    clip = mp.VideoFileClip(directory + "/tmp.mp4")
    fps = int(clip.fps + 0.1)
    audio, _ = librosa.load(directory + "/tmp.wav", sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    len = audio.shape[0]
    len = len - (len % sample_rate)
    fps = 24
    sample_size = int(sample_rate / fps)

    # to get frame
    # clip.get_frame(0)
    # to get image instance from numpy array
    num_frames = int(len / 16000) * fps
    for i in range(num_frames):
        if fps == 30 and i % 5 == 4:
            continue
        img = cv2.blur(clip.get_frame(i),(100,100))
        img = cv2.resize(img,(16,9))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.reshape(144,3)
        Hue_vec = hsv[:, 0]
        fragment = audio[i * sample_size: (i + 1) * sample_size]
        mfcc_feat = mfcc(fragment, 16000)
        # yield a set of data for each frame and corresponding audio's feature
        yield mfcc_feat, Hue_vec

def main(argv):
    # logits = RNN(x, weight, bias)


    # passing only the last step's output
    # output = tf.reshape(outputs[0, -1], [1, n_hidden])

    # logits = tf.cast(tf.add(tf.nn.relu(tf.matmul(output, weight) + bias) * 360, 0.5), tf.int32)

    logits = NN(x)
    prediction = tf.cast(tf.add(logits * 360, 0.5), tf.int32)

    wave = generate_sine_wave()
    mfcc_feat = mfcc(wave, 16000)
    x_ = np.zeros((1, n_steps, n_input))
    x_[0, :n_steps - 3, :] = x_[0, 3:, :]
    x_[0, n_steps - 3:, :] = mfcc_feat

    x__ = np.zeros((1, n_input))
    x__[0, :] = np.sum(mfcc_feat, axis=0)
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
        print("W_1 : %s" % W_1.eval())
        print("b_1 : %s" % b_1.eval())
        print("W_2 : %s" % W_2.eval())
        print("b_2 : %s" % b_2.eval())

        pred = sess.run(logits, feed_dict={x: x__, dropout_1: 1, dropout_2: 1})
        print(pred)
        x_ = np.zeros((1, n_steps, n_input))
        x__ = np.zeros((1, n_input))
        for mfcc_feat, Hue_vec in iterator:
            print("mfcc feature")
            print(mfcc_feat)
            print("Hue vector")
            print(Hue_vec)
            x_[0, :n_steps - 3, :] = x_[0, 3:, :]
            x_[0, n_steps - 3:, :] = mfcc_feat
            x__[0, :] = np.sum(mfcc_feat, axis=0)
            pred = sess.run(logits, feed_dict={x: x__, dropout_1: 1, dropout_2: 1})
            print(pred.astype(int))
            input('Press enter to continue: ')


if __name__ == "__main__":
  main(sys.argv[1:])