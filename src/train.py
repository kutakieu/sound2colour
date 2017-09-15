from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys, getopt
from random import shuffle
import random

from pytube import YouTube
import librosa
import subprocess
import moviepy.editor as mp
from PIL import Image

import cv2
from python_speech_features import mfcc

# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = 50
display_step = 100
num_data = 0
sample_rate = 16000
directory = "../data"
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


def load_generic_audio_video(directory, sample_rate, video_list, video_index, len_video_list):

    # create or load a list of youtube videos (URL)
    # this function gets called every time the model runs out the given training data

    current_video = video_list[int(video_index % len_video_list)]
    download_youtube(directory, video_name=current_video)
    clip = mp.VideoFileClip(directory + "/tmp.mp4")
    clip.audio.write_audiofile(directory + "/tmp.wav")
    fps = int(clip.fps + 0.1)
    if fps not in [24,25,30]:
        return None
    if fps == 30:
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
        if fps == 30 and i % 5 == 4:
            continue
        img = cv2.blur(clip.get_frame(i),(100,100))
        img = cv2.resize(img,(16,9))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.reshape(144,3)
        Hue_vec = hsv[:,0]
        fragment = audio[i * sample_size: (i + 1) * sample_size]
        mfcc_feat = mfcc(fragment, 16000)
        # yield a set of data for each frame and corresponding audio's feature
        yield mfcc_feat, Hue_vec/360

def download_youtube(directory, video_name=None):
    subprocess.call(["rm", directory+"/tmp.wav", directory+"/tmp.mp4"])

    # video_id = "h6yJEHHT5eA"
    try:
        youtube = YouTube(video_name)
        youtube.set_filename('tmp')
    except:
        print("there is no video")

    try:
        video = youtube.get('mp4', '360p')
    except:
        print("there is no video for this setting")

    video.download(directory)


def RNN(x, weight, bias):


    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias

def main(argv):
    file_name = "../data/video_list.txt"
    video_list_file = open(file_name, "r")
    video_list = video_list_file.readlines()
    print(len(video_list))
    print("here")
    # exit()

    """define the model"""
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    prediction = tf.nn.relu(tf.matmul(outputs[-1], weight) + bias)

    # prediction = RNN(x, weight, bias)

    # prediction = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    # prediction = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    # logits = tf.cast(tf.argmax(prediction, 1), tf.float32)

    # Define loss and optimizer
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    loss = tf.losses.mean_squared_error(y, prediction)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluate model
    # correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, "../tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)
        # exit()
        step = 0
        stop = False
        # Keep training until reach max iterations
        while step < training_iters:
            iterator = load_generic_audio_video(directory, sample_rate, video_list, step, len(video_list))
            frame = 0
            x_ = np.zeros((1, n_steps,n_input))
            y_ = np.zeros((1, n_classes))
            for mfcc_feat, Hue_vec in iterator:
                x_[0, :n_steps-3, :] = x_[0, 3:,:]
                x_[0, n_steps-3:,:] = mfcc_feat
                y_[0, :] = Hue_vec
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: x_, y: y_})
                if frame % display_step == 0:
                    # Calculate batch loss
                    # print("LOSS")
                    current_loss = sess.run(loss, feed_dict={x: x_, y: y_})
                    print("Step " + str(step) + ", frame " + str(frame) + ", Minibatch Loss= " + \
                          "{:.6f}".format(current_loss))
                frame += 1
            step += 1
            save_path = saver.save(sess, "../tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)

        print("Optimization Finished!")

        print("Testing Accuracy:", \
            sess.run(loss, feed_dict={x: x_, y: y_}))


if __name__ == "__main__":
   main(sys.argv[1:])