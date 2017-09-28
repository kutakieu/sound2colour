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
# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])
# Define weights

# weight = tf.Variable(tf.random_normal([n_hidden, n_classes]), name="weight")
# bias = tf.Variable(tf.random_normal([n_classes]), name="bias")

# W_1 = tf.get_variable("W_1", [n_input, 360], dtype=tf.float32, initializer=tf.random_normal_initializer())
# b_1 = tf.get_variable("b_1", [360], dtype=tf.float32, initializer=tf.random_normal_initializer())
# W_2 = tf.get_variable("W_2", [360, 144], dtype=tf.float32, initializer=tf.random_normal_initializer())
# b_2 = tf.get_variable("b_2", [144], dtype=tf.float32, initializer=tf.random_normal_initializer())

# weight_1 = tf.Variable(tf.random_normal([n_hidden, 240]), name="weight_1")
# bias_1 = tf.Variable(tf.random_normal([240]), name="bias_1")
# weight_2 = tf.Variable(tf.random_normal([240, n_classes]), name="weight_2")
# bias_2 = tf.Variable(tf.random_normal([n_classes]), name="bias_2")
# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
# outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

dropout_1 = tf.placeholder(tf.float32)
dropout_2 = tf.placeholder(tf.float32)

def RNN(x):
    weight = tf.Variable(tf.random_normal([n_hidden, n_classes]), name="weight")
    bias = tf.Variable(tf.random_normal([n_classes]), name="bias")

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias

def RNN2(x):
    weight_1 = tf.Variable(tf.random_normal([n_hidden, 240]), name="weight_1")
    bias_1 = tf.Variable(tf.random_normal([240]), name="bias_1")
    weight_2 = tf.Variable(tf.random_normal([240, n_classes]), name="weight_2")
    bias_2 = tf.Variable(tf.random_normal([n_classes]), name="bias_2")

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    fc_output = tf.nn.dropout(tf.sigmoid(tf.matmul(outputs[-1], weight_1) + bias_1), dropout_1)

    return tf.nn.dropout(tf.matmul(fc_output, weight_2) + bias_2, dropout_1)

def NN(x):
    W_1 = tf.get_variable("W_1", [n_input, 360], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_1 = tf.get_variable("b_1", [360], dtype=tf.float32, initializer=tf.random_normal_initializer())


    W_2 = tf.get_variable("W_2", [360, 144], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_2 = tf.get_variable("b_2", [144], dtype=tf.float32, initializer=tf.random_normal_initializer())

    # ReLU
    # output_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W_1) + b_1), dropout_1)
    # Sigmoid
    output_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(x, W_1) + b_1), dropout_1)

    # Relu
    # output_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(output_1, W_2) + b_2), dropout_2)
    # Sigmoid
    output_2 = tf.nn.dropout(tf.matmul(output_1, W_2) + b_2, dropout_2)
    # without dropout for this layer
    # output_2 = tf.nn.relu(tf.matmul(output_1, W_2) + b_2)
    return output_2

def NN2(x):
    W_1 = tf.get_variable("W_1", [n_input, 180], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_1 = tf.get_variable("b_1", [180], dtype=tf.float32, initializer=tf.random_normal_initializer())


    W_2 = tf.get_variable("W_2", [180, 360], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_2 = tf.get_variable("b_2", [360], dtype=tf.float32, initializer=tf.random_normal_initializer())

    W_3 = tf.get_variable("W_3", [360, 144], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_3 = tf.get_variable("b_3", [144], dtype=tf.float32, initializer=tf.random_normal_initializer())

    # ReLU
    output_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W_1) + b_1), dropout_1)
    # Sigmoid
    # output_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(x, W_1) + b_1), dropout_1)

    # Relu
    # output_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(output_1, W_2) + b_2), dropout_2)
    # Sigmoid
    output_2 = tf.nn.dropout(tf.sigmoid(tf.matmul(output_1, W_2) + b_2), dropout_2)
    # without dropout for this layer
    # output_2 = tf.nn.relu(tf.matmul(output_1, W_2) + b_2)

    output_3 = tf.nn.dropout(tf.matmul(output_2, W_3) + b_3, dropout_2)
    return output_3

def generate_sine_wave():
    SAMPLE_RATE_HZ = 16000  # Hz
    SAMPLE_DURATION = 1/24  # Seconds
    sample_period = 1.0 / SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)
    wave = 0.6 * np.sin(times * 2.0 * np.pi * 440)
    return wave

def generate_test_audio(directory, sample_rate, len_video):

    # create or load a list of youtube videos (URL)
    # this function gets called every time the model runs out the given training data
    # clip = mp.VideoFileClip(directory + "/tmp_.mp4")
    # fps = int(clip.fps + 0.1)
    audio, _ = librosa.load(directory + "/test.wav", sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    len = audio.shape[0]
    len = len - (len % sample_rate)
    fps = 24
    sample_size = int(sample_rate / fps)

    # to get frame
    # clip.get_frame(0)
    # to get image instance from numpy array
    num_frames = int(len / 16000) * fps
    len_video.append(num_frames)

    for i in range(num_frames):
        # if fps == 30 and i % 5 == 4:
        #     continue
        # img = cv2.blur(clip.get_frame(i),(100,100))
        # img = cv2.resize(img,(16,9))
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv = hsv.reshape(144,3)
        # Hue_vec = hsv[:, 0]
        fragment = audio[i * sample_size: (i + 1) * sample_size]
        mfcc_feat = mfcc(fragment, 16000)
        # yield a set of data for each frame and corresponding audio's feature
        # yield mfcc_feat, Hue_vec
        yield mfcc_feat

def main(argv):
    # logits = RNN(x, weight, bias)


    # passing only the last step's output
    # output = tf.reshape(outputs[0, -1], [1, n_hidden])

    # logits = tf.cast(tf.add(tf.nn.relu(tf.matmul(output, weight) + bias) * 360, 0.5), tf.int32)
    isRNN = True

    if isRNN:
        # x = tf.placeholder("float", [None, n_steps, n_input])
        # weight_1 = tf.Variable(tf.random_normal([n_hidden, 240]), name="weight_1")
        # bias_1 = tf.Variable(tf.random_normal([240]), name="bias_1")
        # weight_2 = tf.Variable(tf.random_normal([240, n_classes]), name="weight_2")
        # bias_2 = tf.Variable(tf.random_normal([n_classes]), name="bias_2")
        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
        # fc_output = tf.nn.dropout(tf.sigmoid(tf.matmul(outputs[-1], weight_1) + bias_1), dropout_1)
        # logits = tf.nn.dropout(tf.matmul(fc_output, weight_2) + bias_2, dropout_1)

        x = tf.placeholder("float", [None, n_steps, n_input])
        weight_1 = tf.get_variable("weight_1", [n_hidden, 240], dtype=tf.float32)
        bias_1 = tf.get_variable("bias_1", [240], dtype=tf.float32)
        weight_2 = tf.get_variable("weight_2", [240, n_classes], dtype=tf.float32)
        bias_2 = tf.get_variable("bias_2", [n_classes], dtype=tf.float32)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
        fc_output = tf.nn.dropout(tf.sigmoid(tf.matmul(outputs[-1], weight_1) + bias_1), dropout_1)
        logits = tf.nn.dropout(tf.matmul(fc_output, weight_2) + bias_2, dropout_1)

        # logits = RNN2(x)
    else:
        W_1 = tf.get_variable("W_1", [n_input, 360], dtype=tf.float32, initializer=tf.random_normal_initializer())
        b_1 = tf.get_variable("b_1", [360], dtype=tf.float32, initializer=tf.random_normal_initializer())
        W_2 = tf.get_variable("W_2", [360, 144], dtype=tf.float32, initializer=tf.random_normal_initializer())
        b_2 = tf.get_variable("b_2", [144], dtype=tf.float32, initializer=tf.random_normal_initializer())
        x = tf.placeholder("float", [None, n_input])
        logits = NN2(x)
    # prediction = tf.cast(tf.add(logits * 360, 0.5), tf.int32)

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
    num_frames = []
    iterator = generate_test_audio("../data", 16000, num_frames)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.

        saver.restore(sess, "../tmp/model.ckpt")
        print("Model restored.")
        # Check the values of the variables
        print("W_1 : %s" % weight_1.eval())
        print(weight_1.eval().shape)
        # exit()
        # print("b_1 : %s" % b_1.eval())
        # print("W_2 : %s" % W_2.eval())
        # print("b_2 : %s" % b_2.eval())

        # pred = sess.run(logits, feed_dict={x: x__, dropout_1: 1, dropout_2: 1})
        # print(pred)
        if isRNN:
            x_ = np.zeros((1, n_steps, n_input))
        else:
            x_ = np.zeros((1, n_input))
        y_ = np.zeros((1, n_classes))
        ratio = 100

        """video writer"""
        # video_writer = cv2.VideoWriter('video.mp4', -1, 1, (9*ratio, 16*ratio))
        video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 24, (16*ratio,9*ratio))
        video_frame = 0
        # for mfcc_feat, Hue_vec in iterator:
        for mfcc_feat in iterator:
            # print(mfcc_feat)
            try:
                if isRNN:
                    x_[0, :n_steps - 3, :] = x_[0, 3:, :]
                    x_[0, n_steps - 3:, :] = mfcc_feat
                else:
                    x_[0, :] = np.sum(mfcc_feat, axis=0)
                # y_[0, :] = Hue_vec
                pred = sess.run(logits, feed_dict={x: x_, dropout_1: 1, dropout_2: 1})
                pred = np.uint8(pred)
                # print(pred.astype(int))
                # print(pred[-1].shape)
                img = np.zeros((9*ratio, 16*ratio ,3),dtype=np.uint8)
                img[:,:,1] = np.uint8(np.random.randint(0,255,size=img.shape[:2]))
                img[:, :, 2] = np.uint8(np.random.randint(0, 255, size=img.shape[:2]))
                for i in range(9*ratio):
                    # print(i)
                    for j in range(16*ratio):
                        img[i,j,0] = pred[-1][int(j/ratio) + int(i/ratio)*16]
                        # print(pred[-1][int(j/ratio) + int(i/ratio)*16])
                rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
                rgb = cv2.blur(rgb,(100,100))
                # cv2.imwrite("out.jpg", rgb)
                video_writer.write(rgb)
                # if video_frame == 60:
                #     break
                if video_frame % 10 == 0:
                    print(str(video_frame) + "/" + str(num_frames[0]))
                # input('Press enter to continue: ')
                video_frame += 1
            except:
                break
        # cv2.destroyAllWindows()
        video_writer.release()


if __name__ == "__main__":
  main(sys.argv[1:])