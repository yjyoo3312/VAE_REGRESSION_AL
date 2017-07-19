'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os
import sys
import glob

import numpy as np
import prettytensor as pt

import tensorflow as tf
from scipy.misc import imsave
from scipy.misc import imread
from scipy.misc import imresize

import pdb

from tensorflow.examples.tutorials.mnist import input_data

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 400, "batch size")
flags.DEFINE_integer("V", 8, "number of videos in the batch")
flags.DEFINE_integer("L", 200, "supervised pair size")
flags.DEFINE_integer("M", 200, "unsupervised z size")
flags.DEFINE_integer("T", 32, "training sample")
flags.DEFINE_integer("domain_size", 64, "dimension of the input domain")
flags.DEFINE_integer("dist_size", 3, "dimension of the input domain")
flags.DEFINE_integer("updates_per_epoch", 200, "number of updates per epoch")
flags.DEFINE_integer("random_size", 32, "dimension of the random input")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_integer("hidden_size", 64, "size of the hidden VAE unit")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("kernel_sig", 0.5, "ard kernel sigma")
flags.DEFINE_float("kernel_sig_error", 0.5, "error term")
flags.DEFINE_float("kernel_base", 1.0, "ard kernel base")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_string("option", 'GP', "Regression option")
flags.DEFINE_boolean("isSave", False, "true: use saved data, false: do training")
flags.DEFINE_boolean("regression", True, "do not learn regression if False")
flags.DEFINE_boolean("pretrain", True, "free training of the data")
flags.DEFINE_boolean("pProc", False, "post processing is yet done?")

# order 1. preTrain On, preTrain Off and genCid On,
FLAGS = flags.FLAGS
Imsize = 64


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

def discriminator(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    input_img_tensor = input_tensor[:, :Imsize*Imsize*3]
    return (pt.wrap(input_img_tensor).
            reshape([None, Imsize, Imsize, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 256, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(1, activation_fn=tf.nn.sigmoid)).tensor # one for gaussian prior, the other for axis prior

def encoder(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    input_img_tensor = input_tensor[:, :Imsize * Imsize * 3]
    return (pt.wrap(input_img_tensor).
            reshape([None, Imsize, Imsize, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 256, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.hidden_size + 2,
                            activation_fn=None)).tensor  # one for gaussian prior, the other for axis prior


def decoder(input_tensor=None, y_tensor=None, e_tensor=None, dist_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, half number of the batch
        y_tensor: domain of the vectors, full number of batch

    Returns:
        A tensor that expresses the decoder network: return batch number of reconstructed image
        half of them is a result of original encoded vector
        the other half is a generation w.r.t gaussian regression
    '''
    # epsilon_c = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size]) #random parameters for co-domain
    # epsilon_d = tf.random_normal([FLAGS.batch_size, FLAGS.domain_size])

    epsilon_c = e_tensor[:, :FLAGS.hidden_size]  # random parameters for co-domain
    epsilon_d = e_tensor[:, FLAGS.hidden_size:]  # random parameters for domain

    if dist_tensor is None:  # and y_tensor is None
        mean = None
        stddev = None
        stddev_y = None

        mean_y = y_tensor[:, :FLAGS.domain_size]
        domain_sample = mean_y + 0.5 * epsilon_d
        latent_sample = epsilon_c

        input_sample = tf.concat([latent_sample, domain_sample], 1)

    else:
        if FLAGS.regression is True:  # learn regression
            z_sup = input_tensor[:FLAGS.L, :FLAGS.hidden_size]
            z_enc = input_tensor[FLAGS.L:, :FLAGS.hidden_size]
            z_sup_stddev = input_tensor[:FLAGS.L, FLAGS.hidden_size: 2 * FLAGS.hidden_size]

            tf_sigma = tf.reduce_mean(
                tf.sqrt(tf.exp(input_tensor[:FLAGS.L, 2 * FLAGS.hidden_size: 2 * FLAGS.hidden_size + 1])))
            # tf_sigma_vec = tf.matmul(tf.ones([FLAGS.batch_size, 1], dtype=np.float32), tf_sigma)

            stepL = int(FLAGS.L / FLAGS.V)
            stepM = int(FLAGS.M / FLAGS.V)

            z_list = []
            z_stddev_list = []
            for ind in range(0, FLAGS.V):  # for the number of video - FLAG later
                # linear regression according to each video
                z_token, z_token_stddev = regression(dist_tensor[ind * stepL:(ind + 1) * stepL, :FLAGS.dist_size],
                                                     dist_tensor[FLAGS.L + ind * stepM:FLAGS.L + (ind + 1) * stepM,
                                                     :FLAGS.dist_size],
                                                     z_sup[ind * stepL:(ind + 1) * stepL, :],
                                                     z_sup_stddev[ind * stepL:(ind + 1) * stepL, :],
                                                     tf_sigma,
                                                     option='GP')
                z_tokens = tf.unstack(z_token)
                z_tokens_stddev = tf.unstack(z_token_stddev)

                for jnd in range(0, len(z_tokens)):
                    z_list.append(z_tokens[jnd])
                    z_stddev_list.append(z_tokens_stddev[jnd])

            z_unsup = tf.stack(z_list)
            z_unsup_stddev = tf.stack(z_stddev_list)

            mean = tf.concat([z_sup, z_unsup], 0)  # total number should be same as FLAGS.batch_size
            stddev = tf.concat([tf.sqrt(tf.exp(z_sup_stddev)), tf.sqrt(z_unsup_stddev)], 0)

            latent_sample = mean + epsilon_c * stddev

            mean_y = y_tensor[:, :FLAGS.domain_size]
            stddev_y = tf.sqrt(tf.exp(input_tensor[:FLAGS.batch_size, 2 * FLAGS.hidden_size + 1:]))

            domain_sample = mean_y + epsilon_d * stddev_y



        else:
            mean = input_tensor[:, :FLAGS.hidden_size]
            stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:2 * FLAGS.hidden_size]))
            latent_sample = mean + epsilon_c * stddev

            mean_y = y_tensor[:, :FLAGS.domain_size]
            domain_sample = mean_y + 0.5 * epsilon_d

        # final sample P(z|y)
        input_sample = tf.concat([latent_sample, domain_sample], 1)

    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size + FLAGS.domain_size]).
            deconv2d(4, 256, edges='VALID').
            deconv2d(5, 128, edges='VALID').
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
            ).tensor, mean, stddev  # last 1 channel means foreground region

def amplifier_enc(input_tensor):
    # amplifying the resultant image - encoder

    #input_img_tensor = input_tensor[:, :Imsize * Imsize * 3]

    tnsor = (pt.wrap(input_tensor).
             reshape([None, Imsize, Imsize, 3]).
             conv2d(4, 32, stride=2).
             conv2d(4, 128, stride=2).
             conv2d(3, 256, stride=1).
             conv2d(3, 512, edges='VALID')).tensor
             #dropout(0.9).
             #flatten().
             #fully_connected(2 * FLAGS.hidden_size, activation_fn=None)).tensor

    return tnsor

def residual_block(input_tensor):

    tnsor = (pt.wrap(input_tensor).
             conv2d(3, 512, stride=1, edges='SAME').
             conv2d(3, 512, stride=1, edges='SAME')).tensor

    return input_tensor + tnsor



def amplifier_dec(input_tensor):
    # amplifying the resultant image - decoder


    tnsor = (pt.wrap(input_tensor). #14 by 14 * 512
            reshape([FLAGS.batch_size, 14, 14, 512]).
            deconv2d(3, 256, edges='VALID').
            deconv2d(5, 128, stride=2).
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 3, stride=1, activation_fn=tf.nn.sigmoid)
            ).tensor  # last 1 channel means foreground region

    return tnsor

def amplifier_discriminator(input_tensor):
    # amplifying the resultant image - encoder

    #input_img_tensor = input_tensor[:, :Imsize *2* Imsize*2 * 3]

    tnsor = (pt.wrap(input_tensor).
             reshape([None, Imsize*2, Imsize*2, 3]).
             conv2d(4, 32, stride=2).
             conv2d(4, 128, stride=2).
             conv2d(3, 256, stride=2).
             conv2d(3, 512, edges='VALID').
             dropout(0.9).
             flatten().
             fully_connected(1, activation_fn=tf.nn.sigmoid)).tensor

    return tnsor


def regression(y_sup, y_unsup, z_sup, z_sup_stddev, tf_sigma, option='Linear'):
    '''
    Regression of the data: we get unsupervised z_unsup, z_stddev_unsup as a result
    Args:
        y_sup: domain of the supervised data
        y_unsup: domain of the unsupervised data
        z_sup: supervised data
        z_sup_stddev: variance of supervised data
        tf_sigma: sigma value for Gaussian ARD Kernel. tensor. not used in Linear
        option: 'GP' -> Gaussian process regression, 'Linear' -> Linear regression

    Returns:
        z_unsup: reconstructed data
        z_unsup_stddev: variance of the reconstructed data

    '''

    if option == 'Linear':  # Linear Regression
        beta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(y_sup), y_sup)),
                                   tf.transpose(y_sup)), z_sup)
        beta_stddev = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(y_sup), y_sup)),
                                          tf.transpose(y_sup)), z_sup_stddev)

        z_unsup = tf.matmul(y_unsup, beta)  # [M D]
        z_unsup_stddev = tf.exp(tf.matmul(y_unsup, beta_stddev))  # [M D] exp according to VAE

    elif option == 'GP':  # Gaussian process Regression

        # Number of batch samples
        L = int(FLAGS.L / FLAGS.V)
        M = int(FLAGS.M / FLAGS.V)
        B = L + M  # total size

        # define domain matirx
        y_tot = tf.concat([y_sup, y_unsup], 0)  # len(y_tot) = B
        y_listT = []
        # s_listT = []
        for ind in range(0, B):
            y_list = tf.unstack(tf.transpose(y_tot))
            # s_list = tf.unpack(tf.transpose(sigma_tot))
            y_listT.append(y_list[0])
            # s_listT.append(s_list[0])

        # s_Mat = tf.mul(tf.pack(s_listT), np.eye(B, dtype=np.float32))
        y_Mat = tf.stack(y_listT)
        y_MatT = tf.transpose(y_Mat)

        # Define Kernel # K  = [k(y_i,y_j),i = 1...L, j = 1...L]
        ko = FLAGS.kernel_sig_error * np.eye(B) + tf_sigma * get_kernel(y_Mat, y_MatT, FLAGS.kernel_base)
        ke = ko[:L, :L]
        ke_ = ko[L:, :L]
        ke__ = (FLAGS.kernel_sig + tf_sigma) * tf.ones([M, FLAGS.hidden_size], dtype=tf.float32)
        # covariance is assumed to be same for all dimension

        # regression result
        z_unsup = tf.matmul(tf.matmul(ke_, tf.matrix_inverse(ke)), z_sup)  # [M D]
        z_unsup_stddev = ke__ + tf.matmul(ke_ * tf.transpose(tf.matmul(tf.matrix_inverse(ke), tf.transpose(ke_))),
                                          tf.ones([L, FLAGS.hidden_size], dtype=np.float32))
    else:  # add other possible regressions
        z_unsup = None
        z_unsup_stddev = None

    return z_unsup, z_unsup_stddev


def get_kernel(x, x_, base, epsilon=1e-8):
    '''Gaussian ERD kernel

    Set ERD kernel for data x and x'

    Args:
        x: data 1
        x_: data 2
        sig: ERD kernel sigma
        base: ERD kernel base. should not be 0

    Returns:

    '''

    return tf.exp((-1 * tf.squared_difference(x, x_) + epsilon) / (2 * base))


def get_vae_cost(mean, stddev, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean:
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                2.0 * tf.log(stddev + epsilon) - 1.0))


def get_kl_cost(mean1, stddev1, mean2, stddev2, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean:
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * ((tf.square(mean1 - mean2) + tf.square(stddev1)) / (tf.square(stddev2) + epsilon) +
                                2.0 * tf.log((stddev2 + epsilon) / (stddev1 + epsilon)) - 1.0))


def get_reconstruction_cost(output_tensor, target_tensor, im_size, epsilon=1e-8):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor: tensor produces by decoder
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''

    output_img_all = tf.reshape(output_tensor, [FLAGS.L, im_size, im_size, 3])
    output_img_rgb = output_img_all[:, :, :, :3]
    outout_tensor_rgb = tf.reshape(output_img_rgb, [FLAGS.L, im_size * im_size * 3])

    target_img_tensor = target_tensor[:, :im_size * im_size * 3]

    return tf.reduce_sum(-target_img_tensor * tf.log(outout_tensor_rgb + epsilon) -
                         (1.0 - target_img_tensor) * tf.log(1.0 - outout_tensor_rgb + epsilon))


def get_regression_cost(output_tensor, target_tensor, im_size, epsilon=1e-8):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor: tensor produces by decoder
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''

    outout_tensor_rgb = tf.reshape(output_tensor, [FLAGS.L, im_size * im_size * 3])

    target_img_tensor = target_tensor[:, :im_size * im_size * 3]

    return tf.reduce_sum(-target_img_tensor * tf.log(outout_tensor_rgb + epsilon) -
                         (1.0 - target_img_tensor) * tf.log(1.0 - outout_tensor_rgb + epsilon))


def get_softmax_cost(output_tensor, target_tensor, epsilon=1e-8):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor: tensor produces by decoder
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''

    return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                         (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))


def resample(image, w, h):
    '''downsample

             Args:
            image: tensor produces by decoder
            w,h: width height of the target data
            epsilon:

        '''
    return tf.image.resize_images(image, h, w)


def return_h36m_order(opt):
    '''return the list of the folder directories in H36m
        opt 1: Greeting
        opt 2: Posing

    '''
    img_list = []
    joint_list = []
    save_list = []

    if opt == 1:
        base = './HumanData/Image/Greeting'
        base_joint = './HumanData/Poses_D2_Positions_Greeting'
    elif opt == 2:
        base = './HumanData/Image/Posing'
        base_joint = './HumanData/Poses_D2_Positions_Posing'
    else:
        base = 'null'
        base_joint = 'null'

    for root, dirs, files in os.walk(base):
        for dr in dirs:
            directory = root + "/" + dr
            if len([sub for sub in os.listdir(directory) \
                    if os.path.isdir(directory + "/" + sub)]) == 0:
                drlist = directory.split('/')
                directory_joint = base_joint + "/" + drlist[len(drlist) - 2] + "/MyPoseFeatures/D2_Positions" + "/" + \
                                  drlist[len(drlist) - 1]
                save_joint = base_joint + "/" + drlist[len(drlist) - 2] + "/MyPoseFeatures/D2_Positions" + "/" + drlist[
                    len(drlist) - 1] + "/encoded"
                img_list.append(directory)
                joint_list.append(directory_joint)
                save_list.append(save_joint)

    return img_list, joint_list, save_list


def generate_train_batch():
    x = np.zeros([FLAGS.batch_size, Imsize * Imsize * 3], dtype=np.float32)
    lx = np.zeros([FLAGS.batch_size, 2*Imsize * 2*Imsize * 3], dtype=np.float32)
    y = np.zeros([FLAGS.batch_size, FLAGS.domain_size], dtype=np.float32)
    d = np.zeros([FLAGS.batch_size, FLAGS.dist_size], dtype=np.float32)

    rand_perm = np.random.permutation(FLAGS.T)

    cnt = 0
    unt = FLAGS.L
    for idx in range(0, FLAGS.V):  # for each video
        rdx = rand_perm[idx]
        if rdx == 2:
            rdx = 8
            # rdx = rdx + 24 + 1

        action_dir = img_list[rdx]
        joint_dir = joint_list[rdx]
        dist_dir = save_list[rdx]
        # dist_dir  =  joint_list[rdx]

        stepL = int(FLAGS.L / FLAGS.V)
        stepM = int(FLAGS.M / FLAGS.V)

        jpg_list = glob.glob(action_dir + '/*.jpg')

        rand_img = np.random.permutation(len(jpg_list))

        if len(jpg_list) == 0:
            pdb.set_trace()

        # add supervised pair
        for jdx in range(0, stepL):
            jpg_set = jpg_list[rand_img[jdx]]
            txt_set = joint_dir + '/' + jpg_set.split('/')[len(jpg_set.split('/')) - 1].split('.')[0] + '.txt'


            imt = imresize(imread(jpg_set), [Imsize, Imsize])
            lmt = imresize(imread(jpg_set), [2*Imsize, 2*Imsize])
            imt = imt / 255.0
            lmt = lmt / 255.0
            imt = imt.reshape([1, Imsize * Imsize * 3])
            lmt = lmt.reshape([1, 2*Imsize *2* Imsize * 3])

            jnt = np.loadtxt(txt_set)
            jnt = jnt.reshape([1, FLAGS.domain_size])

            jnt_x = jnt[:, 0::2]
            jnt_y = jnt[:, 1::2]

            rjnt_x = jnt_x - jnt_x[:, 15]
            rjnt_y = jnt_y - jnt_y[:, 15]

            jnt[:, 0::2] = rjnt_x
            jnt[:, 1::2] = rjnt_y

            jnt = jnt / 100 + 0.3

            # dnt = np.loadtxt(dst_list[rand_img[jdx]])
            # dnt = dnt.reshape([1,2*FLAGS.dist_size])
            # dnt = dnt*10

            x[cnt, :Imsize * Imsize * 3] = imt  # odd image
            lx[cnt, :2*Imsize * 2*Imsize * 3 ] = lmt
            y[cnt, :FLAGS.domain_size] = jnt[:, :FLAGS.domain_size]
            # d[cnt, :FLAGS.dist_size] = dnt[:,:FLAGS.dist_size]

            cnt = cnt + 1

        # add unsupervised regression
        for jdx in range(stepL, stepL + stepM):
            jpg_set = jpg_list[rand_img[jdx - stepL]]
            txt_set = joint_dir + '/' + jpg_set.split('/')[len(jpg_set.split('/')) - 1].split('.')[0] + '.txt'

            imt = imresize(imread(jpg_set), [Imsize, Imsize])
            lmt = imresize(imread(jpg_set), [2 * Imsize, 2 * Imsize])
            imt = imt / 255.0
            lmt = lmt / 255.0
            imt = imt.reshape([1, Imsize * Imsize * 3])
            lmt = lmt.reshape([1, 2 * Imsize * 2 * Imsize * 3])

            jnt = np.loadtxt(txt_set)
            jnt = jnt.reshape([1, FLAGS.domain_size])

            jnt_x = jnt[:, 0::2]
            jnt_y = jnt[:, 1::2]

            rjnt_x = jnt_x - jnt_x[:, 15]
            rjnt_y = jnt_y - jnt_y[:, 15]

            jnt[:, 0::2] = rjnt_x
            jnt[:, 1::2] = rjnt_y

            jnt = jnt / 100 + 0.3

            # dnt = np.loadtxt(dst_list[rand_img[jdx - stepL]])
            # dnt = dnt.reshape([1, 2*FLAGS.dist_size])
            # dnt = dnt*10

            x[unt, :Imsize * Imsize * 3] = imt  # image
            lx[unt, :2 * Imsize * 2 * Imsize * 3] = lmt
            y[unt, :FLAGS.domain_size] = jnt[:, :FLAGS.domain_size]
            # d[unt, :FLAGS.dist_size] = dnt[:,:FLAGS.dist_size]

            unt = unt + 1

    if FLAGS.regression is False:
        x[FLAGS.L:, :] = x[:FLAGS.M:, :]
        lx[FLAGS.L:, :] = lx[:FLAGS.M:, :]

    return x, lx, y, d


def generate_test_batch(idx, tdx):
    x = np.zeros([FLAGS.batch_size, Imsize * Imsize * 3], dtype=np.float32)
    y = np.zeros([FLAGS.batch_size, FLAGS.domain_size], dtype=np.float32)
    d = np.zeros([FLAGS.batch_size, FLAGS.dist_size], dtype=np.float32)

    # for jdx in range(0, min(FLAGS.batch_size, len(img_list)-1)):
    action_dir = img_list[idx]
    joint_dir = joint_list[idx]
    dist_dir = save_list[idx]

    jpg_list = glob.glob(action_dir + '/*.jpg')

    cnt = 0
    rand_img = range(0, len(jpg_list), int(FLAGS.V * len(jpg_list) / FLAGS.L) - 5)

    for jdx in range(0, FLAGS.batch_size):

        if cnt < FLAGS.L:
            jpg_set = jpg_list[rand_img[jdx % int(FLAGS.L / FLAGS.V)]]
            txt_set = joint_dir + '/' + jpg_set.split('/')[len(jpg_set.split('/')) - 1].split('.')[0] + '.txt'
            imt = imread(jpg_set)
            imt = imresize(imt, [Imsize, Imsize])
            imt = imt / 255.0
            imt = imt.reshape([1, Imsize * Imsize * 3])

            jnt = np.loadtxt(txt_set)
            jnt = jnt.reshape([1, FLAGS.domain_size])

            jnt_x = jnt[:, 0::2]
            jnt_y = jnt[:, 1::2]

            rjnt_x = jnt_x - jnt_x[:, 15]
            rjnt_y = jnt_y - jnt_y[:, 15]

            jnt[:, 0::2] = rjnt_x
            jnt[:, 1::2] = rjnt_y

            jnt = jnt / 100 + 0.3

            # dnt = np.loadtxt(dst_list[rand_img[jdx%int(FLAGS.L/FLAGS.V)]])
            # dnt = dnt.reshape([1, 2*FLAGS.dist_size])
            # dnt = dnt*10

            x[cnt, :Imsize * Imsize * 3] = imt  # image
            y[cnt, :FLAGS.domain_size] = jnt[:, :FLAGS.domain_size]
            # d[cnt, :FLAGS.dist_size] = dnt[:,:FLAGS.dist_size]

        else:  # to reconstruct
            tjdx = 5 * (cnt - FLAGS.L) + 1
            ss = '%.4d' % tjdx

            imt = imread(img_list[tdx] + '/' + ss + '.jpg')
            imt = imresize(imt, [Imsize, Imsize])
            imt = imt / 255.0
            imt = imt.reshape([1, Imsize * Imsize * 3])

            jnt = np.loadtxt(joint_list[tdx] + '/' + ss + '.txt')
            jnt = jnt.reshape([1, FLAGS.domain_size])

            jnt_x = jnt[:, 0::2]
            jnt_y = jnt[:, 1::2]

            rjnt_x = jnt_x - jnt_x[:, 15]
            rjnt_y = jnt_y - jnt_y[:, 15]

            jnt[:, 0::2] = rjnt_x
            jnt[:, 1::2] = rjnt_y

            jnt = jnt / 100 + 0.3

            # dnt = np.loadtxt(save_list[tdx]+'/'+ ss + '.txt')
            # dnt = dnt.reshape([1, 2*FLAGS.dist_size])
            # dnt = dnt*10

            x[cnt, :Imsize * Imsize * 3] = imt  # image
            y[cnt, :FLAGS.domain_size] = jnt[:, :FLAGS.domain_size]
            # d[cnt, :FLAGS.dist_size] = dnt[:,:FLAGS.dist_size]

        cnt = cnt + 1

    return x, y, d


def generate_postproc_batch(idx):
    x = np.zeros([FLAGS.batch_size, Imsize * Imsize * 3], dtype=np.float32)
    y = np.zeros([FLAGS.batch_size, FLAGS.domain_size], dtype=np.float32)
    d = np.zeros([FLAGS.batch_size, FLAGS.dist_size], dtype=np.float32)

    stepL = int(FLAGS.L / FLAGS.V)
    stepM = int(FLAGS.M / FLAGS.V)

    rand_video = np.random.permutation(FLAGS.T)
    rand_order = np.random.permutation(FLAGS.V)

    cnt = 0
    unt = FLAGS.L
    for rdx in range(0, FLAGS.V):
        if rand_order[rdx] == 0 or rand_order[rdx] == 5 or rand_order[rdx] == 10:
            vid_idx = idx
            # for jdx in range(0, min(FLAGS.batch_size, len(img_list)-1)):
            action_dir = img_list[vid_idx]
            joint_dir = joint_list[vid_idx]
            dist_dir = save_list[vid_idx]
            # dist_dir  =  joint_list[vid_idx]

            jpg_list = glob.glob(action_dir + '/*.jpg')
            txt_list = glob.glob(joint_dir + '/*.txt')
            dst_list = glob.glob(dist_dir + '/*.txt')

            rand_img = range(0, len(jpg_list), int(FLAGS.V * len(jpg_list) / FLAGS.L) - 5)
        else:
            vid_idx = rand_video[rand_order[rdx]]
            if vid_idx == 2:
                vid_idx = idx

            action_dir = img_list[vid_idx]
            joint_dir = joint_list[vid_idx]
            dist_dir = save_list[vid_idx]
            # dist_dir  =  joint_list[vid_idx]

            jpg_list = glob.glob(action_dir + '/*.jpg')
            txt_list = glob.glob(joint_dir + '/*.txt')
            dst_list = glob.glob(dist_dir + '/*.txt')

            rand_img = range(0, len(jpg_list), int(FLAGS.V * len(jpg_list) / FLAGS.L) - 5)

        rand_ldx = np.random.permutation(stepL)
        rand_mdx = np.random.permutation(stepM)

        # for supervised pair
        for jdx in range(0, stepL):
            imt = imread(jpg_list[rand_img[rand_ldx[jdx]]])
            imt = imresize(imt, [Imsize, Imsize])
            imt = imt / 255.0
            imt = imt.reshape([1, Imsize * Imsize * 3])

            jnt = np.loadtxt(txt_list[rand_img[rand_ldx[jdx]]])
            jnt = jnt.reshape([1, FLAGS.domain_size])

            jnt_x = jnt[:, 0::2]
            jnt_y = jnt[:, 1::2]

            rjnt_x = jnt_x - np.mean(jnt_x)
            rjnt_y = jnt_y - np.mean(jnt_y)

            jnt[:, 0::2] = rjnt_x
            jnt[:, 1::2] = rjnt_y

            jnt = jnt / 100 + 0.3

            dnt = np.loadtxt(dst_list[rand_img[rand_ldx[jdx]]])
            dnt = dnt.reshape([1, 2 * FLAGS.dist_size])
            dnt = dnt * 10

            x[cnt, :Imsize * Imsize * 3] = imt  # image
            y[cnt, :FLAGS.domain_size] = jnt[:, :FLAGS.domain_size]
            d[cnt, :FLAGS.dist_size] = dnt[:, :FLAGS.dist_size]

            cnt = cnt + 1

        # for unsupervised pair
        for jdx in range(0, stepM):
            imt = imread(jpg_list[rand_img[rand_mdx[jdx]]])
            imt = imresize(imt, [Imsize, Imsize])
            imt = imt / 255.0
            imt = imt.reshape([1, Imsize * Imsize * 3])

            jnt = np.loadtxt(txt_list[rand_img[rand_mdx[jdx]]])
            jnt = jnt.reshape([1, FLAGS.domain_size])

            jnt_x = jnt[:, 0::2]
            jnt_y = jnt[:, 1::2]

            rjnt_x = jnt_x - np.mean(jnt_x)
            rjnt_y = jnt_y - np.mean(jnt_y)

            jnt[:, 0::2] = rjnt_x
            jnt[:, 1::2] = rjnt_y

            jnt = jnt / 100 + 0.3

            dnt = np.loadtxt(dst_list[rand_img[rand_ldx[jdx]]])
            dnt = dnt.reshape([1, 2 * FLAGS.dist_size])
            dnt = dnt * 10

            x[unt, :Imsize * Imsize * 3] = imt  # image
            y[unt, :FLAGS.domain_size] = jnt[:, :FLAGS.domain_size]
            d[unt, :FLAGS.dist_size] = dnt[:, :FLAGS.dist_size]
            unt = unt + 1

    return x, y, d


if __name__ == "__main__":

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    # input data_setting

    img_list, joint_list, save_list = return_h36m_order(1)

    train_foldername = 'imgs_h36m_act_all'
    ltrain_foldername = 'limgs_h36m_act_all'
    test_foldername = 'test_h36m_all_test4'

    text_file = open("errors_vanilla_kl_1.txt", "w")

    # folderlist = sorted(os.listdir(data_foldername))
    input_tensor = tf.placeholder(tf.float32, [None, Imsize * Imsize * 3], name="in")
    linput_tensor = tf.placeholder(tf.float32, [None, 2 * Imsize * 2 * Imsize * 3], name="lin")
    gt_tensor = tf.placeholder(tf.float32, [None, Imsize * Imsize * 3], name="in_2")
    y_tensor = tf.placeholder(tf.float32, [None, FLAGS.domain_size], name="y")
    d_tensor = tf.placeholder(tf.float32, [None, FLAGS.dist_size], name="d")

    z_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.hidden_size + FLAGS.domain_size], name="zero")
    e_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.hidden_size + FLAGS.domain_size], name="eps")

    # for train
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("model_g") as scope:
                # insert half number of input_tensor
                so_tensor_z, _, _ = decoder(encoder(input_tensor), y_tensor, z_tensor, d_tensor)
                rec_tensor_z = so_tensor_z[:FLAGS.M, :]
                reg_tensor_z = so_tensor_z[FLAGS.M:, :]

            with tf.variable_scope("model_lg") as scope:
                lo_tensor_z = amplifier_dec(residual_block(residual_block(amplifier_enc(so_tensor_z)))) #lo - 128
                lrec_tensor_z = lo_tensor_z[:FLAGS.M, :]
                lreg_tensor_z = lo_tensor_z[FLAGS.M:, :]

            with tf.variable_scope("model_g", reuse=True) as scope:
                so_tensor_e, mean, stddev = decoder(encoder(input_tensor), y_tensor, e_tensor, d_tensor)
                rec_tensor_e = so_tensor_e[:FLAGS.M, :]
                reg_tensor_e = so_tensor_e[FLAGS.M:, :]

            with tf.variable_scope("model_lg", reuse=True) as scope:
                lo_tensor_e = amplifier_dec(residual_block(residual_block(amplifier_enc(so_tensor_e)))) #lo - 128
                lrec_tensor_e = lo_tensor_e[:FLAGS.M, :]
                lreg_tensor_e = lo_tensor_e[FLAGS.M:, :]

            with tf.variable_scope("model_g", reuse=True) as scope:
                so_tensor_g, _, _ = decoder(encoder(input_tensor), y_tensor, e_tensor)
                rec_tensor_g = so_tensor_g[:FLAGS.M, :]
                reg_tensor_g = so_tensor_g[FLAGS.M:, :]

            with tf.variable_scope("model_lg", reuse=True) as scope:
                lo_tensor_g = amplifier_dec(residual_block(residual_block(amplifier_enc(so_tensor_g)))) #lo - 128
                lrec_tensor_g = lo_tensor_g[:FLAGS.M, :]
                lreg_tensor_g = lo_tensor_g[FLAGS.M:, :]

        with pt.defaults_scope(phase=pt.Phase.train):
            # discriminator
            with tf.variable_scope("model_d") as scope:
                class_d_real = discriminator(input_tensor)

            with tf.variable_scope("model_ld") as scope:
                lclass_d_real = amplifier_discriminator(input_tensor)

            with tf.variable_scope("model_d", reuse=True) as scope:
                # for mode
                class_d_fake_z = discriminator(so_tensor_z)

            with tf.variable_scope("model_ld", reuse=True) as scope:
                lclass_d_fake_z = amplifier_discriminator(lo_tensor_z)

                # VAE diffusion
            with tf.variable_scope("model_d", reuse=True) as scope:
                class_d_fake_e = discriminator(so_tensor_e)

            with tf.variable_scope("model_ld", reuse=True) as scope:
                lclass_d_fake_e = amplifier_discriminator(lo_tensor_e)

                # GAN diffusion
            with tf.variable_scope("model_d", reuse=True) as scope:
                class_d_fake_g = discriminator(so_tensor_g)

            with tf.variable_scope("model_ld", reuse=True) as scope:
                lclass_d_fake_g = amplifier_discriminator(lo_tensor_g)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model_g", reuse=True) as scope:
                sampled_tensor, _, _ = decoder(encoder(input_tensor), y_tensor, e_tensor, d_tensor)
                sampled_rec_tensor = sampled_tensor[:FLAGS.M, :]
                sampled_reg_tensor = sampled_tensor[FLAGS.M:, :]

            with tf.variable_scope("model_lg", reuse=True) as scope:
                lsampled_tensor = amplifier_dec(residual_block(residual_block(amplifier_enc(sampled_tensor))))
                lsampled_rec_tensor = lsampled_tensor[:FLAGS.M, :]
                lsampled_reg_tensor = lsampled_tensor[FLAGS.M:, :]

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model_g", reuse=True) as scope:
                test_tensor, _, _ = decoder(encoder(input_tensor), y_tensor, e_tensor, d_tensor)
                test_rec_tensor = test_tensor[:FLAGS.M, :]
                test_reg_tensor = test_tensor[FLAGS.M:, :]

            with tf.variable_scope("model_lg", reuse=True) as scope:
                ltest_tensor = amplifier_dec(residual_block(residual_block(amplifier_enc(test_tensor))))
                ltest_rec_tensor = ltest_tensor[:FLAGS.M, :]
                ltest_reg_tensor = ltest_tensor[FLAGS.M:, :]


    vae_loss = get_vae_cost(mean, stddev)  # do not consider mean and stddev generated from GP.
    kl_loss = get_kl_cost(mean[:FLAGS.M, :], stddev[:FLAGS.M, :], mean[FLAGS.L:, :],
                          stddev[FLAGS.L:, :])  # vae cost for y, same procedure as x

    # reconstruction loss
    rec_loss_anchor_z = get_reconstruction_cost(rec_tensor_z, input_tensor[:FLAGS.L, :], Imsize)
    rec_loss_regression_z = get_regression_cost(reg_tensor_z, input_tensor[FLAGS.L:, :], Imsize)

    rec_loss_anchor_e = get_reconstruction_cost(rec_tensor_e, input_tensor[:FLAGS.L, :], Imsize)
    rec_loss_regression_e = get_regression_cost(reg_tensor_e, input_tensor[FLAGS.L:, :], Imsize)

    # reconstruction loss
    lrec_loss_anchor_z = get_reconstruction_cost(lrec_tensor_z, linput_tensor[:FLAGS.L, :], Imsize*2)
    lrec_loss_regression_z = get_regression_cost(lreg_tensor_z, linput_tensor[FLAGS.L:, :], Imsize*2)

    lrec_loss_anchor_e = get_reconstruction_cost(lrec_tensor_e, linput_tensor[:FLAGS.L, :], Imsize*2)
    lrec_loss_regression_e = get_regression_cost(lreg_tensor_e, linput_tensor[FLAGS.L:, :], Imsize*2)

    # stage 1
    d_loss_real = tf.reduce_sum(sigmoid_cross_entropy_with_logits(class_d_real, tf.ones_like(class_d_real)))

    d_loss_fake_z = tf.reduce_sum(sigmoid_cross_entropy_with_logits(class_d_fake_z, tf.zeros_like(class_d_fake_z)))
    g_loss_z = tf.reduce_sum(sigmoid_cross_entropy_with_logits(class_d_fake_z, tf.ones_like(class_d_fake_z)))

    d_loss_fake_e = tf.reduce_sum(sigmoid_cross_entropy_with_logits(class_d_fake_e, tf.zeros_like(class_d_fake_e)))
    g_loss_e = tf.reduce_sum(sigmoid_cross_entropy_with_logits(class_d_fake_e, tf.ones_like(class_d_fake_e)))

    d_loss_fake_g = tf.reduce_sum(sigmoid_cross_entropy_with_logits(class_d_fake_g, tf.zeros_like(class_d_fake_g)))
    g_loss_g = tf.reduce_sum(sigmoid_cross_entropy_with_logits(class_d_fake_g, tf.ones_like(class_d_fake_g)))

    # stage 2
    ld_loss_real = tf.reduce_sum(sigmoid_cross_entropy_with_logits(lclass_d_real, tf.ones_like(lclass_d_real)))

    ld_loss_fake_z = tf.reduce_sum(sigmoid_cross_entropy_with_logits(lclass_d_fake_z, tf.zeros_like(lclass_d_fake_z)))
    lg_loss_z = tf.reduce_sum(sigmoid_cross_entropy_with_logits(lclass_d_fake_z, tf.ones_like(lclass_d_fake_z)))

    ld_loss_fake_e = tf.reduce_sum(sigmoid_cross_entropy_with_logits(lclass_d_fake_e, tf.zeros_like(lclass_d_fake_e)))
    lg_loss_e = tf.reduce_sum(sigmoid_cross_entropy_with_logits(lclass_d_fake_e, tf.ones_like(lclass_d_fake_e)))

    ld_loss_fake_g = tf.reduce_sum(sigmoid_cross_entropy_with_logits(lclass_d_fake_g, tf.zeros_like(lclass_d_fake_g)))
    lg_loss_g = tf.reduce_sum(sigmoid_cross_entropy_with_logits(lclass_d_fake_g, tf.ones_like(lclass_d_fake_g)))

    # stage 1
    loss_Gz = g_loss_z + vae_loss + rec_loss_anchor_z + rec_loss_regression_z  # lap_loss  # + shp_loss #+rec_loss
    loss_Dz = d_loss_real + d_loss_fake_z

    loss_Ge = g_loss_e + vae_loss + rec_loss_anchor_e + rec_loss_regression_e  # lap_loss  # + shp_loss #+rec_loss
    loss_De = d_loss_real + d_loss_fake_e

    loss_Gg = g_loss_g  # lap_loss  # + shp_loss #+rec_loss
    loss_Dg = d_loss_real + d_loss_fake_g


    # stage 2
    lloss_Gz = lg_loss_z + lrec_loss_anchor_z + lrec_loss_regression_z  # lap_loss  # + shp_loss #+rec_loss
    lloss_Dz = ld_loss_real + ld_loss_fake_z

    lloss_Ge = lg_loss_e + lrec_loss_anchor_e + lrec_loss_regression_e  # lap_loss  # + shp_loss #+rec_loss
    lloss_De = ld_loss_real + ld_loss_fake_e

    lloss_Gg = lg_loss_g  # lap_loss  # + shp_loss #+rec_loss
    lloss_Dg = ld_loss_real + ld_loss_fake_g

    iter_epoch = FLAGS.max_epoch

    # loss = vae_loss + rec_loss_anchor + rec_loss_regression + kl_loss
    iter_epoch = FLAGS.max_epoch

    # variables
    t_vars = tf.trainable_variables()

    # stage 1
    d_vars = [var for var in t_vars if '_d' in var.name]
    g_vars = [var for var in t_vars if '_g' in var.name]
    de_vars = [var for var in t_vars if 'dec' in var.name]  # only train the deconvolution network

    # stage 2
    ld_vars = [var for var in t_vars if '_ld' in var.name]
    lg_vars = [var for var in t_vars if '_lg' in var.name]

    # optimizers - stage 1
    d_optimz = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(loss_Dz, var_list=d_vars)
    g_optimz = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(loss_Gz, var_list=g_vars)

    d_optime = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(loss_De, var_list=d_vars)
    g_optime = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(loss_Ge, var_list=g_vars)

    d_optimg = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(loss_Dg, var_list=d_vars)
    g_optimg = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(loss_Gg, var_list=de_vars)

    # optimizers - stage 2
    ld_optimz = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(lloss_Dz, var_list=ld_vars)
    lg_optimz = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(lloss_Gz, var_list=lg_vars)

    ld_optime = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(lloss_De, var_list=ld_vars)
    lg_optime = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(lloss_Ge, var_list=lg_vars)

    ld_optimg = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(lloss_Dg, var_list=ld_vars)
    lg_optimg = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0) \
        .minimize(lloss_Gg, var_list=lg_vars)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    if FLAGS.isSave is False:

        with tf.Session() as sess:

            sess.run(init)

            # restore tensor
            # imgs_folder = os.path.join(FLAGS.working_directory, train_foldername)
            # saver.restore(sess, os.path.join(imgs_folder, 'model.ckpt'))

            for epoch in range(iter_epoch):
                training_D = 0.0
                training_G = 0.0
                regression_loss = 0.0
                reconstruction_loss = 0.0
                distance_loss = 0.0

                ltraining_D = 0.0
                ltraining_G = 0.0
                lregression_loss = 0.0
                lreconstruction_loss = 0.0

                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=FLAGS.updates_per_epoch, widgets=widgets)
                pbar.start()
                for i in range(FLAGS.updates_per_epoch):
                    pbar.update(i)

                    # generate batch
                    x, lx,  y, d = generate_train_batch()  # in this case, generate batch considering the clusters.

                    # _, loss_value, lrr, lra, lkl = sess.run([train, loss, rec_loss_regression, rec_loss_anchor, kl_loss],
                    #                                                    {input_tensor: x, y_tensor: y, d_tensor: d})

                    z = np.zeros([FLAGS.batch_size, FLAGS.hidden_size + FLAGS.domain_size], dtype=np.float32)
                    e = np.random.randn(FLAGS.batch_size, FLAGS.hidden_size + FLAGS.domain_size)

                    #stage 1
                    # modality
                    _, d_loss_evalz = sess.run([d_optimz, loss_Dz],
                                           {input_tensor: x, linput_tensor: lx,
                                            y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})
                    _, g_loss_evalz, rec_evalz_recon, rec_evalz_reg, lkl = sess.run(
                        [g_optimz, g_loss_z, rec_loss_anchor_z, rec_loss_regression_z, kl_loss],
                        {input_tensor: x, linput_tensor: lx,
                         y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})

                    # VAE diffusion
                    _, d_loss_evale = sess.run([d_optime, loss_De],
                                               {input_tensor: x, linput_tensor: lx,
                                                y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})
                    _, g_loss_evale, rec_evale_recon, rec_evale_reg = sess.run(
                        [g_optime, g_loss_e, rec_loss_anchor_e, rec_loss_regression_e],
                        {input_tensor: x, linput_tensor: lx,
                         y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})

                    # GAN diffusion
                    _, d_loss_evalg = sess.run([d_optimg, loss_Dg],
                                               {input_tensor: x, linput_tensor: lx,
                                                y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})
                    _, g_loss_evalg = sess.run([g_optimg, g_loss_g],
                                               {input_tensor: x, linput_tensor: lx,
                                                y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})


                    #stage 2
                    # modality
                    _, ld_loss_evalz = sess.run([ld_optimz, lloss_Dz],
                                               {input_tensor: x, linput_tensor: lx,
                                                y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})
                    _, lg_loss_evalz, lrec_evalz_recon, lrec_evalz_reg = sess.run(
                        [lg_optimz, lg_loss_z, lrec_loss_anchor_z, lrec_loss_regression_z],
                        {input_tensor: x, linput_tensor: lx,
                         y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})

                    # VAE diffusion
                    _, ld_loss_evale = sess.run([ld_optime, lloss_De],
                                               {input_tensor: x, linput_tensor: lx,
                                                y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})
                    _, lg_loss_evale, lrec_evale_recon, lrec_evale_reg = sess.run(
                        [lg_optime, lg_loss_e, lrec_loss_anchor_e, lrec_loss_regression_e],
                        {input_tensor: x, linput_tensor: lx,
                         y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})

                    # GAN diffusion
                    _, ld_loss_evalg = sess.run([ld_optimg, lloss_Dg],
                                               {input_tensor: x, linput_tensor: lx,
                                                y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})
                    _, lg_loss_evalg = sess.run([lg_optimg, lg_loss_g],
                                               {input_tensor: x, linput_tensor: lx,
                                                y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})

                    training_D += (d_loss_evalz + d_loss_evale + d_loss_evalg) / 3.0
                    training_G += (g_loss_evalz + g_loss_evale + g_loss_evalg) / 3.0
                    regression_loss += (rec_evalz_reg + rec_evale_reg) / 2.0
                    reconstruction_loss += (rec_evalz_recon + rec_evale_recon) / 2.0
                    distance_loss += lkl

                    ltraining_D += (ld_loss_evalz + ld_loss_evale + ld_loss_evalg) / 3.0
                    ltraining_G += (lg_loss_evalz + lg_loss_evale + lg_loss_evalg) / 3.0
                    lregression_loss += (lrec_evalz_reg + lrec_evale_reg) / 2.0
                    lreconstruction_loss += (lrec_evalz_recon + lrec_evale_recon) / 2.0

                training_D = training_D / \
                             (FLAGS.updates_per_epoch * FLAGS.batch_size)
                training_G = training_G / \
                             (FLAGS.updates_per_epoch * FLAGS.batch_size)
                regression_loss = regression_loss / \
                                  (FLAGS.updates_per_epoch * Imsize * Imsize * 3 * FLAGS.batch_size)
                reconstruction_loss = reconstruction_loss / \
                                      (FLAGS.updates_per_epoch * Imsize * Imsize * 3 * FLAGS.batch_size)
                distance_loss = distance_loss / \
                                (FLAGS.updates_per_epoch * FLAGS.batch_size)

                ltraining_D = ltraining_D / \
                             (FLAGS.updates_per_epoch * FLAGS.batch_size)
                ltraining_G = ltraining_G / \
                             (FLAGS.updates_per_epoch * FLAGS.batch_size)
                lregression_loss = lregression_loss / \
                                  (FLAGS.updates_per_epoch * 4*Imsize * Imsize * 3 * FLAGS.batch_size)
                lreconstruction_loss = lreconstruction_loss / \
                                      (FLAGS.updates_per_epoch * 4* Imsize * Imsize * 3 * FLAGS.batch_size)

                print("Loss %f %f %f %f %f %f %f %f %f" % (
                training_D, training_G, regression_loss, reconstruction_loss,
                ltraining_D, ltraining_G, lregression_loss, lreconstruction_loss,distance_loss))
                text_file.write(
                    "%f %f %f %f %f\n" % (training_D, training_G, regression_loss, reconstruction_loss, distance_loss))


                imgs_folder = os.path.join(FLAGS.working_directory, train_foldername)
                limgs_folder = os.path.join(FLAGS.working_directory, ltrain_foldername)
                    # pdb.set_trace()

                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)

                if not os.path.exists(limgs_folder):
                    os.makedirs(limgs_folder)

                save_path = saver.save(sess, os.path.join(imgs_folder, 'model.ckpt'))


                rec_imgs, reg_imgs = sess.run([sampled_rec_tensor, sampled_reg_tensor],
                                              {input_tensor: x, linput_tensor: lx,
                                               y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})


                lrec_imgs, lreg_imgs = sess.run([lsampled_rec_tensor, lsampled_reg_tensor],
                                                {input_tensor: x, linput_tensor: lx,
                                                 y_tensor: y, d_tensor: d, z_tensor: z, e_tensor: e})


                for k in range(FLAGS.M):
                    one = 3 * k
                    two = 3 * k + 1
                    three = 3 * k + 2

                    rec_imgs[k] = rec_imgs[k] * 255.0
                    img_set = rec_imgs[k].reshape([Imsize, Imsize, 3])
                    reg_imgs[k] = reg_imgs[k] * 255.0
                    rmg_set = reg_imgs[k].reshape([Imsize, Imsize, 3])

                    lrec_imgs[k] = lrec_imgs[k] * 255.0
                    limg_set = lrec_imgs[k].reshape([2*Imsize, 2*Imsize, 3])
                    lreg_imgs[k] = lreg_imgs[k] * 255.0
                    lrmg_set = lreg_imgs[k].reshape([2*Imsize, 2*Imsize, 3])

                    gt_imgs = x[k + FLAGS.L, :] * 255
                    lgt_imgs = lx[k + FLAGS.L, :] * 255

                    gt_imgs = gt_imgs.reshape([Imsize, Imsize, 3])
                    lgt_imgs = lgt_imgs.reshape([2*Imsize, 2*Imsize, 3])

                    imsave(os.path.join(imgs_folder, '%03d.png') % one,
                           rmg_set[0:Imsize, 0:Imsize, 0:3])
                    imsave(os.path.join(imgs_folder, '%03d.png') % two,
                           img_set[0:Imsize, 0:Imsize, 0:3])
                    imsave(os.path.join(imgs_folder, '%03d.png') % three,
                           gt_imgs[0:Imsize, 0:Imsize, 0:3])

                    imsave(os.path.join(limgs_folder, '%03d.png') % one,
                           lrmg_set[0:Imsize*2, 0:Imsize*2, 0:3])
                    imsave(os.path.join(limgs_folder, '%03d.png') % two,
                           limg_set[0:Imsize*2, 0:Imsize*2, 0:3])
                    imsave(os.path.join(limgs_folder, '%03d.png') % three,
                           lgt_imgs[0:Imsize*2, 0:Imsize*2, 0:3])

        text_file.close()
    else:
        with tf.Session() as sess:

            stepL = int(FLAGS.L / FLAGS.V)

            # for post processing of the batch
            # for idx in range(FLAGS.T, FLAGS.T+10):
            # for idx in range(3, FLAGS.T-10):
            # pdb.set_trace()
            for idx in range(3, 32):

                # restore tensor
                imgs_folder = os.path.join(FLAGS.working_directory, train_foldername)
                saver.restore(sess, os.path.join(imgs_folder, 'model.ckpt'))

                # get test sample
                # num_update = FLAGS.updates_per_epoch
                num_update = 200

                # post processing
                tuning_loss = 0.0
                for p_epoch in range(0, 0):
                    widgets = ["video #%d|" % idx, Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=num_update, widgets=widgets)
                    pbar.start()

                    for update in range(0, num_update):
                        pbar.update(update)

                        # generate post processing batch
                        xpost, ypost, dpost = generate_postproc_batch(idx)
                        _, loss_value = sess.run([train, loss], {input_tensor: xpost, y_tensor: ypost, d_tensor: dpost})

                        tuning_loss += loss_value
                    tuning_loss = tuning_loss / (num_update * Imsize * Imsize * 3 * FLAGS.batch_size)
                    print("Loss %f" % tuning_loss)

                for tdx in range(33, 35):

                    # test batch generation
                    print("test images %d-%d" % (idx, tdx))
                    xtest, ytest, dtest = generate_test_batch(idx, tdx)

                    # pdb.set_trace()
                    # see reconstructed results
                    reg_imgs, mapped_z = sess.run([test_reg_tensor, zout_tensor],
                                                  {input_tensor: xtest, y_tensor: ytest, d_tensor: dtest})

                    test_folder = os.path.join(FLAGS.working_directory, test_foldername, str(idx), str(tdx))
                    if not os.path.exists(test_folder):
                        os.makedirs(test_folder)

                    for k in range(FLAGS.M):
                        one = 2 * k
                        two = 2 * k + 1

                        reg_imgs[k] = reg_imgs[k] * 255.0
                        rmg_set = reg_imgs[k].reshape([Imsize, Imsize, 6])
                        gt_imgs = xtest[k + FLAGS.L, :] * 255
                        gt_imgs = gt_imgs.reshape([Imsize, Imsize, 3])

                        imsave(os.path.join(test_folder, '%03d.png') % one,
                               rmg_set[0:Imsize, 0:Imsize, 3:6])
                        imsave(os.path.join(test_folder, '%03d.png') % two,
                               gt_imgs[0:Imsize, 0:Imsize, 0:3])

                    np.savetxt(os.path.join(test_folder, 'mapped_z.txt'), mapped_z)







