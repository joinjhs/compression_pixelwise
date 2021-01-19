import tensorflow as tf
import numpy as np
import os
import random
import cv2
import sys
import tqdm
import time

from config import parse_args
from module import model_conv
from data import read_dir, write_tfrecord, read_tfrecord, read_tfrecord, data_exist
from utils import save_image_data, save_pred_data, save_ctx_data


class Network(object):
    def __init__(self, args):
        self.args = args

        CKPT_DIR = args.ckpt_dir

        # Create ckpt directory if it does not exist
        if not os.path.exists(CKPT_DIR):
            os.mkdir(CKPT_DIR)

        # Print Arguments
        args = args.__dict__
        print("Arguments : ")

        for key, value in sorted(args.items()):
            print('\t%15s:\t%s' % (key, value))


    def build(self):

        # Parameters
        DATA_DIR = self.args.data_dir

        LAYER_NUM = self.args.layer_num
        HIDDEN_UNIT = self.args.hidden_unit

        LAMBDA_CTX = self.args.lambda_ctx
        CHANNEL_NUM = self.args.channel_num

        LR = self.args.lr

        BATCH_SIZE = self.args.batch_size
        CROP_SIZE = self.args.crop_size

        CHANNEL_EPOCH = self.args.channel_epoch
        JOINT_EPOCH = self.args.joint_epoch
        NUM_PATCHES = 11

        # TFRecord
        tfrecord_name = 'train.tfrecord'

        # if train tfrecord does not exist, create dataset
        if not data_exist(DATA_DIR, tfrecord_name):
            img_list = read_dir(DATA_DIR + 'train/')
            write_tfrecord(DATA_DIR, img_list, tfrecord_name)

        self.input_crop, _, _ = read_tfrecord(DATA_DIR, tfrecord_name, num_epochs=3*CHANNEL_EPOCH+JOINT_EPOCH,
                                        batch_size=BATCH_SIZE, min_after_dequeue=10, crop_size=CROP_SIZE)

        self.input = tf.placeholder(tf.uint8, (None, None, None, 3))

        input_yuv = self.rgb2yuv(self.input)

        if CHANNEL_NUM == 1:
            input_img = tf.expand_dims(input_yuv[:,:,:,0],axis=3)
        elif CHANNEL_NUM == 3:
            input_img = input_yuv
        else:
            print("Invalid Channel Num")
            sys.exit(1)

        input_depth = tf.nn.space_to_depth(input_img, 2)
        input_1, input_4, input_3, input_2 = tf.split(input_depth, 4, axis=3)
        """
        if CHANNEL_NUM == 3:
            input_1y, input_1u, input_1v = tf.split(input_1, 3, axis=3)
            input_4y, input_4u, input_4v = tf.split(input_4, 3, axis=3)
            input_3y, input_3u, input_3v = tf.split(input_3, 3, axis=3)
            input_2y, input_2u, input_2v = tf.split(input_2, 3, axis=3)
        """
        type = "1234"
        if type == "1234":
            order = tf.concat([tf.expand_dims(input_1[:,:,:,0],axis=3),tf.expand_dims(input_2[:,:,:,0],axis=3),tf.expand_dims(input_3[:,:,:,0],axis=3),tf.expand_dims(input_4[:,:,:,0],axis=3),
                           tf.expand_dims(input_1[:,:,:,1],axis=3),tf.expand_dims(input_2[:,:,:,1],axis=3),tf.expand_dims(input_3[:,:,:,1],axis=3),tf.expand_dims(input_4[:,:,:,1],axis=3),
                           tf.expand_dims(input_1[:,:,:,2],axis=3),tf.expand_dims(input_2[:,:,:,2],axis=3),tf.expand_dims(input_3[:,:,:,2],axis=3),tf.expand_dims(input_4[:,:,:,2],axis=3)], axis=3)
        elif type == "yuv":
            order = tf.concat([tf.expand_dims(input_1[:, :, :, 0], axis=3), tf.expand_dims(input_1[:, :, :, 1], axis=3),
                           tf.expand_dims(input_1[:, :, :, 2], axis=3), tf.expand_dims(input_2[:, :, :, 0], axis=3),
                           tf.expand_dims(input_2[:, :, :, 1], axis=3), tf.expand_dims(input_2[:, :, :, 2], axis=3),
                           tf.expand_dims(input_3[:, :, :, 0], axis=3), tf.expand_dims(input_3[:, :, :, 1], axis=3),
                           tf.expand_dims(input_3[:, :, :, 2], axis=3), tf.expand_dims(input_4[:, :, :, 0], axis=3),
                           tf.expand_dims(input_4[:, :, :, 1], axis=3), tf.expand_dims(input_4[:, :, :, 2], axis=3)],
                          axis=3)
        print("building order completed.\n")
        #order = tf.concat([tf.expand_dims(input_1[:,:,:,0],axis=3),tf.expand_dims(input_2[:,:,:,0],axis=3),tf.expand_dims(input_3[:,:,:,0],axis=3),tf.expand_dims(input_4[:,:,:,0],axis=3)], axis=3)

        pred_li = []
        ctx_li = []
        error_pred_li = []
        loss_pred_li = []
        loss_ctx_li = []
        loss_li = []
        total_loss = 0

        for i in range(NUM_PATCHES):
            #order1 = tf.concat([tf.expand_dims(input_1[:,:,:,0],axis=3),tf.expand_dims(input_2[:,:,:,0],axis=3)], axis=3)
            if i == 0:
                pred, ctx = model_conv(tf.expand_dims(input_1[:,:,:,0],axis=3), LAYER_NUM, HIDDEN_UNIT, 'model_' + str(i + 1))
            else:
                pred, ctx = model_conv(order[:,:,:,:i], LAYER_NUM, HIDDEN_UNIT, 'model_'+str(i+1))
            error_pred = abs(tf.subtract(pred, tf.expand_dims(order[:,:,:,(i+1)], axis=3)))
            loss_pred = tf.reduce_mean(error_pred)
            loss_ctx = LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx, error_pred)))
            pred_li.append(pred)
            ctx_li.append(ctx)
            error_pred_li.append(error_pred)
            loss_pred_li.append(loss_pred)
            loss_ctx_li.append(loss_ctx)
            loss_li.append(loss_pred + loss_ctx)
            total_loss += loss_pred + loss_ctx

        all_vars = tf.trainable_variables()

        optimizer_li = []
        for j in range(NUM_PATCHES):
            vars = [var for var in all_vars if 'model_'+str(j+1) in var.name]
            optimizer = tf.train.AdamOptimizer(LR).minimize(loss_li[j], var_list=vars)
            optimizer_li.append(optimizer)


        self.pred_li = pred_li
        self.ctx_li = ctx_li
        self.error_pred_li = error_pred_li
        self.loss_pred_li = loss_pred_li
        self.loss_ctx_li = loss_ctx_li
        self.loss_li = loss_li
        self.optimizer_li = optimizer_li
        self.optimizer_all = tf.train.AdamOptimizer(LR).minimize(total_loss, var_list=all_vars)


        '''
        # Prediction of 2
        pred_2, ctx_2 = model_conv(input_1, LAYER_NUM, HIDDEN_UNIT, 'pred_2')

        # Prediction of 3
        concat_1_2 = tf.concat([input_1, input_2], axis=3)
        pred_3, ctx_3 = model_conv(concat_1_2, LAYER_NUM, HIDDEN_UNIT, 'pred_3')

        # Prediction of 4
        concat_1_2_3 = tf.concat([input_1, input_2, input_3], axis=3)
        pred_4, ctx_4 = model_conv(concat_1_2_3, LAYER_NUM, HIDDEN_UNIT, 'pred_4')

        # Prediction error
        error_pred_2 = abs(tf.subtract(pred_2, input_2))
        error_pred_3 = abs(tf.subtract(pred_3, input_3))
        error_pred_4 = abs(tf.subtract(pred_4, input_4))

        # Losses
        loss_pred_2 = tf.reduce_mean(error_pred_2)
        loss_pred_3 = tf.reduce_mean(error_pred_3)
        loss_pred_4 = tf.reduce_mean(error_pred_4)

        loss_ctx_2 = LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_2, error_pred_2)))
        loss_ctx_3 = LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_3, error_pred_3)))
        loss_ctx_4 = LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_4, error_pred_4)))

        loss_2 = loss_pred_2 + loss_ctx_2
        loss_3 = loss_pred_3 + loss_ctx_3
        loss_4 = loss_pred_4 + loss_ctx_4

        total_loss = loss_2 + loss_3 + loss_4

        # Optimizer
        all_vars = tf.trainable_variables()
        vars_2 = [var for var in all_vars if 'pred_2' in var.name]
        vars_3 = [var for var in all_vars if 'pred_3' in var.name]
        vars_4 = [var for var in all_vars if 'pred_4' in var.name]

        self.optimizer_2 = tf.train.AdamOptimizer(LR).minimize(loss_2, var_list=vars_2)
        self.optimizer_3 = tf.train.AdamOptimizer(LR).minimize(loss_3, var_list=vars_3)
        self.optimizer_4 = tf.train.AdamOptimizer(LR).minimize(loss_4, var_list=vars_4)
        self.optimizer_all = tf.train.AdamOptimizer(LR).minimize(total_loss, var_list=all_vars)

        # Variables
        self.loss_2 = loss_2
        self.loss_3 = loss_3
        self.loss_4 = loss_4
        self.loss_all = loss_4 + loss_2 + loss_3

        self.loss_pred_2 = loss_pred_2
        self.loss_pred_3 = loss_pred_3
        self.loss_pred_4 = loss_pred_4
        self.loss_pred_all = loss_pred_2 + loss_pred_3 + loss_pred_4

        self.loss_ctx_2 = loss_ctx_2
        self.loss_ctx_3 = loss_ctx_3
        self.loss_ctx_4 = loss_ctx_4
        self.loss_ctx_all = loss_ctx_2 + loss_ctx_3 + loss_ctx_4

        self.pred_2 = pred_2
        self.pred_3 = pred_3
        self.pred_4 = pred_4

        self.ctx_2 = ctx_3
        self.ctx_3 = ctx_3
        self.ctx_4 = ctx_4
        '''
        # Original images
        self.input_1 = input_1
        self.input_2 = input_2
        self.input_3 = input_3
        self.input_4 = input_4


    def rgb2yuv(self, input_rgb):

        input_rgb = tf.cast(input_rgb, tf.float32)

        r,g,b = tf.split(input_rgb, 3, axis=3)

        u = b - tf.round((87 * r + 169 * g) / 256.0)
        v = r - g
        y = g + tf.round((86 * v + 29 * u) / 256.0)

        input_yuv = tf.concat([y,u,v], axis=3)

        return input_yuv

    def train(self):

        GPU_NUM = self.args.gpu_num
        DATA_DIR = self.args.data_dir
        CKPT_DIR = self.args.ckpt_dir
        LOAD = self.args.load
        CHANNEL_EPOCH = self.args.channel_epoch
        JOINT_EPOCH = self.args.joint_epoch
        BATCH_SIZE = self.args.batch_size
        PRINT_EVERY = self.args.print_every
        SAVE_EVERY = self.args.save_every
        CHANNEL_NUM = self.args.channel_num
        NUM_PATCHES = 11

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        global_step = tf.Variable(0, trainable=False)
        increase = tf.assign_add(global_step, 1)
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True

        # Read dataset
        train_data = read_dir(DATA_DIR + 'train/')

        with tf.Session(config=config) as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord, start=True)

            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

            # Load model if trained before
            if ckpt and LOAD:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model Loaded")

            epoch = sess.run(global_step)

            loss_pred_epoch = np.zeros(NUM_PATCHES)
            loss_ctx_epoch = np.zeros(NUM_PATCHES)

            for a in range(JOINT_EPOCH+NUM_PATCHES*CHANNEL_EPOCH):
                sess.run(increase)

                if epoch < NUM_PATCHES*CHANNEL_EPOCH:
                    if epoch % CHANNEL_EPOCH == 0:
                        print("========== Train Patch {} ==========".format(int(epoch/CHANNEL_EPOCH)+2))
                        optimizer = self.optimizer_li[int(epoch/CHANNEL_EPOCH)]
                else:
                    if epoch == NUM_PATCHES*CHANNEL_EPOCH:
                        print("========== Train All Patches ==========")
                        optimizer = self.optimizer_all

                input_crop = sess.run(self.input_crop)

                feed_dict_train = {
                    self.input : input_crop
                }

                _, loss_p, loss_c =\
                    sess.run([optimizer, self.loss_pred_li, self.loss_ctx_li], feed_dict=feed_dict_train)

                for i1 in range(NUM_PATCHES):
                    loss_pred_epoch[i1] += loss_p[i1]
                    loss_ctx_epoch[i1] += loss_c[i1]

                if (epoch + 1) % PRINT_EVERY == 0:

                    loss_pred_epoch[:] = [x / PRINT_EVERY for x in loss_pred_epoch]
                    loss_ctx_epoch[:] = [x / PRINT_EVERY for x in loss_ctx_epoch]
                    loss_epoch = [x+y for x,y in zip(loss_pred_epoch, loss_ctx_epoch)]
                    print('%04d\n' % (epoch + 1))
                    for i2 in range(NUM_PATCHES):
                        print('*** {} ***   lossPred='.format(i2+1), '{:9.4f}'.format(loss_pred_epoch[i2]), 'lossContext=', '{:9.4f}'.format(
                            loss_ctx_epoch[i2]), 'Loss=', '{:9.4f}'.format(loss_epoch[i2]))
                    print('***all*** lossPred=', '{:9.4f}'.format(sum(loss_pred_epoch)), 'lossContext=',
                      '{:9.4f}'.format(sum(loss_ctx_epoch)), 'Loss=', '{:9.4f}'.format(sum(loss_epoch)))

                if (epoch + 1) % SAVE_EVERY == 0:
                    saver.save(sess, CKPT_DIR + 'model_', global_step=epoch + 1)
                    print("Model Saved")

                epoch = sess.run(global_step)

    def test(self):
        GPU_NUM = self.args.gpu_num
        DATA_DIR = self.args.data_dir
        CKPT_DIR = self.args.ckpt_dir
        CHANNEL_NUM = self.args.channel_num
        NUM_PATCHES = 11

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Read dataset
        test_data = read_dir(DATA_DIR + 'mcm/')

        start = time.time()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

            # Load model
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model Loaded")
            else:
                print("No model to load")
                sys.exit(1)


            loss_pred_epoch = np.zeros(NUM_PATCHES)
            loss_ctx_epoch = np.zeros(NUM_PATCHES)

            img_idx = 0
            elapsed = 0

            for test_name in tqdm.tqdm(test_data):
                starttime = time.time()
                
                test_sample = cv2.cvtColor(cv2.imread(test_name),cv2.COLOR_BGR2RGB)

                test_sample = np.expand_dims(test_sample, axis=0)

                feed_dict_test = {
                    self.input : test_sample
                }

                i1, i2, i3, i4, p, c, loss_p, loss_c = sess.run(
                    [self.input_1, self.input_2, self.input_3, self.input_4, self.pred_li,
                     self.ctx_li, self.loss_pred_li, self.loss_ctx_li], feed_dict=feed_dict_test)

                for i_test in range(NUM_PATCHES):
                    loss_pred_epoch[i_test] += loss_p[i_test]
                    loss_ctx_epoch[i_test] += loss_c[i_test]

                endtime = time.time()
                elapsed += endtime-starttime
                save_image_data(img_idx, 1, i1)
                save_image_data(img_idx, 2, i2)
                save_image_data(img_idx, 3, i3)
                save_image_data(img_idx, 4, i4)
                for i in range(NUM_PATCHES):
                    save_pred_data(img_idx, i, p[i])
                    save_ctx_data(img_idx, i, c[i])

                size = np.zeros(2)
                size[0] = i1.shape[1]
                size[1] = i1.shape[2]
                np.savetxt('../c_compression/ICIP_Compression/data/' + str(img_idx) + '_size.txt', size, fmt='%d')
                
                print('num {}'.format(img_idx))
                img_idx += 1

            elapsed /= img_idx
            loss_pred_epoch[:] = [x / len(test_data) for x in loss_pred_epoch]
            loss_ctx_epoch[:] = [x / len(test_data) for x in loss_ctx_epoch]
            loss_epoch = [x + y for x, y in zip(loss_pred_epoch, loss_ctx_epoch)]

            for i2 in range(NUM_PATCHES):
                print('*** {} ***   lossPred='.format(i2 + 1), '{:9.4f}'.format(loss_pred_epoch[i2]), 'lossContext=',
                      '{:9.4f}'.format(
                          loss_ctx_epoch[i2]), 'Loss=', '{:9.4f}'.format(loss_epoch[i2]))
            print('***all*** lossPred=', '{:9.4f}'.format(sum(loss_pred_epoch)), 'lossContext=',
                  '{:9.4f}'.format(sum(loss_ctx_epoch)), 'Loss=', '{:9.4f}'.format(sum(loss_epoch)))

            print("elapsed time: {}s for {} images, {}s per image.".format(time.time()-start, img_idx, (time.time()-start)/(img_idx)))
            print("purely elapsed time: {}s per image.".format(elapsed))




if __name__ == "__main__":

    args = parse_args()
    my_net = Network(args)
    my_net.build()
    my_net.train()
    my_net.test()