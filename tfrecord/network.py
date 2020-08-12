import tensorflow as tf
import numpy as np
import os
import random
import cv2
import sys
import tqdm

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

            loss_pred_epoch_2 = loss_pred_epoch_3 = loss_pred_epoch_4 = 0
            loss_ctx_epoch_2 = loss_ctx_epoch_3 = loss_ctx_epoch_4 = 0

            while True:
                sess.run(increase)

                if epoch < CHANNEL_EPOCH :
                    if epoch == 0:
                        print("========== Train Patch 2 ==========")
                    optimizer = self.optimizer_2
                elif epoch < 2*CHANNEL_EPOCH:
                    if epoch == CHANNEL_EPOCH:
                        print("========== Train Patch 3 ==========")
                    optimizer = self.optimizer_3
                elif epoch < 3*CHANNEL_EPOCH:
                    if epoch == 2*CHANNEL_EPOCH:
                        print("========== Train Patch 4 ==========")
                    optimizer = self.optimizer_4
                else:
                    if epoch == 3*CHANNEL_EPOCH:
                        print("========== Train All Patches ==========")
                    optimizer = self.optimizer_all

                input_crop = sess.run(self.input_crop)

                feed_dict_train = {
                    self.input : input_crop
                }

                _, loss_p_2, loss_p_3, loss_p_4, loss_c_2, loss_c_3, loss_c_4 =\
                    sess.run([optimizer, self.loss_pred_2, self.loss_pred_3, self.loss_pred_4,
                     self.loss_ctx_2, self.loss_ctx_3, self.loss_ctx_4], feed_dict=feed_dict_train)

                loss_pred_epoch_2 += loss_p_2
                loss_pred_epoch_3 += loss_p_3
                loss_pred_epoch_4 += loss_p_4

                loss_ctx_epoch_2 += loss_c_2
                loss_ctx_epoch_3 += loss_c_3
                loss_ctx_epoch_4 += loss_c_4

                if (epoch + 1) % PRINT_EVERY == 0:
                    loss_pred_epoch_2 /= PRINT_EVERY
                    loss_pred_epoch_3 /= PRINT_EVERY
                    loss_pred_epoch_4 /= PRINT_EVERY

                    loss_ctx_epoch_2 /= PRINT_EVERY
                    loss_ctx_epoch_3 /= PRINT_EVERY
                    loss_ctx_epoch_4 /= PRINT_EVERY

                    loss_epoch_2 = loss_pred_epoch_2 + loss_ctx_epoch_2
                    loss_epoch_3 = loss_pred_epoch_3 + loss_ctx_epoch_3
                    loss_epoch_4 = loss_pred_epoch_4 + loss_ctx_epoch_4

                    print('%04d\n' % (epoch + 1),
                      '***2***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_2), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_2), 'Loss=', '{:9.4f}\n'.format(loss_epoch_2),
                      '***3***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_3), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_3), 'Loss=', '{:9.4f}\n'.format(loss_epoch_3),
                      '***4***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_4), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_4), 'Loss=', '{:9.4f}\n'.format(loss_epoch_4),
                      '***all*** lossPred=', '{:9.4f}'.format(loss_pred_epoch_2 + loss_pred_epoch_3 + loss_pred_epoch_4), 'lossContext=',
                      '{:9.4f}'.format(loss_ctx_epoch_2 + loss_ctx_epoch_3 + loss_ctx_epoch_4), 'Loss=', '{:9.4f}'.format(loss_epoch_2 + loss_epoch_3 + loss_epoch_4))

                if (epoch + 1) % SAVE_EVERY == 0:
                    saver.save(sess, CKPT_DIR + 'model_', global_step=epoch + 1)
                    print("Model Saved")

                epoch = sess.run(global_step)

    def test(self):
        GPU_NUM = self.args.gpu_num
        DATA_DIR = self.args.data_dir
        CKPT_DIR = self.args.ckpt_dir
        CHANNEL_NUM = self.args.channel_num

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Read dataset
        test_data = read_dir(DATA_DIR + 'test/')

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

            pred_2 = []
            pred_3 = []
            pred_4 = []
            ctx_2 = []
            ctx_3 = []
            ctx_4 = []
            loss_pred_epoch_2 = loss_pred_epoch_3 = loss_pred_epoch_4 = 0
            loss_ctx_epoch_2 = loss_ctx_epoch_3 = loss_ctx_epoch_4 = 0

            img_idx = 0

            for test_name in tqdm.tqdm(test_data):
                
                test_sample = cv2.cvtColor(cv2.imread(test_name),cv2.COLOR_BGR2RGB)

                test_sample = np.expand_dims(test_sample, axis=0)

                feed_dict_test = {
                    self.input : test_sample
                }

                i1, i2, i3, i4, p2, p3, p4, c2, c3, c4 = sess.run(
                    [self.input_1, self.input_2, self.input_3, self.input_4, self.pred_2, self.pred_3, self.pred_4,
                     self.ctx_2, self.ctx_3, self.ctx_4], feed_dict=feed_dict_test)

                loss_p_2, loss_p_3, loss_p_4, loss_c_2, loss_c_3, loss_c_4 = sess.run(
                    [self.loss_pred_2, self.loss_pred_3, self.loss_pred_4, self.loss_ctx_2, self.loss_ctx_3,
                     self.loss_ctx_4], feed_dict=feed_dict_test)

                loss_pred_epoch_2 += loss_p_2
                loss_pred_epoch_3 += loss_p_3
                loss_pred_epoch_4 += loss_p_4

                loss_ctx_epoch_2 += loss_c_2
                loss_ctx_epoch_3 += loss_c_3
                loss_ctx_epoch_4 += loss_c_4
                '''
                pred_2.append(p2)
                pred_3.append(p3)
                pred_4.append(p4)
                ctx_2.append(c2)
                ctx_3.append(c3)
                ctx_4.append(c4)
                '''
                save_image_data(img_idx, 1, i1)
                save_image_data(img_idx, 2, i2)
                save_image_data(img_idx, 3, i3)
                save_image_data(img_idx, 4, i4)
                save_pred_data(img_idx, 2, p2)
                save_pred_data(img_idx, 3, p3)
                save_pred_data(img_idx, 4, p4)
                save_ctx_data(img_idx, 2, c2)
                save_ctx_data(img_idx, 3, c3)
                save_ctx_data(img_idx, 4, c4)

                size = np.zeros(2)
                size[0]=i1.shape[1]
                size[1]=i1.shape[2]
                np.savetxt('../c_compression/ICIP_Compression/data/' + str(img_idx) + '_size.txt', size, fmt='%d')
                
                print('num {}'.format(img_idx))
                img_idx += 1


            loss_pred_epoch_2 /= len(test_data)
            loss_pred_epoch_3 /= len(test_data)
            loss_pred_epoch_4 /= len(test_data)
            loss_ctx_epoch_2 /= len(test_data)
            loss_ctx_epoch_3 /= len(test_data)
            loss_ctx_epoch_4 /= len(test_data)
            loss_epoch_2 = loss_pred_epoch_2 + loss_ctx_epoch_2
            loss_epoch_3 = loss_pred_epoch_3 + loss_ctx_epoch_3
            loss_epoch_4 = loss_pred_epoch_4 + loss_ctx_epoch_4

            print('test result: %04d\n' % (len(test_data)),
                  '***2***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_2), 'lossContext=',
                  '{:9.4f}'.format(loss_ctx_epoch_2), 'Loss=', '{:9.4f}\n'.format(loss_epoch_2),
                  '***3***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_3), 'lossContext=',
                  '{:9.4f}'.format(loss_ctx_epoch_3), 'Loss=', '{:9.4f}\n'.format(loss_epoch_3),
                  '***4***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_4), 'lossContext=',
                  '{:9.4f}'.format(loss_ctx_epoch_4), 'Loss=', '{:9.4f}\n'.format(loss_epoch_4),
                  '***all*** lossPred=', '{:9.4f}'.format(loss_pred_epoch_2 + loss_pred_epoch_3 + loss_pred_epoch_4),
                  'lossContext=',
                  '{:9.4f}'.format(loss_ctx_epoch_2 + loss_ctx_epoch_3 + loss_ctx_epoch_4), 'Loss=',
                  '{:9.4f}'.format(loss_epoch_2 + loss_epoch_3 + loss_epoch_4))




if __name__ == "__main__":

    args = parse_args()
    my_net = Network(args)
    my_net.build()
    my_net.train()
    my_net.test()