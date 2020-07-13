import tensorflow as tf
import numpy as np
import os
import random

from config import parse_args
from module import model, model_conv
from data_mono import create_dataset


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

        LAYER_NUM = self.args.layer_num
        HIDDEN_UNIT = self.args.hidden_unit

        LAMBDA_CTX = self.args.lambda_ctx

        LR = self.args.lr

        self.input_1 = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.input_2 = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.input_3 = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.input_4 = tf.placeholder(tf.float32, (None, 128, 128, 1))

        pred_4, ctx_4 = model_conv(self.input_1, LAYER_NUM, HIDDEN_UNIT, 'pred_4')
        concat_1_4 = tf.concat([self.input_1, pred_4], axis=3)

        pred_2, ctx_2 = model_conv(concat_1_4, LAYER_NUM, HIDDEN_UNIT, 'pred_2')
        concat_1_2_4 = tf.concat([self.input_1, pred_4, pred_2], axis=3)

        pred_3, ctx_3 = model_conv(concat_1_2_4, LAYER_NUM, HIDDEN_UNIT, 'pred_3')

        error_pred_4 = abs(tf.subtract(pred_4, self.input_4))*LAMBDA_CTX
        error_pred_3 = abs(tf.subtract(pred_3, self.input_3))*LAMBDA_CTX
        error_pred_2 = abs(tf.subtract(pred_2, self.input_2))*LAMBDA_CTX

        loss_pred_4 = tf.reduce_mean(error_pred_4)
        loss_pred_3 = tf.reduce_mean(error_pred_3)
        loss_pred_2 = tf.reduce_mean(error_pred_2)

        loss_ctx_4 = tf.reduce_mean(abs(tf.subtract(ctx_4, error_pred_4)))
        loss_ctx_3 = tf.reduce_mean(abs(tf.subtract(ctx_3, error_pred_3)))
        loss_ctx_2 = tf.reduce_mean(abs(tf.subtract(ctx_2, error_pred_2)))

        loss_4 = loss_pred_4 + loss_ctx_4
        loss_2 = loss_pred_2 + loss_ctx_2
        loss_3 = loss_pred_3 + loss_ctx_3

        all_vars = tf.trainable_variables()
        vars_4 = [var for var in all_vars if 'pred_4' in var.name]
        vars_2 = [var for var in all_vars if 'pred_2' in var.name]
        vars_3 = [var for var in all_vars if 'pred_3' in var.name]

        self.optimizer_4 = tf.train.AdamOptimizer(LR).minimize(loss_4, var_list=vars_4)
        self.optimizer_2 = tf.train.AdamOptimizer(LR).minimize(loss_2, var_list=vars_2)
        self.optimizer_3 = tf.train.AdamOptimizer(LR).minimize(loss_3, var_list=vars_3)
        self.optimizer_all = tf.train.AdamOptimizer(LR).minimize(loss_4 + loss_2 + loss_3, var_list=all_vars)

        self.loss_4 = loss_4
        self.loss_2 = loss_2
        self.loss_3 = loss_3
        self.loss_all = loss_4 + loss_2 + loss_3
        self.loss_pred_4 = loss_pred_4
        self.loss_pred_2 = loss_pred_2
        self.loss_pred_3 = loss_pred_3
        self.loss_pred_all = loss_pred_4 + loss_pred_2 + loss_pred_3
        self.loss_ctx_4 = loss_ctx_4
        self.loss_ctx_2 = loss_ctx_2
        self.loss_ctx_3 = loss_ctx_3
        self.loss_ctx_all = loss_ctx_4 + loss_ctx_2 + loss_ctx_3
        self.ctx_4 = ctx_4
        self.ctx_2 = ctx_2
        self.ctx_3 = ctx_3


    def train(self):

        GPU_NUM = self.args.gpu_num
        CKPT_DIR = self.args.ckpt_dir
        LOAD = self.args.load
        EPOCH = self.args.epoch
        BATCH_SIZE = self.args.batch_size
        SAVE_INTERVAL = self.args.save_interval

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        #self.check_data_exist()

        global_step = tf.Variable(0, trainable=False)
        increase = tf.assign_add(global_step, 1)
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

            # Load model if trained before
            if ckpt and LOAD:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model Loaded")

            # Load dataset
            x1_train, x2_train, x3_train, x4_train, data_num = self.load_dataset("train")
            x1_valid, x2_valid, x3_valid, x4_valid, _ = self.load_dataset("valid")

            epoch = sess.run(global_step)
            EPOCH = EPOCH + EPOCH % 4

            while epoch < EPOCH:
                sess.run(increase)

                loss_pred_epoch_4 = loss_pred_epoch_3 = loss_pred_epoch_2 = 0
                loss_ctx_epoch_4 = loss_ctx_epoch_3 = loss_ctx_epoch_2 = 0

                step = 0

                if epoch == 0:
                    print("========== Train 4 Patch ==========")
                    optimizer = self.optimizer_4
                elif epoch == EPOCH / 4:
                    print("========== Train 2 Patch ==========")
                    optimizer = self.optimizer_2
                elif epoch == 2 * EPOCH / 4:
                    print("========== Train 3 Patch ==========")
                    optimizer = self.optimizer_3
                elif epoch == 3 * EPOCH / 4:
                    print("========== Train all Patches ==========")
                    optimizer = self.optimizer_all
                else:
                    pass

                for step in range(int(data_num / BATCH_SIZE) + 1):
                    batch_input_1, batch_input_2, batch_input_3, batch_input_4 = self.get_batch(x1_train, x2_train, x3_train, x4_train, step, BATCH_SIZE)

                    if len(batch_input_1) != 0:
                        feed_dict = {
                            self.input_1 : batch_input_1,
                            self.input_2: batch_input_2,
                            self.input_3: batch_input_3,
                            self.input_4: batch_input_4
                        }

                        sess.run(optimizer, feed_dict=feed_dict)
                        loss_p_4, loss_p_2, loss_p_3, loss_c_4, loss_c_2, loss_c_3 = sess.run(
                            [self.loss_pred_4, self.loss_pred_2, self.loss_pred_3, self.loss_ctx_4, self.loss_ctx_2, self.loss_ctx_3], feed_dict=feed_dict)

                        loss_pred_epoch_4 += loss_p_4
                        loss_pred_epoch_2 += loss_p_2
                        loss_pred_epoch_3 += loss_p_3

                        loss_ctx_epoch_4 += loss_c_4
                        loss_ctx_epoch_2 += loss_c_2
                        loss_ctx_epoch_3 += loss_c_3

                loss_pred_epoch_4 /= step
                loss_pred_epoch_2 /= step
                loss_pred_epoch_3 /= step

                loss_ctx_epoch_4 /= step
                loss_ctx_epoch_2 /= step
                loss_ctx_epoch_3 /= step

                loss_epoch_4 = loss_pred_epoch_4 + loss_ctx_epoch_4
                loss_epoch_2 = loss_pred_epoch_2 + loss_ctx_epoch_2
                loss_epoch_3 = loss_pred_epoch_3 + loss_ctx_epoch_3

                print('%04d\n' % (epoch + 1),
                      '***4***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_4), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_4), 'Loss=', '{:9.4f}\n'.format(loss_epoch_4),
                      '***2***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_2), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_2), 'Loss=', '{:9.4f}\n'.format(loss_epoch_2),
                      '***3***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_3), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_3), 'Loss=', '{:9.4f}\n'.format(loss_epoch_3),
                      '***all*** lossPred=', '{:9.4f}'.format(loss_pred_epoch_4 + loss_pred_epoch_2 + loss_pred_epoch_3), 'lossContext=',
                      '{:9.4f}'.format(loss_ctx_epoch_4 + loss_ctx_epoch_2 + loss_ctx_epoch_3), 'Loss=', '{:9.4f}'.format(loss_epoch_4 + loss_epoch_2 + loss_epoch_3))

                if (epoch + 1) % SAVE_INTERVAL == 0:
                    saver.save(sess, CKPT_DIR + 'model_', global_step=epoch + 1)
                    #self.print_weights('4')
                    #self.print_weights('2')
                    #self.print_weights('3')
                    print("Model Saved")

                epoch = sess.run(global_step)

            #self.print_all_weights()

    def check_data_exist(self):
        path = self.args.data_dir + "npy/"
        filelist = os.listdir(path)

        if not (len(filelist) == 6):
            create_dataset(self.args, "train")
            create_dataset(self.args, "valid")
            create_dataset(self.args, "test")

    def load_dataset(self, data_type, shuffle=True):
        path = self.args.data_dir + 'npy/'

        x1 = np.load(path + 'image_' + data_type + '1.npy')
        x2 = np.load(path + 'image_' + data_type + '2.npy')
        x3 = np.load(path + 'image_' + data_type + '3.npy')
        x4 = np.load(path + 'image_' + data_type + '4.npy')

        data_num = x1.shape[0]

        # Shuffle dataset
        if shuffle:
            data_idx = list(range(data_num))
            random.shuffle(data_idx)

            x1 = x1[data_idx]
            x2 = x2[data_idx]
            x3 = x3[data_idx]
            x4 = x4[data_idx]

        return x1, x2, x3, x4, data_num

    @staticmethod
    def get_batch(x1, x2, x3, x4, step, batch_size):
        offset = step * batch_size

        if step == int(len(x1) / batch_size):
            batch_input_1 = x1[offset:]
            batch_input_2 = x1[offset:]
            batch_input_3 = x1[offset:]
            batch_input_4 = x1[offset:]
        else:
            batch_input_1 = x1[offset:(offset + batch_size)]
            batch_input_2 = x2[offset:(offset + batch_size)]
            batch_input_3 = x3[offset:(offset + batch_size)]
            batch_input_4 = x4[offset:(offset + batch_size)]

        return batch_input_1, batch_input_2, batch_input_3, batch_input_4



