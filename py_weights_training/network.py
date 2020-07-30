import tensorflow as tf
import numpy as np
import os
import random

from config import parse_args
from module import model, model_conv
from data_mono import create_dataset
from data import read_dir, create_data, create_test_data


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
        CHANNEL_NUM = self.args.channel_num

        LR = self.args.lr

        self.input = tf.placeholder(tf.float32, (None, None, None, CHANNEL_NUM))

        input_depth = tf.nn.space_to_depth(self.input, 2)

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

    def train(self):

        GPU_NUM = self.args.gpu_num
        DATA_DIR = self.args.data_dir
        CKPT_DIR = self.args.ckpt_dir
        LOAD = self.args.load
        EPOCH = self.args.epoch
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
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

            # Load model if trained before
            if ckpt and LOAD:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model Loaded")

            epoch = sess.run(global_step)
            EPOCH = EPOCH + EPOCH % 4

            loss_pred_epoch_2 = loss_pred_epoch_3 = loss_pred_epoch_4 = 0
            loss_ctx_epoch_2 = loss_ctx_epoch_3 = loss_ctx_epoch_4 = 0

            while True:
                sess.run(increase)

                step = 0

                if epoch == 0:
                    print("========== Train Patch 2 ==========")
                    optimizer = self.optimizer_2
                elif epoch == EPOCH:
                    print("========== Train Patch 3 ==========")
                    optimizer = self.optimizer_3
                elif epoch == 2 * EPOCH:
                    print("========== Train Patch 4 ==========")
                    optimizer = self.optimizer_4
                elif epoch == 3 * EPOCH:
                    print("========== Train all Patches ==========")
                    optimizer = self.optimizer_all
                else:
                    pass

                batch_input = create_data(self.args, train_data, channel=CHANNEL_NUM)

                if len(batch_input) != 0:
                    feed_dict_train = {
                        self.input : batch_input
                    }

                    sess.run(optimizer, feed_dict=feed_dict_train)
                    loss_p_2, loss_p_3, loss_p_4, loss_c_2, loss_c_3, loss_c_4 = sess.run(
                            [self.loss_pred_2, self.loss_pred_3, self.loss_pred_4, self.loss_ctx_2, self.loss_ctx_3, self.loss_ctx_4], feed_dict=feed_dict_train)

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

            pred_2 = []
            pred_3 = []
            pred_4 = []
            ctx_2 = []
            ctx_3 = []
            ctx_4 = []

            for img_idx in range(len(test_data)):
                
                input_data = create_test_data(test_data, img_idx, channel=CHANNEL_NUM)

                feed_dict_test = {
                    self.input : input_data
                }

                p2, p3, p4, c2, c3, c4 = sess.run([self.pred_2, self.pred_3, self.pred_4, self.ctx_2, self.ctx_3, self.ctx_4], feed_dict=feed_dict_test)

                pred_2.append(p2)                
                pred_3.append(p3)
                pred_4.append(p4)
                ctx_2.append(c2)
                ctx_3.append(c3)
                ctx_4.append(c4)

if __name__ == "__main__":

    args = parse_args()
    my_net = Network(args)
    my_net.build()
    #my_net.train()
    my_net.test()