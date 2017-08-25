from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model


import csv
import itertools
import operator
import numpy as np
import math as math
import sys
import os
import time
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)#포맷지
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare', ## 데이터 저장 장소
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models') ##모델 저장소
    parser.add_argument('--log_dir', type=str, default='logs', ## log데이터 저장소
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128,## RNN hidden 사이즈 DF = 128
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=10, ## RNN Layers 지정 DF =2
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm', ## RNN model 정의
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=50, ## batch_size 지정
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50, ## 문자열 길이 지정
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=100, ## epochs Number 지정
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000, ## 결과 저장??
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5., ## gradient를 깎는 정도 ??
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,## learning rate 지정
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97, ##부패되는 정
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=0.7, ## keeping weight output 확률
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=0.7, ## keeping weight input 확률
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--init_from', type=str, default=None, ##초기화 -> pkl 파일형 파이썬 객체를 저장하고, 읽어 들ㅇ
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration; 
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)


def train(args):

    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)## data Proprocessing 하는 부분 -> utils.py(TextLoader Class)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f: ##init_from이 설정이 안되어 있을 경우
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)#--> model.py 에 model class로 진입 RNN의 모델링 과정을 Process

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer()) ## TF 초기화
        saver = tf.train.Saver(tf.global_variables())## TF.variables() 저장
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)##0으로 초기화된 Shape(50,128)

            for b in range(data_loader.num_batches): #0~ 446번을 돈다
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y} ##  x,y shape(50,50)
                for i, (c, h) in enumerate(model.initial_state): ## 초기 스테이트 값과  feed 의 x, y를 받는다
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)

                # instrument for tensorboard
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * data_loader.num_batches + b)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                        or (e == args.num_epochs-1 and
                            b == data_loader.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
