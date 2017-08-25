import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell ##basisLSTMCELL 선택
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):#2단cell추가를 위해
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True) ## 멀티셀 구성하고 입력은 rnn_size =128
        ##cell의 output size는 128 = args.rnn_size
## feed_dict => placeholder 구성 현재 데이터가 (446,50,50) 이므로 input_data= shape(50,50)
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)##초기 state 구성

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", ##shape(128,65)
                                        [args.rnn_size, args.vocab_size],initializer= tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size], initializer= tf.contrib.layers.xavier_initializer())

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size],initializer= tf.contrib.layers.xavier_initializer())#shape(65,128)
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)##shape(50,50,128)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        inputs = tf.split(inputs, args.seq_length, 1) ##seq_length 개수 만큼 잘라줌 inputs=(50,1,128)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs] ## input에 shape가 1인 모든 차원을 제거한다

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size]) ##output = shape(2500,128) output shape => list.each shape =<50,128>  이고 concat의 2번째 매게변수가 1이므로 50개<50,128> -> 2500개로 연결


        self.logits = tf.matmul(output, softmax_w) + softmax_b ##tensor계산
        self.probs = tf.nn.softmax(self.logits)## 활성화 함수 통과
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])##loss 계산
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length ## 하나의 cost 출력

        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=500, prime='', sampling_type=1):## 학습된 모델에서 단어를 생성하는 부분
        sentence_start_token = "CODE_START"
        sentence_end_token = "CODE_END"

        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = ""
        char = sentence_start_token
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed )
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]

            if sentence_end_token == pred or sentence_start_token == pred or pred == "." or pred==";":
                ret+='\n'
            else:
                ret += pred
                ret +=" "
            char = pred
        return ret
