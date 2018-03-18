# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.rnn import RNNCell
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import MultiRNNCell

def get_attn_params(attn_size,initializer = tf.truncated_normal_initializer):
    '''
    Args:
        attn_size: the size of attention specified in https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
        initializer: the author of the original paper used gaussian initialization however I found xavier converge faster

    Returns:
        params: A collection of parameters used throughout the layers
    '''
    with tf.variable_scope("attention_weights"):
        params = {"W_u_Q":tf.get_variable("W_u_Q",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                #"W_ru_Q":tf.get_variable("W_ru_Q",dtype = tf.float32, shape = (2 * attn_size, 2 * attn_size), initializer = initializer()),
                "W_u_P":tf.get_variable("W_u_P",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_v_P":tf.get_variable("W_v_P",dtype = tf.float32, shape = (attn_size, attn_size), initializer = initializer()),
                "W_v_P_2":tf.get_variable("W_v_P_2",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_g":tf.get_variable("W_g",dtype = tf.float32, shape = (4 * attn_size, 4 * attn_size), initializer = initializer()),
                "W_h_P":tf.get_variable("W_h_P",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_v_Phat":tf.get_variable("W_v_Phat",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_h_a":tf.get_variable("W_h_a",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_v_Q":tf.get_variable("W_v_Q",dtype = tf.float32, shape = (attn_size,  attn_size), initializer = initializer()),
                "v":tf.get_variable("v",dtype = tf.float32, shape = (attn_size), initializer =initializer())}
        return params

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.is_training = True
        self.zoneout = args.zoneout
        self.attn_size = args.attn_size
        self.SRU = args.SRU


        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        # [ batch size , time length]
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

    # question 和 passage 经过一个双向lstm的编码
    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    # _match 函数 是唯一被修改的函数，r-net 在 _match函数中实现。
    def _match(self):
        # 将这里 修改为 r-net
        # 输入为 self.sep_p_encodes [batch size , passage length , dim*2] 和 self.sep_q_encodes [ batch size , question length , dim*2]
        # 输出为 self.match_p_encodes [ batch size , passage length , dim*n]

        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        '''''''# attention 用的相关参数初始化 (r-net中有两个attention ，passage question attention 和 passage self attention)'''
        self.params = get_attn_params(self.attn_size, initializer=tf.contrib.layers.xavier_initializer)
        ''''#r-net的主要内容'''
        self._attention_match_rnn()

        # 保留 dureader baseline 的 dropout
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _attention_match_rnn(self):
        # Apply gated attention recurrent network for both query-passage matching and self matching networks
        batch_size = tf.shape(self.sep_q_encodes)[0]
        with tf.variable_scope("attention_match_rnn"):
            memory = self.sep_q_encodes
            inputs = self.sep_p_encodes
            scopes = ["question_passage_matching", "self_matching"]
            params = [(([self.params["W_u_Q"],
                    self.params["W_u_P"],
                    self.params["W_v_P"]],self.params["v"]),
                    self.params["W_g"]),
                (([self.params["W_v_P_2"],
                    self.params["W_v_Phat"]],self.params["v"]),
                    self.params["W_g"])]

            '''这个for循环 ，第一次循环 对应 r-net文章中讲的 passage - question attention ，第二次循环为 passage self attention'''
            for i in range(2):
                args = {"num_units": self.attn_size,
                        "memory": memory,
                        "params": params[i],
                        "self_matching": False if i == 0 else True,
                        "memory_len": self.q_length if i == 0 else self.p_length,
                        "is_training": self.is_training,
                        "use_SRU": self.SRU,
                        "attn_size":self.attn_size}

                '''# 建立一个 cell(因为要双向，所以for 循环两个) ，这个cell 每一次 time 计算 都有 attention 和 drop out'''
                cell = [apply_dropout(gated_attention_Wrapper(**args), size = inputs.shape[-1], is_training = self.is_training,dropout_keep_prob = self.dropout_keep_prob,zoneout = self.zoneout) for _ in range(2)]

                # 用上面的 cell 进行 rnn 操作
                '''将上面生成的两个 cell 应用到下面的attention_rnn 函数中'''
                inputs = attention_rnn(inputs,
                            self.p_length,
                            self.attn_size,
                            cell,
                            scope = scopes[i])
                memory = inputs # self matching (attention over itself)


            '''r-net 计算结果保留在 match p encodes中'''
            self.match_p_encodes = inputs

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob}

            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        self.is_training = True
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        self.is_training = False
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0}
            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, encoding='utf8', ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))

# question ，passage attention 与 passage self attention 的 cell 包装
# 首先 构建 cell ，然后在 __call__中 实现 attention
class gated_attention_Wrapper(RNNCell):
  def __init__(self,
               num_units,
               memory,
               params,
               self_matching = False,
               memory_len = None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
			   is_training = True,
               use_SRU = False,
               attn_size = 150):
    super(gated_attention_Wrapper, self).__init__(_reuse=reuse)
    cell = SRUCell if use_SRU else GRUCell

    # 首先 创建 cell
    self._cell = cell(num_units, is_training = is_training)

    # 其次 设置 参数
    self._num_units = num_units
    self._activation = math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._attention = memory
    self._params = params
    self._self_matching = self_matching
    self._memory_len = memory_len
    self._is_training = is_training
    self.attn_size = attn_size

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope = None):
    """Gated recurrent unit (GRU) with nunits cells."""
    # inputs [batch size , , dim]
    with vs.variable_scope("attention_pool"):
        inputs = gated_attention(self._attention,
                                inputs,
                                state,
                                self._num_units,
                                params = self._params,
                                self_matching = self._self_matching,
                                memory_len = self._memory_len,
                                attn_size=  self.attn_size)
    output, new_state = self._cell(inputs, state, scope)
    return output, new_state

def apply_dropout(cell, size = None, is_training = True,dropout_keep_prob=None,zoneout = None):
    '''
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    '''
    if dropout_keep_prob is None and zoneout is None:
        return cell
    if zoneout is not None:
        return ZoneoutWrapper(cell, state_zoneout_prob= zoneout, is_training = is_training)
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(cell,
                                            output_keep_prob = dropout_keep_prob,
                                            # variational_recurrent = True,
                                            # input_size = size,
                                            dtype = tf.float32)
    else:
        return cell

class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
  """Operator adding zoneout to all states (states+cells) of the given cell."""

  def __init__(self, cell, state_zoneout_prob, is_training=True, seed=None):
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not an RNNCell.")
    if (isinstance(state_zoneout_prob, float) and
        not (state_zoneout_prob >= 0.0 and state_zoneout_prob <= 1.0)):
      raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                       % state_zoneout_prob)
    self._cell = cell
    self._zoneout_prob = state_zoneout_prob
    self._seed = seed
    self.is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    output, new_state = self._cell(inputs, state, scope)
    if self.is_training:
        new_state = (1 - self._zoneout_prob) * tf.nn.dropout(new_state - state, (1 - self._zoneout_prob), seed=self._seed) + state
    else:
        new_state = self._zoneout_prob * state + (1 - self._zoneout_prob) * new_state
    return output, new_state

def gated_attention(memory, inputs, states, units, params, self_matching = False, memory_len = None, scope="gated_attention",attn_size = 150):
    # inputs [ batch size , dim]
    # memory [ batch size , length , dim]
    with tf.variable_scope(scope):
        weights, W_g = params
        # memory [ batch size,  question length , dim*2]
        batch_size = tf.shape(memory)[0]
        inputs_ = [memory, inputs]
        states = tf.reshape(states,(batch_size,attn_size))
        if not self_matching:
            inputs_.append(states)

        scores = attention(inputs_, units, weights, memory_len = memory_len,attn_size = attn_size,batch_size = batch_size)
        scores = tf.expand_dims(scores,-1)
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs,attention_pool),axis = 1)
        g_t = tf.sigmoid(tf.matmul(inputs,W_g))
        return g_t * inputs

def attention(inputs, units, weights, scope = "attention", memory_len = None, reuse = None,attn_size =150,batch_size=None):
    # inputs [ memory , inputs ,state ]
    # memory [ batch size , length ,dim] inputs [ batch size, dim] state [ batch size , dim]
    with tf.variable_scope(scope, reuse = reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp,w) in enumerate(zip(inputs,weights)):
            # shapes = inp.shape.as_list()######################################## 在这里，因为图构建的时候 不知道 batch size 和 time length，所以这两个维度 都是None
            shapes = tf.shape(inp)
            dim_num = len(inp.shape.as_list())
            # shapes = tf.shape(inp)
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],attn_size], initializer = tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if dim_num> 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))   ##################################### 这里的 shapes [0] shapes[1] 都是 None  ，所以没法 reshape 。。。

            else:
                outputs = tf.reshape(outputs, (shapes[0],1,-1))

            outputs_.append(outputs)
        outputs = sum(outputs_)
        if True:# 在 attention中 用 bias
            b = tf.get_variable("b", shape = attn_size, dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            outputs += b
        # scores [batch size , length]
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores) # all attention output is softmaxed now

def mask_attn_score(score, memory_sequence_length, score_mask_value = -1e8):
    score_mask = tf.sequence_mask(
        memory_sequence_length, maxlen= tf.shape(score)[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)

class SRUCell(RNNCell):
  """Simple Recurrent Unit (SRU).
     This implementation is based on:
     Tao Lei and Yu Zhang,
     "Training RNNs as Fast as CNNs,"
     https://arxiv.org/abs/1709.02755
  """

  def __init__(self, num_units, activation=None, is_training=True, reuse=None):
      self._num_units = num_units
      self._activation = activation or tf.tanh
      self._is_training = is_training

  @property
  def output_size(self):
      return self._num_units

  @property
  def state_size(self):
      return self._num_units

  def __call__(self, inputs, state, scope=None):
      """Run one step of SRU."""
      with tf.variable_scope(scope or type(self).__name__):  # "SRUCell"
          with tf.variable_scope("x_hat"):
              x = linear([inputs], self._num_units, False)
          with tf.variable_scope("gates"):
              concat = tf.sigmoid(linear([inputs], 2 * self._num_units, True))
              f, r = tf.split(concat, 2, axis=1)
          with tf.variable_scope("candidates"):
              c = self._activation(f * state + (1 - f) * x)
              # variational dropout as suggested in the paper (disabled)
              # if self._is_training and Params.dropout is not None:
              #     c = tf.nn.dropout(c, keep_prob = 1 - Params.dropout)
          # highway connection
          # Our implementation is slightly different to the paper
          # https://arxiv.org/abs/1709.02755 in a way that highway network
          # uses x_hat instead of the cell inputs. Check equation (7) from the original
          # paper for SRU.
          h = r * c + (1 - r) * x
      return h, c

class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               is_training=True):
      super(GRUCell, self).__init__(_reuse=reuse)
      self._num_units = num_units
      self._activation = activation or math_ops.tanh
      self._kernel_initializer = kernel_initializer
      self._bias_initializer = bias_initializer
      self._is_training = is_training

  @property
  def state_size(self):
      return self._num_units

  @property
  def output_size(self):
      return self._num_units

  def __call__(self, inputs, state, scope=None):
      """Gated recurrent unit (GRU) with nunits cells."""
      if inputs.shape.as_list()[-1] != self._num_units:
          with vs.variable_scope("projection"):
              res = linear(inputs, self._num_units, False, )
      else:
          res = inputs
      with vs.variable_scope("gates"):  # Reset gate and update gate.
          # We start with bias of 1.0 to not reset and not update.
          bias_ones = self._bias_initializer
          if self._bias_initializer is None:
              dtype = [a.dtype for a in [inputs, state]][0]
              bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
          value = math_ops.sigmoid(
              linear([inputs, state], 2 * self._num_units, True, bias_ones,
                     self._kernel_initializer))
          r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
      with vs.variable_scope("candidate"):
          c = self._activation(
              linear([inputs, r * state], self._num_units, True,
                     self._bias_initializer, self._kernel_initializer))
      #   recurrent dropout as proposed in https://arxiv.org/pdf/1603.05118.pdf (currently disabled)
      # if self._is_training and Params.dropout is not None:
      # c = tf.nn.dropout(c, 1 - Params.dropout)
      new_h = u * state + (1 - u) * c
      return new_h + res, new_h

def linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  # 判断 输入 args 是不是 合法
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    # arg的每个输入都必须是一个二维矩阵
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)

def attention_rnn(inputs, inputs_len, units, attn_cell, bidirection = True, scope = "gated_attention_rnn", is_training = True):
    with tf.variable_scope(scope):
        if bidirection:
            # 直接 双向 rnn
            # inputs [batch size , passage length , dim]
            outputs = bidirectional_GRU(inputs,
                                        inputs_len,
                                        cell = attn_cell,
                                        scope = scope + "_bidirectional",
                                        output = 0,
                                        is_training = is_training,
                                        units = units)

        else:
            outputs, _ = tf.nn.dynamic_rnn(attn_cell, inputs,
                                            sequence_length = inputs_len,
                                            dtype=tf.float32)
        return outputs

def bidirectional_GRU(inputs, inputs_len, cell = None, cell_fn = tf.contrib.rnn.GRUCell, units = None, layers = 1, scope = "Bidirectional_GRU", output = 0, is_training = True, reuse = None):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    '''
    with tf.variable_scope(scope, reuse = reuse):
        batch_size = tf.shape(inputs)[0]
        if cell is not None:
            (cell_fw, cell_bw) = cell

        else:
            # if input is character level encoded , input [ batch size , passage length , character length , dim ]
            shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
                inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))

            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
                cell_bw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
            else:
                cell_fw, cell_bw = [apply_dropout(cell_fn(units), size = inputs.shape[-1], is_training = is_training) for _ in range(2)]

		# if input is character level encoded ,then states [ batch size * passage length , dim  ]



        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                        sequence_length = inputs_len,
                                                        dtype=tf.float32)

        if output == 0:
            return tf.concat(outputs, 2)
        elif output == 1:
            return tf.reshape(tf.concat(states,1),(batch_size, shapes[1], 2*units))


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
    if layer_num > 1:
        cell = tc.rnn.MultiRNNCell([cell]*layer_num, state_is_tuple=True)
    return cell