'''
Tensorflow implementation of RCF

@references:
'''
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import LoadData_ML as DATA
from Utilis import get_relational_data


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MF for ML.")

    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--layers', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.5,0.5]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--attention_size', type=int, default=32,
                        help='dimension of attention_size (default: 10)')
    parser.add_argument('--alpha', type=int, default=0.5,
                        help='smoothing factor of softmax')
    parser.add_argument('--reg_t', type=float, default=0.01,
                        help='regulation for translation relational data')
    return parser.parse_args()


class MF(BaseEstimator, TransformerMixin):
    def __init__(self, num_users, num_items, pretrain_flag, hidden_factor, epoch, batch_size, learning_rate,
                 lamda_bilinear, optimizer_type, verbose, layers, activation_function, keep_prob, save_file,
                 random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.num_users = num_users
        self.num_items = num_items
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.verbose = verbose
        self.layers = layers
        self.activation_function = activation_function
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in xrange(len(keep_prob))])
        self.attention_size = args.attention_size
        self.reg_t = args.reg_t
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user = tf.placeholder(tf.int32, shape=[None])  # None
            self.item_pos = tf.placeholder(tf.int32, shape=[None])  # None * 1
            self.item_neg = tf.placeholder(tf.int32, shape=[None])
            self.alpha = tf.placeholder(tf.float32, shape=[None])

            # (positive part)
            self.r0_p = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt0_p = tf.placeholder(tf.float32, shape=[None])
            self.len0_p = tf.placeholder(tf.int32)

            self.r1_p = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt1_p = tf.placeholder(tf.float32, shape=[None])
            self.e1_p = tf.placeholder(tf.int32, shape=[None, None])
            self.len1_p = tf.placeholder(tf.int32)

            self.r2_p = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt2_p = tf.placeholder(tf.float32, shape=[None])
            self.e2_p = tf.placeholder(tf.int32, shape=[None, None])
            self.len2_p = tf.placeholder(tf.int32)

            self.r3_p = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt3_p = tf.placeholder(tf.float32, shape=[None])
            self.e3_p = tf.placeholder(tf.int32, shape=[None, None])
            self.len3_p = tf.placeholder(tf.int32)

            # negative part
            self.r0_n = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt0_n = tf.placeholder(tf.float32, shape=[None])
            self.len0_n = tf.placeholder(tf.int32)

            self.r1_n = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt1_n = tf.placeholder(tf.float32, shape=[None])
            self.e1_n = tf.placeholder(tf.int32, shape=[None, None])
            self.len1_n = tf.placeholder(tf.int32)

            self.r2_n = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt2_n = tf.placeholder(tf.float32, shape=[None])
            self.e2_n = tf.placeholder(tf.int32, shape=[None, None])
            self.len2_n = tf.placeholder(tf.int32)

            self.r3_n = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt3_n = tf.placeholder(tf.float32, shape=[None])
            self.e3_n = tf.placeholder(tf.int32, shape=[None, None])
            self.len3_n = tf.placeholder(tf.int32)

            # relational part
            self.head1 = tf.placeholder(tf.int32, shape=[None])
            self.relation1 = tf.placeholder(tf.int32, shape=[None])
            self.tail1_pos = tf.placeholder(tf.int32, shape=[None])
            self.tail1_neg = tf.placeholder(tf.int32, shape=[None])

            self.head2 = tf.placeholder(tf.int32, shape=[None])
            self.relation2 = tf.placeholder(tf.int32, shape=[None])
            self.tail2_pos = tf.placeholder(tf.int32, shape=[None])
            self.tail2_neg = tf.placeholder(tf.int32, shape=[None])

            self.head3 = tf.placeholder(tf.int32, shape=[None])
            self.relation3 = tf.placeholder(tf.int32, shape=[None])
            self.tail3_pos = tf.placeholder(tf.int32, shape=[None])
            self.tail3_neg = tf.placeholder(tf.int32, shape=[None])

            # Variables.
            self.weights = self._initialize_weights()
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)
            # Model.
            self.user_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user),
                                                 axis=1)  # [B,1,H]
            self.pos_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item_pos)  # [B,H]
            self.neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item_neg)  # [B,H]
            self.r0_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['relation_type_embeddings'], 0),
                                               axis=0)  # [1,H]
            self.r1_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['relation_type_embeddings'], 1),
                                               axis=0)  # [1,H]
            self.r2_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['relation_type_embeddings'], 2),
                                               axis=0)  # [1,H]
            self.r3_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['relation_type_embeddings'], 3),
                                               axis=0)  # [1,H]
            # user's attention for relation type
            self.relation_type_embedding = tf.expand_dims(self.weights['relation_type_embeddings'], axis=0)  # [1,4,H]
            self.type_product = tf.multiply(self.user_embedding, self.relation_type_embedding)  # [B,4,H]
            self.type_w = tf.layers.dense(self.type_product, self.attention_size, name='first_layer_attention_type',
                                          reuse=tf.AUTO_REUSE, use_bias=True, activation=tf.nn.relu)  # [B,4,A]
            self.logits_type = tf.reduce_sum(
                tf.layers.dense(self.type_w, 1, name='final_layer_attention_type', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,4]
            self.logits_exp_type = tf.exp(self.logits_type)  # [B,4]
            self.exp_sum_type = tf.reduce_sum(self.logits_exp_type, axis=-1, keep_dims=True)  # [B,1]
            self.attention_type = self.logits_exp_type / self.exp_sum_type  # [B,4]
            # positive part
            # r0
            self.r0_p_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r0_p)  # [B,?,H]
            self.att0_i_p = tf.layers.dense(self.pos_embedding, self.attention_size, name='w0_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att0_i_p = tf.expand_dims(self.att0_i_p, 1)  # [B,1,A]
            self.att0_j_p = tf.layers.dense(self.r0_p_embedding, self.attention_size, name='w0_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            # self.att0_e_p =tf.layers.dense(self.r0_embedding,self.attention_size, name='w0_e', reuse=tf.AUTO_REUSE,use_bias=False) #[1,A]
            # self.att0_e_p =tf.expand_dims(self.att0_e_p,axis=0)                        #[1,1,A]
            self.att0_sum_p = tf.add(self.att0_i_p, self.att0_j_p)
            # self.att0_sum_p = tf.add(self.att0_sum_p, self.att0_e_p)                    #[B,?,A]
            self.att0_sum_p = tf.tanh(self.att0_sum_p)  # [B,?,A]
            self.att0_losits_p = tf.reduce_sum(
                tf.layers.dense(self.att0_sum_p, 1, name='att0_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att0_losits_p = self.Mask(self.att0_losits_p, self.cnt0_p + 1, self.len0_p,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_0_p = tf.exp(self.att0_losits_p)
            self.sum_0_p = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_0_p, axis=-1), self.alpha), axis=-1)  # [B,1]
            self.att_0_p = tf.expand_dims(self.exp_0_p / self.sum_0_p, axis=1)  # [B,1,?]
            self.item_latent_0_p = tf.matmul(self.att_0_p, self.r0_p_embedding)  # [B,1,H]
            # r1
            self.r1_p_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r1_p)  # [B,?,H]
            self.att1_i_p = tf.layers.dense(self.pos_embedding, self.attention_size, name='w1_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att1_i_p = tf.expand_dims(self.att1_i_p, 1)  # [B,1,A]
            self.att1_j_p = tf.layers.dense(self.r1_p_embedding, self.attention_size, name='w1_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e1_p_embedding = tf.nn.embedding_lookup(self.weights['genre_embeddings'], self.e1_p)  # [B,?,H]
            # self.e1_p_embedding=tf.add(self.e1_p_embedding, tf.expand_dims(self.r1_embedding,axis=0))  #[B,?,H]
            self.att1_e_p = tf.layers.dense(self.e1_p_embedding, self.attention_size, name='w1_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att1_sum_p = tf.add(self.att1_i_p, self.att1_j_p)
            self.att1_sum_p = tf.add(self.att1_sum_p, self.att1_e_p)  # [B,?,A]
            self.att1_sum_p = tf.tanh(self.att1_sum_p)  # [B,?,A]
            self.att1_losits_p = tf.reduce_sum(
                tf.layers.dense(self.att1_sum_p, 1, name='att1_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att1_losits_p = self.Mask(self.att1_losits_p, self.cnt1_p + 1, self.len1_p,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_1_p = tf.exp(self.att1_losits_p)
            self.sum_1_p = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_1_p, axis=-1), self.alpha), axis=-1)  # [B,1]
            self.att_1_p = tf.expand_dims(self.exp_1_p / self.sum_1_p, axis=1)  # [B,1,?]
            self.item_latent_1_p = tf.matmul(self.att_1_p, self.r1_p_embedding)  # [B,1,H]
            # r2
            self.r2_p_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r2_p)  # [B,?,H]
            self.att2_i_p = tf.layers.dense(self.pos_embedding, self.attention_size, name='w2_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att2_i_p = tf.expand_dims(self.att2_i_p, 1)  # [B,1,A]
            self.att2_j_p = tf.layers.dense(self.r2_p_embedding, self.attention_size, name='w2_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e2_p_embedding = tf.nn.embedding_lookup(self.weights['director_embeddings'], self.e2_p)  # [B,?,H]
            # self.e2_p_embedding = tf.add(self.e2_p_embedding, tf.expand_dims(self.r2_embedding, axis=0))  # [B,?,H]
            self.att2_e_p = tf.layers.dense(self.e2_p_embedding, self.attention_size, name='w2_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att2_sum_p = tf.add(self.att2_i_p, self.att2_j_p)
            self.att2_sum_p = tf.add(self.att2_sum_p, self.att2_e_p)  # [B,?,A]
            self.att2_sum_p = tf.tanh(self.att2_sum_p)  # [B,?,A]
            self.att2_losits_p = tf.reduce_sum(
                tf.layers.dense(self.att2_sum_p, 1, name='att2_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att2_losits_p = self.Mask(self.att2_losits_p, self.cnt2_p + 1, self.len2_p,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_2_p = tf.exp(self.att2_losits_p)
            self.sum_2_p = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_2_p, axis=-1), self.alpha), axis=-1)  # [B,1]
            self.att_2_p = tf.expand_dims(self.exp_2_p / self.sum_2_p, axis=1)  # [B,1,?]
            self.item_latent_2_p = tf.matmul(self.att_2_p, self.r2_p_embedding)  # [B,1,H]
            # r3
            self.r3_p_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r3_p)  # [B,?,H]
            self.att3_i_p = tf.layers.dense(self.pos_embedding, self.attention_size, name='w3_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att3_i_p = tf.expand_dims(self.att3_i_p, 1)  # [B,1,A]
            self.att3_j_p = tf.layers.dense(self.r3_p_embedding, self.attention_size, name='w3_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e3_p_embedding = tf.nn.embedding_lookup(self.weights['actor_embeddings'], self.e3_p)  # [B,?,H]
            # self.e3_p_embedding = tf.add(self.e3_p_embedding, tf.expand_dims(self.r3_embedding, axis=0))  # [B,?,H]
            self.att3_e_p = tf.layers.dense(self.e3_p_embedding, self.attention_size, name='w3_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att3_sum_p = tf.add(self.att3_i_p, self.att3_j_p)
            self.att3_sum_p = tf.add(self.att3_sum_p, self.att3_e_p)  # [B,?,A]
            self.att3_sum_p = tf.tanh(self.att3_sum_p)  # [B,?,A]
            self.att3_losits_p = tf.reduce_sum(
                tf.layers.dense(self.att3_sum_p, 1, name='att3_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att3_losits_p = self.Mask(self.att3_losits_p, self.cnt3_p + 1, self.len3_p,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_3_p = tf.exp(self.att3_losits_p)
            self.sum_3_p = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_3_p, axis=-1), self.alpha), axis=-1)  # [B,1]
            self.att_3_p = tf.expand_dims(self.exp_3_p / self.sum_3_p, axis=1)  # [B,1,?]
            self.item_latent_3_p = tf.matmul(self.att_3_p, self.r3_p_embedding)  # [B,1,H]
            # merge all item latent in different relations
            # self.item_latent_p = tf.reduce_sum(self.item_latent_0_p, axis=1)
            self.item_latent_p = tf.concat(
                [self.item_latent_0_p, self.item_latent_1_p, self.item_latent_2_p, self.item_latent_3_p],
                axis=1)  # [B,4,H]
            self.item_latent_p = tf.reduce_sum(
                tf.matmul(tf.expand_dims(self.attention_type, axis=1), self.item_latent_p), axis=1)  # [B,H]
            self.mu_p = tf.add(tf.reduce_sum(self.user_embedding, axis=1), self.item_latent_p)
            # self.pos = tf.reduce_sum(tf.multiply(self.mu_p, self.pos_embedding), 1)
            self.pos = tf.multiply(self.mu_p, self.pos_embedding)
            self.pos = tf.nn.dropout(self.pos, self.dropout_keep[-1])
            for i in range(0, len(self.layers)):
                self.pos = tf.add(tf.matmul(self.pos, self.weights['layer_%d' % i]),
                                  self.weights['bias_%d' % i])  # None * layer[i] * 1
                self.pos = self.activation_function(self.pos)
                self.pos = tf.nn.dropout(self.pos, self.dropout_keep[i])  # dropout at each Deep layer
            self.pos = tf.matmul(self.pos, self.weights['prediction'])  # None * 1

            #################### negative parts ####################
            # r0
            self.r0_n_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r0_n)  # [B,?,H]
            self.att0_i_n = tf.layers.dense(self.neg_embedding, self.attention_size, name='w0_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att0_i_n = tf.expand_dims(self.att0_i_n, 1)  # [B,1,A]
            self.att0_j_n = tf.layers.dense(self.r0_n_embedding, self.attention_size, name='w0_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            # self.att0_e_n =tf.layers.dense(self.r0_embedding,self.attention_size, name='w0_e', reuse=tf.AUTO_REUSE,use_bias=False) #[1,A]
            # self.att0_e_n =tf.expand_dims(self.att0_e_n,axis=0)                        #[1,1,A]
            self.att0_sum_n = tf.add(self.att0_i_n, self.att0_j_n)
            # self.att0_sum_n = tf.add(self.att0_sum_n, self.att0_e_n)                    #[B,?,A]
            self.att0_sum_n = tf.tanh(self.att0_sum_n)  # [B,?,A]
            self.att0_losits_n = tf.reduce_sum(
                tf.layers.dense(self.att0_sum_n, 1, name='att0_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att0_losits_n = self.Mask(self.att0_losits_n, self.cnt0_n + 1, self.len0_n,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_0_n = tf.exp(self.att0_losits_n)
            self.sum_0_n = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_0_n, axis=-1), self.alpha), axis=-1)  # [B,1]
            self.att_0_n = tf.expand_dims(self.exp_0_n / self.sum_0_n, axis=1)  # [B,1,?]
            self.item_latent_0_n = tf.matmul(self.att_0_n, self.r0_n_embedding)  # [B,1,H]
            # r1
            self.r1_n_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r1_n)  # [B,?,H]
            self.att1_i_n = tf.layers.dense(self.neg_embedding, self.attention_size, name='w1_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att1_i_n = tf.expand_dims(self.att1_i_n, 1)  # [B,1,A]
            self.att1_j_n = tf.layers.dense(self.r1_n_embedding, self.attention_size, name='w1_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e1_n_embedding = tf.nn.embedding_lookup(self.weights['genre_embeddings'], self.e1_n)  # [B,?,H]
            # self.e1_n_embedding=tf.add(self.e1_n_embedding, tf.expand_dims(self.r1_embedding,axis=0))  #[B,?,H]
            self.att1_e_n = tf.layers.dense(self.e1_n_embedding, self.attention_size, name='w1_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att1_sum_n = tf.add(self.att1_i_n, self.att1_j_n)
            self.att1_sum_n = tf.add(self.att1_sum_n, self.att1_e_n)  # [B,?,A]
            self.att1_sum_n = tf.tanh(self.att1_sum_n)  # [B,?,A]
            self.att1_losits_n = tf.reduce_sum(
                tf.layers.dense(self.att1_sum_n, 1, name='att1_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att1_losits_n = self.Mask(self.att1_losits_n, self.cnt1_n + 1, self.len1_n,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_1_n = tf.exp(self.att1_losits_n)
            self.sum_1_n = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_1_n, axis=-1), self.alpha), axis=-1)  # [B,1]
            self.att_1_n = tf.expand_dims(self.exp_1_n / self.sum_1_n, axis=1)  # [B,1,?]
            self.item_latent_1_n = tf.matmul(self.att_1_n, self.r1_n_embedding)  # [B,1,H]
            # r2
            self.r2_n_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r2_n)  # [B,?,H]
            self.att2_i_n = tf.layers.dense(self.neg_embedding, self.attention_size, name='w2_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att2_i_n = tf.expand_dims(self.att2_i_n, 1)  # [B,1,A]
            self.att2_j_n = tf.layers.dense(self.r2_n_embedding, self.attention_size, name='w2_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e2_n_embedding = tf.nn.embedding_lookup(self.weights['director_embeddings'], self.e2_n)  # [B,?,H]
            # self.e2_n_embedding = tf.add(self.e2_n_embedding, tf.expand_dims(self.r2_embedding, axis=0))  # [B,?,H]
            self.att2_e_n = tf.layers.dense(self.e2_n_embedding, self.attention_size, name='w2_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att2_sum_n = tf.add(self.att2_i_n, self.att2_j_n)
            self.att2_sum_n = tf.add(self.att2_sum_n, self.att2_e_n)  # [B,?,A]
            self.att2_sum_n = tf.tanh(self.att2_sum_n)  # [B,?,A]
            self.att2_losits_n = tf.reduce_sum(
                tf.layers.dense(self.att2_sum_n, 1, name='att2_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att2_losits_n = self.Mask(self.att2_losits_n, self.cnt2_n + 1, self.len2_n,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_2_n = tf.exp(self.att2_losits_n)
            self.sum_2_n = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_2_n, axis=-1), self.alpha), axis=-1)  # [B,1]
            self.att_2_n = tf.expand_dims(self.exp_2_n / self.sum_2_n, axis=1)  # [B,1,?]
            self.item_latent_2_n = tf.matmul(self.att_2_n, self.r2_n_embedding)  # [B,1,H]
            # r3
            self.r3_n_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r3_n)  # [B,?,H]
            self.att3_i_n = tf.layers.dense(self.neg_embedding, self.attention_size, name='w3_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att3_i_n = tf.expand_dims(self.att3_i_n, 1)  # [B,1,A]
            self.att3_j_n = tf.layers.dense(self.r3_n_embedding, self.attention_size, name='w3_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e3_n_embedding = tf.nn.embedding_lookup(self.weights['actor_embeddings'], self.e3_n)  # [B,?,H]
            # self.e3_n_embedding = tf.add(self.e3_n_embedding, tf.expand_dims(self.r3_embedding, axis=0))  # [B,?,H]
            self.att3_e_n = tf.layers.dense(self.e3_n_embedding, self.attention_size, name='w3_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att3_sum_n = tf.add(self.att3_i_n, self.att3_j_n)
            self.att3_sum_n = tf.add(self.att3_sum_n, self.att3_e_n)  # [B,?,A]
            self.att3_sum_n = tf.tanh(self.att3_sum_n)  # [B,?,A]
            self.att3_losits_n = tf.reduce_sum(
                tf.layers.dense(self.att3_sum_n, 1, name='att3_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att3_losits_n = self.Mask(self.att3_losits_n, self.cnt3_n + 1, self.len3_n,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_3_n = tf.exp(self.att3_losits_n)
            self.sum_3_n = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_3_n, axis=-1), self.alpha), axis=-1)  # [B,1]
            self.att_3_n = tf.expand_dims(self.exp_3_n / self.sum_3_n, axis=1)  # [B,1,?]
            self.item_latent_3_n = tf.matmul(self.att_3_n, self.r3_n_embedding)  # [B,1,H]
            # merge all item latent in different relations
            # self.item_latent_n = tf.reduce_sum(self.item_latent_0_n,axis=1)  # [B,4,H]
            self.item_latent_n = tf.concat(
                [self.item_latent_0_n, self.item_latent_1_n, self.item_latent_2_n, self.item_latent_3_n],
                axis=1)  # [B,4,H]
            self.item_latent_n = tf.reduce_sum(
                tf.matmul(tf.expand_dims(self.attention_type, axis=1), self.item_latent_n), axis=1)  # [B,H]
            self.mu_n = tf.add(tf.reduce_sum(self.user_embedding, axis=1), self.item_latent_n)
            # self.neg = tf.reduce_sum(tf.multiply(self.mu_n, self.neg_embedding), 1)
            self.neg = tf.multiply(self.mu_n, self.neg_embedding)
            self.neg = tf.nn.dropout(self.neg, self.dropout_keep[-1])
            for i in range(0, len(self.layers)):
                self.neg = tf.add(tf.matmul(self.neg, self.weights['layer_%d' % i]),
                                  self.weights['bias_%d' % i])  # None * layer[i] * 1
                self.neg = self.activation_function(self.neg)
                self.neg = tf.nn.dropout(self.neg, self.dropout_keep[i])  # dropout at each Deep layer
            self.neg = tf.matmul(self.neg, self.weights['prediction'])  # None * 1
            # Compute the loss.
            self.loss = -tf.log(tf.sigmoid(self.pos - self.neg))  # [B,1]
            self.loss = tf.reduce_sum(self.loss)

            # regularization on translation data
            self.head1_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.head1)  # [B,H]
            self.translation1_embedding = tf.add(self.r1_embedding,
                                                 tf.nn.embedding_lookup(self.weights['genre_embeddings'],
                                                                        self.relation1))  # [B,H]
            self.tail1_pos_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail1_pos)  # [B,H]
            self.score1_pos = tf.multiply(self.head1_embedding, self.translation1_embedding)  # [B,H]
            self.score1_pos = tf.reduce_sum(tf.multiply(self.score1_pos, self.tail1_pos_embedding), axis=-1)  # [B]

            self.tail1_neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail1_neg)  # [B,H]
            self.score1_neg = tf.multiply(self.head1_embedding, self.translation1_embedding)
            self.score1_neg = tf.reduce_sum(tf.multiply(self.score1_neg, self.tail1_neg_embedding), axis=-1)

            self.rel_loss_1 = tf.reduce_sum(-tf.log(tf.sigmoid(self.score1_pos - self.score1_neg)))
            #########################################################################
            self.head2_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.head2)  # [B,H]
            self.translation2_embedding = tf.add(self.r2_embedding,
                                                 tf.nn.embedding_lookup(self.weights['director_embeddings'],
                                                                        self.relation2))  # [B,H]
            self.tail2_pos_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail2_pos)  # [B,H]
            self.score2_pos = tf.multiply(self.head2_embedding, self.translation2_embedding)  # [B,H]
            self.score2_pos = tf.reduce_sum(tf.multiply(self.score2_pos, self.tail2_pos_embedding), axis=-1)  # [B]

            self.tail2_neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail2_neg)  # [B,H]
            self.score2_neg = tf.multiply(self.head2_embedding, self.translation2_embedding)
            self.score2_neg = tf.reduce_sum(tf.multiply(self.score2_neg, self.tail2_neg_embedding), axis=-1)

            self.rel_loss_2 = tf.reduce_sum(-tf.log(tf.sigmoid(self.score2_pos - self.score2_neg)))
            #########################################################################
            self.head3_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.head3)  # [B,H]
            self.translation3_embedding = tf.add(self.r3_embedding,
                                                 tf.nn.embedding_lookup(self.weights['actor_embeddings'],
                                                                        self.relation3))  # [B,H]
            self.tail3_pos_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail3_pos)  # [B,H]
            self.score3_pos = tf.multiply(self.head3_embedding, self.translation3_embedding)  # [B,H]
            self.score3_pos = tf.reduce_sum(tf.multiply(self.score3_pos, self.tail3_pos_embedding), axis=-1)  # [B]

            self.tail3_neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail3_neg)  # [B,H]
            self.score3_neg = tf.multiply(self.head3_embedding, self.translation3_embedding)
            self.score3_neg = tf.reduce_sum(tf.multiply(self.score3_neg, self.tail3_neg_embedding), axis=-1)

            self.rel_loss_3 = tf.reduce_sum(-tf.log(tf.sigmoid(self.score3_pos - self.score3_neg)))
            self.rel_loss = self.reg_t * (self.rel_loss_1 + self.rel_loss_2 + self.rel_loss_3)

            self.loss = tf.add(self.loss, self.rel_loss)
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_embedding = pretrain_graph.get_tensor_by_name('user_embeddings:0')
            item_embedding = pretrain_graph.get_tensor_by_name('item_embeddings:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie = sess.run([user_embedding, item_embedding])
            all_weights['user_embeddings'] = tf.Variable(ue, dtype=tf.float32)
            all_weights['item_embeddings'] = tf.Variable(ie, dtype=tf.float32)
        else:
            all_weights['user_embeddings'] = tf.Variable(
                tf.random_normal([self.num_users, self.hidden_factor], 0.0, 0.05),
                name='user_embeddings')  # user_num * H
            all_weights['relation_type_embeddings'] = tf.Variable(
                tf.random_normal([4, self.hidden_factor], 0.0, 0.05), name='relation_type_embeddings')
            ie = tf.Variable(tf.random_normal([self.num_items, self.hidden_factor], 0.0, 0.05), name='item_embeddings')
            ge = tf.Variable(tf.random_normal([data.num_genres, self.hidden_factor], 0.0, 0.05),
                             name='genre_embeddings')
            de = tf.Variable(tf.random_normal([data.num_directors, self.hidden_factor], 0.0, 0.05),
                             name='director_embeddings')
            ae = tf.Variable(tf.random_normal([data.num_actors, self.hidden_factor], 0.0, 0.05),
                             name='actor_embeddings')
            mask_embedding = tf.Variable(tf.constant(0.0, shape=[1, self.hidden_factor], dtype=tf.float32),
                                         name='mask_embedding', trainable=False)
            all_weights['item_embeddings'] = tf.concat([ie, mask_embedding], axis=0)
            all_weights['genre_embeddings'] = tf.concat([ge, mask_embedding], axis=0)
            all_weights['director_embeddings'] = tf.concat([de, mask_embedding], axis=0)
            all_weights['actor_embeddings'] = tf.concat([ae, mask_embedding], axis=0)
        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32,
                name='layer_0')
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])),
                                                dtype=np.float32, name='bias_0')  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32,
                    name='layer_%d' % i)  # layers[i-1]*layers[i]
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32,
                    name='bias_%d' % i)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                                                    dtype=np.float32, name='prediction')  # layers[-1] * 1
        return all_weights

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user: data['user'], self.item_pos: data['positive'], self.item_neg: data['negative'],
                     self.alpha: data['alpha'],
                     self.r0_p: data['r0_p'], self.r1_p: data['r1_p'], self.r2_p: data['r2_p'], self.r3_p: data['r3_p'],
                     self.cnt0_p: data['cnt0_p'], self.cnt1_p: data['cnt1_p'], self.cnt2_p: data['cnt2_p'],
                     self.cnt3_p: data['cnt3_p'],
                     self.e1_p: data['e1_p'], self.e2_p: data['e2_p'], self.e3_p: data['e3_p'],
                     self.len0_p: data['len0_p'], self.len1_p: data['len1_p'], self.len2_p: data['len2_p'],
                     self.len3_p: data['len3_p'],
                     self.r0_n: data['r0_n'], self.r1_n: data['r1_n'], self.r2_n: data['r2_n'], self.r3_n: data['r3_n'],
                     self.cnt0_n: data['cnt0_n'], self.cnt1_n: data['cnt1_n'], self.cnt2_n: data['cnt2_n'],
                     self.cnt3_n: data['cnt3_n'],
                     self.e1_n: data['e1_n'], self.e2_n: data['e2_n'], self.e3_n: data['e3_n'],
                     self.len0_n: data['len0_n'], self.len1_n: data['len1_n'], self.len2_n: data['len2_n'],
                     self.len3_n: data['len3_n'],
                     self.head1: data['head1'], self.relation1: data['relation1'], self.tail1_pos: data['tail1_pos'],
                     self.tail1_neg: data['tail1_neg'],
                     self.head2: data['head2'], self.relation2: data['relation2'], self.tail2_pos: data['tail2_pos'],
                     self.tail2_neg: data['tail2_neg'],
                     self.head3: data['head3'], self.relation3: data['relation3'], self.tail3_pos: data['tail3_pos'],
                     self.tail3_neg: data['tail3_neg'],
                     self.dropout_keep: self.keep_prob, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, train_data, batch_size):  # generate a random block of training data
        user, positive, negative, alpha = [], [], [], []
        r0_p, r1_p, r2_p, r3_p = [], [], [], []  # for positive item i, the item set in ru+ which has relationship r0,r1,r2,r3 with i
        cnt0_p, cnt1_p, cnt2_p, cnt3_p = [], [], [], []  # the number of corresponding r, for masking
        e1_p, e2_p, e3_p = [], [], []  # the set of specific attribute value for correspoding r except r0
        # for negative part
        r0_n, r1_n, r2_n, r3_n = [], [], [], []
        cnt0_n, cnt1_n, cnt2_n, cnt3_n = [], [], [], []
        e1_n, e2_n, e3_n = [], [], []
        head1, relation1, tail1_pos, tail1_neg = [], [], [], []
        head2, relation2, tail2_pos, tail2_neg = [], [], [], []
        head3, relation3, tail3_pos, tail3_neg = [], [], [], []
        all_items = data.items.values()
        # get sample
        while len(user) < batch_size:
            index = np.random.randint(0, len(train_data['User']))
            user_id = train_data['User'][index]
            item_id = train_data['Item'][index]
            user.append(user_id)
            positive.append(item_id)
            # uniform sampler
            pos = data.user_positive_list[user_id]  # get positive list for the userID
            # candidates = list(set(all_items) - set(pos))  # get negative set
            neg = np.random.randint(len(all_items))  # uniform sample a negative itemID from negative set
            while (neg in pos):
                neg = np.random.randint(len(all_items))
            negative.append(neg)
            alpha.append(args.alpha)
            # t1 = time()
            r0_temp, r1_temp, r2_temp, r3_temp, e1_temp, e2_temp, e3_temp, cnt0_temp, cnt1_temp, cnt2_temp, cnt3_temp = get_relational_data(
                user_id, item_id, data)
            # t2= time()
            # print ('the time of generating batch:%f' % (t2 - t1))
            r0_p.append(r0_temp)
            r1_p.append(r1_temp)
            r2_p.append(r2_temp)
            r3_p.append(r3_temp)
            e1_p.append(e1_temp)
            e2_p.append(e2_temp)
            e3_p.append(e3_temp)
            cnt0_p.append(cnt0_temp)
            cnt1_p.append(cnt1_temp)
            cnt2_p.append(cnt2_temp)
            cnt3_p.append(cnt3_temp)
            # construct relational data for relational multitask learning
            if cnt1_temp > 0:
                rel_index = np.random.randint(cnt1_temp)
                rel = e1_temp[rel_index]
                tal_pos = r1_temp[rel_index]
                tal_neg = np.random.randint(len(all_items))
                while (tal_neg == tal_pos or rel in data.movie_dict[data.items_traverse[tal_neg]].genre):
                    tal_neg = np.random.randint(len(all_items))
                head1.append(item_id)
                relation1.append(rel)
                tail1_pos.append(tal_pos)
                tail1_neg.append(tal_neg)
            if cnt2_temp > 0:
                rel_index = np.random.randint(cnt2_temp)
                rel = e2_temp[rel_index]
                tal_pos = r2_temp[rel_index]
                tal_neg = np.random.randint(len(all_items))
                while (tal_neg == tal_pos):
                    tal_neg = np.random.randint(len(all_items))
                head2.append(item_id)
                relation2.append(rel)
                tail2_pos.append(tal_pos)
                tail2_neg.append(tal_neg)
            if cnt3_temp > 0:
                rel_index = np.random.randint(cnt3_temp)
                rel = e3_temp[rel_index]
                tal_pos = r3_temp[rel_index]
                tal_neg = np.random.randint(len(all_items))
                while (tal_neg == tal_pos):
                    tal_neg = np.random.randint(len(all_items))
                head3.append(item_id)
                relation3.append(rel)
                tail3_pos.append(tal_pos)
                tail3_neg.append(tal_neg)
            # for negative part
            r0_temp, r1_temp, r2_temp, r3_temp, e1_temp, e2_temp, e3_temp, cnt0_temp, cnt1_temp, cnt2_temp, cnt3_temp = get_relational_data(
                user_id, neg, data)
            r0_n.append(r0_temp)
            r1_n.append(r1_temp)
            r2_n.append(r2_temp)
            r3_n.append(r3_temp)
            e1_n.append(e1_temp)
            e2_n.append(e2_temp)
            e3_n.append(e3_temp)
            cnt0_n.append(cnt0_temp)
            cnt1_n.append(cnt1_temp)
            cnt2_n.append(cnt2_temp)
            cnt3_n.append(cnt3_temp)
        # fill out a fixed length for each batch
        len0_p = max(cnt0_p)
        len1_p = max(cnt1_p)
        len2_p = max(cnt2_p)
        len3_p = max(cnt3_p)
        for index in xrange(len(r0_p)):
            if len(r0_p[index]) < len0_p:
                r0_p[index].extend(np.array([self.num_items]).repeat(len0_p - len(r0_p[index])))
            if len(r1_p[index]) < len1_p:
                r1_p[index].extend(np.array([self.num_items]).repeat(len1_p - len(r1_p[index])))
                e1_p[index].extend(np.array([data.num_genres]).repeat(len1_p - len(e1_p[index])))
            if len(r2_p[index]) < len2_p:
                r2_p[index].extend(np.array([self.num_items]).repeat(len2_p - len(r2_p[index])))
                e2_p[index].extend(np.array([data.num_directors]).repeat(len2_p - len(e2_p[index])))
            if len(r3_p[index]) < len3_p:
                r3_p[index].extend(np.array([self.num_items]).repeat(len3_p - len(r3_p[index])))
                e3_p[index].extend(np.array([data.num_actors]).repeat(len3_p - len(e3_p[index])))
        len0_n = max(cnt0_n)
        len1_n = max(cnt1_n)
        len2_n = max(cnt2_n)
        len3_n = max(cnt3_n)
        for index in xrange(len(r0_n)):
            if len(r0_n[index]) < len0_n:
                r0_n[index].extend(np.array([self.num_items]).repeat(len0_n - len(r0_n[index])))
            if len(r1_n[index]) < len1_n:
                r1_n[index].extend(np.array([self.num_items]).repeat(len1_n - len(r1_n[index])))
                e1_n[index].extend(np.array([data.num_genres]).repeat(len1_n - len(e1_n[index])))
            if len(r2_n[index]) < len2_n:
                r2_n[index].extend(np.array([self.num_items]).repeat(len2_n - len(r2_n[index])))
                e2_n[index].extend(np.array([data.num_directors]).repeat(len2_n - len(e2_n[index])))
            if len(r3_n[index]) < len3_n:
                r3_n[index].extend(np.array([self.num_items]).repeat(len3_n - len(r3_n[index])))
                e3_n[index].extend(np.array([data.num_actors]).repeat(len3_n - len(e3_n[index])))
        return {'user': user, 'positive': positive, 'negative': negative, 'alpha': alpha,
                'r0_p': r0_p, 'r1_p': r1_p, 'r2_p': r2_p, 'r3_p': r3_p,
                'e1_p': e1_p, 'e2_p': e2_p, 'e3_p': e3_p,
                'cnt0_p': cnt0_p, 'cnt1_p': cnt1_p, 'cnt2_p': cnt2_p, 'cnt3_p': cnt3_p,
                'len0_p': len0_p, 'len1_p': len1_p, 'len2_p': len2_p, 'len3_p': len3_p,
                'r0_n': r0_n, 'r1_n': r1_n, 'r2_n': r2_n, 'r3_n': r3_n,
                'e1_n': e1_n, 'e2_n': e2_n, 'e3_n': e3_n,
                'cnt0_n': cnt0_n, 'cnt1_n': cnt1_n, 'cnt2_n': cnt2_n, 'cnt3_n': cnt3_n,
                'len0_n': len0_n, 'len1_n': len1_n, 'len2_n': len2_n, 'len3_n': len3_n,
                'head1': head1, 'relation1': relation1, 'tail1_pos': tail1_pos, 'tail1_neg': tail1_neg,
                'head2': head2, 'relation2': relation2, 'tail2_pos': tail2_pos, 'tail2_neg': tail2_neg,
                'head3': head3, 'relation3': relation3, 'tail3_pos': tail3_pos, 'tail3_neg': tail3_neg}

    def train(self, Train_data):  # fit a dataset
        for epoch in range(self.epoch):
            total_loss = 0
            total_batch = int(len(Train_data['User']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                # t1 = time()
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # t2 = time()
                # print ('the time of generating batch:%f'%(t2-t1))
                # Fit training
                loss = self.partial_fit(batch_xs)
                # t3=time()
                # print ('the time of optimizting:%f' % (t3 - t2))
                total_loss = total_loss + loss
            attention_type = self.get_attention_type_scalar()
            avearge = np.mean(attention_type, axis=0)
            print("the total loss in %d th iteration is: %f, the attentions are %.4f, %.4f, %.4f, %.4f" % (
            epoch, total_loss, avearge[0], avearge[1], avearge[2], avearge[3]))
            # model.evaluate()

    def Mask(self, inputs, seq_len, long, mode):
        if seq_len == None:
            return inputs
        else:
            mask = tf.cast(tf.sequence_mask(seq_len, long), tf.float32)
            # for _ in range(len(inputs.shape) - 2):
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def evaluate(self):
        self.graph.finalize()
        count = [0, 0, 0, 0, 0]
        rank = [[], [], [], [], []]
        hit_user_list = []
        hit_item_list = []
        for index in range(len(data.Test_data['User'])):
            user_id = data.Test_data['User'][index]
            scores = model.get_scores_per_user(user_id)
            # get true item score
            true_item_id = data.Test_data['Item'][index]
            true_item_score = scores[true_item_id]
            # delete visited scores
            visited = data.user_positive_list[user_id]  # get positive list for the userID
            scores = np.delete(scores, visited)
            # whether hit
            sorted_scores = sorted(scores, reverse=True)
            label = [sorted_scores[4]]
            label.append([sorted_scores[9]])
            label.append([sorted_scores[14]])
            label.append([sorted_scores[19]])
            label.append([sorted_scores[24]])

            if true_item_score >= label[0]:
                count[0] = count[0] + 1
                rank[0].append(sorted_scores.index(true_item_score) + 1)
                hit_user_list.append(user_id)
                hit_item_list.append(true_item_id)
            if true_item_score >= label[1]:
                count[1] = count[1] + 1
                rank[1].append(sorted_scores.index(true_item_score) + 1)
            if true_item_score >= label[2]:
                count[2] = count[2] + 1
                rank[2].append(sorted_scores.index(true_item_score) + 1)
            if true_item_score >= label[3]:
                count[3] = count[3] + 1
                rank[3].append(sorted_scores.index(true_item_score) + 1)
            if true_item_score >= label[4]:
                count[4] = count[4] + 1
                rank[4].append(sorted_scores.index(true_item_score) + 1)
            print index
        for i in range(5):
            mrr = 0
            ndcg = 0
            hit_rate = float(count[i]) / len(data.Test_data['User'])
            for item in rank[i]:
                mrr = mrr + float(1.0) / item
                ndcg = ndcg + float(1.0) / np.log2(item + 1)
            mrr = mrr / len(data.Test_data['User'])
            ndcg = ndcg / len(data.Test_data['User'])
            k = (i + 1) * 5
            print("top:%d" % k)
            print("the Hit Rate is: %f" % hit_rate)
            print("the MRR is: %f" % mrr)
            print("the NDCG is: %f" % ndcg)

    def get_scores_per_user(self, user_id):  # evaluate the results for an user context, return scorelist
        scorelist = []
        all_items = data.items.values()
        if len(all_items) % self.batch_size == 0:
            batch_count = len(all_items) / self.batch_size
            flag = 0
        else:
            batch_count = math.ceil(len(all_items) / self.batch_size)
            flag = 1
        j = 0
        for i in range(int(batch_count)):
            user, positive, alpha = [], [], []
            r0, r1, r2, r3 = [], [], [], []  # for positive item i, the item set in ru+ which has relationship r0,r1,r2,r3 with i
            cnt0, cnt1, cnt2, cnt3 = [], [], [], []  # the number of corresponding r, for masking
            e1, e2, e3 = [], [], []  # the set of specific attribute value for correspoding r except r0
            if flag == 1 and i == batch_count - 1:
                k = len(all_items)
            else:
                k = j + self.batch_size
            for itemID in range(j, k):
                user.append(user_id)
                positive.append(itemID)
                alpha.append(args.alpha)
                r0_temp, r1_temp, r2_temp, r3_temp, e1_temp, e2_temp, e3_temp, cnt0_temp, cnt1_temp, cnt2_temp, cnt3_temp = get_relational_data(
                    user_id, itemID, data)
                r0.append(r0_temp)
                r1.append(r1_temp)
                r2.append(r2_temp)
                r3.append(r3_temp)
                e1.append(e1_temp)
                e2.append(e2_temp)
                e3.append(e3_temp)
                cnt0.append(cnt0_temp)
                cnt1.append(cnt1_temp)
                cnt2.append(cnt2_temp)
                cnt3.append(cnt3_temp)
            len0 = max(cnt0)
            len1 = max(cnt1)
            len2 = max(cnt2)
            len3 = max(cnt3)
            for index in xrange(len(r0)):
                if len(r0[index]) < len0:
                    r0[index].extend(np.array([self.num_items]).repeat(len0 - len(r0[index])))
                if len(r1[index]) < len1:
                    r1[index].extend(np.array([self.num_items]).repeat(len1 - len(r1[index])))
                    e1[index].extend(np.array([data.num_genres]).repeat(len1 - len(e1[index])))
                if len(r2[index]) < len2:
                    r2[index].extend(np.array([self.num_items]).repeat(len2 - len(r2[index])))
                    e2[index].extend(np.array([data.num_directors]).repeat(len2 - len(e2[index])))
                if len(r3[index]) < len3:
                    r3[index].extend(np.array([self.num_items]).repeat(len3 - len(r3[index])))
                    e3[index].extend(np.array([data.num_actors]).repeat(len3 - len(e3[index])))
            feed_dict = {self.user: user, self.item_pos: positive, self.alpha: alpha,
                         self.r0_p: r0, self.r1_p: r1, self.r2_p: r2, self.r3_p: r3,
                         self.cnt0_p: cnt0, self.cnt1_p: cnt1, self.cnt2_p: cnt2, self.cnt3_p: cnt3,
                         self.e1_p: e1, self.e2_p: e2, self.e3_p: e3,
                         self.len0_p: len0, self.len1_p: len1, self.len2_p: len2, self.len3_p: len3,
                         self.dropout_keep: self.no_dropout, self.train_phase: False}
            # print(X_item)
            scores = self.sess.run((self.pos), feed_dict=feed_dict)
            scores = scores.reshape(len(user))
            scorelist = np.append(scorelist, scores)
            j = j + self.batch_size
        return scorelist

    def get_attention_type_scalar(self):  # evaluate the results for an user context, return scorelist
        all_users = data.users.values()
        feed_dict = {self.user: all_users}
        results = self.sess.run((self.attention_type), feed_dict=feed_dict)
        return results


if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = DATA.LoadData()
    activation_function = tf.nn.relu
    if args.verbose > 0:
        print(
                "MF:  factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, optimizer=%s"
                % (
                    args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda,
                    args.optimizer))

    save_file = 'pretrain-mf/%s_%d' % ('ml1M', args.hidden_factor)
    # Training
    t1 = time()
    model = MF(data.num_users, data.num_items, args.pretrain, args.hidden_factor, args.epoch,
               args.batch_size, args.lr, args.lamda, args.optimizer, args.verbose, eval(args.layers),
               activation_function, eval(args.keep_prob), save_file)
    # model.evaluate()
    print("begin train")
    model.train(data.Train_data)
    print("end train")
    model.evaluate()
    print("finish")
