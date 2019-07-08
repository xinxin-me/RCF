'''
Tensorflow implementation of NAIS

@references:
'''
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import LoadData_ML as DATA


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MF for ML.")

    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.0001,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--layers', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.8]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--attention_size', type=int, default=32,
                        help='dimension of attention_size (default: 10)')
    parser.add_argument('--alpha', type=int, default=0.5,
                        help='smoothing factor of softmax')
    return parser.parse_args()


class MF(BaseEstimator, TransformerMixin):
    def __init__(self, num_users, num_items, pretrain_flag, hidden_factor, epoch, batch_size, learning_rate,
                 lamda_bilinear, optimizer_type, verbose, layers,activation_function,keep_prob,save_file,random_seed=2016):
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
        self.layers=layers
        self.activation_function = activation_function
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in xrange(len(keep_prob))])
        self.attention_size = args.attention_size

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
            #self.user = tf.placeholder(tf.int32, shape=[None])  # None
            self.item_pos = tf.placeholder(tf.int32, shape=[None])  # None * 1
            self.item_neg = tf.placeholder(tf.int32, shape=[None])
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)
            self.ru = tf.placeholder(tf.int32, shape=[None, None])
            #self.cnt = tf.placeholder(tf.float32, shape=[None])
            self.alpha = tf.placeholder(tf.float32, shape=[None])
            # Variables.
            self.weights = self._initialize_weights()
            # Model.
            # _________ positive part _____________
            #self.user_embedding = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user)
            self.pos_embedding=tf.expand_dims(tf.nn.embedding_lookup(self.weights['item_embeddings_p'], self.item_pos),axis=1)
            self.ru_embedding=tf.nn.embedding_lookup(self.weights['item_embeddings_q'],self.ru)
            self.product_pos=tf.multiply(self.pos_embedding,self.ru_embedding)
            self.w_pos = tf.layers.dense(self.product_pos, self.attention_size, name='first_layer_attention', reuse=tf.AUTO_REUSE, use_bias=True,activation=tf.nn.relu)
            self.logits_pos = tf.reduce_sum(tf.layers.dense(self.w_pos, 1, name='final_layer_attention', reuse=tf.AUTO_REUSE, use_bias=False), axis=-1)
            self.logits_pos_exp = tf.exp(self.logits_pos)
            self.pos_exp_sum = tf.expand_dims(tf.pow(tf.reduce_sum(self.logits_pos_exp,axis=-1), self.alpha), axis=-1)
            self.attention_pos=tf.expand_dims(self.logits_pos_exp/self.pos_exp_sum,axis=1)
            #self.attention_scalar = self.Mask(self.attention_scalar, self.valid, data.longestItem)
            #self.attention =tf.expand_dims(tf.nn.softmax(self.logits),dim=1)
            self.item_latent_pos = tf.matmul(self.attention_pos, self.ru_embedding)
            self.pos=tf.reduce_sum(tf.multiply(self.item_latent_pos,self.pos_embedding),-1)
            #self.pos=tf.reduce_sum(tf.multiply(user_embedding,  pos_embedding), 1)
            # _________ negative part _____________
            self.neg_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['item_embeddings_p'], self.item_neg),axis=1)
            self.product_neg=tf.multiply(self.neg_embedding,self.ru_embedding)
            self.w_neg=tf.layers.dense(self.product_neg, self.attention_size, name='first_layer_attention', reuse=tf.AUTO_REUSE, use_bias=True,activation=tf.nn.relu)
            self.logits_neg=tf.reduce_sum(tf.layers.dense(self.w_neg, 1, name='final_layer_attention', reuse=tf.AUTO_REUSE, use_bias=False), axis=-1)
            self.logits_neg_exp=tf.exp(self.logits_neg)
            self.neg_exp_sum=tf.expand_dims(tf.pow(tf.reduce_sum(self.logits_neg_exp,axis=-1), self.alpha), axis=-1)
            self.attention_neg = tf.expand_dims(self.logits_neg_exp / self.neg_exp_sum, axis=1)
            self.item_latent_neg = tf.matmul(self.attention_neg, self.ru_embedding)
            self.neg = tf.reduce_sum(tf.multiply(self.item_latent_neg, self.neg_embedding), -1)
            # Compute the loss.
            self.loss = -tf.log(tf.sigmoid(self.pos - self.neg))
            self.loss = tf.reduce_sum(self.loss)
            regularization = tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(
                self.weights['item_embeddings_p'])+tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(
                self.weights['item_embeddings_q'])
            self.loss =tf.add(self.loss,regularization)

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

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print ("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            item_embeddings_p = pretrain_graph.get_tensor_by_name('item_embeddings_p:0')
            item_embeddings_q = pretrain_graph.get_tensor_by_name('item_embeddings_q:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                iep, ieq = sess.run([item_embeddings_p, item_embeddings_q])
            all_weights['item_embeddings_p'] = tf.Variable(iep, dtype=tf.float32)
            all_weights['item_embeddings_q'] = tf.Variable(ieq, dtype=tf.float32)
        else:
            all_weights['item_embeddings_p'] = tf.Variable(
                tf.random_normal([self.num_items, self.hidden_factor], 0.0, 0.05), name='item_embeddings_p')  # features_M * 1
            all_weights['item_embeddings_q'] = tf.Variable(
                tf.random_normal([self.num_items, self.hidden_factor], 0.0, 0.05), name='item_embeddings_q')  # features_M * 1
        return all_weights

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.item_pos: data['positive'], self.item_neg: data['negative'],
                     self.dropout_keep: self.keep_prob, self.train_phase: True, self.ru:data['ru'],self.alpha:data['alpha']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, user_id):  # generate a random block of training data
        positive, negative,ru,alpha= [], [], [], []
        all_items = data.items.values()
        # get sample
        pos = data.user_positive_list[user_id]
        count=len(pos)-1
        for item in pos:
            positive.append(item)
            ru_list = list(pos)
            ru_list.remove(item)
            ru.append(ru_list)
            neg = np.random.randint(len(all_items))
            while (neg in pos):
                neg = np.random.randint(len(all_items))
            negative.append(neg)
            alpha.append(args.alpha)
        return {'positive': positive, 'negative': negative, 'ru':ru,'alpha':alpha}


    def spilt_user_batch(self, user_batch):
        num_example=len(user_batch['positive'])
        final_batch=[]
        if num_example % self.batch_size == 0:
            batch_count = num_example / self.batch_size
            flag = 0
        else:
            batch_count = math.ceil(num_example / self.batch_size)
            flag = 1
        j = 0
        for i in range(int(batch_count)):
            if flag == 1 and i == batch_count - 1:
                k = num_example
            else:
                k = j + self.batch_size
            temp={}
            #temp['user']=user_batch['user'][j:k]
            temp['positive'] = user_batch['positive'][j:k]
            temp['negative'] = user_batch['negative'][j:k]
            temp['ru'] = user_batch['ru'][j:k]
            #temp['cnt'] = user_batch['cnt'][j:k]
            temp['alpha']=user_batch['alpha'][j:k]
            final_batch.append(temp)
            j = j + self.batch_size
        return final_batch

    def train(self, Train_data):  # fit a dataset
        for epoch in range(self.epoch):
            total_loss = 0
            for i in range(data.num_users):
                # generate a batch
                batch_xs = self.get_random_block_from_data(i)
                batch_final=self.spilt_user_batch(batch_xs)
                # Fit training
                for batch in batch_final:
                    loss = self.partial_fit(batch)
                    total_loss = total_loss + loss
            print("the total loss in %d th iteration is: %f" % (epoch, total_loss))


    def evaluate(self):
        self.graph.finalize()
        count = [0, 0, 0, 0,0]
        rank = [[], [], [], [],[]]
        for index in range(len(data.Test_data['User'])):
            user = data.Test_data['User'][index]
            scores = model.get_scores_per_user(user)
            # get true item score
            true_item_id=data.Test_data['Item'][index]
            true_item_score = scores[true_item_id]
            # delete visited scores
            visited = data.user_positive_list[user]  # get positive list for the userID
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
            # print index
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
        pos = data.user_positive_list[user_id]
        if len(all_items) % self.batch_size == 0:
            batch_count = len(all_items) / self.batch_size
            flag = 0
        else:
            batch_count = math.ceil(len(all_items) / self.batch_size)
            flag = 1
        j = 0
        for i in range(int(batch_count)):
            positive, ru, alpha = [], [], []
            if flag == 1 and i == batch_count - 1:
                k = len(all_items)
            else:
                k = j + self.batch_size
            for itemID in range(j, k):
                positive.append(itemID)
                ru.append(pos)
                alpha.append(args.alpha)
            feed_dict = {self.item_pos: positive, self.ru: ru, self.alpha:alpha}
            scores = self.sess.run((self.pos), feed_dict=feed_dict)
            scores = scores.reshape(len(positive))
            scorelist = np.append(scorelist, scores)
            # scorelist.append(scores)
            j = j + self.batch_size
        return scorelist


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

    save_file = 'pretrain-fism/%s_%d' % ('ml1M', args.hidden_factor)
    # Training
    t1 = time()
    model = MF(data.num_users, data.num_items, args.pretrain, args.hidden_factor, args.epoch,
               args.batch_size, args.lr, args.lamda,  args.optimizer, args.verbose,eval(args.layers),activation_function,eval(args.keep_prob),save_file)
    #model.evaluate()
    print("begin train")
    model.train(data.Train_data)
    print("end train")
    model.evaluate()
    print("finish")


