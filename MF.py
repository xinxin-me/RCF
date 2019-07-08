'''
Tensorflow implementation of MF

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
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=512,
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
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ positive part _____________
            user_embedding = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user)
            pos_embedding=tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item_pos)
            self.pos = tf.reduce_sum(tf.multiply(user_embedding, pos_embedding), 1)
            # _________ negative part _____________
            neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item_neg)
            self.neg = tf.reduce_sum(tf.multiply(user_embedding, neg_embedding), 1)
            # Compute the loss.
            self.loss = -tf.log(tf.sigmoid(self.pos - self.neg))
            self.loss = tf.reduce_sum(self.loss)
            regularization = tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(
                self.weights['user_embeddings'])+tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(
                self.weights['item_embeddings'])
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
        all_weights['user_embeddings'] = tf.Variable(
            tf.random_normal([self.num_users, self.hidden_factor], 0.0, 0.05), name='user_embeddings')  # features_M * K
        all_weights['item_embeddings'] = tf.Variable(
            tf.random_normal([self.num_items, self.hidden_factor], 0.0, 0.05), name='item_embeddings')  # features_M * 1
        num_layer = len(self.layers)
        return all_weights


    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user: data['user'], self.item_pos: data['positive'], self.item_neg: data['negative'],
                     self.dropout_keep: self.keep_prob, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, train_data, batch_size):  # generate a random block of training data
        user, positive, negative = [], [], []
        all_items = data.items.values()
        # get sample
        while len(user) < batch_size:
            index = np.random.randint(0, len(train_data['User']))
            user.append(train_data['User'][index])
            positive.append(train_data['Item'][index])
            # uniform sampler
            user_id = train_data['User'][index] # get userID
            pos = data.user_positive_list[user_id]  # get positive list for the userID
            # candidates = list(set(all_items) - set(pos))  # get negative set
            neg = np.random.randint(len(all_items))  # uniform sample a negative itemID from negative set
            while (neg in pos):
                neg = np.random.randint(len(all_items))
            negative.append(neg)
        return {'user': user, 'positive': positive, 'negative': negative}


    def train(self, Train_data):  # fit a dataset
        for epoch in range(self.epoch):
            total_loss = 0
            t1 = time()
            total_batch = int(len(Train_data['User']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                loss = self.partial_fit(batch_xs)
                total_loss = total_loss + loss
            t2 = time()
            print("the total loss in %d th iteration is: %f" % (epoch, total_loss))
        if self.pretrain_flag < 0:
            print("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)


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


    def get_scores_per_user(self, user_feature):  # evaluate the results for an user context, return scorelist
        scorelist = []
        all_items = data.items.values()

        if len(all_items) % self.batch_size == 0:
            batch_count = len(all_items) / self.batch_size
            flag = 0
        else:
            batch_count = math.ceil(len(all_items) / self.batch_size)
            flag = 1
        j = 0
        # print(len(all_items))
        # print(batch_count)
        # print(flag)
        for i in range(int(batch_count)):
            X_user, X_item = [], []
            if flag == 1 and i == batch_count - 1:
                k = len(all_items)
            else:
                k = j + self.batch_size
            # print(j)
            # print(k)
            for itemID in range(j, k):
                X_user.append(user_feature)
                X_item.append(itemID)
            feed_dict = {self.user: X_user, self.item_pos: X_item,self.dropout_keep: self.no_dropout, self.train_phase: False}
            # print(X_item)
            scores = self.sess.run((self.pos), feed_dict=feed_dict)
            scores = scores.reshape(len(X_user))
            scorelist = np.append(scorelist, scores)
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

    save_file = 'pretrain-mf/%s_%d' % ('ml1M', args.hidden_factor)
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


