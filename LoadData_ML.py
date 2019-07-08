'''
Utilities for Loading data.
'''
from moive_loader import movie_loader


class LoadData(object):

    # Three files are needed in the path
    def __init__(self):
        self.trainfile = "./ML100K/train.txt"
        self.testfile = "./ML100K/test.txt"
        self.num_users, self.num_items = self.map_features()
        self.user_positive_list = self.get_positive_list(self.trainfile)  # userID positive itemID
        self.Train_data, self.Test_data = self.construct_data()
        loader=movie_loader()
        self.movie_dict = loader.movie_dict
        self.all_genres=loader.genre_list
        self.all_directors = loader.director_list
        self.all_actors = loader.actor_list
        self.num_genres=len(self.all_genres)
        self.num_directors=len(self.all_directors)
        self.num_actors=len(self.all_actors)

    def map_features(self):  # map the feature entries in all files, kept in self.features dictionary
        self.users = {}
        self.items = {}
        self.users_traverse={}
        self.items_traverse={}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        return len(self.users), len(self.items)

    def read_features(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        u = len(self.users)
        i = len(self.items)
        while line:
            contents = line.strip().split('\t')
            user=contents[0]
            item=contents[1]
            if user not in self.users:
                self.users[user] = u
                self.users_traverse[u]=user
                u = u + 1
            if item not in self.items:
                self.items[item] = i
                self.items_traverse[i]=item
                i = i + 1
            line = f.readline()
        f.close()

    def construct_data(self):
        User, Item = self.read_data(self.trainfile)
        Train_data = self.construct_dataset(User, Item)
        print("# of training:", len(User))

        User, Item = self.read_data(self.testfile)
        Test_data = self.construct_dataset(User, Item)
        print("# of test:", len(User))

        return Train_data, Test_data

    def get_positive_list(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        user_positive_list = {}
        while line:
            contents = line.strip().split('\t')
            user_id = self.users[contents[0]]
            item_id = self.items[contents[1]]
            if user_id in user_positive_list:
                user_positive_list[user_id].append(item_id)
            else:
                user_positive_list[user_id] = [item_id]
            line = f.readline()
        f.close()
        return user_positive_list

    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open(file)
        User = []
        Item = []
        line = f.readline()
        while line:
            contents = line.strip().split('\t')
            User.append(self.users[contents[0]])
            Item.append(self.items[contents[1]])
            line = f.readline()
        f.close()
        return User, Item

    def construct_dataset(self, User, Item):
        Data_Dic = {}
        lens = len(User)
        Data_Dic['User'] = [User[i] for i in xrange(lens)]
        Data_Dic['Item'] = [Item[i] for i in xrange(lens)]
        return Data_Dic
