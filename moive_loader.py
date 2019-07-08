import argparse

class movie:
    def __init__(self, genre, director, actor):
        self.genre = genre
        self.director=director
        self.actor=actor


class movie_loader:
    def __init__(self):
        self.movie_dict=self.load_movie()
        self.genre_list, self.director_list, self.actor_list=self.load_attribute()


    def load_movie(self):  # map the feature entries in all files, kept in self.features dictionary
        parser = argparse.ArgumentParser(description=''' load movie data''')

        parser.add_argument('--movie_data_file', type=str, default='./ML100K/auxiliary-mapping.txt')

        parsed_args = parser.parse_args()

        movie_file = parsed_args.movie_data_file
        movie_dict= {}

        fr = open(movie_file, 'r')
        for line in fr:
            lines = line.replace('\n', '').split('|')
            # if len(lines) != 4:
            #     continue
            movie_id = lines[0]
            genre_list = []
            genres = lines[1].split(',')
            for item in genres:
                genre_list.append(int(item))
            director_list=[]
            directors = lines[2].split(',')
            for item in directors:
                director_list.append(int(item))
            actor_list=[]
            actors = lines[3].split(',')
            for item in actors:
                actor_list.append(int(item))
            new_movie = movie(genre_list, director_list, actor_list)
            movie_dict[movie_id]=new_movie
        fr.close()
        # id_list1=[]
        # id_list2=[]
        # fr = open(movie_file, 'r')
        # for line in fr:
        #     lines = line.replace('\n', '').split('|')
        #     # if len(lines) != 4:
        #     #     continue
        #     movie_id = int(lines[0])
        #     if movie_id not in id_list1:
        #         id_list1.append(movie_id)
        # fr.close()
        #
        # fr = open('ML/train.txt', 'r')
        # for line in fr:
        #     lines = line.strip().split('\t')
        #     # if len(lines) != 4:
        #     #     continue
        #     movie_id = int(lines[1])
        #     if movie_id not in id_list2:
        #         id_list2.append(movie_id)
        # fr.close()
        # fr = open('ML/test.txt', 'r')
        # for line in fr:
        #     lines = line.strip().split('\t')
        #     # if len(lines) != 4:
        #     #     continue
        #     movie_id = int(lines[1])
        #     if movie_id not in id_list2:
        #         id_list2.append(movie_id)
        # fr.close()
        # rest=set(id_list2).difference(set(id_list1))
        return movie_dict

    def load_attribute(self):  # map the feature entries in all files, kept in self.features dictionary
        parser = argparse.ArgumentParser(description=''' load movie data''')

        parser.add_argument('--movie_data_file', type=str, default='./ML100K/auxiliary-mapping.txt')

        parsed_args = parser.parse_args()

        movie_file = parsed_args.movie_data_file
        genre_list = []
        director_list=[]
        actor_list=[]
        fr = open(movie_file, 'r')
        for line in fr:
            lines = line.replace('\n', '').split('|')
            # if len(lines) != 4:
            #     continue
            genres = lines[1].split(',')
            for item in genres:
                if int(item) not in genre_list:
                    genre_list.append(int(item))
            directors = lines[2].split(',')
            for item in directors:
                if int(item) not in director_list:
                    director_list.append(int(item))
            actors = lines[3].split(',')
            for item in actors:
                if int(item) not in actor_list:
                    actor_list.append(int(item))
        fr.close()
        return genre_list,director_list,actor_list