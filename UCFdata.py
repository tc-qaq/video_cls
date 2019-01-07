"""
Class for managing our data.
"""
import csv, numpy as np, random, glob, os.path, pandas as pd, sys, operator
from processer import process_image
from keras.utils import np_utils

class DataSet():
    def __init__(self, seq_length=40, class_limit=None, image_shape=(224,224,3)):
        """Constructor.
        seq_length  = (int) nb_frames to consider
        class_limit = (int) nb_classes to limit the data to. None = no limit        
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.seq_path = './data/sequences'
        self.max_frames = 300 # max nb_framses a video can have for us to use it
        
        #Get the data
        self.data = self.getData()
        #Get the classes
        self.classes = self.get_classes()
        #Now do same minor data cleaning
        self.data = self.clean_data()
        self.image_size = image_shape
    
    @staticmethod
    def get_data():
        #Load our data from file
        with open('./data/data_file.csv', 'r') as fin:
            reader = csv.reader(fin)
            data   = list(reader)
        return data
    
    def clean_data(self):
        """
        Limit samples to greater than the sequence length and fewer than N frames. 
        Also limit it to classes we want to use.
        """
        data_clean = []
        for item in self.data:
            if(int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames 
               and item[1] in self.classes):
                data_clean.append(item)
        return data_clean
    
    def get_classes(self):
        classes = []
        for item in self.data:
            if (item[1] not in classes):
                classes.append(item[1])
        classes = sorted(classes)
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes
    
    def get_class_one_hot(self, class_str):
        """
        Given a class as a string, return its number in the classes list.
        """
        #Encode it first
        label_encode = self.classes.index(class_str)
        #Now one-hot it
        label_hot = np_utils.to_categorical(label_encode, len(self.classes))
        label_hot = label_hot[0]  # just get a single row
        return label_hot
    
    def split_train_test(self):
        #Split the data into train and test groups
        train = []
        test  = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test
    
    def get_all_seq_in_memory(self, batch_size, train_test, data_type, concat=False):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        #Get the right dataset
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test
        print("Getting %s data with %d samples." % (train_test, len(data)))
        
        X, y = [], []
        for row in data:
            seq = self.get_exatracted_seq(data_type, row)
            if seq is None:
                print("Can't find sequence. Did you generate them?")
                raise
            if concat:
                #We want to pass the sequence back as a single array. This is used to pass into a CNN
                # or MLP, rather than an RNN.
                seq = np.concatenate(seq).ravel()
            X.append(seq)
            y.append(self.get_class_one_hot(row[1]))
        return np.array(X), np.array(y)
    
    def frame_generator(self, batch_size, train_test, data_type, concat=False):
        """
        Return a generator that we can use to train on. There are a couple different things
        we can return:
                      data_type: 'features', 'images'
        """
        #Get the right dataset for the generator
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test
        print("Creating %s generator with %d samples" %(train_test, len(data)))
        
        while 1:
            X, y = [], []
            #Generate batch_size smaples
            for _ in range(batch_size):
                #Reset to be safe
                seq = None
                sample = random.choice(data)
                if data_type is "images":
                    #Get and resample frames
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)
                    #Build the image sequence
                    seq = self.build_image_seq(frames)
                else:
                    seq = self.get_extracted_seq(data_type, sample)
                if seq is None:
                    print("Can't find sequence. Did you generate them?")
                    sys.exit()  # TODO this should raise
                if concat:
                    #We want to pass the sequence back as a single array. This is used to pass into a CNN
                    # or MLP, rather than an RNN.
                    seq = np.concatenate(seq).ravel()
                X.append(seq)
                y.append(self.get_class_one_hot(sample[1]))
            yield np.array(X), np.array(y)
    
    def build_image_seq(self, frames):
        #Given a set of frames, build our sequence
        return [process_image(x, self.image_shape) for x in frames]
    
    def get_extracted_seq(self, data_type, sample):
        #Get the saved extracted features
        filename = sample[2]
        path     = self.seq_path + filename + '-' + str(self.seq_length) + '-' + data_type + '.txt'
        if os.path.isfile(path):
            #Use a dataframe/read_csv for speed increase over numpy
            features = path.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None
    
    @staticmethod
    def get_frames_for_sample(sample):
        #Given a sample row from the data file, get all the corresponding frame filenames
        path   = './data/' + sample[0] + '/' + sample[1] + '/' + sample[2] + '*jpg'
        images = sorted(glob.glob(path)) 
        return images
    
    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split('/')
        return parts[-1].replace('.jpg', '')
    
    @staticmethod
    def rescale_list(input_list, size):
        #Given a list and a size, return a rescaled/samples list. 
        assert len(input_list) >= size
        #Get the number to skip between iterations
        skip = len(input_list)//size
        #Build our new output
        output = [input_list[i] for i in range(0, len(input_list), skip)]
        return output[:size]