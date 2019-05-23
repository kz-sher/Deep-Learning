import os
import zipfile
import collections
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class DataManager:
    def __init__(self, verbose=True, random_state=6789):
        self.verbose = verbose
        self.max_sentence_len = 0
        self.questions = list()
        self.qlen = list()
        self.str_labels = list()
        self.numeral_labels = list()
        self.numeral_data = list()
        self.cur_pos=0
        self.pad_word = '_PAD'
        self.random_state = random_state
        self.random = np.random.RandomState(random_state)
        
    def maybe_download(self, dir_name, file_name, url):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not os.path.exists(os.path.join(dir_name, file_name)):
            urlretrieve(url + file_name, os.path.join(dir_name, file_name))
        if self.verbose:
            print("Downloaded successfully {}".format(file_name))
    
    def read_data(self, dir_name, file_name):
        file_path= os.path.join(dir_name, file_name)
        self.questions= list(); self.labels= list()
        with open(file_path, "r", encoding="latin-1") as f:
            for row in f:
                row_str= row.split(":")
                label, question= row_str[0], row_str[1]
                question= question.lower()
                self.labels.append(label)
                self.questions.append(question.split()[1:])
                self.qlen.append(len(self.questions[-1]))
                if self.max_sentence_len < len(self.questions[-1]):
                    self.max_sentence_len= len(self.questions[-1])
        
        # turns question length list to numpy array
        self.qlen = np.array(self.qlen)
        
        # turns labels into numbers
        le= preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.numeral_labels = np.array(le.transform(self.labels))
        self.str_classes= le.classes_
        self.num_classes= len(self.str_classes)
        if self.verbose:
            print("Sample questions \n")
            print(self.questions[0:5])
            print("Labels {}\n\n".format(self.str_classes))
    
    def padding(self, length):
        for question in self.questions:
            question= question.extend([self.pad_word]*(length- len(question)))
    
    def build_numeral_data(self, dictionary):
        self.numeral_data= list()
        for question in self.questions:
            self.numeral_data.append([dictionary[word] for word in question])
        self.numeral_data = np.array(self.numeral_data)
        if self.verbose:
            print('Sample numeral data \n')   
            print(self.numeral_data[0:5])
    
    def train_valid_split(self, train_size=0.9):
        X_train, X_valid, y_train, y_valid, qlen_train, qlen_valid = train_test_split(
            self.numeral_data, self.numeral_labels, self.qlen, test_size = 1-train_size, random_state= self.random_state)
        self.train_numeral = X_train
        self.train_labels = y_train
        self.valid_numeral = X_valid
        self.valid_labels = y_valid
        self.train_qlen = qlen_train
        self.valid_qlen = qlen_valid
        
    @staticmethod
    def build_dictionary_count(questions):
        count= []
        dictionary= dict()
        words= []
        for question in questions:
            words.extend(question)
        count.extend(collections.Counter(words).most_common())
        for word,freq in count:
            dictionary[word]= len(dictionary)
        reverse_dictionary= dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary, count
    
    def next_batch(self, batch_size, split='train'):
        if split == 'train':
            idx = self.random.randint(len(self.train_numeral), size=batch_size)
            return self.train_numeral[idx], self.train_labels[idx], self.train_qlen[idx]
        elif split == 'valid':
            idx = self.random.randint(len(self.valid_numeral), size=batch_size)
            return self.valid_numeral[idx], self.valid_labels[idx], self.valid_qlen[idx]
        else:
            idx = self.random.randint(len(self.numeral_data), size=batch_size)
            return self.numeral_data[idx], self.numeral_labels[idx], self.qlen[idx]
        

def load_data(verbose=False):
    print('Loading data...')
    # download data if needed
    train_dm = DataManager(random_state=6789, verbose=verbose)
    train_dm.maybe_download("Data", "train_1000.label", "http://cogcomp.org/Data/QA/QC/")
    test_dm = DataManager(random_state=6789, verbose=verbose)
    test_dm.maybe_download("Data", "TREC_10.label", "http://cogcomp.org/Data/QA/QC/")

    # read data
    train_dm.read_data("Data/", "train_1000.label")
    test_dm.read_data("Data/", "TREC_10.label")

    # pad question so that questions are of the same length
    pad_len = max(train_dm.max_sentence_len, test_dm.max_sentence_len)
    train_dm.padding(pad_len)
    test_dm.padding(pad_len)

    # get the list of questions
    all_questions= list(train_dm.questions) 
    all_questions.extend(test_dm.questions)

    # tockenize questions and create a dictionary to map words to numbers
    dictionary, id2word, _= DataManager.build_dictionary_count(all_questions)

    # map questions to sequence of word ids in the dictionary created
    train_dm.build_numeral_data(dictionary)
    test_dm.build_numeral_data(dictionary)
    train_dm.train_valid_split()
    data_batch, label_batch, qlen = train_dm.next_batch(batch_size=5)
    if verbose:
        print("Sample data batch- label batch \n")
        print("Question sequences: ", data_batch)
        print("Question labels: ", label_batch)
        print("Quenstion length: ", qlen)    
    print('Finished loading data!')
    return train_dm, test_dm, dictionary, id2word

he_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN')
normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
zero_initializer = tf.constant_initializer(0.0)

class Layers:
    @staticmethod
    def dense(inputs, output_dim, act=None, name='dense'):
        """
        Dense layer
    
        inputs: of shape [batch size, D]
        output_dim: The number of hidden units in the output layer
        act: Apply activation function if act is not None
        
        Return: a tensor of shape [batch size, output_dim]
        """
        with tf.variable_scope(name):
            W = tf.get_variable('W', [inputs.get_shape()[1], output_dim], initializer=he_initializer)
            b = tf.get_variable('b', [output_dim], initializer=zero_initializer)
            Wxb= tf.matmul(inputs, W) + b
            return Wxb if act is None else act(Wxb)

    @staticmethod
    def conv2D(inputs, output_dim, kernel_size=3, strides=1, padding="SAME", act=None, name= "conv"):
        """
        2D Convolutional layer
        
        inputs: a feature map of shape [batch size, H, W, C]
        output_dim: the number of feature maps in the output
        kernel_size: a tuple (h, w) specifying the heigh and width of the convolution window.
                     Can be a integer if the heigh and width are equal
        strides: a tuple (h, w) specifying the strides of the convolution along the height and width.
                 Can be a integer if the stride is the same along the height and width
        act: Apply activation function if act is not None
        
        Return: a tensor of shape [batch_size, H', W', C']
        """
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        strides = (strides, strides) if isinstance(strides, int) else strides
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                [kernel_size[0], kernel_size[1], inputs.get_shape()[-1], output_dim],
                                initializer=he_initializer)
            b = tf.get_variable('b', [output_dim], initializer=zero_initializer)
            conv = tf.nn.conv2d(input=inputs, filter=W, strides=[1, strides[0], strides[1], 1], padding= padding)
            conv = conv + b
            return conv if act is None else act(conv)
        
            
    @staticmethod
    def conv1D(inputs, filter_width, output_dim, stride=1, padding="SAME", name="conv1", act=None):
        """
        1D Convolutional layer. Source: https://www.tensorflow.org/api_docs/python/tf/nn/conv1d
        
        inputs: a feature map of shape [batch, in_width, in_channels]
        output_dim: the number of channels in the output
        strides: a number specifying the strides of the convolution along the width axis
        act: Apply activation function if act is not None
        
        Return: a tensor of shape [batch_size, out_width, output_dim]
        """        
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                [filter_width, inputs.get_shape()[-1], output_dim],
                                initializer=he_initializer)
            b = tf.get_variable('b', [output_dim], initializer=zero_initializer)
            Wxb= tf.nn.conv1d(value=inputs, filters=W, stride=stride, padding=padding) + b
            if act is None:
                return Wxb
            else:
                return act(Wxb)        
        
    @staticmethod
    def max_pool(inputs, ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name='max_pool'):
        return tf.nn.max_pool(value= inputs, ksize=ksize, strides= strides, padding= padding, name=name)
    
    @staticmethod
    def mean_pool(inputs, ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name='avg_pool'):
        return tf.nn.avg_pool(value= inputs, ksize=ksize, strides= strides, padding= padding, name=name)
    
    @staticmethod
    def dropout(inputs, keep_prob, name='drop_out'):
        return tf.nn.dropout(inputs, keep_prob=keep_prob, name=name)
    
    @staticmethod
    def batch_norm(inputs, phase_train, name='batch_norm'):
        return tf.contrib.layers.batch_norm(inputs, decay= 0.99, epsilon=1e-5, updates_collections=None, 
                                            is_training=phase_train, center= True, scale=True, reuse= False, scope=name)
    
       
class BaseModel():
    def __init__(self,
                 vocab_size,
                 ebd_size=3,
                 num_layers=1,
                 state_size=1,
                 cell_type='gru',
                 net_type='uni-directional',
                 batch_size=32,
                 num_epochs=100,
                 num_classes=5,
                 verbose= True,
                 optimizer= 'adam',
                 learning_rate=0.001,
                 num_subsample=20,
                 random_state=6789,
                 name='base_CNN'):
        self.name = name
        self.vocab_size = vocab_size
        self.ebd_size = ebd_size
        self.num_layers = num_layers
        self.state_size = state_size
        self.cell_type = cell_type
        self.net_type = net_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.verbose = verbose
        self.random_state = random_state
        self.num_subsample = num_subsample
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            
        self.tf_graph = tf.Graph()
        self.session = tf.Session(graph=self.tf_graph)
        with self.tf_graph.as_default():
            if random_state is not None:
                tf.set_random_seed(random_state)
            
            self.build_graph()
            self.session.run(tf.global_variables_initializer())
            
        # create log_path
        self.root_dir = 'models/{}'.format(self.name)
        self.log_path = os.path.join(self.root_dir, 'logs')
        self.model_path = os.path.join(self.root_dir, 'saved/model.ckpt')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)            
            
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))            
    
    def build_graph(self):
        self.X = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.qlen = tf.placeholder(shape=[None], dtype=tf.int32)
        
        # Embedding layer
        self.embeddings = tf.get_variable("embeddings", [self.vocab_size, self.ebd_size])
        inputs = tf.nn.embedding_lookup(self.embeddings, self.X)
        
        return
        
            
    def fit(self, data_manager, batch_size=None, num_epochs=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        num_epochs = self.num_epochs if num_epochs is None else num_epochs
        iters_per_epoch= data_manager.train_numeral.shape[0] // batch_size + 1
        
        # initialize list to store history of training
        self.best_val_accuracy = 0
        self.history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": [], "best_valid_acc": []}   

        # bootstrapping the loss and accuracy
        def compute_loss_acc(split, num_subsample=self.num_subsample):
            loss, acc = 0.0, 0.0
            for i in range(num_subsample):
                X_batch, y_batch, qlen_batch = data_manager.next_batch(batch_size=batch_size, split=split)
                _loss, _acc= self.session.run([self.loss, self.accuracy],
                                              feed_dict={self.X: X_batch, self.y: y_batch, self.qlen: qlen_batch})
                loss += _loss / num_subsample
                acc += _acc / num_subsample
            return loss, acc        
        
        for epoch in range(num_epochs):
            # run optimization steps
            for i in range(iters_per_epoch):
                X_batch, y_batch, qlen_batch = data_manager.next_batch(batch_size= batch_size, split='train')
                self.session.run([self.train], feed_dict={self.X: X_batch, self.y: y_batch, self.qlen: qlen_batch})
            
            # evaluate training progress
            train_loss, train_acc= compute_loss_acc(split='train')
            valid_loss, valid_acc= compute_loss_acc(split='valid')
            for key, value in zip(['train_loss', 'train_acc', 'valid_loss', 'valid_acc'],
                                  [train_loss, train_acc, valid_loss, valid_acc]):
                self.history[key].append(value)
                    
            if epoch == 0 or self.history['best_valid_acc'][-1] < valid_acc:
                self.history['best_valid_acc'].append(valid_acc)
                self.save(self.model_path)
            else:
                self.history['best_valid_acc'].append(self.history['best_valid_acc'][-1])
                
            if self.verbose:
                print('Epoch {:03d}'.format(epoch + 1))
                print('Train loss: {:.4f}   Train accuracy: {:.4f}'.format(train_loss, train_acc))
                print('Valid loss: {:.4f}   Valid accuracy: {:.4f}   Best valid accuracy: {:.4f}'.
                      format(valid_loss, valid_acc, self.history['best_valid_acc'][-1]))
                
    def save(self, model_path):
        with self.tf_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, model_path)       
            
    def load(self, model_path):
        with self.tf_graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, model_path)
                
    def predict(self, X, y=None, qlen=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        if qlen is None:
            qlen = np.zeros(X.shape[0])
            
        y_pred = np.zeros(len(X))
        for i in range(0, len(X), batch_size):
            j = min(i + batch_size, len(X))
            y_pred[i: j] = self.session.run(self.y_pred, feed_dict={self.X: X[i: j], self.qlen: qlen[i: j]})
            
        if y is None:
            return y_pred
        
        acc = np.mean(y_pred == y)
        return y_pred, acc
    
    def plot_history(self):
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        ax[0].plot(np.arange(len(self.history['train_acc'])), self.history['train_acc'], "g")
        ax[0].plot(np.arange(len(self.history['valid_acc'])), self.history['valid_acc'], "b")
        ax[0].set_title('Accuracy over epoch')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend(['Train Accuracy', 'Val Accuracy'], loc='best')

        ax[1].plot(np.arange(len(self.history['train_loss'])), self.history['train_loss'], "g")
        ax[1].plot(np.arange(len(self.history['valid_loss'])), self.history['valid_loss'], "b")
        ax[1].set_title('Loss over epochs')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend(['Train Loss', 'Val Loss'], loc='best')
        plt.show()