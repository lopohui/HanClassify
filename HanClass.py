import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
class HanClassify(object):
    def __init__(self,vocab,label):
        self.n_words  = len(vocab) 
        self.n_labels = len(label)

    def feedforward(self,dropout,emd_dim,lstm_dim):
        self.sents_id  =  tf.placeholder(tf.int32,shape=[None,None],name="sents_id") #shape batch_size*max_paras_len,max_sent_len
        self.sents_pos = tf.placeholder(tf.int32,shape=[None],name="sents_pos")#shape batch_size*max_paras_len
        self.paras_pos = tf.placeholder(tf.int32,shape=[None],name="paras_pos")#shape batch_size
        self.labels    = tf.placeholder(tf.int32,shape=[None],name="labels")#shape batch_size
        self.max_sen_len = tf.placeholder(tf.int32,shape=[],name="max_sen_len") # max sentences length
        self.max_para_len = tf.placeholder(tf.int32,shape=[],name="max_para_len") #max paras length
        with tf.variable_scope("embed"):
            embed_shape=(self.n_words,emd_dim)
            drange = np.sqrt(6./(np.sum(embed_shape))) 
            embeddings =  tf.get_variable("embedding",embed_shape,tf.float32,tf.random_uniform_initializer(-drange,drange))
            inputs = tf.nn.embedding_lookup(embeddings,self.sents_id)
        with tf.variable_scope("lstm"):
            lstm_cell_fw1 = tf.contrib.rnn.LSTMCell(lstm_dim,use_peepholes=True)
            lstm_cell_bw1 = tf.contrib.rnn.LSTMCell(lstm_dim,use_peepholes=True)
            output1,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw1,cell_bw=lstm_cell_bw1,inputs=inputs,sequence_length=self.sents_pos,dtype=tf.float32)
        output1 = tf.concat(output1,2)
        with tf.variable_scope("self-attn"):
            h1 = layers.fully_connected(output1,lstm_dim*2,activation_fn=tf.nn.tanh)
            u1_context = tf.get_variable("u_context",(lstm_dim*2,1))
            h1 = tf.matmul(tf.reshape(h1,[-1,lstm_dim*2]),u1_context)
            h1 = tf.reshape(h1,[-1,self.max_sen_len])
        return h1

    def build(self,dropout=0.5,clip_norm=10,lr_method="sgd",lr_rate=0.1,emd_dim=256,lstm_dim=256,is_train=True,**kwargs):
        f_scores = self.feedforward(dropout,emd_dim,lstm_dim)   
        return f_scores
