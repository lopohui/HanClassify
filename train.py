import os
import ioutils
import optparse
import pdb
import os
from ioutils import gen_sentence_info,save_info,load_info,gen_id_sentence,load_byte_pos,read_random_data
from utils import create_input_batch
import numpy as np
import tensorflow as tf
import logging
from HanClass import HanClassify

def set_log():
    logging.basicConfig(filename='logger.log', level=logging.INFO)

def parse_args():
    optparser = optparse.OptionParser()
    optparser.add_option("--train_path",default="train",help="train path")
    optparser.add_option("--dev_path",default="dev",help="dev path")
    optparser.add_option("--test_path",default="test",help="test path")
    optparser.add_option("--vocab_path",default="vocab",help="vocab path")
    optparser.add_option("--tags_path",default="tags",help="tags path")

    optparser.add_option("--n_epoch",default=10,help="num of epochs")
    optparser.add_option("--batch_size",default=2,help="batch size")
    
    optparser.add_option("--emd_dim",default=256,help="embedding size")
    optparser.add_option("--lstm_dim",default=128,help="lstm hidden dim")
    optparser.add_option("--lr_method",default="sgd",help="learning method")
    optparser.add_option("--lr_rate",default=0.1,help="learning rate")

    opts = optparser.parse_args()[0]
    # pdb.set_trace()
    parameters=vars(opts)
    return parameters
    # pdb.set_trace()

def initial_tf_config():
    os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement     = True
    return config

def print_train_var():
    for var in tf.trainable_variables():    
        print var.name+":"+str(var.get_shape())

if __name__=="__main__":
    parameters=parse_args()
    set_log()
    # vocab,tags=load_info(parameters["vocab_path"],parameters["tags_path"])
    vocab,tags=gen_sentence_info(parameters["train_path"],min_fre=5) 
    print "%i words %i tags in train dataset." %(len(vocab),len(tags))
    # save_info(vocab,tags,parameters["vocab_path"],parameters["tags_path"])
    #prevent from loading the entire dataset into memory
    # byte_pos_train = gen_id_sentence(parameters["train_path"],vocab,tags)
    # byte_pos_dev   = gen_id_sentence(parameters["dev_path"],vocab,tags)
    # byte_pos_test  = gen_id_sentence(parameters["test_path"],vocab,tags)
    byte_pos_train   = load_byte_pos(parameters["train_path"])
    byte_pos_dev     = load_byte_pos(parameters["dev_path"])
    byte_pos_test    = load_byte_pos(parameters["test_path"])
    config=initial_tf_config() 
    model = HanClassify(vocab,tags)
    with tf.device("gpu:0"):
        cost = model.build(**parameters)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print_train_var()
        batch_size = parameters["batch_size"]
        n_batch = len(byte_pos_train)/batch_size
        logging.info("total batch is %i",n_batch)
        for epoch in xrange(parameters["n_epoch"]):
            train_data_count = 0
            logging.info("start epoch %i",epoch)
            permuate_index = np.random.permutation(len(byte_pos_train))
            for i in range(n_batch):
                batch_index = permuate_index[i*batch_size:(i+1)*batch_size]
                batch_pos   = [byte_pos_train[index] for index in batch_index]
                batch_data  = read_random_data(parameters["train_path"],batch_pos) 
                input_ = create_input_batch(batch_data)
                # pdb.set_trace()
                feed_dict_={}
                feed_dict_[model.sents_id]  = input_[0]
                feed_dict_[model.sents_pos] = input_[1]
                feed_dict_[model.paras_pos] = input_[2]
                feed_dict_[model.labels]    = input_[3]
                feed_dict_[model.max_sen_len] = max(input_[1])
                feed_dict_[model.max_para_len] = max(input_[2])
                # pdb.set_trace()
                f_score = sess.run([cost],feed_dict=feed_dict_)
                pdb.set_trace()
