import json
import pdb
import numpy as np

def _parse_json(data):
    content = json.loads(data)
    return content["text"],content["stars"]

def _gen_max_len(paras):
    max_paras_len = 0
    max_sent_len  = 0
    for para in paras:
        text,label=_parse_json(para)
        if len(text)>max_paras_len:
            max_paras_len = len(text)
        for sentence in text:
            assert len(sentence) > 0
            if len(sentence) > max_sent_len:
                max_sent_len = len(sentence)
    return max_paras_len,max_sent_len

def _padding_words_ids(max_paras_len,max_sents_len,batch_size,paras):
    sents_id   =  np.zeros([max_paras_len*batch_size,max_sents_len])
    sents_pos  = np.zeros([max_paras_len*batch_size,])
    paras_pos  = np.zeros([batch_size,])
    labels     = np.zeros([batch_size,])
    for para_idx,para in enumerate(paras):
        text,label = _parse_json(para)
        labels[para_idx] = label
        paras_pos[para_idx] = len(text)
        for sent_idx,sentence in enumerate(text):
            index = para_idx*max_paras_len+sent_idx
            sents_id[index,0:len(sentence)]=sentence
            sents_pos[index] = len(sentence)
    return sents_id,sents_pos,paras_pos,labels

def create_input_batch(paras):
    batch_size = len(paras)
    max_paras_len,max_sent_len = _gen_max_len(paras)
    return _padding_words_ids(max_paras_len,max_sent_len,batch_size,paras)
    # pdb.set_trace()
    # max_word_length = 
