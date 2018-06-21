import os
import pdb
import json
import nltk
from nltk import WordPunctTokenizer
nltk.data.path.append("/media/hui/DATA/linux_data/data")
from collections import defaultdict
import pickle
pos_suffix = ".pos"
id_suffix  = ".new"
UNK="<unk>"
PAD="<pad>"

sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

word_tokenizer = WordPunctTokenizer()

def _parse_json(data):
    content = json.loads(data)
    return content["text"],content["stars"]

def gen_sentence_info(data_path,min_fre=3,lower=True):
    with open(data_path,'r') as f_data:
        word_fre=defaultdict(int)
        tag_dict={}
        for line in f_data:
            data,label=_parse_json(line)
            if label not in tag_dict:
                tag_dict[label]=label
            sents=sent_tokenizer.tokenize(data)
            words=[word_tokenizer.tokenize(sent) for sent in sents]
            for sent in words:
                for word in sent:
                    if lower==True:
                        word=word.lower()
                    word_fre[word] +=1
        vocab = {}
        i = 2
        vocab[PAD]=0
        vocab[UNK]=1
        for word,fre in  word_fre.iteritems():
            if fre>=min_fre:
                vocab[word]=i
                i=i+1
        return vocab,tag_dict

def save_info(vocab,tags,vocab_path,tags_path):
    with open(vocab_path,'w') as g:
        pickle.dump(vocab,g)
        print "vocab save finished"
    with open(tags_path,'w') as g:
        pickle.dump(tags,g)
        print "tags save finished"

def load_info(vocab_path,tags_path):
    with open(vocab_path,'r') as g:
        vocab=pickle.load(g)
        print "vocab load finished"
    with open(tags_path,'r') as g:
        tags=pickle.load(g)
        print "tags load finished"
    return vocab,tags

def _tokenize(text):
    sents=sent_tokenizer.tokenize(text)
    words=[word_tokenizer.tokenize(sent) for sent in sents]
    return words

def gen_id_sentence(infile,vocab,tags,lower=True):
    def f(x): return x.lower() if lower else x
    outfile=infile+id_suffix
    data_byte_pos=[]
    total_byte=0
    with open(infile,'r') as f_in,open(outfile,'w') as f_out:
        for line in f_in:
            data={}
            text,label=_parse_json(line)
            label_id=tags[label]
            sents=_tokenize(text)
            sents_id=[]
            for sent in sents:
                sent_id=[]
                for word in sent:
                    word_id = vocab[f(word) if f(word) in vocab else UNK]
                    sent_id.append(word_id)
                sents_id.append(sent_id)
            data["text"]  = sents_id
            data["stars"] = label_id
            data_str=json.dumps(data)
            f_out.write(data_str+'\n')
            byte_len = len(data_str.decode('utf8'))+1
            data_byte_pos.append([total_byte,byte_len])
            total_byte += byte_len
    pos_file=infile+pos_suffix
    with open(pos_file,'w') as f_pos:
        pickle.dump(data_byte_pos,f_pos)
    return data_byte_pos

def load_byte_pos(path):
    with open(path+pos_suffix,'r') as f_pos:
        data_byte_pos=pickle.load(f_pos)
        return data_byte_pos

def read_random_data(file,index):
    batch_data=[]
    with open(file+id_suffix,'r') as f_in:
        for pos,size in index:
            f_in.seek(pos)
            data = f_in.read(size)
            batch_data.append(data.strip('\n'))
        return batch_data
