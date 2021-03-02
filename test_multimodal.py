'''
This Program is developed to run pretrained model on a test dataset for our VAG-NMT
'''
import torch
from torch.autograd import Variable
from preprocessing import *
from train import *
from bleu import *
import os
from machine_translation_vision.utils import im_retrieval_eval
from machine_translation_vision.meteor.meteor import Meteor
import argparse

SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 80

use_cuda = torch.cuda.is_available()
print("Whether GPU is available: {}".format(use_cuda))

######################User Defined Area#########################
# data_path = '/directory/of/data' #Define the Directory of the Test Data Path
# source_language = 'en'
# target_language = 'fr'
# model_path = "/path/to/model" #The full path to the trained model that you want to test on
# output_path = "/path/to/save/prediction" #Directory to save the translation results from a trained model

PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
PARSER.add_argument('--data_path',required=True, help='path to multimodal machine translation dataset')
PARSER.add_argument('--trained_model_file',required=True, help='path to trained model file')
PARSER.add_argument('--sr',type=str,required=True,help='the source language')
PARSER.add_argument('--tg',type=str,required=True, help='the target language')
PARSER.add_argument('--output_path',type=str,required=True,help='directory to save translation results from a trained model')

PARSER.add_argument('--batch_size',type=int, default=32, help='batch size during generation of corresponding text and image features')
PARSER.add_argument('--eval_batch_size',type=int, default=16, help='batch size during evaluation')
PARSER.add_argument('--beam_size',type=int, default=12, help='The beam size for beam search')

ARGS = PARSER.parse_args()

data_path = ARGS.data_path
source_language = ARGS.sr
target_language = ARGS.tg
model_path = ARGS.trained_model_file
output_path = ARGS.output_path

batch_size = ARGS.batch_size
eval_batch_size = ARGS.eval_batch_size
beam_size = ARGS.beam_size
################################################################

BPE_dataset_suffix = '.norm.tok.lc.10000bpe'
dataset_suffix = '.norm.tok.lc'
dataset_im_suffix = '.norm.tok.lc.10000bpe_ims'

#Initilalize a Meteor Scorer
Meteor_Scorer = Meteor(target_language)

#Create the directory for the trained_model_output_path
if not os.path.isdir(output_path):
    os.mkdir(output_path)

#Load the test dataset
test_source = load_data(os.path.join(data_path,'test'+BPE_dataset_suffix+'.'+source_language))
test_target = load_data(os.path.join(data_path,'test'+BPE_dataset_suffix+'.'+target_language))
print('The size of Test Source and Test Target is: {},{}'.format(len(test_source),len(test_target)))

#Load the original test dataset
test_ori_source = load_data(os.path.join(data_path,'test'+dataset_suffix+'.'+source_language))
test_ori_target = load_data(os.path.join(data_path,'test'+dataset_suffix+'.'+target_language))

#Create the paired test_data
test_data = [[x.strip(),y.strip()] for x,y in zip(test_source,test_target)]

#Creating List of pairs in the format of [[en_1,de_1], [en_2, de_2], ....[en_3, de_3]] for original data
test_ori_data = [[x.strip(),y.strip()] for x,y in zip(test_ori_source,test_ori_target)]

#Filter the data
test_data = data_filter(test_data,MAX_LENGTH)

#Filter the original data
test_ori_data = data_filter(test_ori_data,MAX_LENGTH)

print("The size of Test Data after filtering: {}".format(len(test_data)))


#Load the Vocabulary File and Create Word2Id and Id2Word dictionaries for translation
vocab_source = load_data(os.path.join(data_path,'vocab.'+source_language))
vocab_target = load_data(os.path.join(data_path,'vocab.'+target_language))

#Construct the source_word2id, source_id2word, target_word2id, target_id2word dictionaries
s_word2id, s_id2word = construct_vocab_dic(vocab_source)
t_word2id, t_id2word = construct_vocab_dic(vocab_target)

print("The vocabulary size for source language: {}".format(len(s_word2id)))
print("The vocabulary size for target language: {}".format(len(t_word2id)))

#Generate Train, Val and Test Indexes pairs
test_data_index = create_data_index(test_data,s_word2id,t_word2id)

test_y_ref = [[d[1].split()] for d in test_ori_data]
test_y_ref_meteor = dict((key,[value[1]]) for key,value in enumerate(test_ori_data)) # Define the test data

#Load the vision features
test_im_feats = np.load(os.path.join(data_path,'test'+dataset_im_suffix+'.npy'))

#Load the model
best_model = torch.load(model_path)

if use_cuda:
    best_model.cuda()

#Convert best_model to eval phase
best_model.eval()

test_translations = []
for test_x,test_y,test_im,test_x_lengths,test_y_lengths,test_sorted_index in data_generator_mtv(test_data_index,test_im_feats,eval_batch_size):
    test_translation = best_model.beamsearch_decode(test_x,test_x_lengths,test_im,beam_size,MAX_LENGTH)

    #Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation,test_sorted_index,t_id2word) 
    test_translations += test_translation_reorder

#Evaluate the Image Retrieval Results
test_sample_size = len(test_data_index)
lim,ltxt = torch.FloatTensor(test_sample_size,best_model.shared_embedding_size),torch.FloatTensor(test_sample_size,best_model.shared_embedding_size)
if use_cuda:
    lim = lim.cuda()
    ltxt = ltxt.cuda()

#Start to generate corresponding im and text features
for test_x,test_y,test_im,test_x_lengths,index_retrieval in data_generator_tl_mtv_imretrieval(test_data_index,test_im_feats,batch_size):
    index_reorder = [int(x) for x in index_retrieval]
    test_im_vecs, test_txt_vecs = best_model.embed_sent_im_test(test_x,test_x_lengths,test_im, max_length=MAX_LENGTH)
    #Update the Two Matrix
    lim[index_reorder] = test_im_vecs
    ltxt[index_reorder] = test_txt_vecs

test_r1,test_r5,test_r10,test_medr = im_retrieval_eval.t2i(lim,ltxt)
#Generate the test results. 
print("Image Retrieval Accuracy with the trained multimodal model is:")
print("r1: {}, r5: {}, r10: {}".format(test_r1, test_r5, test_r10))

#Compute the test bleu score and test meteor score
test_bleu = compute_bleu(test_y_ref,test_translations)
#Compute the METEOR Score
test_translations_meteor = dict((key,[' '.join(value)]) for key,value in enumerate(test_translations))
test_meteor = Meteor_Scorer.compute_score(test_y_ref_meteor,test_translations_meteor)

print("test bleu from the trained multimodal model: {}".format(test_bleu[0]))
print("test meteor from the trained multimodal model: {}".format(test_meteor[0]))
#Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(output_path,'test_multimodal_model_prediction.'+target_language)

with open(test_prediction_path,'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')
