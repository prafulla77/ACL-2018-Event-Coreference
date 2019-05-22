from parse_stanford import get_data
import pickle, numpy as np
from keras.models import load_model
import os
from collections import OrderedDict as OD
import tensorflow as tf
tf.python.control_flow_ops = tf

NER_map = {'PERSON':0, 'LOCATION':1, 'ORGANIZATION':2, 'MISC':3,'MONEY':4, 'NUMBER':5, 'ORDINAL':6, 'PERCENT':7}
NER_map_inv = {0:'P', 1:'L', 2:'ORG', 3:'MI', 4:'MO', 5:'N', 6:'ORD', 7:'%'}

TYPE_map = {0:'Conflict_Attack',1:'Conflict_Demonstrate',2:'Contact_Meet',3:'Contact_Correspondence',
            4:'Contact_Broadcast',5:'Contact_Contact',6:'Life_Injure',7:'Life_Die',8:'Justice_Arrest-Jail',
            9:'Manufacture_Artifact',11:'Movement_Transport-Artifact',10:'Movement_Transport-Person',
            12:'Personnel_Elect',13:'Personnel_End-Position',14:'Personnel_Start-Position',15:'Transaction_Transfer-Money',
            16:'Transaction_Transfer-Ownership',17:'Transaction_Transaction'}

with open('../vocab/2015_eval_.pkl', 'rb') as fp:
    word_vecs = pickle.load(fp)
with open('../vocab/POS.pkl', 'rb') as fp:
    pos_vecs = pickle.load(fp)
with open('../vocab/deprel.pkl', 'rb') as fp:
    dep_vecs = pickle.load(fp)

word_vecs['PADDING'] = [0.0]*300
word_vecs['UNKNOWN_MY'] = [0.50]*300

def _get_prefix_sufix(word):
    suffix = ['te', 'tor', 'or', 'ing', 'cy', 'id', 'ed', 'en', 'er', 'ee', 'pt', 'de', 'on', 'ion', 'tion', 'ation',
              'ction', 'de', 've', 'ive', 'ce', 'se', 'ty', 'al', 'ar', 'ge', 'nd', 'ize', 'ze', 'it', 'lt'] #31
    prefix = ['re', 'in', 'at', 'tr', 'op'] #5
    ans = []
    for suf in suffix:
        if word[-len(suf):] == suf: ans.append(1.0)
        else: ans.append(0.0)
    for pref in prefix:
        if word[:len(pref)] == pref: ans.append(1.0)
        else: ans.append(0.0)
    return ans

def _get_joint(data):
    data_x = []
    for filename in data:
        doc = data[filename]
        for sent_no in doc:
            for token_no in doc[sent_no].tokens:
                feat_temp = _get_prefix_sufix(doc[sent_no].tokens[token_no].word.lower())
                try: feat_temp += word_vecs[doc[sent_no].tokens[token_no].word.lower()]
                except KeyError: feat_temp += word_vecs['UNKNOWN_MY']
                try: feat_temp += word_vecs[doc[sent_no].tokens[token_no].lemma.lower()]
                except KeyError: feat_temp += word_vecs['UNKNOWN_MY']

                child_feats_temp = [0.0]*208
                for child in doc[sent_no].tokens[token_no].children_deprel:
                    try: child_feats_temp[max( (v, i) for i, v in enumerate(dep_vecs[child]))[1]] += 1.0
                    except KeyError: pass
                feat_temp += child_feats_temp

                child_feats_temp = [0.0] * 8 #PERSON, LOCATION, ORGANIZATION, MISC, MONEY, NUMBER, ORDINAL, PERCENT
                for child in doc[sent_no].tokens[token_no].children_token_nos:
                    if doc[sent_no].tokens[child].NER in NER_map:
                        child_feats_temp[NER_map[doc[sent_no].tokens[child].NER]] += 1.0
                feat_temp += child_feats_temp

                # NER related features
                data_x.append(feat_temp)
    print len(data_x)
    return data_x

def _get_joint_test(data):
    data_x = []
    all_data = []
    for filename in data:
        doc = data[filename]
        for sent_no in doc:
            sent_temp = ['PADDING', 'PADDING']
            for token_no in doc[sent_no].tokens:
                all_data.append(doc[sent_no].tokens[token_no])
                sent_temp += [doc[sent_no].tokens[token_no]]
            sent_temp += ['PADDING', 'PADDING']
            for i in range(2, len(sent_temp) - 2):
                feat_temp = _get_prefix_sufix(sent_temp[i].word.lower()) #[]

                for j in range(-2, 3, 1):
                    try: feat_temp += pos_vecs[sent_temp[i + j].POS] + dep_vecs[sent_temp[i + j].parent_deprel]
                    except KeyError: feat_temp += pos_vecs[sent_temp[i + j].POS] + dep_vecs['UNKNOWN']
                    except AttributeError: feat_temp += pos_vecs[sent_temp[i + j]] + dep_vecs[sent_temp[i + j]]

                if 1:
                    try: feat_temp += list(np.array(word_vecs[sent_temp[i].word.lower()]) - np.array(word_vecs[sent_temp[i].lemma.lower()]))
                    except KeyError: feat_temp += word_vecs['UNKNOWN_MY']
                if 1:
                    try: feat_temp += word_vecs[sent_temp[i].lemma.lower()]
                    except KeyError: feat_temp += word_vecs['UNKNOWN_MY']

                try: feat_temp += pos_vecs[sent_temp[i].POS] +  pos_vecs[sent_temp[i].parent_POS] +  dep_vecs[sent_temp[i].parent_deprel]
                except KeyError:
                    if sent_temp[i].parent_POS: feat_temp += pos_vecs[sent_temp[i].POS] +  pos_vecs[sent_temp[i].parent_POS] +  dep_vecs['UNKNOWN']
                    else: feat_temp += pos_vecs[sent_temp[i].POS] +  pos_vecs['ROOT'] +  dep_vecs['UNKNOWN']

                child_feats_temp = [0.0]*208
                for child in sent_temp[i].children_deprel:
                    try: child_feats_temp[max( (v, i) for i, v in enumerate(dep_vecs[child]))[1]] += 1.0
                    except KeyError: pass
                feat_temp += child_feats_temp

                child_feats_temp = [0.0]*47
                for child in sent_temp[i].children_POS:
                    child_feats_temp[max( (v, i) for i, v in enumerate(pos_vecs[child]))[1]] += 1.0
                feat_temp += child_feats_temp

                data_x.append(feat_temp)
    return data_x, all_data

def _parse_data():
    train_data = OD()
    test_data = OD()
    """
    tag_folder = '../data/train/event_tags/'
    data_folder = '../data/train/stanford_parse/'
    data = get_data(tag_folder, data_folder)
    for doc in data:
        train_data[doc] = OD()
        event_num = 0
        for sent in data[doc]:
            train_data[doc][sent] = OD()
            for tok in data[doc][sent].tokens:
                if data[doc][sent].tokens[tok].coref_id:
                    event_num += 1
                    train_data[doc][sent][tok] = data[doc][sent].tokens[tok]
                    train_data[doc][sent][tok].event_no = event_num
    """

    #tag_folder = '../data/test/event_tags/'
    data_folder = '../data/test/2015_out/'
    data = get_data(data_folder)

    models = os.listdir('type_models/')
    type_prediction = []
    type_test_x = _get_joint(data)
    for model in models:
        if ".DS" not in model:
            print model
            my_model = load_model('type_models/' + model)
            type_prediction.append(my_model.predict(np.array(type_test_x)))
    type_predict = type_prediction[0]
    for pred in type_prediction[1:]:
        type_predict += pred

    models = os.listdir('realis_models/')
    prediction = []
    test_x, all_data = _get_joint_test(data)

    for model in models:
        if ".DS" not in model:
            print model
            my_model = load_model('realis_models/' + model)
            prediction.append(my_model.predict(np.array(test_x)))

    predicted = prediction[0]
    for pred in prediction[1:]:
        predicted += pred

    # test data generation
    file = ''
    event_num = 0
    for i in range(len(all_data)):
        if type_predict[i][18] > 2.95: continue
        else: type_predict[i][18] = 0.0
        if all_data[i].filename[:-4] != file:
            file = all_data[i].filename[:-4]
            event_num = 0
            test_data[file] = OD()
        if predicted[i][-1] < 6.1:
            predicted[i][-1] = 0
        ind = max((v, i) for i, v in enumerate(predicted[i]))[1]
        type_ind = max((v, i) for i, v in enumerate(type_predict[i]))[1]
        if ind == 0:
            all_data[i].predict_realis = 'Actual'
        if ind == 1:
            all_data[i].predict_realis = 'Generic'
        if ind == 2:
            all_data[i].predict_realis = 'Other'
        if ind < 3:
            event_num += 1
            all_data[i].event_no = event_num
            all_data[i].predict_subtype = TYPE_map[type_ind]
            try: test_data[file][all_data[i].sent_no][all_data[i].token_no] = all_data[i]
            except KeyError:
                test_data[file][all_data[i].sent_no] = OD()
                test_data[file][all_data[i].sent_no][all_data[i].token_no] = all_data[i]
    return train_data, test_data

train_data, test_data = _parse_data()
print test_data

#with open('../coref_codes/training_data.pkl', 'wb') as fp:
    #pickle.dump(train_data, fp)
with open('../coref_codes/test_data_2015.pkl', 'wb') as fp:
    pickle.dump(test_data, fp)
