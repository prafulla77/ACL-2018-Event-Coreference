from parse_stanford import get_data
import collections
import pickle

# Sub-events: sentence ID: [event IDs]
# 2 or more events in sentences
# events have dobj or
with open('../coref_codes/test_data_2017.pkl', 'rb') as fp:
    test_data = pickle.load(fp)
print test_data

import itertools
def _parse_data(fol):
    data_folder = '../data/'+fol+'/2017_out/'
    data = get_data(data_folder)
    all_subevents = {}
    for doc in data:
        if 'DF' in doc: continue
        subevents = list()
        for sent in data[doc]:
            if sent in test_data[doc[:-8]] and len(test_data[doc[:-8]][sent]) > 2:
                toks = []
                for tok in test_data[doc[:-8]][sent]:
                    #print data[doc][sent].tokens[tok].parent_deprel
                    if 'VB' in data[doc][sent].tokens[tok].POS and data[doc][sent].tokens[tok].parent_deprel in ['ccomp', 'conj:and']:
                        toks.append(tok)
                if len(toks) > 1:
                    for elem in toks:
                        subevents.append(sent+'__'+elem)

                #if sent in test_data[doc[:-8]] and tok in test_data[doc[:-8]][sent]:
                    #print sent, tok, data[doc][sent].tokens[tok].POS, data[doc][sent].tokens[tok].parent_deprel, data[doc][sent].tokens[tok].parent_POS, data[doc][sent].tokens[tok].parent_token

        all_subevents[doc[:-8]] = subevents

    with open('file_subevent_2017.pkl', 'wb') as fp:
        pickle.dump(all_subevents, fp)
_parse_data('test')


