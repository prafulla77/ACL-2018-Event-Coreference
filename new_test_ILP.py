import pickle
from pulp import *
import math, numpy as np
clstr_no = 111

def sigmoid(x):
    return 1 / (1 + (math.e ** (-x)))

def _sent_sim_mapping(sent_sim):
    sent_sim_map = {}
    for key in sent_sim:
        temp = key.split('__')
        try: sent_sim_map[temp[1]][temp[0]] = sigmoid(sent_sim[key][0])
        except KeyError:
            sent_sim_map[temp[1]] = {temp[1]:0.99}
            sent_sim_map[temp[1]][temp[0]] = sigmoid(sent_sim[key][0])
    return sent_sim_map

def _return_coref(prob):
    global clstr_no
    inv_map = {}
    cluster_map = {}
    for v in prob.variables():
        temp = v.name.split('_|_')
        if len(temp) == 2 and temp[0] != 'ROOT' and v.varValue == 1:
            if temp[0] not in inv_map and temp[1] not in inv_map:
                clstr_no += 1
                inv_map[temp[0]] = inv_map[temp[1]] = clstr_no
                cluster_map[clstr_no] = set(temp)
            elif temp[0] in inv_map and temp[1] not in inv_map:
                inv_map[temp[1]] = inv_map[temp[0]]
                cluster_map[inv_map[temp[0]]].add(temp[1])
            elif temp[1] in inv_map and temp[0] not in inv_map:
                inv_map[temp[0]] = inv_map[temp[1]]
                cluster_map[inv_map[temp[1]]].add(temp[0])
            elif inv_map[temp[0]] != inv_map[temp[1]]:
                clstr_no += 1
                temp_mems = set(temp)
                x, y = inv_map[temp[0]], inv_map[temp[1]]
                for elem in cluster_map[inv_map[temp[0]]]:
                    temp_mems.add(elem)
                    inv_map[elem] = clstr_no
                for elem in cluster_map[inv_map[temp[1]]]:
                    temp_mems.add(elem)
                    inv_map[elem] = clstr_no
                cluster_map[clstr_no] = temp_mems
                cluster_map.pop(x, None)
                cluster_map.pop(y, None)
    #print cluster_map
    return cluster_map

def _solve_ilp_news(_scores, _sent_sim, subevents, delta_sent_percent):
    SENT_SIM = 1
    STRETCH = 1
    CROSS_CHAIN = 1
    TRANSITIVITY = 1
    STRETCH_TO_SIZE = 1
    SIZE_TO_STRETCH = 0
    SUBEVENT_BASED_CONSTRAINTS = 1
    DISTRIBUTIONAL_CONSTRAINTS = 1

    print "Number of Events: {}".format(len(_scores))
    sent_sim = _sent_sim_mapping(_sent_sim)
    delta_sent = len(sent_sim)/delta_sent_percent
    print "Delta sentences for current document: {}".format(delta_sent)
    prob = LpProblem("Coref", LpMinimize)
    x, stretch = {}, {} # x: LP variable for event pair similarity
    events_in_sent_pair, sentence_event_num = collections.defaultdict(set), {}
    vertices = []
    for key_1 in _scores: vertices.append(key_1)
    N = len(vertices)

    for key_1 in _scores:
        for key_2 in _scores[key_1]:
            x[(key_2, key_1)] = LpVariable('{}_|_{}'.format(key_2, key_1), 0, 1, LpInteger)
            if STRETCH:
                stretch[(key_2, key_1)] = LpVariable('{}_||_{}'.format(key_2, key_1), 0, 1, LpInteger)
            if int(key_1.split('__')[0]) - int(key_2.split('__')[0]) > 0:#delta_sent:
                events_in_sent_pair[(key_2.split('__')[0], key_1.split('__')[0])].add(x[(key_2, key_1)])
        try: sentence_event_num[key_1.split('__')[0]] += 1
        except KeyError: sentence_event_num[key_1.split('__')[0]] = 1
    temp_objective = [-math.log(_scores[elem[1]][elem[0]]) * x[elem] - math.log(1 - _scores[elem[1]][elem[0]]) * (1-x[elem]) for elem in x.keys()]

    if DISTRIBUTIONAL_CONSTRAINTS:
        pos_component = set()
        neg_component = set()
        for key_2,key_1 in x:
            if float(key_1.split('__')[0])/len(sent_sim) < 0.3 and float(key_2.split('__')[0])/len(sent_sim) < 0.3:
                pos_component.add(x[(key_2, key_1)])
            if float(key_1.split('__')[0])/len(sent_sim) > 0.5 and float(key_2.split('__')[0])/len(sent_sim) > 0.5:
                neg_component.add(x[(key_2, key_1)])
        temp_objective += [-2.5*elem for elem in pos_component]
        temp_objective += [2.5*elem for elem in neg_component]

    if SUBEVENT_BASED_CONSTRAINTS:
        for key in x:
            if key[0] in subevents or key[1] in subevents:
                temp_objective.append(10*x[key])

    if SENT_SIM:
        w = {} # w: LP variable for sentence similarity
        for sent in sent_sim:
            for prev_sent in sent_sim[sent]:
                if int(sent)-int(prev_sent) > delta_sent:
                    w[(prev_sent, sent)] = LpVariable('{}_||_{}'.format(prev_sent, sent), 0, 15, LpInteger)
        for prev_sent,sent in w:
            #print sent_sim[sent][prev_sent]
            temp_objective.append(-math.log(sent_sim[sent][prev_sent])*w[(prev_sent,sent)] - math.log(1-sent_sim[sent][prev_sent])*(1-w[(prev_sent,sent)]))

    if STRETCH:
        predecessors_key, successors_key = collections.defaultdict(set), collections.defaultdict(set)
        for pred_key, suc_key in x:
            successors_key[pred_key].add(suc_key)
            predecessors_key[suc_key].add(pred_key)
        predecessors, successors, len_pred_succ = {}, {}, {}
        for key in x:
            predecessors[key] = [x[(elem, key[0])] for elem in predecessors_key[key[0]]]
            successors[key] = [x[(key[1], elem)] for elem in successors_key[key[1]]]
            len_pred_succ[key] = len(predecessors[key]) + len(successors[key])

    if CROSS_CHAIN:
        all_sent_pairs = {}
        for sent in sent_sim:
            for prev_sent in sent_sim[sent]:
                all_sent_pairs[(prev_sent, sent)] = LpVariable('{}_|_{}'.format(prev_sent, sent), 0, 15, LpInteger)
        for prev_sent,sent in all_sent_pairs:
            if (prev_sent, sent) in events_in_sent_pair:
                if len(events_in_sent_pair[(prev_sent, sent)]) > 3:
                    temp_objective.append(-0.5*all_sent_pairs[(prev_sent,sent)])

    if STRETCH_TO_SIZE:
        stretch_to_size_predecessors = collections.defaultdict(set)
        slack_stretch_to_size, size_global = {}, {}
        for key in stretch:
            size_global[key] = LpVariable('{}_|_|_{}'.format(key[0], key[1]), 0, 10, LpInteger)
            slack_stretch_to_size[key] = LpVariable('{}_|_||_|_{}'.format(key[0], key[1]), 0, 1, LpInteger)
            for x_key in x:
                if (key[0] in x_key or key[1] in x_key):
                    stretch_to_size_predecessors[key].add(x[x_key])
            temp_objective.append(-0.5*size_global[key]) #0.0005

    if SIZE_TO_STRETCH:
        size_to_stretch = collections.defaultdict(set)
        slack_size_to_stretch, stretch_global = {}, {}
        for key in stretch:
            stretch_global[key] = LpVariable('{}_|_|_|_{}'.format(key[0], key[1]), 0, None, LpContinuous)
            slack_size_to_stretch[key] = LpVariable('{}_||_||_{}'.format(key[0], key[1]), 0, 1, LpInteger)
        for xs_key in x:
            for ss_key in stretch_global:
                if ss_key[0] in xs_key:
                    size_to_stretch[ss_key].add(x[xs_key])
        for key in stretch_global:
            temp_objective.append(stretch_global[key])

    prob += lpSum(temp_objective)

    #CONSTRAINTS
    if TRANSITIVITY:
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                for k in range(j + 1, N):
                    prob += (1 - x[(vertices[i], vertices[j])] + 1 - x[(vertices[j], vertices[k])] >= 1 - x[(vertices[i], vertices[k])])

    if SENT_SIM:
        for prev_sent, sent in w:
            if (prev_sent, sent) in events_in_sent_pair:
                prob += sum(events_in_sent_pair[(prev_sent, sent)]) >= w[(prev_sent,sent)]

    if CROSS_CHAIN:
        for prev_sent, sent in all_sent_pairs:
            if (prev_sent, sent) in events_in_sent_pair:
                prob += all_sent_pairs[(prev_sent,sent)] == sum(events_in_sent_pair[(prev_sent, sent)])

    if STRETCH:
        for key in stretch:
            prob +=  - lpSum(predecessors[key]) - lpSum(successors[key]) + x[key] - (1+len_pred_succ[key])*stretch[key] <= 0
            prob +=  0 <= len_pred_succ[key] - lpSum(predecessors[key]) - lpSum(successors[key]) + x[key] - (1+len_pred_succ[key])*stretch[key]

    if STRETCH_TO_SIZE:
        for (prev_id, cur_id) in stretch:
            stretch_coef = float(cur_id.split('__')[0])- float(prev_id.split('__')[0])
            prob += 100000*(1-slack_stretch_to_size[(prev_id, cur_id)]) >= stretch_coef*stretch[(prev_id, cur_id)] - (3*len(sent_sim))/4#2*delta_sent
            prob += size_global[(prev_id, cur_id)] - lpSum(stretch_to_size_predecessors[(prev_id, cur_id)]) <= 100000*slack_stretch_to_size[(prev_id, cur_id)]

    if SIZE_TO_STRETCH:
        for (prev_id, cur_id) in stretch:
            stretch_coef = math.log(1/(1.+float(cur_id.split('__')[0])- float(prev_id.split('__')[0])))
            prob += 26000*(1-slack_size_to_stretch[(prev_id, cur_id)]) >= lpSum(size_to_stretch[(prev_id, cur_id)]) - 2
            prob += stretch_global[(prev_id, cur_id)] - stretch_coef*stretch[(prev_id, cur_id)] + 27000*slack_size_to_stretch[(prev_id, cur_id)] >= 0
            #prob += stretch_global[(prev_id, cur_id)] <= 28000*(1-slack_size_to_stretch[(prev_id, cur_id)])

    print "problem created"
    prob.solve()
    print("Status:", LpStatus[prob.status])

    if 0:
        for key in stretch:
            if stretch[key].varValue: print key
        print "---"
        for key in slack_stretch_to_size:
            if not slack_stretch_to_size[key].varValue: print key
        print "---"
        for key in size_global:
            if size_global[key].varValue: print key, size_global[key].varValue
        quit()

    #print "--", [v.varValue for v in stretch.values()]
    #print [v.varValue for v in xs.values()]
    #quit()
    return _return_coref(prob)

def output_to_file(doc, data, cls):
    temp1 = "#BeginOfDocument "+doc+"\n"
    temp2 = "#BeginOfDocument "+doc+"\n"
    for key_1 in data:
        for key_2 in data[key_1]:
            temp2 += "s1\t" + doc + '\t' + key_1+'__'+key_2 + '\t' + str(data[key_1][key_2].CharacterOffsetBegin) + ',' \
                    + str(data[key_1][key_2].CharacterOffsetEnd) + '\t' + data[key_1][key_2].word.encode('utf-8') + '\t' + 'Contact_Contact\tActual\n'
            temp1 += "s1\t" + doc + '\t' + key_1+'__'+key_2 + '\t' + str(data[key_1][key_2].CharacterOffsetBegin) + ',' \
                    + str(data[key_1][key_2].CharacterOffsetEnd) + '\t' + data[key_1][key_2].word + '\t' + data[key_1][key_2].predict_subtype + \
                    '\t' + data[key_1][key_2].predict_realis + '\n'
    for key in cls:
        temp1 += '@Coreference\tC' + str(key) + '\t' +','.join(cls[key])+'\n'
        temp2 += '@Coreference\tC' + str(key) + '\t' +','.join(cls[key])+'\n'
    temp1 += "#EndOfDocument\n"
    temp2 += "#EndOfDocument\n"
    return temp1, temp2

def _get_coref(_scores):
    cls = {}
    cls_no = 1
    inv_cls = {}
    for ev_ind in _scores:
        best_match = ''
        max_score = 0.0
        for prev_id in _scores[ev_ind]:
            score = _scores[ev_ind][prev_id]
            if score > max_score:
                best_match = prev_id
                max_score = score
        if max_score > 0.5:
            cls[inv_cls[best_match]].append(ev_ind)
            inv_cls[ev_ind] = inv_cls[best_match]
        else:
            cls_no += 1
            cls[cls_no] = [ev_ind]
            inv_cls[ev_ind] = cls_no
    return cls

def _test_():
    f1 = open('../output/news/ilp_new.txt', 'w')
    f2 = open('../output/news/ilp_new_.txt', 'w')
    with open('../pairwise_scores/simple_score.pkl', 'rb') as fp:
        scores = pickle.load(fp)
    with open('../data/test_data.pkl', 'rb') as fp:#test_data
        data = pickle.load(fp)
    with open('../data/test_sent_sim.pkl', 'rb') as fp:
        sent_sim = pickle.load(fp)
    with open('../data/file_subevent.pkl', 'rb') as fp:
        subevents = pickle.load(fp)
    for doc in scores:
        # print sent_pairs_doc.keys()
        if 'DF' not in doc:
            predicted_coref_cluster = _solve_ilp_news(scores[doc], sent_sim[doc], subevents[doc], 5) #_get_coref(scores[doc])#_solve_ilp_news(scores[doc], sent_sim[doc], 5) #_solve_ilp_baseline(scores[doc])#
            temp1, temp2 = output_to_file(doc, data[doc], predicted_coref_cluster)
            f1.write(temp1)
            f2.write(temp2)

_test_()