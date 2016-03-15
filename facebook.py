'''
Created on 13 Mar 2016

@author: af
'''
import pdb
from os import path
import glob
import codecs
from collections import defaultdict, Counter
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import vstack, csr_matrix
import networkx as nx
import numpy as np
import logging
import embedding_optimiser as op
from sklearn.cross_validation import KFold
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

data_home = '/home/arahimi/datasets/ego-networks/facebook/'
features_home = path.join(data_home, 'facebook')
graph_file = path.join(data_home, 'facebook_combined.txt')
ego_id_feature = {}
features_to_exclude = set(['concentration', 'year', 'gender', 'degree', 'work_position', 'work_location', 'birthday', 'date', 'projects_', 'with_', '_from', 'religion', 'political', 'languages', 'locale', 'name', 'education_type', 'education_classes'])

def exclude_features(features):
    for feature in features.keys():
        for f in features_to_exclude:
            if f in feature:
                del features[feature]
    return features
#####################################  read feature types and indices for each eago circle #################################
featnames_files = glob.glob(features_home + '/*.featnames') 
for f in featnames_files:
    egoid = int(path.basename(f).split('.')[0])
    id_feature = {}
    feature_id = {}
    with codecs.open(f, 'r', encoding='utf-8') as inf:
        for line in inf:
            line = line.strip()
            fields = line.split()
            id = int(fields[0])
            feature_num = int(fields[-1])
            feature = '_'.join(fields[1].split(';')[0:-1])+ '_' + str(feature_num)
            id_feature[id] = feature
            feature_id[feature] = id 
    ego_id_feature[egoid] = (id_feature, feature_id)

#####################################  read main ego features   #################################
egofeat_files = glob.glob(features_home + '/*.egofeat')
egoid_features = {}
for f in egofeat_files:
    egoid = int(path.basename(f).split('.')[0])
    with codecs.open(f, 'r', encoding='utf-8') as inf:
        egofeats = inf.read().strip()
    cells = egofeats.split()
    features = {ego_id_feature[egoid][0][index]:1 for index, value in enumerate(cells) if int(value)==1}
    features = exclude_features(features)
    egoid_features[egoid] = features

#####################################  read friend ego features   ##############################
friendfeat_files = glob.glob(features_home + '/*.feat')
for f in friendfeat_files:
    egoid = int(path.basename(f).split('.')[0])
    with codecs.open(f, 'r', encoding='utf-8') as inf:
        for line in inf:
            line = line.strip()
            cells = line.split()
            friendid = int(cells[0])
            features = {ego_id_feature[egoid][0][index]:1 for index, value in enumerate(cells[1:]) if int(value)==1}
            features = exclude_features(features)
            egoid_features[friendid] = features
#################################### vectorizing features ##################################
vectorizer = DictVectorizer(sparse=True)
user_ids = np.array(sorted(egoid_features))
category_dicts = [egoid_features[id] for id in user_ids.tolist()]
X = vectorizer.fit_transform(category_dicts)
label_names = vectorizer.get_feature_names()
label_types = Counter(['_'.join(f.split('_')[0:-1]) for f in label_names])
print label_types.most_common()

label_slices = []
start_index = 0
for label_type in sorted(label_types):
    end_index = start_index + label_types[label_type]
    label_slices.append([start_index, end_index])
    start_index = end_index
##################################### social graph ####################################
G = nx.read_edgelist(graph_file, nodetype=int, create_using=nx.Graph())
A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
assert sorted(G.nodes())==user_ids.tolist(), 'the node order in adjacency matrix does not match the node order in representation'

print 'adjacency matrix', A.shape
print 'label matrix', X.shape
############################################# k-fold corss validation ##############################
kf = KFold(len(user_ids), 2, shuffle=True, random_state=0)
results = []
for train_original_indices, test_original_indices in kf:
    print 'loooop', len(results) + 1
    X_train, X_test = X[train_original_indices], X[test_original_indices]
    test_random_to_original_index = dict(zip(range(len(train_original_indices), len(train_original_indices) + len(test_original_indices)), test_original_indices.tolist()))
    test_original_to_random_index = {o:r for r, o in test_random_to_original_index.iteritems()}
    X_randomized = vstack([X_train, X_test])
    print 'X_train', X_train.shape, 'X_test', X_test.shape
    train_ids, test_ids = user_ids[train_original_indices], user_ids[test_original_indices]
    id_random_index = dict(zip(train_ids.tolist() + test_ids.tolist(), range(0, len(train_original_indices)) + range(len(train_original_indices), len(train_original_indices) + len(test_original_indices))))
    random_index_id = {index:id for id, index in id_random_index.iteritems()}
    #add empty test matrix to the x_train for prediction
    X_test_empty = csr_matrix((X_test.shape[0], X_test.shape[1]))
    X_input = vstack([X_train, X_test_empty])
    #X_output = vstack([X_train, X_test])
    training_indices = np.array(range(0, X_train.shape[0]))
    #X_output = op.edgexplain_profiler(X_input, training_indices, A, label_slices=label_slices, iterations=20, learning_rate=0.05, alpha=10, C=0, lambda1=0.1, text_dimensions=None)
    X_output = op.iterative_profiler(X_input, G, train_ids.tolist(), test_ids.tolist(), id_random_index, label_slices=label_slices, preserve_coef=0.9, iterations=5, 
                                     alpha=0.01, c=0, node_order='random', keep_topK=0.1, edgexplain__scaler=True)
    

    #print 'first test id', test_ids[0], 'first test index', id_random_index[test_ids[0]], 'nbrs:', G[test_ids[0]], 'known labels\n', X_randomized[id_random_index[test_ids[0]]], '\npredicted:\n', X_output[id_random_index[test_ids[0]]]
    #for nbr in G[test_ids[0]]:
    #    print 'nbr', nbr, 'nbr_index', id_random_index[nbr], 'known\n', X_randomized[id_random_index[nbr]], '\npredicted\n', X_output[id_random_index[nbr]]
    
    print 'evaluating by precision @ K'
    k = 1
    print 'K', k
    accuracies = {}
    for i, label_slice in enumerate(label_slices):
        label_type = sorted(label_types)[i]
        total = 0
        correct = 0
        start, end = label_slice
        for test_original_index in test_original_indices.tolist():
            all_known_labels = category_dicts[test_original_index]
            known_labels = [l for l in all_known_labels if l.startswith(label_type)]
            test_random_index = test_original_to_random_index[test_original_index]
            predicted_label_indices = np.argsort(X_output[test_random_index, start: end].toarray()[0])[-k:][::-1]
            predicted_labels = [label_names[i + start] for i in predicted_label_indices.tolist()]
            if len(known_labels) != 0:
                total += min(k, len(known_labels))
                for l in predicted_labels:
                    if l in known_labels:
                        correct += 1
        if total == 0:
            print 'total equals zero**************'
            acc = 0
        else:
            acc = float(correct) / total
        #print label_type, total, correct, acc
        accuracies[label_type] = acc
    results.append(accuracies)
    print accuracies
print '############################# evaluation #########################' 
for lt in results[0].keys():
    avg = int(100 * sum([i[lt] for i in results]) / len(results))
    print lt, avg