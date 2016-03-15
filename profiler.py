'''
Created on 7 Mar 2016

@author: af
'''
import zipfile
from os import path
import pandas as pd
import pdb
import networkx as nx
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
import scipy as sp
from scipy.sparse import csr_matrix, vstack
import numpy as np
import embedding_optimiser as op
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

data_home = '/home/arahimi/datasets/gplus/imc12'
graph_file = path.join(data_home, 'small-graph.txt.gz') 
node_attr_file = path.join(data_home, 'small-node_attri.txt')
attr_types_file = path.join(data_home, 'attri_type.txt')

##############################################  graph ############################################
G = nx.read_edgelist(graph_file, nodetype=int, create_using=nx.Graph())
node_degree = G.degree()
print '#nodes', len(node_degree)
degree_1 = [n for n, deg in node_degree.iteritems() if deg == 1]
print '#nodes with degree 1', len(degree_1)
G.remove_nodes_from(degree_1)
node_degree = G.degree()
degree_0 = [n for n, deg in node_degree.iteritems() if deg == 0]
G.remove_nodes_from(degree_0)
print '#nodes after removing degree 1 and 0: ', nx.number_of_nodes(G), '#edges', nx.number_of_edges(G)
degrees = Counter(G.degree().values())
#print degrees.most_common()
############################################## create small dataset ##############################3
create_small_node_attr = False
if create_small_node_attr:
    nodes = set(G.nodes())
    with open(node_attr_file + '.small', 'w') as outf:
        with open(node_attr_file, 'r') as inf:
            for l in inf:
                fields = l.strip().split()
                if int(fields[0]) in nodes:
                    outf.write(l.strip() + '\n')

###############################################   attributes #####################################
print 'loading attri types...'
attr_type = pd.read_csv(attr_types_file, index_col=None, header=None, usecols=[0, 1], delimiter=' ', names=['attr', 'attr_name'])
#print attr_type.head(5)

print 'loading node attributes...'
node_attr = pd.read_csv(node_attr_file, index_col=None, header=None, usecols=[0, 1], delimiter=' ', names=['node', 'attr'])
#print node_attr.head(5)


print 'merging attributes and attribute types...'
n_a = pd.merge(node_attr, attr_type, on='attr', how='inner')
n_a['a'] = n_a['attr_name'] + n_a['attr'].map(str)
n_a.drop(['attr', 'attr_name'], 1, inplace=True)
n_a.columns = ['node', 'attr']

#add a fake label to all nodes for simplicity during vectorization
unknown_records = [[n, 'unknown-0'] for n in G.nodes()]
unknown_df = pd.DataFrame(unknown_records, columns=['node', 'attr'])
n_a = n_a.append(unknown_df, ignore_index=True)
n_a.head()
grouped = n_a.groupby('node').attr.apply(lambda lst: tuple((k, 1) for k in lst))
grouped.head()
category_dicts = [dict(tuples) for tuples in grouped]
vectorizer = DictVectorizer(sparse=True)
#remove the fake label from X vectors and feature_names
X = vectorizer.fit_transform(category_dicts)
X = X[:, :-1]
label_names = vectorizer.get_feature_names()
label_names.remove('unknown-0')
label_types = Counter([f.split('-')[0] for f in label_names])
print label_types.most_common()
user_ids = grouped.index.values
assert set([label_names.index(a[0]) for a in grouped.values[0] if a[0]!='unknown-0']) == set(X[0].nonzero()[1].tolist())
label_slices = []
start_index = 0
for label_type in sorted(label_types):
    end_index = start_index + label_types[label_type]
    label_slices.append([start_index, end_index])
    start_index = end_index
A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
A = A.tocsr()

print 'adjacancy matrix', A.shape
print 'label matrix', X.shape
    
############################################# k-fold corss validation ##############################
kf = KFold(len(user_ids), 2, shuffle=True, random_state=0)
results = []
for train_original_indices, test_original_indices in kf:
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
    #X_output = op.edgexplain_profiler(X_input, training_indices, A, label_slices=label_slices, iterations=10, learning_rate=0.05, alpha=10, C=0, lambda1=0.1, text_dimensions=None)
    X_output = op.iterative_profiler(X_input, G, train_ids.tolist(), test_ids.tolist(), id_random_index, label_slices=label_slices, preserve_coef=0.9, iterations=5, 
                                     alpha=2, c=0, node_order='random', keep_topK=0.1, edgexplain__scaler=False)
    

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
        acc = float(correct) / total
        print label_type, total, correct, acc
        accuracies[label_type] = acc
    results.append(accuracies)
    print accuracies

for lt in results[0].keys():
    avg = sum([i[lt] for i in results]) / len(results)
    print lt, avg

