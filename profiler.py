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
print degrees.most_common()
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
print attr_type.head(5)

print 'loading node attributes...'
node_attr = pd.read_csv(node_attr_file, index_col=None, header=None, usecols=[0, 1], delimiter=' ', names=['node', 'attr'])
print node_attr.head(5)


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
feature_names = vectorizer.get_feature_names()
feature_names.remove('unknown-0')
feature_prefixes = Counter([f.split('-')[0] for f in feature_names])
print feature_prefixes.most_common()
user_ids = grouped.index.values
assert set([feature_names.index(a[0]) for a in grouped[5] if a[0]!='unknown-0']) == set(X[0].nonzero()[1].tolist())
############################################# k-fold corss validation ##############################
A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
kf = KFold(len(user_ids), 2, shuffle=True, random_state=0)
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    train_ids, test_ids = user_ids[train_index], user_ids[test_index]
    
pdb.set_trace()

