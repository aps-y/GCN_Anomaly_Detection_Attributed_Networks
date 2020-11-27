# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import networkx as nx
import pandas as pd
import numpy as np
import scipy.io
import os
import re
import matplotlib.pyplot as plt
import random
from sklearn import metrics


# %%
name_list=[]
for i in os.listdir():
    if len(i.split('.'))>1 and i.split('.')[1]=='feat':
        name_list.append(i.split('.')[0])
name_list=set(name_list)
name_list=list(name_list)
name_list


# %%
num_lines=0
for i in name_list:
    num_lines += sum(1 for line in open(i+".edges"))
#print("#edges combining all egonets :",num_lines)
#print("#edges in combined file :",sum(1 for line in open("facebook_combined.txt")))


# %%
S=set()
for line in open("0.featnames"):
    S.add(' '.join(line.split(' ')[1:]))
for name in name_list:
    T=set()
    for line in open(name+".featnames"):
        T.add(' '.join(line.split(' ')[1:]))
    S=S.intersection(T)
print("Common features from all egonnets ")
common_features=list(S)
common_features


# %%
Mem={}
def extract_feature_number(name,feature):
    if Mem.get((name,feature)!=None):
        return Mem.get((name,feature))
    line=re.findall("[0-9]* "+feature,open(name+".featnames").read())[0]
    Mem[(name,feature)]=int(line.split(' ')[0])
    return Mem[(name,feature)]

def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr

extract_feature_number('0',common_features[0])


# %%
features=[]
for name in name_list:
    for line in open(name+".feat"):
        t=line[:-1].split(' ')
        t.insert(0,name)
        features.append(t)


# %%
l=[i[1] for i in features]
print("total nodes :",len(l))
print("total distinct nodes :",len(list(set(l))))
l=[int(i) for i in l]
missing=[]
for i in range(4039):
    if i not in l:
        missing.append(i)
print("missing numbers :",missing)
print("max node number : ",max(l))


# %%
A={}
for i in missing:
        A[i]=[0.0 for k in range(len(common_features))]
for i in features:
    name=i[0]
    ind=int(i[1])
    temp=[]
    for feat in common_features:
        f_ind=extract_feature_number(name,feat)+2
        temp.append(float(i[f_ind]))
        if int(i[f_ind])!=0 and int(i[f_ind])!=1:
            print("Error at ",ind)
    A[ind]=temp
print("total nodes in A :",len(A.keys()))


# %%
G = nx.Graph()
G.add_nodes_from([i for i in range(4039)])
edge_df=pd.read_csv("facebook_combined.txt",sep=' ',header=None)
for i in range(len(edge_df)):
    G.add_edge(edge_df.loc[i,0], edge_df.loc[i,1])
    G.add_edge(edge_df.loc[i,1], edge_df.loc[i,0])
Adj=nx.adjacency_matrix(G)
Adj=Adj.toarray()
Adj=Adj.astype('float')
print(np.shape(Adj))
Adj


# %%
nodes=[i for i in range(4039)]
nodes=random.choices(nodes,k=300)


# %%
for i in range(15):
    clique=nodes[i*15:i*15+10]
    for node1 in clique:
        for node2 in clique:
            if node1!=node2:
                G.add_edge(node1,node2)


# %%
node_list=[k for k in range(4039)]

for i in nodes[150:300]:
    random_set=random.choices(node_list,k=50)
    node_J=-1
    node_dist=0
    for d in random_set:
        dist=np.linalg.norm(np.array(A[i]) - np.array(A[d]))
        if dist>node_dist:
            node_dist=dist
            node_J=d
    A[i]=A[node_J]


# %%
Adj=nx.adjacency_matrix(G)
Adj=Adj.toarray()
Adj=Adj.astype('float')
print(np.shape(Adj))
Adj


# %%
#final Atribute matrix
Arr=[]
for i in range(len(A.keys())):
    Arr.append(A[i])
Arr=np.array(Arr)
print(np.shape(Arr))
Arr


# %%
temp=[0 for i in range(4039)]
print(len(nodes))
for i in nodes:
    temp[i]=1
for i in range(len(temp)):
    temp[i]=[temp[i]]
gnd=np.array(temp)
gnd=gnd.astype('float')
print(np.shape(gnd))
gnd


# %%
scipy.io.savemat("./matfiles/"+"facebook"+'.mat', mdict={'X':Arr,'A':Adj,'gnd':gnd})


# %%
np.save("anomalous_nodes", nodes)

# %% [markdown]
# # Rough Work

# %%
labels=[]
with open('facebook-ranking.txt') as fb:
    a=fb.readlines()
    labels=[int(i[0]) for i in a]


# %%
detected_anomalies=np.array(labels)


# %%
skplt.metrics.plot_roc_curve(gnd, detected_anomalies)


# %%
fpr=[1,2,3,3,4]
tpr=[12,1,22,1,21]

import matplotlib.pyplot as plt

plt.plot(fpr,tpr)


# %%



