import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.model_selection import StratifiedKFold
from statistics import pstdev
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import pandas as pd 
import matplotlib.pyplot as plt
from time import perf_counter as pc
from datetime import timedelta
import seaborn as sns
import numpy as np
import umap

batch_size=32
nfolds=5
seed=0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using",device)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim,allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_dim, 512,allow_zero_in_degree=True)
        self.conv3 = dglnn.GraphConv(512, 256,allow_zero_in_degree=True)
        self.conv4 = dglnn.GraphConv(256, 128,allow_zero_in_degree=True)
        self.conv5 = dglnn.GraphConv(128, 64,allow_zero_in_degree=True)
        self.conv6 = dglnn.GraphConv(64,32,allow_zero_in_degree=True)
        self.classify = nn.Linear(32, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        h = F.relu(self.conv5(g, h))
        h = F.relu(self.conv6(g, h))

        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)


def collate(batch):
    g = dgl.batch([example[0] for example in batch])
    l = torch.LongTensor([example[1] for example in batch])
    return g, l



def load_data():
    print("Data upload...")
    with open("../graphs/graphs_no_dup", 'rb') as f:
       graphs=pickle.load(f)

    labels=[]

    for _,value in graphs.items():
        labels.append(value[1])

    paths=list(graphs.keys())

    return graphs,paths,labels

def create_train_test_loader(train_index,test_index,paths,graphs):
    x_train,y_train,x_test,y_test=[],[],[],[]
    for i in train_index:
        p=paths[i]
        x_train.append(graphs[p][0])
        y_train.append(torch.tensor(graphs[p][1]))
    
    data = list(zip(x_train,y_train))
    train_loader = DataLoader(data, batch_size, shuffle=True,collate_fn=collate)

    for i in test_index:
        p=paths[i]
        x_test.append(graphs[p][0])
        y_test.append(torch.tensor(graphs[p][1]))   

    data = list(zip(x_test,y_test))
    test_loader = DataLoader(data, batch_size, shuffle=False,collate_fn=collate)  

    return train_loader,test_loader

def load_model(f):
    path="../models/gcn_f%s" %(f)
    model = Classifier(2048,1024,3)
    model.load_state_dict(torch.load(path))
    model.classify=nn.Identity()
    return model

def find_embeddings(model,train_loader,test_loader):            
    x_train,x_test,y_train,y_test=[],[],[],[]
    model.eval()
    with torch.no_grad():
        for iter,(batched_graph, labels) in enumerate(train_loader):
            batched_graph.to(device)
            feats = batched_graph.ndata['feat']
            aus = labels.tolist()
            y_train+=aus
            outputs = model(batched_graph,feats)
            outputs=outputs.numpy()
            aus=list(outputs)
            x_train+=aus
    
    with torch.no_grad():
        for iter,(batched_graph, labels) in enumerate(test_loader):
            batched_graph.to(device)
            feats = batched_graph.ndata['feat']
            aus = labels.tolist()
            y_test+=aus
            outputs = model(batched_graph,feats)
            outputs=outputs.numpy()
            aus=list(outputs)
            x_test+=aus
    
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)

    return x_train,y_train,x_test,y_test

def plot_embedding(x_train,y_train,x_test,y_test,f):
    mapping={0: "Grade 1",1:"Grade 2",2:"Grade 3"}
    order=["Grade 1","Grade 2","Grade 3"]
    aus = y_train.tolist()
    y_train = list(map(mapping.get, aus))
    y_train = np.asarray(y_train)

    aus = y_test.tolist()
    y_test = list(map(mapping.get, aus))
    y_test = np.asarray(y_test)

    embedder = umap.UMAP(random_state=42)
    emb_train = embedder.fit_transform(x_train)
    emb_test = embedder.transform(x_test)

    title = "Fold %s - Agios Pavlos Dataset" %(f)
    path="../results/emb_vis_f%s.png" %(f)

    plt.figure()
    plt.suptitle(title)
    plt.subplot(211)
    plt.title("Training set")
    sns.scatterplot(x=emb_train[:,0],y=emb_train[:,1],marker='o',palette='Set1',hue=y_train,hue_order=order,s=40,alpha=0.5)
    plt.legend()
    plt.subplot(212)
    plt.title("Test set")
    sns.scatterplot(x=emb_test[:,0],y=emb_test[:,1],hue=y_test,marker='o',palette='Set1',hue_order=order,s=40,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path,dpi=1000)
            

def main():
    graphs,paths,labels=load_data()
    f = 1
    skf = StratifiedKFold(nfolds,random_state=seed,shuffle=True)
    for train_index , test_index in skf.split(paths , labels):
        print("Fold:",f)
        train_loader,test_loader=create_train_test_loader(train_index,test_index,paths,graphs)
        model=load_model(f)
        x_train,y_train,x_test,y_test=find_embeddings(model,train_loader,test_loader)
        plot_embedding(x_train,y_train,x_test,y_test,f)
        f+=1
    print("Done!")

if __name__ == "__main__":
    main()
