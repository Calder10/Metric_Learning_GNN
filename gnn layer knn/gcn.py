import torch
import random
import numpy as np
import argparse
import os
import pickle
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from time import perf_counter as pc
from datetime import timedelta
import torchextractor as tx

seed=0
minibatchsize=32
epochs=50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using",device)

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GNN, self).__init__()
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fold", type=int,required=True, choices=[1,2,3,4,5], help="Fold")
    args = parser.parse_args()
    f = args.fold
    return f

def collate(batch):
    g = dgl.batch([example[0] for example in batch])
    l = torch.LongTensor([example[1] for example in batch])
    return g, l

def upload_data():
    graphs={}
    path="../graphs"

    for p in sorted(os.listdir(path)):
        data_path=os.path.join(path,p)
        with open(data_path,"rb") as f:
            g=pickle.load(f)
            graphs.update(g)
    
    print("Number of graphs in the dataset:",len(graphs))
    return graphs

def create_train_test_set(graphs,f):
    path="dsfold%d.txt" %(f)
    x_train,y_train=[],[]
    x_test,y_test=[],[]
    paths=[]
    mapping = {'b':0,'m':1}

    db=open(path)

    for row in db.readlines():
        columns = row.split('|')
        imgname = columns[0]
        grp = columns[3].strip()
        s = imgname.split('-')
        label = s[0].split("_")[1].lower()
        g=graphs[imgname]
        l=torch.tensor(mapping[label])
        if grp=="train":
            x_train.append(g)
            y_train.append(l)
        else:
            x_test.append(g)
            y_test.append(l)
            paths.append(imgname)

    print("Number of graphs in the training set:",len(y_train))
    print("Number of graphs in the test set:",len(y_test))

    data=list(zip(x_train,y_train))
    train_loader=DataLoader(data,batch_size=minibatchsize,shuffle=True,collate_fn=collate)

    data=list(zip(x_test,y_test))
    test_loader=DataLoader(data,batch_size=minibatchsize,shuffle=False,collate_fn=collate)

    out_path="../features/paths/path_f%d" %(f)

    with open(out_path,"wb") as f:
        pickle.dump(paths,f)

    return train_loader,test_loader

def train_net(net,train_loader,f):
    opt = torch.optim.Adam(net.parameters())
    start = pc()
    for epoch in range(epochs):
        epoch_loss=0
        for iter,(batched_graph, labels) in enumerate(train_loader):
            batched_graph.to(device)
            feats = batched_graph.ndata['feat']
            logits = net(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch+1, epoch_loss))

    end = pc()-start
    training_time = timedelta(seconds=end)
    print("Training ended in:", str(training_time))

    path="../models/gcn_f%s" %(f)
    torch.save(net.state_dict(),path)
    return net

def get_layer_feature(net,train_loader,test_loader,f):
    features_train={
        "conv1":[],
        "conv2":[],
        "conv3":[],
        "conv4":[],
        "conv5":[],
        "conv6":[]
    }

    features_test={
        "conv1":[],
        "conv2":[],
        "conv3":[],
        "conv4":[],
        "conv5":[],
        "conv6":[]
    }

    labels_train,labels_test=[],[]

    layer_names=tx.list_module_names(net)
    layer_names.remove("")
    layer_names.remove("classify")
    net = tx.Extractor(net, layer_names)

    print("Extracting the features....")

    for _,(bg,labels) in enumerate(train_loader):
        graphs=dgl.unbatch(bg)
        for g,l in zip(graphs,labels):
            feats=g.ndata['feat']
            model_output, features = net(g,feats)
            labels_train.append(int(l))
            for k,v in features.items():
                v=v.detach().numpy()
                readout=np.mean(v,axis=0)
                features_train[k].append(readout)
    
    labels_train=np.array(labels_train)
    for k,v in features_train.items():
        v=np.array(v)
        out_path="../features/%d/train/%s.npz" %(f,k)
        np.savez_compressed(out_path,x=v,y=labels_train)


    for _,(bg,labels) in enumerate(test_loader):
        graphs=dgl.unbatch(bg)
        for g,l in zip(graphs,labels):
            feats=g.ndata['feat']
            model_output, features = net(g,feats)
            labels_test.append(int(l))
            for k,v in features.items():
                v=v.detach().numpy()
                readout=np.mean(v,axis=0)
                features_test[k].append(readout)
    
    labels_test=np.array(labels_test)
    for k,v in features_test.items():
        v=np.array(v)
        out_path="../features/%d/test/%s.npz" %(f,k)
        np.savez_compressed(out_path,x=v,y=labels_test)

    print("Done!")

    
def main():
    set_seed(seed)
    f=parse_arguments()
    print("=================================================================")
    print("FOLD",f)
    graphs=upload_data()
    train_loader,test_loader=create_train_test_set(graphs,f)
    net=GNN(2048,1024,2)
    summary(net)
    net=train_net(net,train_loader,f)
    get_layer_feature(net,train_loader,test_loader,f)
    print("=================================================================")



if __name__=="__main__":
    main()