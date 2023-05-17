import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.model_selection import StratifiedKFold,KFold
from statistics import pstdev
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import pandas as pd 
import matplotlib.pyplot as plt
from time import perf_counter as pc
from datetime import timedelta
import seaborn as sns
import numpy as np

batch_size=32
nfolds=5
seed=0
nepochs=60

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

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



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

def plot_loss(losses,f):
    title="GCN Loss - Fold %d" %(f)
    epochs = range(1, len(losses) + 1)
    plt.figure()
    plt.plot(epochs, losses, 'b', label="Training loss")
    plt.title(title)
    plt.legend()
    plt.ylabel('cross entropy')
    plt.xlabel('epoch')
    plt.tight_layout()
    path="../results/train_loss_f%d.png" %(f)
    plt.savefig(path,dpi=300)


def train_net(net,trainloader,testloader,f):
    losses=[]    
    y_test=[]
    y_pred=[]
    opt = torch.optim.Adam(net.parameters())
    print("Training...")
    start = pc()
    for epoch in range(nepochs):
        epoch_loss=0
        for iter,(batched_graph, labels) in enumerate(trainloader):
            batched_graph.to(device)
            feats = batched_graph.ndata['feat']
            print(feats.size())
            logits = net(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        losses.append(epoch_loss)
        print('Epoch {}, loss {:.4f}'.format(epoch+1, epoch_loss))

    end = pc()-start
    training_time = timedelta(seconds=end)
    print("Training ended in:", str(training_time))

    path="../models/gcn_f%s" %(f)
    torch.save(net.state_dict(),path)
 
    with torch.no_grad():
        for iter,(batched_graph, labels) in enumerate(testloader):
                batched_graph.to(device)
                feats = batched_graph.ndata['feat']
                aus = labels.tolist()
                y_test+=aus
                # calculate outputs by running images through the network
                outputs = net(batched_graph,feats)
                # the class with the highest energy is what we choose as prediction
                #_, predicted = torch.max(outputs.data, 1)
                softmax = torch.exp(outputs).cpu()
                prob = list(softmax.numpy())
                predictions = np.argmax(prob, axis=1)
                #aus = predicted.tolist()
                aus=list(predictions)
                y_pred+=aus
    plot_loss(losses,f)
    return y_test,y_pred


def plot_cm(cm):
    title="Confusion Matrix"
    path="../results/cm.png"
    labels = ['Grade 1','Grade 2','Grade 3']
    plt.figure(figsize=(4,4))
    plt.suptitle(title)
    ax=sns.heatmap(np.array(cm), annot=True,fmt='g',cmap='Blues',cbar=False,annot_kws={"size": 14})
    ax.set_xticklabels(labels,rotation=45,fontsize=12)
    ax.set_yticklabels(labels,rotation=45,fontsize=12)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.tight_layout()
    plt.savefig(path,dpi=1000)

def compute_class_metrics(y_true,y_pred,f):
    metrics=[]

    acc=accuracy_score(y_true,y_pred)
    print("Accuracy",acc);
    metrics.append(("Accuracy",acc))

    precision=precision_score(y_true,y_pred,average="weighted")
    print("Precision:",precision)
    metrics.append(("Precision",precision))


    recall=recall_score(y_true,y_pred,average="weighted")
    print("Recall:",recall)
    metrics.append(("Recall",recall))

    f1score=f1_score(y_true,y_pred,average="weighted")
    print("F1-Score:",f1score)
    metrics.append(("F1-Score",f1score))

    cm=confusion_matrix(y_true,y_pred)

    metric=[f,acc,precision,recall,f1score]
    return metric,cm


def save_metrics(metrics):
    avg_metrics=[]
    avgs=[]
    sds=[]
    labels=['Accuracy',"Precision","Recall","F1-Score"]
    df=pd.DataFrame(metrics,columns=["Fold","Accuracy","Precision","Recall","F1-Score"])
    df.to_csv("../results/metrics.csv",index=False)

    acc=df['Accuracy'].values
    avg_acc=sum(acc)/nfolds
    sd=pstdev(acc)
    avgs.append(avg_acc)
    sds.append(sd)
    avg_metrics.append(("Accuracy",avg_acc,sd))



    prec=df['Precision'].values
    avg_prec=sum(prec)/nfolds
    sd=pstdev(prec)
    avgs.append(avg_prec)
    sds.append(sd)
    avg_metrics.append(("Precision",avg_prec,sd))

    rec=df['Recall'].values
    avg_rec=sum(rec)/nfolds
    sd=pstdev(rec)
    avgs.append(avg_rec)
    sds.append(sd)
    avg_metrics.append(("Recall",avg_rec,sd))

    f1=df['F1-Score'].values
    avg_f1=sum(f1)/nfolds
    sd=pstdev(f1)
    avgs.append(avg_f1)
    sds.append(sd)
    avg_metrics.append(("F1-Score",avg_f1,sd))

    path="../results/avg_metrics.txt"
    f = open(path,'w')
    print("==========================================================")
    for m in avg_metrics:
        f.write("AVG. %s = %f SD = %f \n" %(m[0],m[1],m[2]))
        print("AVG. %s = %f SD = %f \n" %(m[0],m[1],m[2]))
    f.close()
    print("==========================================================")

    colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
    plt.figure()
    plt.title("Classification metrics - Mean and Standard Deviation")
    plt.bar(labels, avgs, yerr=sds ,align='center', alpha=0.5, color=colors,ecolor='black', capsize=12)
    plt.ylim([0.8,1.0])
    plt.xticks(labels)
    plt.savefig("../results/class_metrics.png",dpi=1000)


def main():
    graphs,paths,labels=load_data()
    print("Number of graphs in the dataset:",len(paths))
    metrics=[]
    cms=[]
    f = 1
    #skf = StratifiedKFold(nfolds, random_state=None, shuffle=False)
    skf = StratifiedKFold(nfolds, random_state=seed, shuffle=True)
    for train_index , test_index in skf.split(paths , labels):
        print("Fold:",f)
        train_loader,test_loader=create_train_test_loader(train_index,test_index,paths,graphs)
        net=Classifier(2048,1024,3)
        np=get_n_params(net)
        print("Number of parameters of the net:",np)
        y_test,y_pred=train_net(net,train_loader,test_loader,f)
        metric,cm=compute_class_metrics(y_test,y_pred,f)
        metrics.append(metric)
        cms.append(cm)
        f+=1

    avg_cm = sum(cms)
    avg_cm = (avg_cm/nfolds)
    plot_cm(avg_cm)
    save_metrics(metrics)


if __name__ == '__main__':
    main()
