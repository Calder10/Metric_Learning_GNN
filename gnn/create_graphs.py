import os
import pickle
import torch
import dgl
from img2ragdeep import create_graph
from time import perf_counter as pc
from datetime import timedelta
import numpy as np


def collate(batch):
    g = dgl.batch([example[0] for example in batch])
    l = torch.LongTensor([example[1] for example in batch])
    return g, l


def create_graphs():
    graphs={}
    failed=[]
    measures=[]
    print("Creating the graph starting from the tissue...")
    path="../Dataset/Grade_1"
    start = pc()
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        print("Processing image:",img)
        try:
            g=create_graph(img_path,300,20,1,"G1",img,96)
            #measures.append(stats)
            if img not in graphs:
                graphs[img]=(g,0)
        except:
            print("Failed,",img)
            failed.append(img)
            continue
        
    path="../Dataset/Grade_2"
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        print("Processing image:",img)
        try:
            g=create_graph(img_path,300,20,1,"G2",img,96)
            #measures.append(stats)
            if img not in graphs:
                graphs[img]=(g,1)
        except:
            print("Failed,",img)
            failed.append(img)
            continue

    path="../Dataset/Grade_3"
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        print("Processing image:",img)
        try:
            g=create_graph(img_path,300,20,1,"G3",img,96)
            #measures.append(stats)
            if img not in graphs:
                graphs[img]=(g,2)
        except:
            print("Failed,",img)
            failed.append(img)
            continue
    
    end = pc()-start
    time = timedelta(seconds=end)
    print("Ended in:", str(time))

    with open("../graphs/graphs", 'wb') as f:
        pickle.dump(graphs, f)
    print(len(graphs))
    print("Failed: \n",failed);
    print("Done!")

    #measures=np.array(measures)
    #m=measures.mean(axis=0)
    #return m

def save_metrics(metrics):
    names=[
        "Number of nodes",
        "Number of edges",
        "Degree",
        "Node betweenness",
        "Edge betweenness",
        "Average short path legth",
        "Local cluster coefficient",
        "Global cluster coefficient"
    ]

    path="../results/graphs_stats.txt"
    f = open(path,'w')

    for n,m in zip(names,metrics):
        f.write("%s: %s \n" %(n,m))
    f.close()

def main():
    create_graphs()
    #save_metrics(list(m))
    

if __name__ == '__main__':
    main()
