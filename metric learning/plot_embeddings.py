import os
import pickle
from cv2 import resize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import im
from tqdm import tqdm
import umap
from pprint import pprint
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances


def upload_data(train_path,test_path,test_path_p):
    mapping={0: "Benign",1:'Malignant'}
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_test = test_data['arr_0']
    y_test = test_data['arr_1']
    y_train = np.reshape(y_train,-1)
    y_test = np.reshape(y_test,-1)
    aus = y_train.tolist()
    y_train = list(map(mapping.get, aus))
    y_train = np.asarray(y_train)
    aus = y_test.tolist()
    y_test = list(map(mapping.get, aus))
    y_test = np.asarray(y_test)
    with open(test_path_p, 'rb') as f:
        test_paths = pickle.load(f)
    return x_train, y_train, x_test, y_test,test_paths

def plot_b_mispaced_images(b_m,b_m_c,paths):


    root_dir = '../BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}
    
    b_images = []
    labels = []
    k=0

    for i in b_m:
        imgname = paths[int(i[0])]
        info = paths[int(i[0])].split("-")
        tumor = info[0].split('_')[-1]
        srcfile = srcfiles[tumor]
        sub = info[0] + '_' + info[1] + '-' + info[2]
        mag = info[3]
        srcfile = srcfile % (root_dir, sub, mag, imgname)
        img = np.array(Image.open(srcfile).resize((224,224)))
        b_images.append(img)
        labels.append(tumor)
    
    fig, axs = plt.subplots(2, 5)
    fig.suptitle("Misplaced benign images")
    for i in range(2):
        for j in range(5):
            axs[i,j].imshow(b_images[k])
            axs[i,j].axis('off')
            axs[i,j].set_title("%s" %(labels[k]))
            k+=1
    plt.show()

    b_c_images = []
    labels = []
    k=0

    for i in b_m_c:
        imgname = paths[int(i[0])]
        info = paths[int(i[0])].split("-")
        tumor = info[0].split('_')[-1]
        srcfile = srcfiles[tumor]
        sub = info[0] + '_' + info[1] + '-' + info[2]
        mag = info[3]
        srcfile = srcfile % (root_dir, sub, mag, imgname)
        img = np.array(Image.open(srcfile).resize((224,224)))
        b_c_images.append(img)
        labels.append(tumor)
    
    fig, axs = plt.subplots(2, 5)
    fig.suptitle("Misplaced benign images - Central cluster")
    k=0
    for i in range(2):
        for j in range(5):
            axs[i,j].imshow(b_c_images[k])
            axs[i,j].axis('off')
            axs[i,j].set_title("%s" %(labels[k]))
            k+=1
    plt.show()



def plot_embedding(x_train,y_train,x_test,y_test,paths,f):
    path = "../plots/emb vis/emb_vis_f"+str(f)+".png"
    order=['Benign','Malignant']
    embedder = umap.UMAP(random_state=42)
    emb_train = embedder.fit_transform(x_train)
    emb_test = embedder.transform(x_test)


    plt.figure()
    plt.suptitle("Embedding - Triplet net \n Fold " + str(f) ,fontsize = 14)
    plt.subplot(211)
    plt.title("Training set")
    sns.scatterplot(x=emb_train[:,0],y=emb_train[:,1],marker='s',palette='Set1',hue=y_train,hue_order=order,s=10,alpha=0.5)
    plt.legend()
    plt.subplot(212)
    plt.title("Test set")
    sns.scatterplot(x=emb_test[:,0],y=emb_test[:,1],hue=y_test,marker='s',palette='Set1',hue_order=order,s=10,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path,dpi=300)

def plot_distance_matrix(x_train,x_test,f):
    train_dist = pairwise_distances(x_train,metric='euclidean')
    plt.figure()
    plt.title("Train set distance matrix")
    sns.heatmap(train_dist,xticklabels=False,yticklabels=False)
    path = "train_dist_f%s.png" %(f)
    plt.savefig(path,dpi=100)

    test_dist = pairwise_distances(x_test,metric='euclidean')
    plt.figure()
    plt.title("Test set distance matrix")
    sns.heatmap(test_dist,xticklabels=False,yticklabels=False)
    path = "test_dist_f%s.png" %(f)
    plt.savefig(path,dpi=300)


def create_plots():
    global train_path,test_path
    prefix_train = "../embeddings/train"
    prefix_test = "../embeddings/test"
    train = os.listdir(prefix_train)
    train.sort(reverse=False)
    test = os.listdir(prefix_test)
    test.sort(reverse=False)
    print("Saving emebeddings visulazations...")
    for x,y in zip(train,test):
        train_path = os.path.join(prefix_train,x)
        test_path = os.path.join(prefix_test,y)
        print(test_path)
        f=(x.split('f')[1].split('.npz')[0])
        test_paths_p = "../embeddings/test/test_images_path_f"+str(f)
        x_train, y_train, x_test, y_test , paths = upload_data(train_path,test_path,test_paths_p)
        plot_embedding(x_train, y_train, x_test, y_test,paths,f)
        #plot_distance_matrix(x_train,x_test,f)
        
    print("Done!")

if __name__ == '__main__':
    create_plots()
