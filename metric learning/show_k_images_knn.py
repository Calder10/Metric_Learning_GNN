import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def upload_data(train_path,test_path,images_path):
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_test = test_data['arr_0']
    y_test = test_data['arr_1']
    y_train = np.reshape(y_train,-1)
    y_test = np.reshape(y_test,-1)

    images = np.load(images_path,allow_pickle=True)
    train_images = images['arr_0']
    test_images = images['arr_1']
    return x_train,y_train,x_test,y_test,train_images,test_images

def show_k_images(x_train,y_train,x_test,y_test,train_images,test_images,f):
    mapping = {0:'Benign',1:'Malignant'}
    clf = KNeighborsClassifier(3)
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    nn = clf.kneighbors(x_test)
    i = 0
    
    for test_image , dist,index,y,p in zip(test_images,nn[0],nn[1],y_test,pred):
        if (y==p):
            path = "../results knn/%d/ti%d.png" %(f,i)
        else:
            path = "../results knn/%d/ti%d_m.png" %(f,i)
        fig, axs = plt.subplots(1,4)
        axs[0].imshow(test_image)
        axs[0].set_title("Test Image \n Prediction: %s \n True label: %s" %(mapping[p],mapping[y]),fontsize=5,color="red")
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        axs[3].axis('off')

        axs[1].imshow(train_images[index[0]])
        axs[1].set_title("Training image label: \n %s \n Distance: %f" %(mapping[y_train[index[0]]],dist[0]),fontsize=5)
        

        axs[2].imshow(train_images[index[1]])
        axs[2].set_title("Training image label: \n %s \n Distance: %f" %(mapping[y_train[index[1]]],dist[1]),fontsize=5)

        axs[3].imshow(train_images[index[2]])
        axs[3].set_title("Training image label: \n %s \n Distance: %f" %(mapping[y_train[index[2]]],dist[2]),fontsize=5)
        fig.savefig(path, bbox_inches='tight',dpi=1000)
        plt.close()
        i+=1

def run_folds():
    for f in range(1,6):
        print("Fold %d" %(f))
        train_path = "../embeddings/train/train_emb_f"+str(f)+".npz"
        test_path = "../embeddings/test/test_emb_f"+str(f)+".npz"
        images_path = "../images npz/images_f"+str(f)+".npz"
        x_train,y_train,x_test,y_test,train_images,test_images = upload_data(train_path,test_path,images_path)
        show_k_images(x_train,y_train,x_test,y_test,train_images,test_images,f)

if __name__ == '__main__':
    run_folds()
