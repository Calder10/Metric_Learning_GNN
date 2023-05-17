import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from time import perf_counter as pc
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from keras.applications.resnet import ResNet152
from keras_balanced_batch_generator import make_generator
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

img_size = 224
batch_size = 32
epochs = 20
lr = 1e-5
wd = 1e-4
m = 0.1
emb_size = 512
distance ='L2'

def resize_image(img):
    return img.resize((img_size,img_size))

def create_fold(f):
    x_train = []
    x_test = []
    x_val=[]
    y_val=[]
    y_train = []
    y_test = []
    p_test = []

    mapping = {'b':0,'m':1}

    root_dir = '../BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}
    
    path ="../src/dsfold"+str(f)+".txt"
    db = open(path)
    print("Training and test set creation....")
    for row in db.readlines():
        columns = row.split('|')
        imgname = columns[0]
        mag = columns[1]  # 40, 100, 200, or 400
        grp = columns[3].strip()  # train or test
        tumor = imgname.split('-')[0].split('_')[-1]
        srcfile = srcfiles[tumor]
        s = imgname.split('-')
        pi=s[2]
        label = s[0].split("_")[1].lower()
        sub = s[0] + '_' + s[1] + '-' + s[2]
        srcfile = srcfile % (root_dir, sub, mag, imgname)
        image = Image.open(srcfile)
        image = resize_image(image)
        x = np.asarray(image)
        if grp == 'train':
            x_train.append(x)
            y_train.append(mapping[label])
        else:
            x_test.append(x)
            y_test.append(mapping[label])
            p_test.append(imgname)
            
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)


    path = "../embeddings/test/test_images_path_f"+str(f)

    with open(path, "wb") as fp:   
        pickle.dump(p_test, fp)
    

    path = "../images npz/images_f%s.npz" %(f) 
    np.savez_compressed(path,x_train,x_test)
    print("Done !")
    return x_train,y_train,x_test,y_test

def create_triplet_net():
    triplet_net = tf.keras.models.Sequential()
    resnet152=ResNet152(include_top=False,
                   input_shape=(224,224,3),
                   pooling='max',classes=None,
                   weights='imagenet')

    triplet_net.add(resnet152)
    triplet_net.add(tf.keras.layers.Flatten())
    triplet_net.add(tf.keras.layers.Dense(emb_size, activation=None)) 
    triplet_net.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    opt = tf.keras.optimizers.Adam(learning_rate = lr, decay = wd)
    loss_fn = tfa.losses.TripletSemiHardLoss(margin=m, distance_metric = distance)
    triplet_net.compile(optimizer=opt,loss=loss_fn)
    return triplet_net






def plot_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label="Training loss")
    plt.title('Triplet net loss')
    plt.legend()
    plt.ylabel('triplet margin loss')
    plt.xlabel('epoch')
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()


def train_net(triplet_net,x_train,y_train,f):
    steps_per_epoch = int( np.ceil(x_train.shape[0] / batch_size) )
    y_cat = tf.keras.utils.to_categorical (y_train)
    train_gen = make_generator(x_train, y_cat, batch_size=batch_size,
               categorical=False,
               seed=None)
    print("Training.....")
    start = pc()
    history = triplet_net.fit(train_gen,epochs=epochs,steps_per_epoch=steps_per_epoch)
    end = pc()-start
    training_time = timedelta(seconds=end)
    print("Training ended in:", str(training_time))
    path = "../models/triplet_net_"+str(f)+".h5"
    triplet_net.save(path)
    print("Model saved !")
    plot_loss(history)
    return triplet_net

def save_embedding(triplet_net,f,x_train,y_train,x_test,y_test):
    print("Embeddings creation....")
    x_train_emb = triplet_net.predict(x_train,verbose=1)
    x_test_emb = triplet_net.predict(x_test,verbose=1)
    
    train_emb = "../embeddings/train/train_emb_f"+str(f)+".npz"
    test_emb = "../embeddings/test/test_emb_f"+str(f)+".npz"

    np.savez_compressed(train_emb,x_train_emb,y_train)
    np.savez_compressed(test_emb,x_test_emb,y_test)
    print("Embedding saved !")

def run_5_folds():
    for i in range(1,6):
        print("Fold",i)
        x_train,y_train,x_test,y_test= create_fold(i)
        triplet_net=create_triplet_net()
        triplet_net = train_net (triplet_net,x_train,y_train,i)
        save_embedding(triplet_net,i,x_train,y_train,x_test,y_test)
        print("Done!")


if __name__ == '__main__':
    run_5_folds()
