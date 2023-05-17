import numpy as np
from PIL import Image
from skimage.segmentation import slic
import networkx as nx
from skimage import color, measure
from skimage.future import graph
from dgl import from_networkx
import torch
import cv2
import tensorflow as tf
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def save_slic_image(img, segments_slic,grade,name):
    path="../graphs/img/slic/%s/%s" %(grade,name)
    im=mark_boundaries(img, segments_slic)
    im = Image.fromarray((im * 255).astype(np.uint8))
    im.save(path)

def save_final_image(labels,img,grade,name):
    final_img = color.label2rgb(labels,img, kind="avg")
    path ="../graphs/img/final images/%s/%s" %(grade,name)
    im = Image.fromarray(final_img.astype(np.uint8))
    im.save(path)

def save_graph_img(g,grade,name):
    path ="../graphs/img/graph images/%s/%s" %(grade,name)
    nx.draw(g, cmap = plt.get_cmap('jet'))
    plt.savefig(path,dpi=300)


def load_resnet50(patch_size):
    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(patch_size,patch_size,3),
                   pooling='avg',classes=3,
                   weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable=False
    return pretrained_model

def stats_graph(g):
    non=g.number_of_nodes()
    noe=g.number_of_edges()
    aus_degree=g.degree()
    d=sum(map(lambda x: x[1], aus_degree)) / len(aus_degree)
    aus_nb=nx.betweenness_centrality(g,normalized=False)
    nb = sum(aus_nb.values()) / len(aus_nb)
    aus_eb=nx.edge_betweenness_centrality(g,normalized=False)
    eb=sum(aus_eb.values()) / len(aus_eb)
    aspl=nx.average_shortest_path_length(g)
    gcc=nx.transitivity(g)
    aus_lcc=nx.clustering(g)
    lcc=sum(aus_lcc.values()) / len(aus_lcc)

    return [non,noe,d,nb,eb,aspl,gcc,lcc]

def process_patch(patch_center,img_path,net,patch_size):
    image = cv2.imread(img_path)
    """
    smaller_dim = np.min(image.shape[0:2])
    patch_scale = 0.1
    patch_size = int(patch_scale * smaller_dim)
    """
    if (patch_center[0]<patch_size):
        patch_center[0]+=patch_center[0]+patch_size
    if(patch_center[1]<patch_size):
        patch_center[1]+=patch_center[1]+patch_size
    patch_x = int(patch_center[0] - patch_size / 2.)
    patch_y = int(patch_center[1] - patch_size / 2.)
    patch_image = image[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
    img = Image.fromarray(patch_image) #for the resnet
    img = np.array(img)
    x=net.predict(img[None,...],verbose=0)[0]
    return x

def create_graph(img_path,ns,c,s,grade,name,patch_size):
    net = load_resnet50(patch_size)
    img = np.array(Image.open(img_path))
    segments_slic = slic(img, n_segments=ns, compactness=c, sigma=s,start_label=1)
    #save_slic_image(img,segments_slic,grade,name);
    rag = graph.rag_mean_color(img,segments_slic)
    labels = graph.cut_threshold(segments_slic, rag, thresh=0)
    rag = graph.rag_mean_color(img, labels)
    labels = labels + 1
    properties = measure.regionprops(labels)
    #save_final_image(labels,img,grade,name)

    X = []
    for region in rag.nodes:
        idx = region
        props = properties[idx]
        centroid_x, centroid_y = props.centroid
        patch_center = np.array([centroid_x,centroid_y])
        x = process_patch(patch_center,img_path,net,patch_size)
        X.append(x)

    X = np.stack(X)

    g = nx.DiGraph(rag)
    #stats=stats_graph(g)
    #save_graph_img(g,grade,name)
    G = from_networkx(g)
    G.ndata['feat'] = torch.from_numpy(X)
    return G
