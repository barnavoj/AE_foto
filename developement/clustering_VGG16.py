# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

from scipy.spatial import distance_matrix

path = "C:\\Users\\VBarnat\\Desktop\\AE_foto\\database\\cernobyl"
# change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
images = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.jpg'):
          # adds only the image files to the images list
            images.append(file.name)
            
            
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}
p = r"C:\Users\VBarnat\Desktop\AE_foto"

# lop through each image in the dataset
for image in images:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(image,model)
        data[image] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
          
 
# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=5)
pca.fit(feat)
x = pca.transform(feat)


def find_closest(target_img_index, num):
    d_mat = distance_matrix(x, x)
    distances = d_mat[target_img_index]
    order = np.argsort(distances)
    result_indecies = order[1:num+1]
    return result_indecies


# function that lets you view reslut      
def view_result(target_img_index, result_indecies):
    plt.figure(figsize = (16,8))
    target_file = images[target_img_index]
    result_files = [ images[result_index] for result_index in  result_indecies]
    num = len(result_files)
    plt.subplot(2,num,(1,num))
    img = load_img(target_file)
    img = np.array(img)
    plt.imshow(img)
    plt.axis('off')
    for index, file in enumerate(result_files):
        plt.subplot(2,num,num+index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("C:\\Users\\VBarnat\\Desktop\\AE_foto\\results\\PCA_distance\\result_target_" + str(target_img_index) + ".png")
    #plt.show()

num = 5
target_img_index = 15
for target_img_index in range(len(images[::5])):
    i = target_img_index*5
    result_indecies = find_closest(i, num)
    view_result(i, result_indecies)

def cluster():
    # cluster feature vectors
    unique_labels = 10
    kmeans = KMeans(n_clusters=unique_labels)
    kmeans.fit(x)

    # holds the cluster id and the images { id: [images] }
    groups = {}
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    return groups

# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster, groups):
    plt.figure(figsize = (10,10))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    n = 9
    if len(files) > n:
        print(f"Clipping cluster size from {len(files)} to n")
        files = files[:n]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(3,3,index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("C:\\Users\\VBarnat\\Desktop\\AE_foto\\cluster_" + str(cluster) + ".png")
    plt.show()

#groups = cluster()
#for i in range(unique_labels, groups):
#    view_cluster(i)

# # this is just incase you want to see which value for k might be the best 
# sse = []
# list_k = list(range(2, 220))

# for k in list_k:
#     km = KMeans(n_clusters=k)
#     km.fit(x)
    
#     sse.append(km.inertia_)

# # Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse)
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance')
# plt.show()