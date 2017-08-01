import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
sns.set(style = "whitegrid", palette = "muted")

def get_dataset(location, dimension, visualize = False):
    
    list_folder = os.listdir(location)
    list_images = []
    for i in xrange(len(list_folder)):
        images = os.listdir(location + list_folder[i])
        for x in xrange(len(images)):
            image = [list_folder[i] + '/' + images[x], list_folder[i]]
            list_images.append(image)
            
    if visualize:
        fig = plt.figure(figsize = (30, 30))
        
        num = 1
        for i in xrange(len(list_folder)):
            plt.subplot(10, 10, num)
            image = misc.imread(location + list_folder[i] + '/image_0001.jpg')
            image = misc.imresize(image, (dimension, dimension))
            plt.imshow(image)
            plt.title(list_folder[i])
            num += 1
        
        plt.savefig('sample.png')
        plt.savefig('sample.pdf')
    
    list_images = np.array(list_images)
    np.random.shuffle(list_images)
    
    print "before cleaning got: " + str(list_images.shape[0]) + " data"
    
    list_temp = []
    for i in xrange(list_images.shape[0]):
        image = misc.imread(location + list_images[i, 0])
        if len(image.shape) < 3:
            continue
        list_temp.append(list_images[i, :].tolist())
        
    list_images = np.array(list_temp)
    
    print "after cleaning got: " + str(list_images.shape[0]) + " data"
    
    label = np.unique(list_images[:, 1]).tolist()
    
    list_images[:, 1] = LabelEncoder().fit_transform(list_images[:, 1])
    
    return list_images, np.unique(list_images[:, 1]).shape[0], label
