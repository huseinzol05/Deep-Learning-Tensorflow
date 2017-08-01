import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "whitegrid", palette = "muted")
import numpy as np
import matplotlib.gridspec as gridspec
import csv
import pandas as pd
from scipy import misc

def generategraph(x, accuracy, lost):
    
    fig = plt.figure(figsize = (15, 5))
    
    plt.subplot(1, 2, 1)
    
    plt.plot(x, lost)
    plt.xlabel('Epoch')
    plt.ylabel('lost')
    plt.title('LOST')
    
    plt.subplot(1, 2, 2)
    
    plt.plot(x, accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('ACCURACY')
    
    fig.tight_layout()
    plt.savefig('graph.png')
    plt.savefig('graph.pdf')
    plt.cla()
    
def generateoutput(image, output, label, title, i):
    
    with open('output.csv', 'wb') as fopen:
        writer = csv.writer(fopen, delimiter=',')
        writer.writerow(['class', 'value'])
        
        for i in xrange(output.shape[1]):
            row = [label[i], output[0, i]]
            writer.writerow(row)
            
    fig = plt.figure(figsize = (10, 25))
    
    gridspec.GridSpec(20, 1)
    
    plt.subplot2grid((20, 1), (0,0))
    plt.imshow(image)
    plt.title(title)

    plt.subplot2grid((20, 1), (1,0), colspan = 1, rowspan = 19)
    dataset = pd.read_csv('output.csv')
    print dataset.head()
    sns.set_color_codes("pastel")
    sns.barplot(x = "value", y = "class", data = dataset, color = "b")
    
    fig.tight_layout()
    plt.savefig('probs' + str(i) + '.png')
    plt.savefig('probs' + str(i) + '.pdf')
    plt.cla()
    
    