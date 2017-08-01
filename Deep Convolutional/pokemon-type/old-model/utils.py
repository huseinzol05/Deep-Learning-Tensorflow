import numpy as np
import os

def gettype(location):
    
    with open(location + '/type', 'r') as fopen:
        type_pokemon = fopen.read().split('\n')
        type_pokemon = [i.split('\t')[4:] for i in type_pokemon]
        
        for i in xrange(len(type_pokemon)):
            if len(type_pokemon[i]) == 1:
                type_pokemon[i].append('none')
        
        type_pokemon = np.array(type_pokemon)
        
        type_list = np.array(np.unique(type_pokemon[:, 0]).tolist() + np.unique(type_pokemon[:, 1]).tolist())
        
        return type_pokemon, np.unique(type_list).tolist()
        
def getpictures(location):
    
    list_folder = os.listdir(location)
    list_folder = [int(i.replace('.png', '')) for i in list_folder]
    list_folder.sort()
    list_folder = [str(i) + '.png' for i in list_folder]
    return list_folder