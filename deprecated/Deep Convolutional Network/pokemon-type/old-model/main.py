import tensorflow as tf
import numpy as np
import model
import utils
import graph
from scipy import misc
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split

current_location = os.getcwd()
learning_rate = 0.001
epoch = 1000
batch_size = 5
split_percentage = 0.2

Train = False

test_number = 10

type_pokemon, unique_type = utils.gettype(current_location)
pokemon_pictures = utils.getpictures(current_location + '/pokemon')

output_dimension = len(unique_type)
picture_dimension = 28

pokemon_pictures_train, pokemon_pictures_test, pokemon_types_train, pokemon_types_test = train_test_split(pokemon_pictures, type_pokemon, test_size = split_percentage)

sess = tf.InteractiveSession()
model = model.Model(picture_dimension, learning_rate, output_dimension)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

try:
    saver.restore(sess, current_location + "/model.ckpt")
    print "load model.."
except:
    if Train:
        print "start from fresh variables"
    else:
        print "please train first, exiting.."
        exit(0)
        
def train():

    ACCURACY = []; EPOCH = []; LOST = []
    
    for i in xrange(epoch):
        total_cost = 0
        total_accuracy = 0
        last_time = time.time()
        EPOCH.append(i)
        
        for k in xrange(0, len(pokemon_pictures_train) - batch_size, batch_size):
            
            emb_data = np.zeros((batch_size, picture_dimension, picture_dimension, 4), dtype = np.float32)
            emb_data_label_1 = np.zeros((batch_size, output_dimension), dtype = np.float32)
            emb_data_label_2 = np.zeros((batch_size, output_dimension), dtype = np.float32)
            
            for x in xrange(batch_size):
                
                image = misc.imread(current_location + '/pokemon/' + pokemon_pictures_train[k + x])
                image = misc.imresize(image, (picture_dimension, picture_dimension))
                emb_data_label_1[x, unique_type.index(pokemon_types_train[k + x, 0])] = 1.0
                emb_data_label_2[x, unique_type.index(pokemon_types_train[k + x, 1])] = 1.0
                
                emb_data[x, :, :, :] = image
            
            _, loss = sess.run([model.optimizer, model.cost], feed_dict = {model.X : emb_data, model.Y_1 : emb_data_label_1, model.Y_2 : emb_data_label_2})
            accuracy_1, accuracy_2 = sess.run([model.accuracy_1, model.accuracy_2], feed_dict = {model.X : emb_data, model.Y_1 : emb_data_label_1, model.Y_2 : emb_data_label_2})
            total_cost += loss
            total_accuracy += ((accuracy_1 + accuracy_2) / 2.0) 
        
        
        accuracy = total_accuracy / ((len(pokemon_pictures_train) - batch_size) / (batch_size * 1.0))
        loss = total_cost / ((len(pokemon_pictures_train) - batch_size) / (batch_size * 1.0))
        ACCURACY.append(accuracy)
        LOST.append(loss)
        
        print "epoch: " + str(i + 1) + ", loss: " + str(loss) + ", accuracy: " + str(accuracy) + ", s / epoch: " + str(time.time() - last_time)
        graph.generategraph(EPOCH, ACCURACY, LOST)
        saver.save(sess, current_location + "/model.ckpt")

def test():
    
    import matplotlib.pyplot as plt
    
    num_print = int(np.sqrt(len(pokemon_pictures_test)))
    fig = plt.figure(figsize = (1.5 * num_print, 1.5 * num_print))
    
    for k in xrange(0, num_print * num_print):
        
        plt.subplot(num_print, num_print, k + 1)
        
        emb_data = np.zeros((1, picture_dimension, picture_dimension, 4), dtype = np.float32)
            
        image = misc.imread(current_location + '/pokemon/' + pokemon_pictures_test[k])
        image = misc.imresize(image, (picture_dimension, picture_dimension))
                
        emb_data[0, :, :, :] = image
           
        y_hat_1, y_hat_2 = sess.run([model.y_hat_1, model.y_hat_2], feed_dict = {model.X : emb_data})
        
        label_1 = unique_type[np.argmax(y_hat_1[0])]
        label_2 = unique_type[np.argmax(y_hat_2[0])]
        
        plt.imshow(image)
        plt.title(label_1 + " + " + label_2)
    
    fig.tight_layout()    
    plt.savefig('output.png')
    plt.savefig('output.pdf') 
    plt.cla()
    
    print "printing diamond-pearl.."
    
    list_folder = os.listdir(current_location + '/diamond-pearl')
    
    num_print = int(np.sqrt(len(list_folder)))
    fig = plt.figure(figsize = (1.5 * num_print, 1.5 * num_print))
    
    for k in xrange(0, num_print * num_print):
        
        plt.subplot(num_print, num_print, k + 1)
        
        emb_data = np.zeros((1, picture_dimension, picture_dimension, 4), dtype = np.float32)
            
        image = misc.imread(current_location + '/diamond-pearl/' + list_folder[k])
        image = misc.imresize(image, (picture_dimension, picture_dimension))
                
        emb_data[0, :, :, :] = image
           
        y_hat_1, y_hat_2 = sess.run([model.y_hat_1, model.y_hat_2], feed_dict = {model.X : emb_data})
        
        label_1 = unique_type[np.argmax(y_hat_1[0])]
        label_2 = unique_type[np.argmax(y_hat_2[0])]
        
        plt.imshow(image)
        plt.title(label_1 + " + " + label_2)
    
    fig.tight_layout()    
    plt.savefig('output_diamond_pearl.png')
    plt.savefig('output_diamond_pearl.pdf') 
    plt.cla()
        
    
def main():
    if Train:
        train()
    else:
        test()
        
        
main()
