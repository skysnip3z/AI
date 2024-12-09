# -*- coding: utf-8 -*-
"""
CNN - CIFAR-10

0 = airplane, 1 = automobile, 2 = bird, 3 = cat,  4 = deer  

"""
# Imports - All available in Conda
import numpy as np
import matplotlib.pyplot as plt
import keras as K
from keras import models, layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

#%%  DataManager Class - Manages cifar-10 dataset

class DataManager:
    def __init__(self):
        self.subsets = self.generate_subsets()
        self.x_train = self.subsets[0][0]
        self.y_train = self.subsets[0][1]
        self.x_test = self.subsets[1][0]
        self.y_test = self.subsets[1][1]
        self.x_val = self.subsets[2][0]
        self.y_val = self.subsets[2][1]
        
    # returns train, test, and validation sets based on five classes 
    def generate_subsets(self):
        # load unaltered cifar dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # ensure only specified classes are in the dataset
        (x_train, y_train) = self.extract_five_classes(x_train, y_train)
        (x_test, y_test) = self.extract_five_classes(x_test, y_test)
        
        # Ensure data compatability with keras
        x_train = x_train.astype("float32")/255
        x_test = x_test.astype("float32")/255
        
        # Ensure labels are categorical
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        
        # split test data/labels to create validation set
        (x_val, y_val) = x_train[:1000], y_train[:1000]
        (x_train, y_train) = x_train[1000:], y_train[1000:]
        
        return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    # extracts classes from subsets - ordered then shuffled
    def extract_five_classes(self, data, labels):
        # Main arrays
        x = np.empty((0, 32, 32, 3), int)
        y = np.empty((0, 1), int)
        
        # each specified class is extracted individually
        for i in range(0, 5):
            # Bool arr representing idicies of class data in dataset
            class_indicies = (labels == i).reshape(data.shape[0])
            
            # specified class data and labels are retrieved
            x_data = data[class_indicies]
            y_labels = labels[class_indicies]
        
            # specified class data and labels added to main arrays
            x = np.concatenate((x, x_data))
            y = np.concatenate((y, y_labels))
            
        self.shuffle_subsets(x, y)
            
        return x, y
    
    # Shuffle dataset - ensures same random seed is used
    def shuffle_subsets(self, data, labels):
        seed = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(seed)
        np.random.shuffle(labels)
    
    # Shuffles train, test, and val sets
    def shuffle_all_subsets(self):
        seed = np.random.get_state()
        np.random.shuffle(self.x_train)
        np.random.set_state(seed)
        np.random.shuffle(self.y_train)
        np.random.set_state(seed)
        np.random.shuffle(self.x_test)
        np.random.set_state(seed)
        np.random.shuffle(self.y_test)
        np.random.set_state(seed)
        np.random.shuffle(self.x_val)
        np.random.set_state(seed)
        np.random.shuffle(self.y_val)
        
    
    # Plot images from dataset - 6 by default
    def display_images(self, data, labels, amount=6, rows=2, cols=3):
        fig = plt.figure(figsize=(32,32))
        for i in range(1, amount+1):
            img = data[i]
            ax = fig.add_subplot(rows, cols, i)
            plt.subplots_adjust(hspace=None, wspace=0.02)
            ax.title.set_text(str(i)+ " label=" + str(labels[i]))
            plt.imshow(img)
        plt.show()

    # Displays a sample of six training images
    def display_train_images(self):
        self.display_images(self.x_train, self.y_train)
        
    # Displays a sample of six testing images
    def display_test_images(self):
        self.display_images(self.x_test, self.y_test)
        
    # Displays a sample of six validation images
    def display_val_images(self):
        self.display_images(self.x_val, self.y_val)
        
    # Returns Train, Val, Test Generator Iterators
    def generate_gen_iterators(self, batch_size=64):
        train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                   fill_mode="nearest")
        test_gen = ImageDataGenerator()
        val_gen = ImageDataGenerator()
        
        train_it = train_gen.flow(self.x_train, self.y_train,
                                  batch_size=batch_size)
        val_it = val_gen.flow(self.x_val, self.y_val,
                              batch_size=batch_size)
        test_it = test_gen.flow(self.x_test, self.y_test,
                                batch_size=batch_size)
        
        return train_it, val_it, test_it
            

#%% Functions - Test Results & Diagnostics

# Display results from network.fit in plots
def plot_fit_results(h_dict, test_no=0):
    epochs = range(1, len(h_dict["loss"])+1)
    
    # Test No.
    title = "Test: " + str(test_no + 1)
    
    # Plot loss
    plt.title(title + " Loss")
    plt.plot(epochs, h_dict["loss"], "r", label="Training Loss")
    plt.plot(epochs, h_dict["val_loss"], "g", label="Validation Loss")
    plt.legend()
    plt.show()
    
    # Plot accuracies
    plt.title(title + " Accuracy")
    plt.plot(epochs, h_dict["accuracy"], "r", label="Training Acc")
    plt.plot(epochs, h_dict["val_accuracy"], "g", label="Validation Acc")
    plt.legend()
    plt.show()
    

def plot_all_fit_results(all_h_dicts):
    for i in range(len(all_h_dicts)):
        plot_fit_results(all_h_dicts[i], i)
    
# Prints results from network.evaluate to console     
def print_eval_results(results, test_no=0):
    print("#", str(test_no+1), " test loss, test acc:", results)

def print_all_eval_results(all_results):
    for i in range(len(all_results)):
        print_eval_results(all_results[i], i)

def get_list_avg(list_):
    avg = sum(list_) / len(list_)
    return avg

def display_all_results(all_h_dicts, all_results, all_loss, all_acc):
    plot_all_fit_results(all_h_dicts)
    print_all_eval_results(all_results)
    print("Average Loss: " + str(get_list_avg(all_loss)))
    print("Average Accuracy: " + str(get_list_avg(all_acc)))


#%% Functions - Models defined inside functions for iterative testing

# Baseline model - no alterations
def run_baseline_test(dm, count, epochs, batch_size, return_net=False):
    
    all_h_dicts = []
    all_results = []
    all_loss = []
    all_acc = []
    network_ = None
    
    for i in range(count):
        #  Baseline Network
        print("Test #", str(i+1))
        network = models.Sequential()
        network.add(layers.Conv2D(32, (3,3), activation="relu",
                                  input_shape=(32,32,3)))
        network.add(layers.MaxPooling2D((2,2)))
        network.add(layers.Conv2D(128, (3,3), activation="relu"))
        network.add(layers.MaxPooling2D((2,2)))
        network.add(layers.Conv2D(64, (3,3), activation="relu"))
        network.add(layers.Flatten())
        network.add(layers.Dense(64, activation="relu"))
        network.add(layers.Dense(5, activation="softmax"))
        
        # Compile network with optimizer
        network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                        metrics=["accuracy"])
        
        # Train network and store training history
        hist = network.fit(dm.x_train,  dm.y_train, epochs=epochs,
                           batch_size=batch_size,
                           validation_data=(dm.x_test, dm.y_test))
        
        # Evaluate network
        results = network.evaluate(dm.x_val, dm.y_val)
        
        # To set network to var if network
        if return_net and i == 0:
            network_ = network
        
        # Shuffle data for next iteration
        dm.shuffle_all_subsets()
        
        # Collect each iterations results
        all_h_dicts.append(hist.history)
        all_results.append(results)
        all_loss.append(results[0])
        all_acc.append(results[1])

    display_all_results(all_h_dicts, all_results, all_loss, all_acc)
    if return_net:
        return network_
    
# Baseline model with added dropout post-pooling    
def run_exp_dropout(dm, count, epochs, batch_size, return_net=False):
    all_h_dicts = []
    all_results = []
    all_loss = []
    all_acc = []
    network_ = None
    
    for i in range(count):
        print("Test #", str(i+1))
        network = models.Sequential()
        network.add(layers.Conv2D(32, (3,3), activation="relu",
                                  input_shape=(32,32,3)))
        network.add(layers.MaxPooling2D((2,2)))
        network.add(layers.Dropout(0.1))
        network.add(layers.Conv2D(128, (3,3), activation="relu"))
        network.add(layers.MaxPooling2D((2,2)))
        network.add(layers.Dropout(0.2))
        network.add(layers.Conv2D(64, (3,3), activation="relu"))
        network.add(layers.Flatten())
        network.add(layers.Dense(64, activation="relu"))
        network.add(layers.Dropout(0.3))
        network.add(layers.Dense(5, activation="softmax"))
        
        # Compile network
        network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                        metrics=["accuracy"])
        
        # Train network and store training history
        hist = network.fit(dm.x_train,  dm.y_train, epochs=epochs,
                           batch_size=batch_size,
                           validation_data=(dm.x_test, dm.y_test))
        
        # Evaluate network
        results = network.evaluate(dm.x_val, dm.y_val)
        
        # To set network to var if network
        if return_net and i == 0:
            network_ = network
        
        # Shuffle data for next iteration
        dm.shuffle_all_subsets()
        
        # Collect each iterations results
        all_h_dicts.append(hist.history)
        all_results.append(results)
        all_loss.append(results[0])
        all_acc.append(results[1])

    display_all_results(all_h_dicts, all_results, all_loss, all_acc)
    if return_net:
        return network_

# Baseline model with dropout and data augmentation
def run_exp_augmentation(dm, count, epochs, batch_size=64, return_net=False):
    all_h_dicts = []
    all_results = []
    all_loss = []
    all_acc = []
    network_ = None
    
    for i in range(count):
        train_it, val_it, test_it = dm.generate_gen_iterators(batch_size)
        
        print("Test #", str(i+1))
        network = models.Sequential()
        network.add(layers.Conv2D(32, (3,3), activation="relu",
                                  input_shape=(32,32,3)))
        network.add(layers.MaxPooling2D((2,2)))
        network.add(layers.Dropout(0.1))
        network.add(layers.Conv2D(128, (3,3), activation="relu"))
        network.add(layers.MaxPooling2D((2,2)))
        network.add(layers.Dropout(0.2))
        network.add(layers.Conv2D(64, (3,3), activation="relu"))
        network.add(layers.Flatten())
        network.add(layers.Dense(64, activation="relu"))
        network.add(layers.Dropout(0.3))
        network.add(layers.Dense(5, activation="softmax"))
        
        # Compile network
        network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                        metrics=["accuracy"])
        
        # Train network and store training history
        hist = network.fit_generator(train_it, steps_per_epoch=100,
                                     epochs=epochs, validation_data=val_it,
                                     validation_steps=50)
        
        # Evaluate network
        results = network.evaluate_generator(test_it, steps=50, verbose=1)
        
        # To set network to var if network
        if return_net and i == 0:
            network_ = network
        
        # Shuffle data for next iteration
        dm.shuffle_all_subsets()
        
        # Collect each iterations results
        all_h_dicts.append(hist.history)
        all_results.append(results)
        all_loss.append(results[0])
        all_acc.append(results[1])

    display_all_results(all_h_dicts, all_results, all_loss, all_acc)
    if return_net:
        return network_

# Transfered model with droupout, data augmentation
def run_exp_augmentation_transfer(dm, count, epochs, batch_size=64, return_net=False):
    all_h_dicts = []
    all_results = []
    all_loss = []
    all_acc = []
    network_ = None
    
    # Transfered Model - VGG16
    vgg_base = VGG16(weights = 'imagenet', include_top= False,
                     input_shape=(32,32,3))
    
    for i in range(count):
        train_it, val_it, test_it = dm.generate_gen_iterators(batch_size)
        
        print("Test #", str(i+1))
        network = models.Sequential()
        network.add(vgg_base)  
        network.add(layers.Flatten())
        network.add(layers.Dense(512, activation='relu'))
        network.add(layers.Dropout(0.1))
        network.add(layers.Dense(5, activation="softmax"))
        
        # Compile network
        network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                        metrics=["accuracy"])
        
        # Train network and store training history
        hist = network.fit_generator(train_it, steps_per_epoch=100,
                                     epochs=epochs, validation_data=val_it,
                                     validation_steps=50)
        
        # Evaluate network
        results = network.evaluate_generator(test_it, steps=50, verbose=1)
        
        # To set network to var if network
        if return_net and i == 0:
            network_ = network
        
        # Shuffle data for next iteration
        dm.shuffle_all_subsets()
        
        # Collect each iterations results
        all_h_dicts.append(hist.history)
        all_results.append(results)
        all_loss.append(results[0])
        all_acc.append(results[1])

    display_all_results(all_h_dicts, all_results, all_loss, all_acc)
    if return_net:
        return network_   

# Transfered model with dropout, data augmentation, and learning rate adjustment
def run_exp_augmentation_transfer_lr(dm, count, epochs, batch_size=64, return_net=False):
    all_h_dicts = []
    all_results = []
    all_loss = []
    all_acc = []
    network_ = None
    
    # Transfered Model - VGG16
    vgg_base = VGG16(weights = 'imagenet', include_top= False,
                     input_shape=(32,32,3))
    
    for i in range(count):
        train_it, val_it, test_it = dm.generate_gen_iterators(batch_size)
        
        print("Test #", str(i+1))
        network = models.Sequential()       
        network.add(vgg_base)
        network.add(layers.Flatten())
        network.add(layers.Dense(512, activation='relu'))
        network.add(layers.Dropout(0.1))
        network.add(layers.Dense(5, activation="softmax"))
        
        # Compile network)
        opt = optimizers.RMSprop(learning_rate=1e-5)
        network.compile(optimizer=opt, loss="categorical_crossentropy",
                        metrics=["accuracy"])
        
        # Train network and store training history
        hist = network.fit_generator(train_it, steps_per_epoch=100,
                                     epochs=epochs, validation_data=val_it,
                                     validation_steps=50)
        
        # Evaluate network
        results = network.evaluate_generator(test_it, steps=50, verbose=1)
        
        # To set network to var if network
        if return_net and i == 0:
            network_ = network
        
        # Shuffle data for next iteration
        dm.shuffle_all_subsets()
        
        # Collect each iterations results
        all_h_dicts.append(hist.history)
        all_results.append(results)
        all_loss.append(results[0])
        all_acc.append(results[1])

    display_all_results(all_h_dicts, all_results, all_loss, all_acc)
    if return_net:
        return network_   

# Transfered model with improvements documented above function call
def run_exp_performance(dm, count, epochs, batch_size=64, return_net=False):
    all_h_dicts = []
    all_results = []
    all_loss = []
    all_acc = []
    network_ = None
    
    # Transfered Model - VGG16
    vgg_base = VGG16(weights = 'imagenet', include_top= False,
                     input_shape=(32,32,3))
    
    for i in range(count):
        train_it, val_it, test_it = dm.generate_gen_iterators(batch_size)
        
        print("Test #", str(i+1))
        network = models.Sequential()       
        network.add(vgg_base)
        network.add(layers.Flatten())
        network.add(layers.Dense(512, activation='relu'))
        network.add(layers.Dropout(0.5))
        network.add(layers.Dense(5, activation="softmax"))
        
        # Compile network)
        opt = optimizers.RMSprop(learning_rate=1e-5)
        network.compile(optimizer=opt, loss="categorical_crossentropy",
                        metrics=["accuracy"])
        
        # Train network and store training history
        hist = network.fit_generator(train_it, steps_per_epoch=100,
                                     epochs=epochs, validation_data=val_it,
                                     validation_steps=50)
        
        # Evaluate network
        results = network.evaluate_generator(test_it, steps=50, verbose=1)
        
        # To set network to var if network
        if return_net and i == 0:
            network_ = network
        
        # Shuffle data for next iteration
        dm.shuffle_all_subsets()
        
        # Collect each iterations results
        all_h_dicts.append(hist.history)
        all_results.append(results)
        all_loss.append(results[0])
        all_acc.append(results[1])
        
        # Clear Keras Backend session, attempts to ensure it does not intefere with tests
        K.backend.clear_session()

    display_all_results(all_h_dicts, all_results, all_loss, all_acc)
    if return_net:
        return network_   

#%% Data manager which manages data subsets: train, test, val sets

dm = DataManager()

#%% Tests to observe behaviour

"""
Test Descriptions ~ #Tx

acc avg = sum of all test  / model instances tested 

#T1 : baseline test, no improvement measures (~79.0% acc avg)

#T2 : baseline + dropout (~79.5% acc avg)

#T3 : baseline + dropout + data augmentation (~64.9% acc avg)

#T4 : Transfer learning (TL) + dropout + data augmentation 
      (Trainable VGG) ~19.9% acc avg, (Untrainable VGG) ~69.7% acc avg
      ## after testing both, functions for TL are always trainable ##

#T5 : Run transfer learning + dropout + data augmentation + learning rate change 
      (~87.6% acc avg)

~Each test is documented and reproducible below
"""


"""T1"""
#run_baseline_test(dm, 10, 10, 64)

"""T2"""
#run_exp_dropout(dm, 10, 10, 64)

"""T3"""
#run_exp_augmentation(dm, 10, 10, 64)

"""T4"""
#run_exp_augmentation_transfer(dm, 10, 10, 64)

"""T5"""
#run_exp_augmentation_transfer_lr(dm, 10, 10, 64)


#%% Tests to improve performance - based on observed behaviours

"""
Improved Baseline from previous tests: ~87.6%

#T1 : Increased batch size to 128 (~88.9% / +1.3% from prev), dropout=0.5, 10 epochs

#T2 : Testing T1 @ 35 epochs (~90.9% / +2.0% from prev)

~Each change is documented, only highest accuracy (last) test is repeatable
"""

#run_exp_performance(dm, 10, 35, 128)


# For data to provide a better comparison between models (~79.29% acc avg)

#run_baseline_test(dm, 10, 35, 128)


