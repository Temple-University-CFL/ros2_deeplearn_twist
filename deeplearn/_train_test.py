#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Race-car Deep Learning Class.

This script contains all deep learning tools to train and predict speed and 
steering value from a provided image. 

Revision History:
        2020-05-10 (Animesh): Baseline Software.
        2020-07-30 (Animesh): Updated Docstring.

Example:
        from _train_test import NNTools

"""


#___Import Modules:
import os
import json
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from _datagen import Datagen
from _parser import ParseData
from racecarNet import ServoNet, MotorNet


#___Global Variables:
TYPE = ["servo", "test"]
SETTINGS = 'settings.json'
ODIR = "output/"
SEED = 717
WEIGHT = 'models/servo.pth'


#__Classes:
class NNTools:
    """Neural Network Tool Class.
    
    This class contains all methods to complete whole deep learing session
    containing training, testing and prediction-make sessions.
    
    """

    def __init__(self, settings=SETTINGS, types=TYPE, weight=WEIGHT):
        """Constructor.
        
        Args:
            settings (JSON file): Contains all settings manually provided.
            types (list): Contains settings to determine the session is for
                training or testing.

        """

        self.type = types[0]

        # set hyperparameters
        with open(settings) as fp:
            content = json.load(fp)[types[0]][types[1]]

            self.shape = content["shape"]
            self.batch_size = content["batch"]
            self.cuda = content["cuda"]

            if types[1] == "train":
                self.epochs = content["epoch"]
            else:
                self.weights_path = weight

        # set neural net by type
        torch.manual_seed(SEED)
        if self.type == "servo":
            self.model = ServoNet(self.shape)
        elif self.type == "motor":
            self.model = MotorNet(self.shape)

        # set output and load weights if required
        if types[1] == "train":
            self.log = self.set_output()
        else:
            self.load_weights(self.weights_path)

        # set output folders and required classes
        self.parsedata = ParseData()
        self.datagen = Datagen(shape=self.shape)

        return None


    def set_output(self):
        """Output Manager.
        
        This method checks files and directories for producing output during
        training session and creates them if they don't exist.
        
        Returns:
            log (file): Location of log file to dump results during training 
                session.

        """

        # checks and creates output directories
        if not os.path.exists(ODIR):
            os.mkdir(ODIR)        
        if not os.path.exists(os.path.join(ODIR,"curves")):
            os.mkdir(os.path.join(ODIR,"curves"))        
        if not os.path.exists(os.path.join(ODIR,"weights")):
            os.mkdir(os.path.join(ODIR,"weights"))

        # checks and creates log file to dump results
        log = os.path.join(ODIR,"result.csv")
        if os.path.exists(log):
            os.remove(log)
            open(log, 'a').close()
        else:
            open(log, 'a').close()

        return log


    def train(self, trainset, devset):
        """Mathod to run Training Session.
        
        This method runs the complete training session and produces plots and
        results in every epoch.
        
        Args:
            trainset (pandas dataframe): Contains training data.
            devset (pandas dataframe): Contains validation data.

        """
        
        # loads training dataset
        trainset = pd.read_csv(trainset)

        # set neural network model and loss function
        # GPU support if required
        if (self.cuda):
            model = self.model.cuda()
            criterion = nn.MSELoss().cuda()
        else:
            model = self.model
            criterion = nn.MSELoss()
        
        # sets optimizer and data generator
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        dataloader = DataLoader(dataset=Datagen(trainset, self.shape), \
                                    batch_size=self.batch_size, shuffle=True)
       
        # initialize conunter and result holder
        total_loss = []
        dev_accuracy = []
        epoch_loss = 0.0
        accuracy = 0.0

        # loop over the dataset multiple times
        for epoch in range(1, self.epochs+1):

            # initialize train loss and running loss
            batch = 0
            running_loss = 0.0
            start = timeit.default_timer()

            for image, servo, motor in dataloader:

                batch += self.batch_size

                # set target
                if self.type == "servo":
                    target = servo
                else:
                    target = motor
                
                # implement GPU support
                if (self.cuda):
                    image = image.cuda()
                    target = target.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(image)
                loss = criterion(target.unsqueeze(1), output)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # print status for every 100 mini-batches
                if batch % 10240 == 0:                    
                    stop = timeit.default_timer()
                    print('[{0: 4d}, {1: 6d}] loss: {2: 2.7f} time: {3: 2.3f} dev: {4: 2.0f}'\
                        .format(epoch, batch, running_loss/10240, stop-start, accuracy))

                    epoch_loss = running_loss/10240
                    running_loss = 0.0
                    start = timeit.default_timer()
                
                # Free memory
                del image, servo, motor, target, output

            # accuracy count on dev set
            accuracy = self.test(devset)
            dev_accuracy.append(accuracy)

            # total loss count
            total_loss.append(epoch_loss)
            weights_path = "weights/{0}_epoch_{1:04d}.pth".format(self.type, epoch)
            self.save_model(os.path.join(ODIR, weights_path))
            
            # plotting loss vs epoch curve, produces log file
            self.plot_result(epoch, total_loss, dev_accuracy)
        
        #show finish message
        print("Training (" + self.type + ") Finished!")

        return None


    def plot_result(self, epoch, total_loss, dev_accuracy):
        """Managing Result.
        
        This method produces result with required plots in proper format at 
        each epoch.
        
        Args:
            epoch (int): Indicator of epoch count.
            total loss (float): The accumulated loss.
            dev_accuracy (float): Accuracy percentage on validation data.

        """

        # plotting loss vs epoch curve
        plt.figure()
        plt.plot(range(1,epoch+1), total_loss, linewidth = 4)
        plt.title("Training (" + self.type + ")")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(ODIR + "/curves/" + self.type + " loss curve.png")
        plt.close()

        # dev accuracy vs epoch curve
        plt.figure()
        plt.plot(range(1,epoch+1), dev_accuracy, linewidth = 4)
        plt.title("Training (" + self.type + ")")
        plt.xlabel("Epoch")
        plt.ylabel("Dev Accuracy")
        plt.savefig(ODIR + "/curves/" + self.type + " accuracy curve.png")
        plt.close()
        
        # save accuracy values and show finish message
        content = "[{0:04d}, {1:02.2f}] {3}: epoch = {0:04d}, accuracy = {1:02.2f}, best = {2:04d}\n"\
                .format(epoch, dev_accuracy[epoch-1], np.argmax(dev_accuracy)+1, self.type)

        # write in log
        with open(self.log, 'a') as fp:
            fp.write(content)

        return None


    def test(self, testset, display=False):
        """Mathod to run Testing Session.
        
        This method runs the complete testing session producing results.
        
        Args:
            testset (pandas dataframe): Contains testing data.
            display (boolian): Flag to display result or not.
        
        Returns:
            (float): Accuracy percentage.

        """

        # loads testing dataset
        testset = pd.read_csv(testset)
        
        # set neural network model
        if (self.cuda):
            model = self.model.cuda()
        else:
            model = self.model

        # set dataloader
        dataloader = DataLoader(dataset=Datagen(testset, self.shape), \
                                    batch_size=self.batch_size, shuffle=False) 


        # initialize train loss and running loss
        total_accuracy = 0.0
        count = 0

        for image, servo, motor in dataloader:

            count += self.batch_size

            # set target
            if self.type == "servo":
                target = servo
            else:
                target = motor
            
            # implement GPU support
            if (self.cuda):
                image = image.cuda()
                target = target.cuda()

            # accuracy calculation
            output = model(image).round()
            accuracy = abs(target.unsqueeze(1) - output) <= 1

            total_accuracy += sum(accuracy).item()

            if display and count%128 == 0:
                print("[{0: 5d}] accuracy: {1: 2.2f}"\
                      .format(count, total_accuracy*100/count))
            
            # Free memory
            del image, servo, motor, target, output

        if display:
            print("total accuracy = %2.2f" % (total_accuracy*100/len(testset)))

        return total_accuracy*100/len(testset)


    def save_model(self, weights_path='weights/servo.pth'):
        """Mathod to save Trained Model.
        
        This method saves weights of a trained model.
        
        Args:
            weights_path (pth file): Trained Weights.

        """
        
        torch.save(self.model.state_dict(), weights_path)
        print("Saving Weights (" + self.type + ")")
    
        return None


    def load_weights(self, weights_path):
        """Mathod to load a Model.
        
        Args:
            weights_path (pth file): Pretrained Weights.

        """

        self.model.load_state_dict(torch.load(weights_path, \
                                             map_location=torch.device('cpu')))

        return None


    def robust_test(self, testset, edir="output/"):
        """Mathod to Calculate Accuracy.
        
        This method runs the complete testing session producing accuracy in
        different type of basis and produces a list for error predicted data.
        
        Args:
            testset (pandas dataframe): Contains testing data.
            edir (diectory path): Directory path to create error file.

        """
        
        # loads testing dataset
        testset = pd.read_csv(testset)
        
        # initialize data holder and result holder
        count_10 = 0
        count_LFR = 0
        content_10 = []
        content_LFR = []
        
        for index in range(len(testset)):
            
            # parse data and make prediction
            if self.type == "servo":
                image,target,_ = self.parsedata.parse_data(testset["image"][index])
                prediction =  self.predict(image)
            else:
                image,_,target = self.parsedata.parse_data(testset["image"][index])
                prediction =  self.predict(image)

            # 10% tolerence
            if abs(target - prediction) <= 1:
                count_10 += 1
            else:
                content_10.append(testset["image"][index])
            
            # LFR/BSF
            if target == 5:
                if prediction != 5:
                    count_LFR += 1
                else:
                    content_LFR.append(testset["image"][index])
            elif target < 5:
                if prediction < 5:
                    count_LFR += 1
                else:
                    content_LFR.append(testset["image"][index])
            elif target > 5:
                if prediction > 5:
                    count_LFR += 1
                else:
                    content_LFR.append(testset["image"][index])
            
            # print result for every 100 contents
            if (index+1)%100 == 0:
                print('[{0: 5d}] {3}: {1: 2.2f} {2: 2.2f}' \
                          .format(index+1, 100*count_10/(index+1), \
                                      100*count_LFR/(index+1), self.type))
        
        # print final result
        print("{2}: {0: 2.2f} {1: 2.2f}" \
            .format(100*count_10/(index+1), 100*count_LFR/(index+1), self.type))
        
        # checks and create output directories
        if not os.path.exists(edir):
            os.makedirs(edir)
        
        # create error file
        pd.DataFrame(content_10, 
                     columns =['image']).to_csv(edir+'error_10.csv', index=False)
        if self.type == "servo":
            pd.DataFrame(content_LFR, 
                     columns =['image']).to_csv(edir+'error_LFR.csv', index=False)
        else:
            pd.DataFrame(content_LFR, 
                     columns =['image']).to_csv(edir+'error_BSF.csv', index=False)
        
        return None
    
    
    def predict(self, image):
        """Mathod for Prediction.
        
        This method predicts streering or speed value from a provided single
        image.
        
        Args:
            image (image file): Image as input.
        
        Returns:
            (int): Predicted steering or speed value.

        """

        image = self.datagen.get_image(image).unsqueeze(0)
        
        # implement GPU support if required
        if (self.cuda):
            model = self.model.cuda()
        else:
            model = self.model

        # return prediction
        if (self.cuda):
            return model(image.cuda()).round().int().item()
        else:
            return model(image).round().int().item()


    def time_count(self, testset):
        """Mathod to Count Runtime.
        
        This method runs a testing session to count time to produce result from
        neural network.
        
        Args:
            testset (pandas dataframe): Contains testing data.

        """
        
        testset = pd.read_csv(testset)
        
        # timer start
        start = timeit.default_timer()
        
        # make prediction
        for i in range(10):
            for index in range(len(testset)): 
                image,_,_ = self.parsedata.parse_data(testset["image"][index])
                _ =  self.predict(image)
        
        # timer stop
        stop = timeit.default_timer()
        
        # calculate required time and display
        time = 0.1*(stop-start)/(index+1)
        print(time)
        
        return None


#                                                                              
# end of file
"""ANI717"""