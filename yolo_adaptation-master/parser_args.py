# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:40:31 2017

@author: FDarmon
"""
from argparse import ArgumentParser

class Parser():
    def __init__(self):
        self.mainParser=ArgumentParser()
        self.subparsers=self.mainParser.add_subparsers(dest="action",help='Action performed')
        self.mainParser.add_argument('--weights',type=str,default='./weights/weights.ckpt',
                            help='Where to save or load the weights')
        
        
        self.train_parser=self.subparsers.add_parser("train",
                            help='Train a new network and store the weights learnt')
                
        self.test_parser=self.subparsers.add_parser("test",
                            help='Test the model loaded on a single image')
        
        self.score_parser=self.subparsers.add_parser("score",
                            help= 'Compute metrics on a labelled dataset evaluating the model loaded')
        

        self.train_parser.add_argument('--training_pathfile', type=str, 
                            help='Txt file containing paths to training images')
      
        self.train_parser.add_argument('--batch_size', type=int, default=32,
                            help='How many images to train on at a time.')
        self.train_parser.add_argument('--epochs', type=int, default=5,
                            help='How many epoch.')
        self.train_parser.add_argument('--log_dir', type=str, default='./logs',
                            help='Directory for logs')
        self.train_parser.add_argument('--restore',action='store_true',
                                  help='Restore weights for training from these weights')
        self.train_parser.add_argument('--learning_rate',type=float,default=0.001)
        
        
        self.test_parser.add_argument('--path_test_img',type=str,default='./test.jpg',
                            help='Single image to test')
       
        self.test_parser.add_argument('--thresh',type=float,default=0.5,
                            help='Threshold used selection of the prediction')
        
        
        self.score_parser.add_argument('--thresh',type=float,default=0.5,
                            help='Threshold used selection of the prediction if no PRcurve')
        self.score_parser.add_argument('--test_pathfile',
                            help='Path to a txt file containing the paths toward the test images')
        
        self.score_parser.add_argument('--PRCurve',action='store_true')
       
        
        
        
    def parse_args(self):
        
        return(self.mainParser.parse_args())
