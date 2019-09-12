#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:48:00 2017

@author: FDarmon
"""
import cv2
import numpy as np

class Utils :
    def __init__(self,SUBDIVISIONS,SHAPE,B):
        self.SUBDIVISIONS=SUBDIVISIONS
        self.SHAPE=SHAPE
        self.B=B
        
    def print_bb(self,img,out,thresh,offset=(0,0),verbose=True,thickness=1):
        l=self.list_bb(out,thresh,offset,verbose)
        for bb in l:
            
            img=cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color=(0,0,255),thickness=thickness)
            
        return(img)
            
    def list_bb(self,out, thresh,offset=(0,0),verbose=True,regroup=True):
        """
        Creates a list of length 4 tuple for each bounding box detected in out
        ######################### HORIZONTAL INDEX FIRST ##########################
        x1,y1,x2,y2 : upper left corner  coordinate lower right corner coordinate
        
        
        """
        y,x=offset
        res=[]
        for i in range(self.SUBDIVISIONS):
            for j in range(self.SUBDIVISIONS):
                for k in range(self.B):
                    if out[i,j,k,4]>thresh :
                        j1=int(((i+out[i,j,k,0])/self.SUBDIVISIONS-out[i,j,k,2]/2)*self.SHAPE)
                        j2=int(((i+out[i,j,k,0])/self.SUBDIVISIONS+out[i,j,k,2]/2)*self.SHAPE)
                        i1=int(((j+out[i,j,k,1])/self.SUBDIVISIONS-out[i,j,k,3]/2)*self.SHAPE)
                        i2=int(((j+out[i,j,k,1])/self.SUBDIVISIONS+out[i,j,k,3]/2)*self.SHAPE)
                        detected_bb=[i1+x,j1+y,i2+x,j2+y]
                        
                        if verbose:
                            print('Detected in (%d,%d) with IOU score %.2f'%(i,j,out[i,j,k,4]))
                    
                        if regroup:
                            for index,bb in enumerate(res) :
                                cond1=max(detected_bb[0],bb[0])<min(detected_bb[2],bb[2])
                                cond2=max(detected_bb[1],bb[1])<min(detected_bb[3],bb[3])
                                if cond1 and cond2:
                                    x_upperleft=min(bb[0],detected_bb[0])
                                    y_upperleft=min(bb[1],detected_bb[1])
                                    x_lowerright=max(bb[2],detected_bb[2])
                                    y_lowerright=max(bb[3],detected_bb[3])
                                    detected_bb=[x_upperleft,y_upperleft,x_lowerright,y_lowerright]
                                    if verbose:
                                        print('Merge with bounding box number %d'%index)
                                    res.remove(bb)                            
                                
                        res.append(tuple(detected_bb))
        return(res)
    
    def list_bb_from_label(self,label):
        """
        Creates a list of length 4 tuple for each bounding box of the label
        ###### Horizontal index first ####
        """
        res=[]
        for i in range(self.SUBDIVISIONS):
            for j in range(self.SUBDIVISIONS):
                if not label[i,j,0]==0:
                    j1=int(((i+label[i,j,1])/self.SUBDIVISIONS-label[i,j,3]/2)*self.SHAPE)
                    j2=int(((i+label[i,j,1])/self.SUBDIVISIONS+label[i,j,3]/2)*self.SHAPE)
                    i1=int(((j+label[i,j,2])/self.SUBDIVISIONS-label[i,j,4]/2)*self.SHAPE)
                    i2=int(((j+label[i,j,2])/self.SUBDIVISIONS+label[i,j,4]/2)*self.SHAPE)
                    res.append([i1,j1,i2,j2])
                    
        
        return res
    
    def premetrics(self,pred,lbl,overlap=0.5):
        """
        Returns a tuple (TP,FP,FN) for metrics calculation
        """
        TP=0
        FN=0
        while len(lbl)>0:
            bb_lbl=lbl[0]
            lbl=lbl[1:]
            iou_max=0
            bb_max=None
            for bb_pred in pred:
                iou=self.compute_IOU(bb_lbl,bb_pred)
                if iou_max<iou:
                    iou_max=iou
                    bb_max=bb_pred
        
            if iou_max>=overlap:
                TP+=1
                pred.remove(bb_max)
            else:
                FN+=1
        FP=len(pred)
        return(TP,FP,FN)
        
    def compute_IOU(self,bb1,bb2):
        """
        returns the IOU of bb1 with bb2
        """
        
        up_left_corner=np.array((max(bb1[0],bb2[0]),max(bb1[1],bb2[1])))
        low_right_corner=np.array((min(bb1[2],bb2[2]),min(bb1[3],bb2[3])))
        
            
        dist=low_right_corner-up_left_corner
        if np.any(dist<=0):
            inter_area=0
        else:
            inter_area=np.prod(dist)
        
        union_area=(bb1[2]-bb1[0])*(bb1[2]-bb1[0])+\
            (bb2[2]-bb2[0])*(bb2[2]-bb2[0])
            
        return(inter_area/union_area)
        