#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:47:26 2020

@author: yiningma
"""
import torch
import os
from matplotlib import pyplot as plt
import cv2
import io
import numpy as np

def plot_grad_flow(model):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    named_parameters = model.named_parameters() 
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.ioff()
    fig = plt.figure(figsize=(8,6))
    plt.plot(ave_grads, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=60)
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_improve_pg(initial_value, reward):
    
    plt.ioff()
    fig = plt.figure(figsize=(4,3))
    plt.plot(initial_value.mean() - np.cumsum(reward.cpu().mean(0)))
    
    plt.xlabel("T")
    plt.ylabel("Cost")
    plt.title("Avg Improvement Progress")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=60)
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_tour(city_tour, coordinates, dpi = 300, show = True):
    
    if not show: plt.ioff()
    
    fig = plt.figure(figsize=(8,6))
  
    index = torch.cat((
        city_tour.view(-1,1),
        city_tour.view(-1,1)[None,0]),0).repeat(1,2).long()

    xy = torch.gather(coordinates,0,index)   
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([-0.05, 1.05]*2)
    plt.plot(xy[:,0], xy[:,1], color = 'black', zorder = 1)

    g1 = plt.scatter(xy[:,0], xy[:,1], marker = 'H', s = 55, c = 'blue', zorder = 2)
    g2 = plt.scatter(xy[0,0], xy[0,1], marker = 'H', s = 55, c = 'red', zorder = 2)
    handle = [g1,g2]
    plt.legend(handle, ['node', 'depot'], fontsize = 12)
    
    #    plot show        
    if not show:
        buf = io.BytesIO()
        plt.savefig(buf, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        plt.show()
        return None

def plot_heatmap(problem, solutions, predicted_feasibility):
    
    from problems.problem_pdp_mp import PDP as PDPmp
    problem_mp = PDPmp(problem.size)
    
    true_feasibility = (problem_mp.get_swap_mask(solutions).bool()).float()
    
    import seaborn as sns; sns.set()
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize = (10,4))

    sns.heatmap(predicted_feasibility.detach(), ax = ax1)
    sns.heatmap(true_feasibility[0], ax = ax2)
    
    plt.show()
    
    
