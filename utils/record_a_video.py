#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:47:10 2020

@author: yiningma
"""

from utils.plots import plot_tour
import imageio
import torch
from tqdm import tqdm
        
def record_gif(batch, history, filename = 'ep_gif', dpi = 30):
    
    batch_size, ep_len, p_size = history.size()
    
    for batch_i in tqdm(range(batch_size),desc = 'record for each instance',  position=1, leave=True):
        
        solutions_per_instance = history[batch_i]
        
        with imageio.get_writer(f'./outputs/{filename}_{batch_i}.gif', mode='I') as writer:
        
            for tour in tqdm(solutions_per_instance, desc = 'tour completed', position=0, leave=True):
            
                img = plot_tour(tour, batch[batch_i], show = False, dpi = dpi)
            
                writer.append_data(img)