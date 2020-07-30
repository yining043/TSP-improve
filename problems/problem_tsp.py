from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle

class TSP(object):

    NAME = 'tsp'
    
    def __init__(self, p_size=20, with_assert = False):

        self.size = p_size          # the number of nodes in tsp 
        self.do_assert = with_assert
        print(f'TSP with {self.size} nodes.')
    
    def step(self, rec, exchange):
        
        device = rec.device
        
        exchange_num = exchange.clone().cpu().numpy()
        rec_num = rec.clone().cpu().numpy()
         
        for i in range(rec.size()[0]):
            
            loc_of_first = np.where(rec_num[i] == exchange_num[i][0])[0][0]
            loc_of_second = np.where(rec_num[i] == exchange_num[i][1])[0][0]
            
            if( loc_of_first < loc_of_second):
                rec_num[i][loc_of_first:loc_of_second+1] = np.flip(
                        rec_num[i][loc_of_first:loc_of_second+1])
            else:
                temp = rec_num[i][loc_of_first]
                rec_num[i][loc_of_first] = rec_num[i][loc_of_second]
                rec_num[i][loc_of_second] = temp
                
        return torch.tensor(rec_num, device = device)
            
    
    def get_costs(self, dataset, rec):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param rec: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        if self.do_assert:
            assert (
                torch.arange(rec.size(1), out=rec.data.new()).view(1, -1).expand_as(rec) ==
                rec.data.sort(1)[0]
            ).all(), "Invalid tour"
        
        # Gather dataset in order of tour
        d = dataset.gather(1, rec.long().unsqueeze(-1).expand_as(dataset))
        length =  (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
        
        return length
    
    
    def get_swap_mask(self, rec):
        _, graph_size = rec.size()
        return torch.eye(graph_size).view(1,graph_size,graph_size)

    def get_initial_solutions(self, methods, batch):
        
        def seq_solutions(batch_size):
            graph_size = self.size
            solution = torch.linspace(0, graph_size-1, steps=graph_size)
            return solution.expand(batch_size,graph_size).clone()
        
        batch_size = len(batch)
        
        if methods == 'seq':
            return seq_solutions(batch_size)
            
    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)   



class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=20, num_samples=10000):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in data[:num_samples]]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
