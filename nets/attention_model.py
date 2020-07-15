import torch
from torch import nn
from nets.graph_layers import MultiHeadAttentionLayer, MultiHeadDecoder, EmbeddingNet

class AttentionModel(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 device,
                 ):
        super(AttentionModel, self).__init__()

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.device = device

        # Problem specific placeholders
        
        if self.problem.NAME == 'tsp':
            self.node_dim = 2  # x, y
        else:
            assert False, "Unsupported problem: {}".format(self.problem.NAME)
        
        

        # networks
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim,
                            self.device)
        
        self.encoder = nn.Sequential(*(
            MultiHeadAttentionLayer(self.n_heads, 
                                    self.embedding_dim, 
                                    self.hidden_dim, 
                                    self.normalization)
            for _ in range(self.n_layers))) 
            
            
        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)  
        
        self.decoder = MultiHeadDecoder(self.problem,
                                        input_dim = self.embedding_dim, 
                                        embed_dim = self.embedding_dim)    
            
        
    def forward(self, x, solutions, exchange, do_sample = False, best_solutions = None):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """         
         
        # the embedded input x
        bs, gs, in_d = x.size()
        
        # pass through embedder: current solutions
        x_embed = self.embedder(x, solutions)
        
        # pass through encoder: current solutions
        h_em = self.encoder(x_embed) # torch.Size([2, 50, 128])
        
        # get embed graph feature
        max_pooling = h_em.max(1)[0] # max Pooling
        graph_feature = self.project_graph(max_pooling)[:, None, :]
    
        # get embed node feature
        node_feature = self.project_node(h_em)
        
        # pass through decoder, get log_likelihood and M matrix
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([2, 50, 128])

        log_likelihood, M_value = self.decoder(fusion, exchange, solutions) # att torch.Size([1, 2, 2500])          
        
        # sample or select current pair for actions
        pair_index = M_value.multinomial(1) if do_sample else M_value.max(-1)[1].view(-1,1)
        
        selected_log_likelihood = log_likelihood.gather(1, pair_index)
        
        col_selected = pair_index % gs
        row_selected = pair_index // gs
        pair = torch.cat((row_selected,col_selected),-1)  # pair: no_head bs, 2
        
        return pair, selected_log_likelihood.squeeze()