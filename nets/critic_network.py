from torch import nn
from nets.graph_layers import MultiHeadAttentionLayer, EmbeddingNet


class CriticNetwork(nn.Module):

    def __init__(self,
             problem,
             embedding_dim,
             hidden_dim,
             n_heads,
             n_layers,
             normalization,
             device,
             best_incumbent = False
             ):
        
        super(CriticNetwork, self).__init__()
        
        self.best_incumbent = best_incumbent

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
        
        self.value_head = nn.Sequential(
                nn.Linear(embedding_dim, self.hidden_dim),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Dropout(0.1)
            )   
            

    def forward(self, x, solutions):
        """
        :param inputs: (x, graph_size, input_dim)
        :return:
        """
        # pass through embedder
        x_embed = self.embedder(x, solutions)
        
        # pass through encoder
        h_em = self.encoder(x_embed) # torch.Size([2, 50, 128])
        
        # get embed feature
        max_pooling = h_em.max(1)[0]   # max Pooling
        graph_feature = self.project_graph(max_pooling)[:, None, :]
        
        # get embed node feature
        node_feature = self.project_node(h_em) 
        
        # pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([2, 50, 128])
        value = self.value_head(fusion.mean(dim=1))

        return value
    