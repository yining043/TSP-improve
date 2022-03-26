import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
                                        # description="the algorithm name"
                                    )

    ### overall settings
    parser.add_argument('--problem', default='tsp', choices = ['tsp'],
                        help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, 
                        help="The size of the problem graph")
    parser.add_argument('--eval_only', action='store_true', 
                        help='used only if to evaluate a model')
    parser.add_argument('--init_val_met', choices = ['seq'], default = 'seq',
                        help='method to generate initial solutions while validation')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_tb', action='store_true', help='Disable Tensorboard')
    parser.add_argument('--no_assert', action='store_true', help='Disable Assertion')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    
    
    # resume and load models
    parser.add_argument('--load_path', default = None,
                        help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None,
                        help='Resume from previous checkpoint file')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')


    ### training AND validation
    parser.add_argument('--n_step', type=int, default=4)
    parser.add_argument('--T_train', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='Number of instances per batch during training')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='The number of epochs to train')
    parser.add_argument('--epoch_size', type=int, default=10000,
                        help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=1000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=1000,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--val_dataset', type=str, default = './datasets/tsp_20_10000.pkl', 
                        help='Dataset file to use for validation')

    
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    
    
    ### network
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--n_heads_encoder', type=int, default=1)
    parser.add_argument('--n_heads_decoder', type=int, default=1) 
    parser.add_argument('--tanh_clipping', type=float, default=0.001,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--gamma', type=float, default=0.8, help='decrease future reward')
    parser.add_argument('--T_max', type=int, default=1000, help='number of steps to swap')
    
    ### logs to tensorboard and screen
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=25, # 50
                        help='Log info every log_step steps')
    ### outputs
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--run_name', default='run_name', help='Name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    return opts

if __name__ == "__main__":
    opts = get_options()