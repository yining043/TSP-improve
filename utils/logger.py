import torch
import math
from utils.plots import plot_grad_flow, plot_improve_pg

def log_to_screen(time_used, init_value, best_value, reward, improvement,
                  batch_size, dataset_size, T):
    # reward
    print('\n', '-'*60)
    print('Avg total reward:'.center(35), '{:<10f} +- {:<10f}'.format(
            reward.sum(1).mean(), torch.std(reward.sum(1)) / math.sqrt(batch_size)))
    print('Avg step reward:'.center(35), '{:<10f} +- {:<10f}'.format(
            reward.mean(), torch.std(reward) / math.sqrt(batch_size)))
    # cost
    print('-'*60)
    print('Avg init cost:'.center(35), '{:<10f} +- {:<10f}'.format(
            init_value.mean(), torch.std(init_value) / math.sqrt(batch_size)))
    for per in range(20,101,20):
        improved = improvement[:,:round(T*per/100)].sum(1).view(-1,1)
        cost_ = init_value - improved
        print(f'Avg cost after {per}% improvement:'.center(35), '{:<10f} +- {:<10f}'.format(
                cost_.mean(), 
                torch.std(cost_) / math.sqrt(batch_size)))
    print('Avg best cost:'.center(35), '{:<10f} +- {:<10f}'.format(
            best_value.mean(), torch.std(best_value) / math.sqrt(batch_size)))
    # time
    print('-'*60)
    print('Avg used time:'.center(35), '{:f}s'.format(
            time_used.mean() / dataset_size))
    print('-'*60, '\n')
    
def log_to_tb_val(tb_logger, time_used, init_value, best_value, reward, improvement,
                  batch_size, dataset_size, T, no_figures, epoch):
    if not no_figures:
        tb_logger.log_images('validation/improve_pg',[plot_improve_pg(init_value, reward)], epoch)

    tb_logger.log_value('validation/avg_time',  time_used.mean() / dataset_size, epoch)
    tb_logger.log_value('validation/avg_total_reward', reward.sum(1).mean(), epoch)
    tb_logger.log_value('validation/avg_step_reward', reward.mean(), epoch)
    for per in range(20,101,20):
        improved = improvement[:,:round(T*per/100)].sum(1).view(-1,1)
        cost_ = init_value - improved
        tb_logger.log_value(f'validation/avg_.{per}_cost', cost_.mean(), epoch)
    tb_logger.log_value(f'validation/avg_init_cost', init_value.mean(), epoch)
    tb_logger.log_value(f'validation/avg_best_cost', best_value.mean(), epoch)

def log_to_tb_train(tb_logger, optimizer, model, baseline, total_cost, grad_norms, reward, 
               exchange_history, reinforce_loss, baseline_loss, log_likelihood, initial_cost, no_figures, mini_step):
    
    tb_logger.log_value('learnrate_pg', optimizer.param_groups[0]['lr'], mini_step)            
    avg_cost = (total_cost).mean().item()
    tb_logger.log_value('train/avg_cost', avg_cost, mini_step)
    avg_reward = torch.stack(reward, 0).sum(0).mean().item()
    max_reward = torch.stack(reward, 0).sum(0).max().item()
    tb_logger.log_value('train/avg_reward', avg_reward, mini_step)
    tb_logger.log_value('train/init_cost', initial_cost.mean(), mini_step)
    tb_logger.log_value('train/max_reward', max_reward, mini_step)
    grad_norms, grad_norms_clipped = grad_norms
    
    tb_logger.log_value('loss/actor_loss', reinforce_loss.item(), mini_step)
    tb_logger.log_value('loss/nll', -log_likelihood.mean().item(), mini_step)
    
    tb_logger.log_value('grad/actor', grad_norms[0], mini_step)
    tb_logger.log_value('grad_clipped/actor', grad_norms_clipped[0], mini_step)
    tb_logger.log_value('loss/critic_loss', baseline_loss.item(), mini_step)
    tb_logger.log_value('loss/total_loss', (reinforce_loss+baseline_loss).item(), mini_step)
    tb_logger.log_value('grad/critic', grad_norms[1], mini_step)
    tb_logger.log_value('grad_clipped/critic', grad_norms_clipped[1], mini_step)
    exchange_history = torch.stack(exchange_history)
    tb_logger.log_histogram('exchange', (exchange_history.view(-1).tolist()), mini_step)
    
    if not no_figures:
        tb_logger.log_images('grad/actor',[plot_grad_flow(model)], mini_step)
        tb_logger.log_images('grad/critic',[plot_grad_flow(baseline.critic)], mini_step)