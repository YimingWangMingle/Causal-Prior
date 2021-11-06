import sys
import time
import signal
import argparse
import os
import torch
import env_init
import numpy as np
from pathlib import Path
import gym


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
np.set_printoptions(precision=2)
gym.logger.set_level(40)


import wandb
from multi_processing import MultiProcessTrainer
from model_basic import AC, Random, RNN
from model_tie import Tie
import trainer
import trainer_baselines
import utils
from action_utils import parse_action_args



parser = argparse.ArgumentParser(description='TieMARL')

# training
parser.add_argument('--num_epochs', default=500, type=int, help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10, help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500, help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=1, help='How many processes to run')

# model
parser.add_argument('--hid_size', default=128, type=int, help='hidden layer size')
parser.add_argument('--nagents', type=int, default=10, help="number of agents")
parser.add_argument('--mean_ratio', default=0, type=float,
                    help='how much cooperative to do? 1.0 means fully cooperative')
parser.add_argument('--detach_gap', default=10, type=int,
                    help='detach hidden state and cell state for rnns at this interval')
parser.add_argument('--comm_init', default='uniform', type=str, help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--comm_mask_zero', action='store_true', default=False, help="whether block the communication")

# optimization
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--normalize_rewards', action='store_true', default=False, help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
parser.add_argument('--entr', type=float, default=0, help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01, help='coefficient for value loss term')

# environment
parser.add_argument('--env_name', default='traffic_junction', type=str, help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int, help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str, help='the number of agent actions')
parser.add_argument('--action_scale', default=1.0, type=float, help='scale action output from model')

# other
parser.add_argument('--save', action="store_true", default=False, help='save the model after training')
parser.add_argument('--save_every', default=0, type=int, help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str, help='load the model')
parser.add_argument('--display', action="store_true", default=False, help='display epoch result')
parser.add_argument('--algo', default='ac', type=str, choices=['random', 'tie', 'ac', 'rnn-ac', 'lstm-ac'])
parser.add_argument('--memo', default='baselines', type=str)
parser.add_argument('--debug', action='store_true', default=False, help="enable wandb")
parser.add_argument('--random', default=False)

utils.init_args_for_env(parser)
args = parser.parse_args()

args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = env_init.init(args.env_name, args, False)

args.obs_size = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)):  # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0, 10000)
torch.manual_seed(args.seed)

if args.debug:
    os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(project='TieMARL', save_code=True, tags=['Ming'],
           name=args.memo + '_' + args.env_name + '_' + args.algo + '_' + str(args.seed))
wandb.config.update(args)
print(args)



if args.algo == 'random':
    policy_net = Random(args)
    args.random = True
    Trainer = trainer_baselines.Trainer
elif args.algo == 'ac':
    policy_net = AC(args)
    Trainer = trainer_baselines.Trainer
elif args.algo in ['rnn-ac', 'lstm-ac']:
    policy_net = RNN(args)
    Trainer = trainer_baselines.Trainer
elif args.algo == 'tie':
    policy_net = Tie(args)
    Trainer = trainer.Trainer
else:
    raise RuntimeError("Wrong algo!")

utils.display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, env_init.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, env_init.init(args.env_name, args))



log = dict()
log['epoch'] = utils.LogField(list(), False, None, None)
log['reward'] = utils.LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = utils.LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = utils.LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = utils.LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = utils.LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = utils.LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = utils.LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = utils.LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = utils.LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = utils.LogField(list(), True, 'epoch', 'num_steps')



model_dir = Path('./saved_model') / args.env_name / args.algo / args.memo
if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                     model_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
run_dir = model_dir / curr_run


def save(final, epoch=0):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        torch.save(d, run_dir / 'final_model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' % (epoch)))



def run(total_training_epoches, epoch_size):

    num_episodes = 0
    if args.save:
        os.makedirs(run_dir)

    for epoch in range(total_training_epoches):
        epoch_begin_time = time.time()
        log_dict = dict()
        for n in range(epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True

            epoch_result = trainer.train_batch(epoch)
            utils.merge_stat(epoch_result, log_dict)
            trainer.display = False
            print('Epoch: {0}/{1} | {2}/{3}'.format(epoch, total_training_epoches, n, epoch_size))


        epoch_time = time.time() - epoch_begin_time
        num_episodes += log_dict['num_episodes']
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch+1)
            else:
                if k in log_dict and v.divide_by is not None and log_dict[v.divide_by] > 0:
                    log_dict[k] = log_dict[k] / log_dict[v.divide_by]
                v.data.append(log_dict.get(k, 0))

        print('Epoch {}'.format(epoch))
        print('Episode: {}'.format(num_episodes))
        print('Reward: {}'.format(log_dict['reward']))
        print('Time: {:.2f}s'.format(epoch_time))
        print('Steps-Taken: {:.2f}'.format(log_dict['steps_taken']))



        if args.env_name == 'traffic_junction':
            print('Success: {:.4f}'.format(log_dict['success']))
            print('Add-Rate: {:.2f}'.format(log_dict['add_rate']))

            wandb.log({"Epoch": epoch,
                       'Episode': num_episodes,
                       'Num_step': log_dict['num_steps'],
                       'Reward_mean': np.mean(log_dict['reward']),
                       'Reward_sum': np.sum(log_dict['reward']),
                       "Success": log_dict['success'],
                       "Add-Rate": log_dict['add_rate'],
                       "Steps_Taken": log_dict['steps_taken'],
                       'Action_loss': log_dict['action_loss'],
                       'Value_loss': log_dict['value_loss'],
                       'Entropy': log_dict['entropy']
                       })


        elif args.env_name == 'predator_prey':
            wandb.log({"Epoch": epoch,
                       'Episode': num_episodes,
                       'Num_step': log_dict['num_steps'],
                       "Steps_Taken": log_dict['steps_taken'],
                       'Reward_mean': np.mean(log_dict['reward']),
                       'Reward_sum': np.sum(log_dict['reward']),
                       'Action_loss': log_dict['action_loss'],
                       'Value_loss': log_dict['value_loss'],
                       'Entropy': log_dict['entropy'],
                       })
        else:
            raise RuntimeError("Wrong algo!")


        if args.save:
            if epoch == total_training_epoches:
                save(final=True)
            elif (epoch + 1) % args.save_every == 0:
                save(final=False, epoch=epoch + 1)
            else:
                pass
        else:
            pass



def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])


def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exiting gracefully.')
    if args.display:
        env.end_display()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


if args.load != '':
    load(args.load)

run(args.num_epochs, args.epoch_size)


if args.display:
    env.end_display()


if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    os._exit(0)

























