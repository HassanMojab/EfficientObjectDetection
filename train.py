"""
How to train the Policy Network :
    python train.py
        --lr 1e-4
        --cv_dir checkpoint directory
        --batch_size 512 (more is better)
        --data_dir directory to contain csv file
        --alpha 0.6
"""
import os
import torch
import torch.utils.data as torchdata
import torch.nn as nn

import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import argparse
from tensorboard_logger import configure, log_value
from torch.distributions import Bernoulli

from utils import utils, utils_detector
from constants import num_actions

parser = argparse.ArgumentParser(description='PolicyNetworkTraining')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--img_size', type=int, default=448, help='PN Image Size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers')
parser.add_argument('--test_epoch', type=int, default=10, help='At every N epoch test the network')
parser.add_argument('--parallel', action='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--beta', type=float, default=0.1, help='Coarse detector increment')
parser.add_argument('--sigma', type=float, default=0.5, help='cost for patch use')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):
    agent.train()
    rewards, rewards_baseline, policies = [], [], []
    for (inputs, _, offset_fd, offset_cd, object_counts) in tqdm.tqdm(trainloader, total=len(trainloader)):
        if not args.parallel:
            inputs = inputs.cuda()
            offset_fd = offset_fd.cuda()
            offset_cd = offset_cd.cuda()
            object_counts = object_counts.cuda()

        # Actions by the Agent
        probs = torch.sigmoid(agent.forward(inputs))
        alpha_hp = np.clip(args.alpha + epoch * 0.001, 0.6, 0.95)
        probs = probs*alpha_hp + (1-alpha_hp) * (1-probs)

        # Sample the policies from the Bernoulli distribution characterized by agent
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0

        # Find the reward for baseline and sampled policy
        reward_map = utils.compute_reward(offset_fd, offset_cd, object_counts, policy_map.data, args.beta, args.sigma)
        reward_sample = utils.compute_reward(offset_fd, offset_cd, object_counts, policy_sample.data, args.beta, args.sigma)
        advantage = reward_sample.cuda().float() - reward_map.cuda().float()

        # Find the loss for only the policy network
        loss = -distr.log_prob(policy_sample)
        loss = loss * advantage.expand_as(policy_sample)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())
        policies.append(policy_sample.data.cpu())

    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)

    print('Train: %d | Rw: %.2E | S: %.3f | V: %.3f | #: %d' % (epoch, reward, sparsity, variance, len(policy_set)))

    log_value('train_reward', reward, epoch)
    log_value('train_sparsity', sparsity, epoch)
    log_value('train_variance', variance, epoch)
    log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    log_value('train_unique_policies', len(policy_set), epoch)

def test(epoch):
    agent.eval()
    rewards, metrics, policies, set_labels = [], [], [], []

    with torch.no_grad():
        for (inputs, targets, offset_fd, offset_cd, object_counts) in tqdm.tqdm(testloader, total=len(testloader)):
            if not args.parallel:
                inputs = inputs.cuda()
                offset_fd = offset_fd.cuda()
                offset_cd = offset_cd.cuda()
                object_counts = object_counts.cuda()

            # Actions by the Policy Network
            probs = torch.sigmoid(agent(inputs))

            # Sample the policy from the agents output
            policy = probs.data.clone()
            policy[policy<0.5] = 0.0
            policy[policy>=0.5] = 1.0

            reward = utils.compute_reward(offset_fd, offset_cd, object_counts, policy.data, args.beta, args.sigma)
            metrics, set_labels = utils.get_detected_boxes(policy, targets, metrics, set_labels)

            rewards.append(reward)
            policies.append(policy.data)

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(true_positives, pred_scores, pred_labels, set_labels)
    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)

    print('Test - AP: %.3f | AR : %.3f' % (AP[0], recall.mean()))
    print('Test - Rw: %.2E | S: %.3f | V: %.3f | #: %d' % (reward, sparsity, variance, len(policy_set)))

    log_value('test_reward', reward, epoch)
    log_value('test_AP', AP[0], epoch)
    log_value('test_AR', recall.mean(), epoch)
    log_value('test_sparsity', sparsity, epoch)
    log_value('test_variance', variance, epoch)
    log_value('test_unique_policies', len(policy_set), epoch)

    # save the model --- agent
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': reward,
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_R_%.2E'%(epoch, reward))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.img_size, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
agent = utils.get_model(num_actions)

# ---- Load the pre-trained model ----------------------
start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print('loaded agent from %s' % args.load)

# Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance
if args.parallel:
    agent = nn.DataParallel(agent)
agent.cuda()

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Save the args to the checkpoint directory
configure(args.cv_dir+'/log', flush_secs=5)

# Start training and testing
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch % args.test_epoch == 0:
        test(epoch)
