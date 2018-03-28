#!/usr/bin/env python

import gym
import cv2
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.nn import functional as F
from torch.autograd import Variable
from collections import namedtuple, deque
from itertools import count
from atari_wrapper import wrap_deepmind
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Epsilon parameters
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
MEMORY_SIZE = 1000000
GAMMA = 0.99
UPDATE_PARAM_TAU = 1 #0.001
UPDATE_TARGET_STPS = 10000

LEARNING_RATE = 0.00025
SQRD_GRAD_MOMNTM = 0.95
MIN_SQ_GR_MOMNTM = 0.01
MOMENTUM = 0.95
OPTIM_RATE = 4
PLOT_EP_RATE = 20
SAVE_MODEL_FR = 500000

ENV_NAME = "Breakout-v4"
NUM_EPISODES = 1000000
MAX_NOOP = 30
INIT_EXPLORATION_FRAMES = 50000


use_gpu = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor


# Network
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lin1 = nn.Linear(7*7*64, 500)
        self.head = nn.Linear(500, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x)


# Process input screen
def process_input(screen, prev_state=None):
    imgr = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)[40:200,:]
    imgr = cv2.resize(imgr, (84,84))
    if prev_state is None:
        tstate = np.expand_dims(imgr, 0)
        tstate = np.expand_dims(tstate, 0)
        tstate = np.repeat(tstate, 4, axis=1)
        #state = Tensor(state)
    else:
        tstate = np.zeros_like(prev_state)
        tstate[:,:-1] = prev_state[:,1:]
        tstate[:,-1] = imgr
    return tstate


# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position +1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Select action
def select_action(state, network, steps_done):
    sample = random.random()
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.max([EPS_DECAY-steps_done, 0])/EPS_DECAY
    if sample > eps_threshold:
        return network(Variable(state, volatile=True).type(Tensor)/255.0).data.max(1)[1].view(1,1).cpu()
    else:
        return LongTensor([[random.randrange(n_actions)]])

q_averages = []
r_averages = []
def optimise_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    non_final_next_states = Variable(torch.cat([Tensor(s.astype(float)) for s in batch.next_state if s is not None]).type(Tensor)/255.0,
                                    volatile=True)
    state_batch = Variable(torch.cat(Tensor(np.array(batch.state,dtype=float)))/255.0)
    actions_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a)  | The model computes Q(s_t) and we select
    # the column of actions taken
    state_action_values = Q(state_batch).gather(1, actions_batch)
    q_averages.append(state_action_values.mean().data.cpu().numpy()[0])
    r_averages.append(reward_batch.sum().data.cpu().numpy()[0])

    # Compute V(s_{t+1}) for all next states
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = Qp(non_final_next_states).max(1)[0]

    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    bellman_error = expected_state_action_values - state_action_values.squeeze()
    clip_bellman_error = bellman_error.clamp(-1, 1)
    d_error = clip_bellman_error * -1.0
    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimise the model
    optimizer.zero_grad()
    state_action_values.backward(d_error.data.unsqueeze(1))
    #loss.backward()
    #nn.utils.clip_grad_norm(Q.parameters(), 1)
    optimizer.step()

def update_target_network(current, target, tau):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)

def plot_rewards(episode_rewards):
    plt.figure(1)
    plt.clf()
    x_ax = np.arange(len(episode_rewards)) * PLOT_EP_RATE
    rewards_t = torch.FloatTensor(episode_rewards)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(x_ax, rewards_t.cpu().numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(x_ax, means.numpy())
    plt.pause(0.001) # Pause to update plots
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


env = gym.make(ENV_NAME)
#env = wrap_deepmind(env)

n_actions = env.action_space.n
Q = DQN(n_actions)
Qp = DQN(n_actions)
if use_gpu:
    Q = Q.cuda()
    Qp = Qp.cuda()
update_target_network(Q,Qp,1) # Copy the params (tau=1)
memory = ReplayMemory(MEMORY_SIZE)
optimizer = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE, alpha=SQRD_GRAD_MOMNTM, eps=MIN_SQ_GR_MOMNTM)#, momentum=MOMENTUM)

n_performed_steps = 0
episode_rewards = []
for i_episode in range(NUM_EPISODES): 
    # Initialise environment
    print("Episode " + str(i_episode))
    obs = env.reset()
    #env.render()
    state = process_input(obs.squeeze())
    next_state = None
    ep_reward = 0
    started = False
    n_noops = 0
    for t in count():
        n_performed_steps += 1
        # Select and perform an action
        if n_performed_steps < INIT_EXPLORATION_FRAMES:
            action_n = random.randrange(n_actions)
        else:
            action_n = select_action(Tensor(state), Q, n_performed_steps - INIT_EXPLORATION_FRAMES)[0,0]

        # Force a limit on max no-op actions at the start of the episode
        if not started:
            if (action_n != 1):
                n_noops += 1
            elif n_noops > MAX_NOOP:
                action_n = 1
                started = True
            else:
                started = True

        obs, reward, done, _ = env.step(action_n)
        ep_reward += reward
        #env.render()
        # Observe new state
        next_state = process_input(obs.squeeze(), state)
        if done:
            next_state = None

        # Store transition in memory
        memory.push(state, LongTensor([[action_n]]), next_state, Tensor([reward]))

        # Move to the next state
        state = deepcopy(next_state)

        if (n_performed_steps % OPTIM_RATE) == 0:
            optimise_model()

        if (n_performed_steps % UPDATE_TARGET_STPS) == 0:
            update_target_network(Q, Qp, UPDATE_PARAM_TAU)
        if (n_performed_steps % SAVE_MODEL_FR) == 0:
            print("Saving model at "+ str(n_performed_steps) + " steps.")
            fname = "Qp_model_" + str(n_performed_steps) + ".dat"
            torch.save(Qp.state_dict(), fname)

        if done:
            # plot things
            break
    if (i_episode % PLOT_EP_RATE) == 0:
            episode_rewards.append(ep_reward)
            plot_rewards(episode_rewards)
            plt.figure(5)
            plt.clf()
            plt.plot(q_averages)
            plt.figure(6)
            plt.clf()
            plt.plot(r_averages)
            plt.pause(0.001) # Pause to update plots

torch.save(Q.state_dict(), "Q_model.dat")
torch.save(Qp.state_dict(), "Qp_model.dat")
print("Finished")