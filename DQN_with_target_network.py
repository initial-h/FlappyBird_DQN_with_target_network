# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:24:12 2018

@author: len
"""
#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import tflearn 
import cv2
import sys
sys.path.append("game/")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import matplotlib.pylab as plt
import pygame
import time

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon 探索度，epsilon贪心策略
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1


def createNetwork():
    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers   
    h_conv1 = tflearn.conv_2d(s, 32, 8, strides=4, activation='relu',weights_init=tflearn.initializations.truncated_normal(stddev=0.01))  
    h_pool1 = tflearn.max_pool_2d(h_conv1,2,2)

    h_conv2 = tflearn.conv_2d(h_pool1,64,4,strides=2,activation='relu',weights_init=tflearn.initializations.truncated_normal(stddev=0.01))
   
    h_conv3 = tflearn.conv_2d(h_conv2,64,3,strides=1,activation='relu',weights_init=tflearn.initializations.truncated_normal(stddev=0.01))

    h_conv3_flat = tflearn.reshape(h_conv3,[-1,1600])
    h_fc1 = tflearn.fully_connected(incoming=h_conv3_flat,n_units=512,activation='relu',weights_init=tflearn.initializations.truncated_normal(stddev=0.01))

    readout = tflearn.fully_connected(incoming=h_fc1,n_units=ACTIONS)

    return s, readout

def plot_score(score_list):
    #plt.plot(np.arange(len(score_list)), score_list)
    plt.semilogx(np.arange(len(score_list)), score_list)
    plt.ylabel('score')
    plt.xlabel('game')
    plt.show()
    
def plot_cost(cost_list):
    #plt.plot(np.arange(len(cost_list)), cost_list)
    plt.semilogx(np.arange(len(cost_list)), cost_list)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
    
def plot_maxvalue(value_list):
    #plt.plot(np.arange(len(value_list)),value_list)
    plt.semilogx(np.arange(len(value_list)),value_list)
    plt.ylabel('max_value')
    plt.xlabel('training step')
    plt.show()

score_list = []
cost_list = []
value_list = []

def trainNetwork(s,q_values,st,target_q_values,reset_target_network_params,sess,train,acceleration):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])#action
    y = tf.placeholder("float", [None])#Q现实
    readout_action = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)
    
    #readout是在s状态下，采取各个动作的Q值，a是实际采取的动作，
    # multiply一下再reduce_sum就是网络预测的在状态s下采取动作a的Q值，也就是Q估计
    #readout is the Q value of each action at state s
    #the result of multiply and reduce_sum is the Q value from network at state s with action a,that is Q evaluate 
    cost = tf.reduce_mean(tf.square(y - readout_action))
    #这里就是Q现实与Q估计的差值
    #the diffence between Q real and Q evaluate
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()
    game_state.acceleration = acceleration#if acceleration
    # store the previous observations in replay memory
    D = deque()
    cost_tmp = deque(maxlen = 1000)##save a length of cost to judge if we should adjust the step parameter C
    cost_tmp.append(0)#to avoid np.mean nan

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)#image_data, reward, terminal
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    if not os.path.exists("saved_networks"):
        os.makedirs("saved_networks")
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
     
    t = 0
    score = 0
    start_time = time.time()#to compute the time to avoid ploting too frequently 
    ##################################train the network#######################################        
    if train:
        sess.run(reset_target_network_params)
#        print('reset_target_network_params!!!!!!!')

        # start training
        epsilon = INITIAL_EPSILON
        game_num = 0
        C = 1#every C step reset target Q network
        while "flappy bird" != "angry bird":
            # choose an action epsilon greedily
            readout_t = q_values.eval(feed_dict={s : [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
#                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t[random.randrange(ACTIONS)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1
            else:
                a_t[0] = 1 # do nothing
    
            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    
            # run the selected action and observe next state and reward
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
    
            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
    
            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)
    
                # get the batch variables
                #s_t, a_t, r_t, s_t1
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]
    
                y_batch = []#Q现实
                readout_j1_batch = target_q_values.eval(feed_dict = {st : s_j1_batch})
                #tensorflow还可以这样操作，这个eval可以转变BN吗，可以试试
                for i in range(0, len(minibatch)):
                    terminal_ = minibatch[i][4]#this terminal_ should be different from terminal
                    # if terminal_, only equals reward
                    if terminal_:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
    
                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch}
                )
                cost_list.append(cost.eval(feed_dict = {y:y_batch,a:a_batch,s:s_j_batch}))
                cost_tmp.append(cost.eval(feed_dict = {y:y_batch,a:a_batch,s:s_j_batch}))
                
            # update the old values
            s_t = s_t1
            t += 1
            
    #         save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)
                
            #adjust the step C by cost_tmp
            if t % C == 0:
                sess.run(reset_target_network_params)
#                print('reset_target_network_params!!!!!!!',C)
                if np.mean(cost_tmp)>0.5:
                    C*=2
                else:
                    C = np.ceil(C/2)
    
            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
    
    #        print("TIMESTEP", t, "/ STATE", state, \
    #            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
    #            "/ Q_MAX %e" % np.max(readout_t))
            # write info to files
            '''
            if t % 10000 <= 100:
                a_file.write(",".join([str(x) for x in readout_t]) + '\n')
                h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
                cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
            '''
            score+=r_t
            value_list.append(np.max(readout_t))
            if terminal:
                game_state.acceleration = acceleration#if acceleration
                game_num+=1
                score_list.append(score)
                if time.time()-start_time>60:#don't print too frequently
                    plot_cost(cost_list)
                    plot_maxvalue(value_list)
                    plot_score(score_list)
                    print('---------game:',game_num,state,'train_step:',t,'score:',score)
                    print('C is:',C,'mean cost:',np.mean(cost_tmp))
                    start_time = time.time()
                score=0
    ##########################test network###############################################3
    else:
        terminal = False
        while not terminal:
            # choose an action epsilon greedily
            readout_t = q_values.eval(feed_dict={s : [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if t % FRAME_PER_ACTION == 0:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
            else:
                a_t[0] = 1 # do nothing

            # run the selected action and observe next state and reward
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            s_t = s_t1
            
            score+=r_t
            if time.time()-start_time>10:
                print('current score:',score)
                start_time = time.time()
            if terminal:
                print('total score:',score)


#for human to play the game 
###################space key to fly, do nothing to fall down#################################
#just take a try to feel the game's difficulty
def play_by_human():
    game_state = game.GameState()
    game_state.acceleration = False
    game_1 = True
    score = 0
    
    start_time = time.time()
    while game_1:
        a_t = np.zeros([2])
        keystate = pygame.key.get_pressed() 
        if keystate[pygame.K_SPACE]:
            a_t = np.array([0,1])
        else:
            a_t = np.array([1,0])
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        score += r_t
        if time.time()-start_time>10:
            print('current score:',score)
        if terminal:
            print('total score:',score)
#            print('time:',time.time()-start)
            game_1 = False              

def playGame(train):
    sess = tf.InteractiveSession()
    s, q_network= createNetwork()
    network_params = tf.trainable_variables()
    q_values = q_network
    
    st, target_q_network = createNetwork()
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network
    
    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]  
    trainNetwork(s, q_values, st,target_q_values,reset_target_network_params, sess,train=train,acceleration=train)


#if play_by_human = True, play it by yourself with space key 
#if train = True, train the network with acceleration(do not use FPS)
#if train = False, test the network model
# human_play = False       
# train = True    

def main():
    if human_play:  
        play_by_human()
    elif train:
        playGame(train)
    else:
        playGame(train)
# main()

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--train',type =bool,default = False)
   parser.add_argument('--human_play',type=bool,default = False)
   parser.add_argument('--test',type = bool,default = True)
   
   args = parser.parse_args()
   if args.human_play:
       play_by_human()
   elif args.train:
       playGame(train=True)
   elif args.test:
       playGame(train=False)
