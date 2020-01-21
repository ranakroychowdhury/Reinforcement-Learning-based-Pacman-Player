Answer to 9.4 (b)
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:18:57 2019

@author: Ranak Roy Chowdhury
"""
import numpy as np
import math
from random import randrange

def readFiles():
    prob = []
    with open("rewards.txt", "r") as file:
        rew = [int(line.split()[0]) for line in file]
    with open("prob_a1.txt", "r") as file:
        west = [[float(x) for x in line.split()] for line in file]
    with open("prob_a2.txt", "r") as file:
        north = [[float(x) for x in line.split()] for line in file]
    with open("prob_a3.txt", "r") as file:
        east = [[float(x) for x in line.split()] for line in file]
    with open("prob_a4.txt", "r") as file:
        south = [[float(x) for x in line.split()] for line in file]
    prob.append(west)
    prob.append(north)
    prob.append(east)
    prob.append(south)
    return prob, rew
    
def makeMatrices(prob, rew, n, a):
    trans = np.zeros((a, n, n))
    for i in range(a):
        for j in range(len(prob[i])):
            row = int(prob[i][j][0]) - 1
            col = int(prob[i][j][1]) - 1
            val = prob[i][j][2]
            trans[i][row][col] = val
    rewards = np.zeros((n, 1))
    for i in range(n):
        rewards[i][0] = rew[i]
    return trans, rewards

def policyEvaluation(trans, rewards, n, a, gamma, value):
    for s in range(n):
        arr = []
        for i in range(a):
            tot = 0
            for s_prime in range(n):
                tot += trans[i][s][s_prime] * value[s_prime]
            arr.append(tot)
        value[s] = rewards[s] + gamma * max(arr)
    return value
             
def optimalPolicy(trans, n, a, value):
    p = []
    for s in range(n):
        arr = []
        for i in range(a):
            tot = 0
            for s_prime in range(n):
                tot += trans[i][s][s_prime] * value[s_prime]
            arr.append(tot)
        p.append(arr.index(max(arr)))
    p = np.array(p)
    return p
        
def valueIteration(trans, rewards, n, a, gamma, iteration):
    value = np.zeros(n)
    for i in range(iteration):
        if i%1000 == 0:
            print(i)
        value = policyEvaluation(trans, rewards, n, a, gamma, value)
    policy = optimalPolicy(trans, n, a, value)
    r = int(math.sqrt(n))
    value = value.reshape((r, r))
    value = value.transpose()
    policy = policy.reshape((r, r))
    policy = policy.transpose()
    return value, policy

def printResult(value, policy, n):
    np.savetxt('value_value.out', value, delimiter=' ')
    np.savetxt('policy_value.out', policy, delimiter=' ')
    string = []
    r = int(math.sqrt(n))
    for i in range(r):
        s = []
        for j in range(r):
            if value[i][j] == 0:
                s.append('0 ')
            else:
                if policy[i][j] == 0:
                    s.append('< ')
                elif policy[i][j] == 1:
                    s.append('^ ')
                elif policy[i][j] == 2:
                    s.append('> ')
                else:
                    s.append('v ')
        string.append(s)
    with open('action_value.out', 'w') as file:
        file.writelines('\t'.join(str(j) for j in i) + '\n' for i in string)
        
if __name__ == "__main__":
    print("Reading Files")
    prob, rew = readFiles()
    n = 81;
    a = 4;
    gamma = 0.9925
    iteration = 10000
    trans, rewards = makeMatrices(prob, rew, n, a)
    value, policy = valueIteration(trans, rewards, n, a, gamma, iteration)
    printResult(value, policy, n)

