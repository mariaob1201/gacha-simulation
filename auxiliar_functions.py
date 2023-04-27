import numpy as np
import pandas as pd
import streamlit as st
from rewards import *






def human_format(num: object) -> object:
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])



def first_roll_out_dynamics(N, die, probabilities):
    # np.random.seed(1)
    initiall_roll = np.random.choice(die, size=N, p=probabilities)
    # print("The Outcome of the throw is: {}".format(initiall_roll))
    return initiall_roll


# Use np.random.choice to throw the die once and record the outcome
def second_roll_out_dynamics(frou, all_rewards):
    die = all_rewards[frou]['Types']
    probabilities = all_rewards[frou]['Probabilities']
    sec = np.random.choice(die, size=1, p=probabilities)

    return sec

def complete_dynamics(N, fr, all_rewards):
    plots = 0
    plots_8 = 0
    plots_16 = 0
    plots_32 = 0

    first_rou = first_roll_out_dynamics(N, fr['categories'], fr['probabilities'])

    #salida = []
    new = {}
    new['Poor'] = []
    new['Regular'] = []
    new['Amazing'] = []

    for i in first_rou:
        secnd = second_roll_out_dynamics(i, all_rewards)
        #salida.append(secnd[0])
        new[i].append(secnd[0])
        if 'Plot' in secnd[0]:
            plots += 1
        if '8*8' in secnd[0]:
            plots_8 += 1
        if '16*16' in secnd[0]:
            plots_16 += 1
        if '32*32' in secnd[0]:
            plots_32 += 1

    plots_types = [plots, plots_8, plots_16, plots_32]




    return new, plots_types
