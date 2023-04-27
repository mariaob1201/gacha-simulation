import streamlit as st
import numpy as np
from scipy.special import comb
import pandas as pd
import plotly.express as px

def human_format(num: object) -> object:
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def first_roll_out_dynamics(N, die, probabilities):
    initiall_roll = np.random.choice(die, size=N, p=probabilities)
    return initiall_roll


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

def hypergeom_pmf(N, A, n, x):
    '''
    Probability Mass Function for Hypergeometric Distribution
    :param N: population size
    :param A: total number of desired items in N
    :param n: number of draws made from N
    :param x: number of desired items in our draw of n items
    :returns: PMF computed at x
    '''
    Achoosex = comb(A, x)
    NAchoosenx = comb(N - A, n - x)
    Nchoosen = comb(N, n)

    return (Achoosex) * NAchoosenx / Nchoosen


def hypergeom_cdf(N, A, n, t, min_value=None):
    '''
    Cumulative Density Funtion for Hypergeometric Distribution
    :param N: population size
    :param A: total number of desired items in N
    :param n: number of draws made from N
    :param t: number of desired items in our draw of n items up to t
    :returns: CDF computed up to t
    '''
    if min_value:
        return np.sum([hypergeom_pmf(N, A, n, x) for x in range(min_value, t + 1)])

    return np.sum([hypergeom_pmf(N, A, n, x) for x in range(t + 1)])


def hypergeom_plot2(N, n, K, ps):
    '''
    Visualization of Hypergeometric Distribution for given parameters
    :param N: population size
    :param n: total number of desired items in N
    :param K: number of draws made from N
    :returns: Plot of Hyper geometric Distribution for given parameters
    '''
    x = np.arange(0, K + 1)
    y = [(hypergeom_pmf(N, n, K, x)) for x in range(K + 1)]
    df = pd.DataFrame({f"k events": x, 'Probability': y})
    fig = px.scatter(df, x=f"k events", y="Probability",
                     title=f"Plot Size {ps} - Probabilities of k events")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)