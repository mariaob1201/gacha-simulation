import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.special import comb
from scipy.stats import binom
import requests

import logging

import random

from numpy.random import choice


rewards_dict = {'Plots': ['Plot 8*8', 'Plot 16*16', 'Plot 32*32', 'Plot 64*64'],
               'MysteryBoxes': ['Familiar Bronze Mystery Box (5)',
                                        'Mount Bronze Mystery Box (5)',
                                        'Architecture Bronze Mystery Box (5)',
                                        'Item Style Bronze Mystery Box (5)',
                                        'Familiar Silver Mystery Box (3)',
                                        'Mount Silver Mystery Box (3)',
                                        'Architecture Silver Mystery Box (3)',
                                        'Item Style Silver Mystery Box (3)',
                                        'Familiar Gold Mystery Box (2)',
                                        'Mount Gold Mystery Box (2)',
                                        'Architecture Gold Mystery Box (2)',
                                        'Item Style Gold Mystery Box (2)'],
               'NoMysteryBoxes': ['Material Pack 1', 'Material Pack 2', 'Material Pack 3',
                        'Resource Pack 1', 'Resource Pack 2', 'Resource Pack 3','Recipe (3)']}


def ether_to_usd(id):
    '''

    :param url: api request url
    :return: the conversion rate between USD and ethereum currencies
    '''
    url = f"https://min-api.cryptocompare.com/data/price?fsym=ETH&tsyms={id}"
    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json"},
        )
        data = response.json()
        USD = data['USD']
    except Exception as e:
        logging.error(f'''In ether to usd function <<<<< {e} >>>>>>''')
        USD = None
    return USD


class CurrenciesConversion():
    def __init__(self, eth_rate, rolls_by_usd_price, one_roll_mana_price, pity_list):
        self.eth_rate=eth_rate
        self.rolls_by_usd_price = rolls_by_usd_price
        self.one_roll_mana_price = one_roll_mana_price
        self.pity_list = pity_list

    def master_conversion_function(self, usd_spent):
        """

        :param usd_spent: usd spent by player
        :param rolls_by_usd_price: number of rolls that can be used by this spent
        :param one_roll_mana_price: conversion rate between one roll and mana units
        :param eth_rate: conversion rate between ethereum and usd
        :return: ordered table with currencies
        """
        try:
            conversion = {'USD': usd_spent,
                          'ETH': (usd_spent / self.eth_rate),
                          'Mana': self.one_roll_mana_price * int(usd_spent / self.rolls_by_usd_price)}
            rolls = int(usd_spent / self.rolls_by_usd_price)

            conv = {'Currencies': [key for key, val in conversion.items()],
                    'Spent': [val for key, val in conversion.items()],
                    'Total ROLLS': [rolls, rolls, rolls]}
            df = pd.DataFrame(conv)
            st.dataframe(df)
            return conversion, rolls

        except Exception as e:
            logging.error(f'''ERROR in master_conversion_function function <<<<< {e} >>>>>>''')
            return None, None


def human_format(num: object) -> object:
    '''

    :param num: gets a number
    :return: human format based on its magnitude
    '''
    try:
        num = float('{:.3g}'.format(num))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

    except Exception as e:
        logging.error(f'''In ether to usd function <<<<< {e} >>>>>>''')
        return None


class RollOut():
    def __init__(self, N):
        """

        :param N: Times the roll out was called
        """
        self.N = N

    def first_roll_out_dynamics(self, die, probabilities, flag):
        '''
            Roll out simplification
            :param die: vector of events that can occur
            :param probabilities: vector with probabilities of each element in die vector
            :param flag: Times the roll out was called N or 1, 1 value is for gacha specific rewards
            :returns: a vector of occurrences
        '''
        try:
            if flag:
                tot = self.N
            else:
                tot = 1

            if sum(probabilities)!=1:
                raise TypeError("Input should sum 1:")
            else:
                initiall_roll = np.random.choice(die, size=tot, p=probabilities)
                return list(initiall_roll)
        except Exception as e:
            raise TypeError("Error : ", e)

    def complete_dynamics(self, fr, all_rewards):
        """

        :param N: Number of rolls
        :param fr: dictionary with all rewards types and its probabilities
        :param all_rewards: dictionary with specific rewards per type and its probabilities
        :return: occurrences on each reward type and specific rewards
        """
        try:
            plots = 0
            plots_8 = 0
            plots_16 = 0
            plots_32 = 0
            plots_64 = 0
            first_rou = self.first_roll_out_dynamics(fr['categories'], fr['probabilities'], True)

            new = {}
            new['NoMysteryBoxes'] = []
            new['MysteryBoxes'] = []
            new['Plots'] = []

            for i in first_rou:
                secnd = self.first_roll_out_dynamics(all_rewards[i]['Types'], all_rewards[i]['Probabilities'], False)
                new[i].append(secnd[0])
                if 'Plot' in secnd[0]:
                    plots += 1
                if '8*8' in secnd[0]:
                    plots_8 += 1
                if '16*16' in secnd[0]:
                    plots_16 += 1
                if '32*32' in secnd[0]:
                    plots_32 += 1
                if '64*64' in secnd[0]:
                    plots_64 += 1

            plots_types = [plots, plots_8, plots_16, plots_32, plots_64]
            return new, plots_types
        except Exception as e:
            raise TypeError("complete_dynamics:", e)

class HypergeometricDistributionFunction():
    def __init__(self, K, N):
        self.K = K
        self.N = N

    def hypergeom_pmf(self, N, A, n, x):
        '''
        Probability Mass Function for Hypergeometric Distribution
        :param N: population size
        :param A: total number of desired items in N
        :param n: number of draws made from N
        :param x: number of desired items in our draw of n items
        :returns: PMF computed at x
        '''
        try:
            Achoosex = comb(A, x)
            NAchoosenx = comb(N - A, n - x)
            Nchoosen = comb(N, n)

            return (Achoosex) * NAchoosenx / Nchoosen

        except Exception as e:
            logging.error(f'''ERROR in hypergeom_cdf function <<<<< {e} >>>>>>''')

    def hypergeom_cdf(self, N, A, n, t, min_value=None):
        '''
        Cumulative Density Funtion for Hypergeometric Distribution
        :param N: population size
        :param A: total number of desired items in N
        :param n: number of draws made from N
        :param t: number of desired items in our draw of n items up to t
        :returns: CDF computed up to t
        '''
        try:
            if min_value:
                return np.sum([hypergeom_pmf(N, A, n, x) for x in range(min_value, t + 1)])

            return np.sum([hypergeom_pmf(N, A, n, x) for x in range(t + 1)])

        except Exception as e:
            logging.error(f'''ERROR in hypergeom_cdf function <<<<< {e} >>>>>>''')
            return None

    def hypergeom_plot(self, N, n, K, ps, mean, std):
        '''
        Visualization of Hypergeometric Distribution for given parameters
        :param N: population size
        :param n: total number of desired items in N
        :param K: number of draws made from N
        :returns: Plot of Hyper geometric Distribution for given parameters
        '''
        try:
            x = np.arange(0, K + 1)
            y = [(self.hypergeom_pmf(N, n, K, x)) for x in range(K + 1)]
            df = pd.DataFrame({f"Number of events": x, 'Probability': y})
            fig = px.scatter(df, x=f"Number of events", y="Probability",
                             title=f"Plot Size {ps} - Probabilities on {K} events")
            fig.add_vline(x=mean, line_width=3, line_dash="dash", line_color="green",
                          annotation_text=f'''Mean {"{:.2f}".format(mean)}''', annotation_position="bottom left")
            fig.add_vrect(x0=mean - 2 * abs(std), x1=mean + 2 * abs(std),
                          annotation_text=f'''CI''', annotation_position="top left",
                          fillcolor="red", opacity=0.25, line_width=0
                          )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        except Exception as e:
            logging.error(f'''ERROR in hypergeom_plot function <<<<< {e} >>>>>>''')

    def hypergeometric_stats(self, N, n, K):
        return {'mean': n * K / N,
                'variance': n * K * (N - n) * (N - K) / ((N * N * (N - 1)))}

    def plots_stat(self, ps, n):
        """
        Plot statistics graphs on an hypergeometric distribution function
        :param ps: type of rewards
        :param n: number of groups
        :param K: size of sample of interest
        :param N: Sample size
        :return:
        """
        try:
            stats_hg = self.hypergeometric_stats(self.N, n, self.K)
            if stats_hg['mean']>=0:
                mean = stats_hg['mean']
                variance = stats_hg['variance']
                std = np.sqrt(variance)
            else:
                mean=0
                variance=0
                std=0

            self.hypergeom_plot(self.N, n, self.K, ps, mean, std)
            st.write(f''':green[{ps} plots] After {self.K} events Statistics are:

                            - The mean is having {"{:.2f}".format(max(0,mean))} plots of that type
    - 95% Confidence interval: [{"{:.2f}".format(max(0, mean - 2 * abs(std)))}, {"{:.2f}".format(max(0,mean + 2 * abs(std)))}] plots of that type
    - Variance {"{:.2f}".format(variance)}''')

        except Exception as e:
            logging.error(f'''ERROR in plots_stat function <<<<< {e} >>>>>>''')


class BinomialDistributionFunction():
    def __init__(self, n):
        self.n = n

    def binomial_distribution(self, p):
        try:
            dist = [binom.pmf(r, self.n, p) for r in list(range(self.n + 1))]
            mean = self.n * p
            variance = self.n * p * (1 - p)
            return {'distribution': dist,
                    'binomial_mean': mean,
                    'binomial_variance': variance}

        except Exception as e:
            logging.error('Error in binomial distribution ', e)
            return {}

    def binomial_plot(self, p, title):
        """

        :param n: sample size
        :param p: probability of sucess
        :param title: title for specific reward
        :return:
        """
        try:
            bin = self.binomial_distribution(p)
            if 'distribution' in bin.keys():
                dist = bin['distribution']
                mean = bin['binomial_mean']
                variance = bin['binomial_variance']
                df = pd.DataFrame({f"Number of events": list(range(self.n + 1)), 'Probability': dist})
                fig = px.scatter(df, x=f"Number of events", y="Probability",
                                 title=f"{title}")
                fig.add_vline(x=mean, line_width=3, line_dash="dash", line_color="green",
                              annotation_text=f'''Mean {"{:.2f}".format(mean)}''', annotation_position="bottom left")
                fig.add_vrect(x0=mean - 2 * abs(np.sqrt(variance)), x1=mean + 2 * abs(np.sqrt(variance)),
                              annotation_text=f'''CI : [{"{:.2f}".format(mean - 2 * abs(np.sqrt(variance)))}, {"{:.2f}".format(mean + 2 * abs(np.sqrt(variance)))}]''',
                              annotation_position="top right",
                              fillcolor="red", opacity=0.25, line_width=0
                              )
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                st.write(f'''Statistics are:
    
    - The mean is having {"{:.2f}".format(mean)} rewards of this type
    - 95% Confidence interval: [{"{:.2f}".format(max(0, mean - 2 * abs(np.sqrt(variance))))}, {"{:.2f}".format(mean + 2 * abs(np.sqrt(variance)))}]
    - Variance {"{:.2f}".format(variance)}''')

        except Exception as e:
            raise TypeError(f"ERROR in binomial_plot function <<<<< {e} >>>>>>")

def rew_type_f(item):
    res = 'Plot'
    if 'Mystery' in item:
        if 'Bronze' in item:
            res = 'MB: Bronze'
        elif 'Gold' in item:
            res = 'MB: Gold'
        elif 'Silver' in item:
            res = 'MB: Silver'
    elif 'Material' in item:
        res = 'Pack: Material'
    elif 'Resource' in item:
        res = 'Pack: Resource'
    elif 'Recipe' in item:
        res = 'Pack: Recipe'
    return res

def function(new, rtype, plots_earned, plot_base_price, rolls, reserve_multiplier,
             conv_fun, plots_earn):
    """

    :param new:
    :param rtype:
    :param plots_earned:
    :param plot_base_price:
    :param plots_8:
    :param plots_16:
    :param plots_32:
    :param rolls:
    :param reserve_multiplier:
    :param conv_fun:
    :param plots_earn:
    :return:
    """
    try:
        dict0 = {}
        for i in new[rtype]:
            if i not in dict0.keys():
                dict0[i] = 1
            else:
                dict0[i] += 1

        sorted_ = sorted(dict0.items(), key=lambda x: x[0])
        dict1 = dict(sorted_)
        dict_f = {'Rewards': dict1.keys(),
                  'Quantity': dict1.values(),
                  'RewardType': [rew_type_f(i) for i in dict1.keys()]}

        df = pd.DataFrame(dict_f)
        rtype2 = 'Soft Pity'
        if 'No' in rtype:
            rtype2='Normal'
        if 'Plot' in rtype:
            rtype2='Hard Pity'

        fig = px.bar(df, x="Rewards", y="Quantity", text="Quantity", color='RewardType',
                     title=f"""{rtype2} Rewards on {rolls} ROLLS""",
                     height=400
                     )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        if plots_earned[0] > 0:
            plots_earns = plot_base_price * ((plots_earn['8x8'] * reserve_multiplier['8x8']) + (
                    plots_earn['16x16'] * reserve_multiplier['16x16']) + (plots_earn['32x32'] * reserve_multiplier['32x32'])
                                             + (plots_earn['64x64'] * reserve_multiplier['64x64']))
            runi_earns = conv_fun['USD'] - plots_earns
            st.write(f''':green[Player earns {int(plots_earned[0])} plots] as follows: {plots_earn}. That means in USD: 

                        - Player earnings by plots: {human_format(plots_earns)} USD 
            - Runiverse {'earns' if runi_earns > 0 else 'loses'} about {human_format(runi_earns)} USD''')

    except Exception as e:
        logging.error(f'''ERROR in function <<<<< {e} >>>>>>''')


def normal_distribution(items_list, probabilities_list, N, title):
    dict_items = {}
    fin_dict = {}
    try:
        randomNumberList = choice(
            items_list, N, p=probabilities_list)

        for l in randomNumberList:
            if l not in dict_items.keys():
                dict_items[l]=1
            else:
                dict_items[l]+=1

        fin_dict = {'Reward':list(dict_items.keys()),
                'Amount':list(dict_items.values()),
                    'Type':[rew_type_f(k) for k in list(dict_items.keys())]}

        fig = px.bar(
            fin_dict, x=f"Type", y="Amount",color='Reward',text="Amount",
            title=f"Rewards on {title}")
        fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    except Exception as e:
        logging.error('Error ', e)

    return dict_items
