import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.special import comb
from scipy.stats import binom
import requests

import logging


class CurrenciesConversion():
    def __init__(self, url, rolls_by_usd_price, one_roll_mana_price):
        self.url = url
        self.rolls_by_usd_price = rolls_by_usd_price
        self.one_roll_mana_price = one_roll_mana_price
    def ether_to_usd(self):
        '''

        :param url: api request url
        :return: the conversion rate between USD and ethereum currencies
        '''
        try:
            response = requests.get(
                self.url,
                headers={"Accept": "application/json"},
            )
            data = response.json()
            USD = data['USD']

        except Exception as e:
            logging.error(f'''In ether to usd function <<<<< {e} >>>>>>''')
            USD = None

        return USD

    def master_conversion_function(self, eth_rate, usd_spent):
        """

        :param usd_spent: usd spent by player
        :param rolls_by_usd_price: number of rolls that can be used by this spent
        :param one_roll_mana_price: conversion rate between one roll and mana units
        :param eth_rate: conversion rate between ethereum and usd
        :return: ordered table with currencies
        """
        try:
            conversion = {'USD': usd_spent,
                          'ETH': (usd_spent / eth_rate),
                          'Mana': self.one_roll_mana_price * int(usd_spent / self.rolls_by_usd_price)}
            rolls = int(usd_spent / self.rolls_by_usd_price)

            conv = {'Currencies': [key for key, val in conversion.items()],
                    'Spent': [val for key, val in conversion.items()],
                    'ROLLS Equivalence': [rolls,rolls,rolls]}
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

        :param N: Times the roll out is mande
        """
        self.N=N
    def first_roll_out_dynamics(self, die, probabilities, flag):
        '''
            Roll out simplification
            :param die: vector of events that can occur
            :param probabilities: vector with probabilities of each element in die vector
            :param flag: Times the roll out is mande N or 1, last value is for sub cases
            :returns: a vector of occurrences
        '''
        try:
            if flag:
                tot = self.N
            else:
                tot = 1
            initiall_roll = np.random.choice(die, size=tot, p=probabilities)
            return initiall_roll
        except Exception as e:
            logging.error(f'''ERROR in first_roll_out_dynamics function <<<<< {e} >>>>>>''')
            return []

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
            first_rou = self.first_roll_out_dynamics(fr['categories'], fr['probabilities'], True)

            new = {}
            new['Poor'] = []
            new['Regular'] = []
            new['Amazing'] = []

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

            plots_types = [plots, plots_8, plots_16, plots_32]

            return new, plots_types
        except Exception as e:
            logging.error(f'''ERROR in complete_dynamics function <<<<< {e} >>>>>>''')
            return None, None

class HypergeometricDistributionFunction():
    def __init__(self, K, N):
        self.K=K
        self.N=N
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
            fig.add_vrect(x0=mean-2*abs(std), x1=mean+2*abs(std),
                              annotation_text=f'''CI''', annotation_position="top left",
                              fillcolor="red", opacity=0.25, line_width=0
                              )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        except Exception as e:
            logging.error(f'''ERROR in hypergeom_plot function <<<<< {e} >>>>>>''')


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
            N=self.N
            K=self.K
            mean = n * K / N
            variance = n * K * (N - n) * (N - K) / ((N * N * (N - 1)))
            std = np.sqrt(variance)
            self.hypergeom_plot(N, n, K, ps, mean, std)
            st.write(f''':green[{ps} plots] After {K} events Statistics are:
        
                            - The mean is having {"{:.2f}".format(mean)} plots of that type
    - 95% Confidence interval: [{"{:.2f}".format(max(0, mean - 2 * abs(std)))}, {"{:.2f}".format(mean + 2 * abs(std))}] plots of that type
    - Variance {"{:.2f}".format(variance)}''')
        except Exception as e:
            logging.error(f'''ERROR in plots_stat function <<<<< {e} >>>>>>''')

class BinomialDistributionFunction():
    def __init__(self, n):
        self.n=n
    def binomial_plot(self, p, title):
        """

        :param n: sample size
        :param p: probability of sucess
        :param title: title for specific reward
        :return:
        """
        try:
            r_values = list(range(self.n + 1))
            dist = [binom.pmf(r, self.n, p) for r in r_values]
            mean = self.n * p
            variance = self.n * p * (1 - p)
            df = pd.DataFrame({f"Number of events": r_values, 'Probability': dist})
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
            logging.error(f'''ERROR in binomial_plot function <<<<< {e} >>>>>>''')


def function(new,rtype, plots_earned, plot_base_price, plots_8, plots_16, plots_32, rolls, reserve_multiplier, conv_fun, plots_earn):
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
        dict_f = {'Rewards': dict1.keys(), 'Quantity': dict1.values()}

        df = pd.DataFrame(dict_f)

        fig = px.bar(df, x="Rewards", y="Quantity", text="Quantity",
                     title=f"""{rtype} Rewards on {rolls} ROLLS""",
                     height=400
                     )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        if plots_earned[0] > 0:
            plots_earns = plot_base_price * ((plots_8 * reserve_multiplier['8x8']) + (
                    plots_16 * reserve_multiplier['16x16']) + (plots_32 * reserve_multiplier['32x32']))
            runi_earns = conv_fun['USD'] - plots_earns
            st.write(f''':green[Player earns {int(plots_earned[0])} plots] as follows: {plots_earn}. That means in USD: 
    
                        - Player earns by plots: {human_format(plots_earns)} USD 
            - Runiverse {'earns' if runi_earns > 0 else 'loses'} about {human_format(runi_earns)} USD''')
    except Exception as e:
        logging.error(f'''ERROR in function <<<<< {e} >>>>>>''')


