import streamlit as st
import pandas as pd
import logging
import requests
from auxiliar_functions import *
import plotly.express as px
from scipy.stats import binom
import time

reserve_multiplier = {'8x8': 1,
                      '16x16': 2.75,
                      '32x32': 41.67,
                      '64x64': 66.67,
                      '128x128': 666.67}
plot_base_price = 480

p_32 = 88
p_16 = 1800
p_8 = 6000
N = (p_32 + p_16 + p_8)



###################### REWARDS
st.sidebar.markdown("## 1. Controls")
n_players = st.sidebar.slider('Number of players', min_value=1, max_value=100000, value=80, step=1, key=121117711123)


st.sidebar.markdown("## 2.1 Hard Pity Odds")

p8_p = st.sidebar.slider('8x8 Plot', min_value=0.00, max_value=1.0, value=0.9, step=.01, key=1322911)
p16_p = st.sidebar.slider('16x16 Plot', min_value=0.00, max_value=min(0.0,1.00 - p8_p), value=0.05, step=.01, key=1392921)
p32_p = st.sidebar.slider('32x32 Plot', min_value=0.00, max_value=min(0.0,1.00-(p8_p+p16_p)), value=0.03, step=.01, key=1392192)

st.sidebar.markdown("## 2.2 Soft Pity Odds")
MB_sp = st.sidebar.slider('Mystery Boxes Chances on Soft Pity',min_value=0.00, max_value=1.0,
                           value=.9, step=.01, key=1213)
p8_sp = st.sidebar.slider('Plot 8*8 on Soft Pity',min_value=0.00, max_value=min(0.0,1.0-(MB_sp)),
                           value=.06, step=.01, key=1214)
p16_sp = st.sidebar.slider('Plot 16*16 on Soft Pity',min_value=0.00, max_value=min(0.0,1.0-(MB_sp+p8_sp)),
                            value=.004, step=.001, key=3)
p32_sp = st.sidebar.slider('Plot 32*32 Chances on Soft Pity',min_value=0.00,
                            max_value=min(0.0,1.0-(MB_sp+p8_sp+p16_sp)), value=0.001, step=.01, key=1)

st.sidebar.markdown("## 2.3 Normal Rewards Odds")

NoMB_nor = st.sidebar.slider('No Mystery Boxes Chances on Normal', min_value=0.00, max_value=1.00,
                              value=0.9, step=.01, key=11)
MB_nor = st.sidebar.slider('Mystery Boxes Chances on Normal',min_value=0.00, max_value=1.0-(NoMB_nor),
                           value=.06, step=.01, key=1212)
p8_nor = st.sidebar.slider('Plot 8*8 on Normal',min_value=0.00, max_value=1.0-(MB_nor+NoMB_nor),
                           value=.03, step=.01, key=12144)
p16_nor = st.sidebar.slider('Plot 16*16 on Normal',min_value=0.00, max_value=1.0-(MB_nor+NoMB_nor+p8_nor),
                            value=.006, step=.001, key=2)
p32_nor = st.sidebar.slider('Plot 32*32 Chances on Normal',min_value=0.00,
                            max_value=1.0-(MB_nor+NoMB_nor+p8_nor+p16_nor), value=.0, step=.01, key=100000000)
###################### REWARDS

probabilities_for_amazing = [p8_p, p16_p, p32_p, 1 - (p32_p + p8_p + p16_p)]

all_rewards = {'Plots': {'Types': ['Plot 8*8', 'Plot 16*16', 'Plot 32*32', 'Plot 64*64'],
                           'Probabilities': probabilities_for_amazing},
               'MysteryB': {'Types': ['Familiar Bronze Mystery Box (5)',
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
                           'Probabilities': [1/12]*12},
               'NMysteryB': {'Types': ['Material Pack 1', 'Material Pack 2', 'Material Pack 3',
                                  'Resource Pack 1', 'Resource Pack 2', 'Resource Pack 3','Recipe (3)'],
                        'Probabilities': [1/7,1/7,1/7,1/7,1/7,1/7,1-(1/7+1/7+1/7+1/7+1/7+1/7)]}}

##################################################################################################################
#################################MASTER CONVERSION FUNCTION ######################################################
##################################################################################################################

st.title(":blue[Gacha Rolls Dynamics]")

st.title("1. Master Conversion Function")

st.write(f'''The equivalence between Mana and hard currencies can be changed by control usage.''')

usd_spent = st.number_input('USD player Spent (AVG)', step=1., value=100., format="%.2f", key=12112)
rolls_by_usd_price = st.number_input('One roll equals USD', step=1., value=2.5, format="%.2f", key=122)
one_roll_mana_price = st.number_input('One roll equals Mana Units', step=1., value=10.0, format="%.2f", key=12221)


hpity = st.number_input('Hard Pity Rolls (over 1000)', step=1.0, value=4.0, format="%.2f", key=12333221)
hpity_p = hpity/1000
softp = st.number_input('Soft Pity Rolls (over 1000)', step=5.0, value=100.0, format="%.2f", key=1221121)
softp_p = softp/1000

eth_rate = ether_to_usd('USD')
conversionfunction = CurrenciesConversion(eth_rate, rolls_by_usd_price, one_roll_mana_price, [hpity_p, softp_p])



st.write(f''':green[Equivalence between currencies:]

    - REAL: The Ethereum corresponds to {eth_rate} USD at this moment.
    - IN GAME: One Roll price is {rolls_by_usd_price} USD or {one_roll_mana_price} Mana units (can change by control).''')

conv_fun, rolls = conversionfunction.master_conversion_function(usd_spent)


st.title("2. Reward Types")



n = rolls * n_players

plots_summary = {'Plot 8*8':0, 'Plot 16*16':0, 'Plot 32*32':0, 'Plot 64*64':0}
if rolls > 0:
    st.subheader('''2.1 Example (one player) VERSION 1''')
    st.write(
        f'''Reward Chances Link [here](https://docs.google.com/spreadsheets/d/1mDwwNCCeqi0cMjmO2SBg9iaw0gQaITJQoKw0Q2ozO1c/edit#gid=0).''')

    st.write(f''':green[One single player Paid {rolls} rolls], then the given rewards are distributed as follows:''')
    chart_data1 = pd.DataFrame(
        {'Pity Types': ['HardPity','SoftPity',  'Normal'],
         'Probabilities': [str(hpity_p*100)+' %', str(softp_p*100)+' %', str(100 - 100*(hpity_p + softp_p))+' %'],
         'Number of rolls of type': [int(rolls*hpity_p), int(rolls*softp_p),
                                      rolls-(int(rolls*softp_p)+int(rolls*hpity_p))]
         })

    st.dataframe(chart_data1)


    rtype = 'HardPity'
    st.caption(f''':moneybag: :moneybag: :red[Community Tier 5  - {rtype[:4] + ' ' + rtype[4:]} Odds] :moneybag: :moneybag:''')
    if chart_data1['Number of rolls of type'][0] > 0:
        nd = normal_distribution(
            all_rewards['NMysteryB']['Types'] + all_rewards['MysteryB']['Types'] + all_rewards['Plots']['Types'],
            19*[0]+probabilities_for_amazing,
            chart_data1['Number of rolls of type'][0], rtype[:4] + ' ' + rtype[4:])

        for k, val in nd.items():
            if 'Plot' in k:
                plots_summary[k] += val
    else:
        st.write('No Rewards on Hard Pity')


    rtype = 'SoftPity'
    st.caption(f''':moneybag: :moneybag: :red[Community Tier 5  - {rtype[:4] + ' ' + rtype[4:]} Odds] :moneybag: :moneybag:''')
    if chart_data1['Number of rolls of type'][1] > 0:
        try:
            nd=normal_distribution(
                all_rewards['NMysteryB']['Types'] + all_rewards['MysteryB']['Types'] + all_rewards['Plots']['Types'],
                7*[0] + 12*[MB_sp/12]+[p8_sp,p16_sp,p32_sp, 1-(p8_sp+p16_sp+p32_sp+MB_sp)],
                chart_data1['Number of rolls of type'][1], rtype[:4] + ' ' + rtype[4:])

            for k, val in nd.items():
                if 'Plot' in k:
                    plots_summary[k] += val
        except Exception as e:
            logging.error('In soft pity ', e)
    else:
        st.write('No Rewards on Soft Pity')

    rtype = 'Normal'
    st.caption(f''':moneybag: :moneybag: :red[Community Tier 5  - {rtype} Odds] :moneybag: :moneybag:''')
    if chart_data1['Number of rolls of type'][2] > 0:
        nd = normal_distribution(
            all_rewards['NMysteryB']['Types'] + all_rewards['MysteryB']['Types'] + all_rewards['Plots']['Types'],
            (7 * [NoMB_nor / 7]) + (12 * [MB_nor / 12]) + [p8_nor, p16_nor, p32_nor,
                                                       1 - (p8_nor + p16_nor + p32_nor + MB_nor + NoMB_nor)],
            chart_data1['Number of rolls of type'][2], rtype)

        for k, val in nd.items():
            if 'Plot' in k:
                plots_summary[k] += val
    else:
        st.write('No Rewards on Normal')

    def fun_prince(plot_str):
        if '8*8' in plot_str:
            return 480
        elif '16*16' in plot_str:
            return 1320
        elif '32*32' in plot_str:
            return 20000
        elif '64*64' in plot_str:
            return 32000


    d_ = {'Plots': list(plots_summary.keys()),
            'Amount': list(plots_summary.values()),
                'Total USD': [plots_summary[l]*fun_prince(l) for l in plots_summary.keys()]}
    pd_ = pd.DataFrame.from_dict(d_)
    st.write(f":red[BALANCE ON PLOTS REWARDS] {human_format(usd_spent-sum(d_['Total USD']))}: Spent by player {human_format(usd_spent)}, "
             f"Earned on Plots {human_format(sum(d_['Total USD']))} ")
    st.dataframe(pd_)

    ########################################################################################################################
    ################################2.2 Probability Distribution Function ##################################################
    ########################################################################################################################

    dynamics = RollOut(rolls)
    Amazing = hpity_p
    Regular = softp_p
    fr = {
        'categories': ['Plots', 'MysteryB', 'NMysteryB'],
        'probabilities': [Amazing, Regular, 1 - (Amazing+Regular)]
    }
    new, plots_earned = dynamics.complete_dynamics(fr, all_rewards)

    st.header("## 3. Other Mathematical Models of Gacha Dynamics")




    plots_8 = plots_earned[1]
    plots_16 = plots_earned[2]
    plots_32 = plots_earned[3]

    st.subheader('''2.1 Example (one player) VERSION 0''')



    plots_earn = {'8x8': plots_8, '16x16': plots_16, '32x32': plots_32}

    rtype = 'Plots'
    st.caption(f''':moneybag: :moneybag: :moneybag: :red[{rtype}] :moneybag: :moneybag: :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new,rtype, plots_earned, plot_base_price, plots_8, plots_16, plots_32, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no Plots rewards.''')


    rtype = 'MysteryB'
    st.caption(f''':moneybag: :moneybag: :red[{rtype}] :moneybag: :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new,rtype, plots_earned, plot_base_price, plots_8, plots_16, plots_32, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no Regular rewards.''')


    rtype = 'NMysteryB'
    st.caption(f''':moneybag: :red[{rtype}] :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new, rtype, plots_earned, plot_base_price, plots_8, plots_16, plots_32, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no NoMysteryB rewards.''')




st.subheader(f" 2.2 Probability Distribution Function")


st.write(f'''The chance for each reward class is:  

        - Plots: {str(Amazing * 100) + '%'} probability
    - Mystery Boxes: {str(Regular * 100) + '%'} probability
    - No Mystery Boxes: {str((1 - (Amazing + Regular)) * 100) + '%'} probability (as complement)''')


if n > 0:
    binomial_pt = BinomialDistributionFunction(n)
    st.write(f'''By {n_players} player{'s' if n_players>1 else ''}, {rolls} rolls each of them (:green[{n} total rolls]), we have.''')
    if Amazing > 0:
        binomial_pt.binomial_plot(Amazing, f'''Plots Reward - Chances on {n} rolls''')

    if Regular > 0:
        binomial_pt.binomial_plot(Regular, f'''Regular Reward - Chances on {n} rolls''')

    if 1 - (Amazing + Regular) > 0:
        binomial_pt.binomial_plot(1 - (Amazing + Regular), f'''NMysteryB Reward - Chances on {n} rolls''')

else:
    st.write(f'''ZERO rolls''')

########################################################################################################################
####################### Amazing Rewards - Probability Distribution Function on plots ###################################
########################################################################################################################

try:
    st.title("3. Plots Rewards - Probability Distribution Function")
    st.write(f'''With focus on :green[Plots Rewards] we fit an probability distribution function to explain how many plots we give to players after certain number of rolls.''')
    st.subheader("3.1 Explanation")
    st.write(f'''At the beginning, there are {N} plots as rewards distributed as follows:

            - 8x8: {round(100 * p_8 / N)} % ({p_8} plots)
    - 16x16: {round(100 * p_16 / N)} % ({p_16} plots)
    - 32x32: {round(100 * p_32 / N)} % ({p_32} plots)
once one is giving as a reward, the collection decreases (no replacement) and then its probability changes.''')

    st.write(
        f''':moneybag: {n_players} players, each of them spending {usd_spent} USD on average means:''')

    #total_spent = n_players * usd_spent
    #conversionfunction2 = CurrenciesConversion(url, rolls_by_usd_price, one_roll_mana_price)

    tolls_per_tspent, totalrolls = conversionfunction.master_conversion_function(n_players * usd_spent)

    dynamics = RollOut(totalrolls)
    dyn, plots_earn = dynamics.complete_dynamics(fr, all_rewards)

    plots_earned_by_rolls = plots_earn[0]

    ether = "{:.2f}".format(tolls_per_tspent['ETH'])
    players_earnings_usd = plot_base_price * (
                (plots_earn[1] * reserve_multiplier['8x8']) + (plots_earn[2] * reserve_multiplier['16x16']) + (
                    plots_earn[-1] * reserve_multiplier['32x32']))


    if plots_earned_by_rolls > 0:
        #Earnings player vs Runiverse
        st.write(f''':house: :green[Runiverse incomes:] 

                    - {human_format(n_players * usd_spent)} USD
    - {ether} ETHER''')


        st.write(f''':video_game: :green[Players Earnings:]

        - {totalrolls} rolls (In average {int(totalrolls/ n_players)} per player)
    - {human_format(tolls_per_tspent['Mana'])} Mana Units 
    - {plots_earned_by_rolls} plots ({human_format(players_earnings_usd)} USD) as a reward as follows:
        - 8x8: {plots_earn[1]} plots ({plot_base_price * plots_earn[1] * reserve_multiplier['8x8']} USD)
        - 16x16: {plots_earn[2]} plots ({plot_base_price * plots_earn[2] * reserve_multiplier['16x16']} USD)
        - 32x32: {plots_earn[3]} plots ({plot_base_price * plots_earn[3] * reserve_multiplier['32x32']} USD) ''')

        st.write(f''':red[Balance: {':arrow_up:' if (n_players * usd_spent)-players_earnings_usd>0 else ':arrow_down:'} {human_format((n_players * usd_spent)-players_earnings_usd)} USD]''')

        K = plots_earn[0]
        if plots_earn[0] > 0:
            st.subheader(f"3.2 Probability to have plot types as a reward for the given number of plots {(K)}.")
            st.write(
                f''':green[Probabilities are given on {K} events.] \n
                We use an hyper geometric distribution function to our dynamics.''')
            hyp_pl = HypergeometricDistributionFunction(K, N)
            hyp_pl.plots_stat('8x8', p_8)

            hyp_pl.plots_stat('16x16', p_16)

            hyp_pl.plots_stat('32x32', p_32)

    else:
        st.write(''':red[No plots were given by this spent and number of players]''')

except Exception as e:
    logging.error(f'''Error----> {e} <--- in 3rd section ''')
    pass
