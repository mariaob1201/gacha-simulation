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
                      '32x32': 42,
                      '64x64': 67,
                      '128x128': 667}
plot_base_price = 480

p_32 = 88
p_16 = 1800
p_8 = 6000
N = (p_32 + p_16 + p_8)



###################### REWARDS


st.sidebar.markdown("## 2.1 Hard Pity Odds")

p8_p = st.sidebar.slider('8x8 Plot', min_value=0.00, max_value=1.0, value=0.9, step=.01, key=1322911)
p16_p = st.sidebar.slider('16x16 Plot', min_value=0.00, max_value=1.00 - p8_p+0.001, value=0.05, step=.01, key=1392921)
p32_p = st.sidebar.slider('32x32 Plot', min_value=0.00, max_value=1.00-(p8_p+p16_p)+0.001, value=0.03, step=.01, key=1392192)



#st.sidebar.markdown("## 2.3 Normal Rewards Odds")


###################### REWARDS

probabilities_for_amazing = [p8_p, p16_p, p32_p, 1 - (p32_p + p8_p + p16_p)]

all_rewards = {'Plots': {'Types': ['Plot 8*8', 'Plot 16*16', 'Plot 32*32', 'Plot 64*64'],
                           'Probabilities': probabilities_for_amazing},
               'MysteryBoxes': {'Types': ['Familiar Bronze Mystery Box (5)',
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
               'NoMysteryBoxes': {'Types': ['Material Pack 1', 'Material Pack 2', 'Material Pack 3',
                                  'Resource Pack 1', 'Resource Pack 2', 'Resource Pack 3','Recipe (3)'],
                        'Probabilities': [1/7,1/7,1/7,1/7,1/7,1/7,1-(1/7+1/7+1/7+1/7+1/7+1/7)]}}

##################################################################################################################
#################################MASTER CONVERSION FUNCTION ######################################################
##################################################################################################################

st.title(":red[Monetization System based on Gacha Process]")

st.title(":blue[1. Master Conversion Function]")

n_players = st.number_input('Number of players', step=1.0, value=100.0, format="%.0f", key=121117711123)

st.write(f'''The equivalence between Mana and hard currencies can be changed by control usage.''')

usd_spent = st.number_input('USD player Spent', step=1., value=50., format="%.2f", key=12112)
rolls_by_usd_price = st.number_input('One roll price (USD)', step=1., value=2.5, format="%.2f", key=122)
one_roll_mana_price = st.number_input('One roll price (Mana Units)', step=1., value=10.0, format="%.2f", key=12221)


hpity = st.number_input('Hard Pity Rolls (ON 1000R)', step=1.0, value=4.0, format="%.2f", key=12333221)
hpity_p = hpity/1000
softp = st.number_input('Soft Pity Rolls (ON 1000R)', step=5.0, value=100.0, format="%.2f", key=1221121)
softp_p = softp/1000

eth_rate = ether_to_usd('USD')
conversionfunction = CurrenciesConversion(eth_rate, rolls_by_usd_price, one_roll_mana_price, [hpity_p, softp_p])



st.write(f''':green[Equivalence between currencies:]

    - REAL: The Ethereum corresponds to {eth_rate} USD at this moment.
    - IN GAME: One Roll price is {rolls_by_usd_price} USD or {one_roll_mana_price} Mana units (can change by control).''')

conv_fun, rolls = conversionfunction.master_conversion_function(usd_spent)


st.title(":blue[2. Reward Types]")



n = rolls * n_players

plots_summary = {'Plot 8*8':0, 'Plot 16*16':0, 'Plot 32*32':0, 'Plot 64*64':0}
if rolls > 0:
    st.subheader('''2.1 Example (one player) VERSION 1''')
    st.write(
        f'''Reward Chances Link [here](https://docs.google.com/spreadsheets/d/1mDwwNCCeqi0cMjmO2SBg9iaw0gQaITJQoKw0Q2ozO1c/edit#gid=0).''')

    st.write(f''':green[{n} rolls were paid in total ({n_players} players)], then the given rewards are distributed as follows:''')
    hrolls = int(n*hpity_p)
    srolls = int((n-int(n*hpity_p))*softp_p)
    chart_data1 = pd.DataFrame(
        {'Pity Types': ['HardPity','SoftPity',  'Normal'],
         'Probabilities': [str(hpity_p*100)+' %', str(softp_p*100)+' %', str(100 - 100*(hpity_p + softp_p))+' %'],
         'Number of rolls of type': [hrolls, srolls, n-(hrolls+srolls)]
         })

    st.dataframe(chart_data1)


    rtype = 'HardPity'
    st.markdown(f''':trophy: :trophy: :trophy: :trophy: :red[Community Tier 5  - {rtype[:4] + ' ' + rtype[4:]} Odds] :trophy: :trophy: :trophy: :trophy:''')
    if chart_data1['Number of rolls of type'][0] > 0:
        nd = normal_distribution(
            all_rewards['NoMysteryBoxes']['Types'] + all_rewards['MysteryBoxes']['Types'] + all_rewards['Plots']['Types'],
            (19*[0])+probabilities_for_amazing, int(hrolls), rtype[:4] + ' ' + rtype[4:])

        for k, val in nd.items():
            if 'Plot' in k:
                plots_summary[k] += val
    else:
        st.write('No Rewards on Hard Pity')


    rtype = 'SoftPity'
    st.markdown(f''':trophy: :trophy: :trophy: :red[Community Tier 5  - {rtype[:4] + ' ' + rtype[4:]} Odds] :trophy: :trophy: :trophy:''')
    if chart_data1['Number of rolls of type'][1] > 0:
        MB_sp_b = st.number_input(
            'Bronze Mystery Boxes Chances on Soft Pity', step=0.001, value=.24, format="%.4f", key=100000)
        MB_sp_s = st.number_input(
            'Silver Mystery Boxes Chances on Soft Pity', step=0.001, value=.28, format="%.4f", key=1100000)
        MB_sp_g = st.number_input(
            'Gold Mystery Boxes Chances on Soft Pity', step=0.001, value=.36, format="%.4f", key=1010000)
        #min_value=0.00, max_value=1.0, value=.9, step=.01, key=1213)
        p8_sp = st.number_input(
            'Plot 8*8 on Soft Pity', step=0.001, value=.08, format="%.4f", key=123332212)
            #min_value=0.00, max_value=min(0.0, 1.0 - (MB_sp) + 0.001), value=.06, step=.01, key=1214)
        p16_sp = st.number_input(
            'Plot 16*16 on Soft Pity', step=0.001, value=.03, format="%.4f", key=123332213)
            #min_value=0.00, max_value=min(0.0, 1.0 - (MB_sp + p8_sp) + 0.001), value=.004, step=.001, key=3)
        p32_sp = st.number_input(
            'Plot 32*32 Chances on Soft Pity', step=0.001, value=.01, format="%.4f", key=123332214)
        #min_value=0.00,max_value=min(0.0, 1.0 - (MB_sp + p8_sp + p16_sp) + 0.001), value=0.001, step=.01, key=1)
        try:
            nd=normal_distribution(
                all_rewards['NoMysteryBoxes']['Types'] + all_rewards['MysteryBoxes']['Types'] + all_rewards['Plots']['Types'],
                7*[0] + 4*[MB_sp_b/4]+4*[MB_sp_s/4]+4*[MB_sp_g/4]+[p8_sp,p16_sp,p32_sp,
                                                                   1-(p8_sp+p16_sp+p32_sp+MB_sp_s+MB_sp_g+MB_sp_b)],
                int(srolls), rtype[:4] + ' ' + rtype[4:])

            for k, val in nd.items():
                if 'Plot' in k:
                    plots_summary[k] += val
        except Exception as e:
            logging.error('In soft pity ', e)
    else:
        st.write('No Rewards on Soft Pity')

    rtype = 'Normal'
    st.markdown(f''':trophy: :trophy: :red[Community Tier 5  - {rtype} Odds] :trophy: :trophy:''')

    if chart_data1['Number of rolls of type'][2] > 0:
        NoMB_nor = st.number_input(
            'No Mystery Boxes Chances on Normal', step=0.001, value=.9, format="%.4f", key=2011)
        MB_nor_b = st.number_input(
            'Bronze Mystery Boxes Chances on Normal', step=0.001, value=.01, format="%.4f", key=2021)
        MB_nor_s = st.number_input(
            'Silver Mystery Boxes Chances on Normal', step=0.001, value=.02, format="%.4f", key=2022)
        MB_nor_g = st.number_input(
            'Gold Mystery Boxes Chances on Normal', step=0.001, value=.03, format="%.4f", key=20231)
        p8_nor = st.number_input(
            'Plot 8*8 on Normal', step=0.001, value=.02, format="%.4f", key=203)
        p16_nor = st.number_input(
            'Plot 16*16 on Normal', step=0.001, value=.01, format="%.4f", key=204)
        p32_nor = st.number_input(
            'Plot 32*32 Chances on Normal', step=0.001, value=.01, format="%.4f", key=205)

        nd = normal_distribution(
            all_rewards['NoMysteryBoxes']['Types'] + all_rewards['MysteryBoxes']['Types'] + all_rewards['Plots']['Types'],
            (7 * [NoMB_nor / 7]) + (4 * [MB_nor_b / 4]) + (4 * [MB_nor_s / 4]) + (4 * [MB_nor_g / 4])
            + [p8_nor, p16_nor, p32_nor,1 - (p8_nor + p16_nor + p32_nor + MB_nor_b + MB_nor_s + MB_nor_g + NoMB_nor)],
            int(n-(hrolls+srolls)), rtype)

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

    st.write(
        f":green[BALANCE ON PLOTS REWARDS:] {':arrow_up:' if usd_spent > sum(d_['Total USD']) else ':arrow_down:'} {human_format(usd_spent - sum(d_['Total USD']))} USD.")
    st.write(
        f''':house: :green[Runiverse incomes] (one player): {human_format(usd_spent)} USD''')

    st.write(
        f''':video_game: :green[Players Earnings] ({n} rolls): {human_format(sum(d_['Total USD']))} USD ({sum(list(plots_summary.values()))} plots) as a reward.''')


    st.dataframe(pd_)

    ########################################################################################################################
    ################################2.2 Probability Distribution Function ##################################################
    ########################################################################################################################

    dynamics = RollOut(rolls)
    Amazing = hpity_p
    Regular = softp_p
    fr = {
        'categories': ['Plots', 'MysteryBoxes', 'NoMysteryBoxes'],
        'probabilities': [hpity_p, softp_p, 1 - (softp_p+hpity_p)]
    }

    all_rewards['MysteryBoxes']['Probabilities'] = (4*[.16/4]) + (4*[.24/4]) + (4*[.60/4])
    all_rewards['NoMysteryBoxes']['Probabilities'] = [1/7,1/7,1/7,1/7,1/7,1/7,1-((1/7)+(1/7)+(1/7)+(1/7)+(1/7)+(1/7))]

    new, plots_earned = dynamics.complete_dynamics(fr, all_rewards)

    st.header(":blue[3. The Optimal Case of Gacha Dynamics]")

    st.write(f"We are supposing that Normal Pity Odds are only related to No Mystery Boxes rewards, that means that having a mystery box or plot has 0 probability. "
             f"The same for Soft Pity where we are supposing only Mystery Boxes rewards and so on for Plots related to hard pity. :blue[In this way we ensure only give plots as a rewards when a player complete {hpity} on 1K rolls.]"
             f" Aditionally, once we give a plot as a reward we have no replacement of them, the chances must decrease, so we use an hypergeometrical distribution function on that case.")
    plots_8 = plots_earned[1]
    plots_16 = plots_earned[2]
    plots_32 = plots_earned[3]
    plots_64 = plots_earned[4]

    st.subheader('''2.1 Example (one player) VERSION 0''')



    plots_earn = {'8x8': plots_8, '16x16': plots_16, '32x32': plots_32, '64x64': plots_64}

    rtype = 'Plots'
    st.caption(f''':moneybag: :moneybag: :moneybag: :red[{rtype}] :moneybag: :moneybag: :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new,rtype, plots_earned, plot_base_price, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no Plots (Hard Pity) rewards.''')


    rtype = 'MysteryBoxes'
    st.caption(f''':moneybag: :moneybag: :red[{rtype}] :moneybag: :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new,rtype, plots_earned, plot_base_price, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no Soft Pity rewards.''')


    rtype = 'NoMysteryBoxes'
    st.caption(f''':moneybag: :red[{rtype}] :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new, rtype, plots_earned, plot_base_price, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no Normal rewards.''')




st.subheader(f" 2.2 Probability Distribution Function")


st.write(f'''The chance for each reward class is:  

        - Plots (Hard Pity): {str(hpity_p * 100) + '%'} probability
    - Mystery Boxes (Soft Pity): {str(softp_p * 100) + '%'} probability
    - No Mystery Boxes (Normal): {str(round((1 - (hpity_p + softp_p)) * 100)) + '%'} probability (as complement)''')


if n > 0:
    binomial_pt = BinomialDistributionFunction(int(n))
    st.write(f'''By {n_players} player{'s' if n_players>1 else ''}, {rolls} rolls each of them (:green[{n} total rolls]), we have.''')
    if hpity_p > 0:
        binomial_pt.binomial_plot(hpity_p, f'''Plots Reward (Hard Pity) - Chances on {n} rolls''')

    if softp_p > 0:
        binomial_pt.binomial_plot(softp_p, f'''Regular Rewards (Soft Pity) - Chances on {n} rolls''')

    if 1 - (hpity_p + softp_p) > 0:
        binomial_pt.binomial_plot(1 - (hpity_p + softp_p), f'''No Mystery Boxes Reward (Normal Pity) - Chances on {n} rolls''')

else:
    st.write(f'''ZERO rolls''')

########################################################################################################################
####################### Amazing Rewards - Probability Distribution Function on plots ###################################
########################################################################################################################

try:
    st.subheader("2.2 Plots Rewards - Probability Distribution Function")
    st.write(f'''With focus on :green[Plots Rewards] we fit an probability distribution function to explain how many plots we give to players after certain number of rolls.''')
    st.write(f'''EXPLANATION: At the beginning, there are {N} plots as rewards distributed as follows:

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
                (plots_earn[1] * reserve_multiplier['8x8']) + (plots_earn[2] * reserve_multiplier['16x16']) +
                    (plots_earn[-2] * reserve_multiplier['32x32'])+(plots_earn[-1] * reserve_multiplier['64x64']))


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
        - 32x32: {plots_earn[3]} plots ({plot_base_price * plots_earn[3] * reserve_multiplier['32x32']} USD)
        - 64x64: {plots_earn[4]} plots ({plot_base_price * plots_earn[4] * reserve_multiplier['64x64']} USD)''')

        st.write(f''':red[Balance: {':arrow_up:' if (n_players * usd_spent)-players_earnings_usd>0 else ':arrow_down:'} {human_format((n_players * usd_spent)-players_earnings_usd)} USD]''')

        K = plots_earn[0]
        if plots_earn[0] > 0:
            st.subheader(f"2.2.3 Probability to have plot types as a reward for the given number of plots {(K)}.")
            st.write(
                f''':green[Probabilities are given on {K} events.] \n
                We use an hyper geometric distribution function to our dynamics.''')
            hyp_pl = HypergeometricDistributionFunction(K, N)
            hyp_pl.plots_stat('8x8', p_8)

            hyp_pl.plots_stat('16x16', p_16)

            hyp_pl.plots_stat('32x32', p_32)

            hyp_pl.plots_stat('64x64', 1-(p_8+p_16+p_32))

    else:
        st.write(''':red[No plots were given by this spent and number of players]''')

except Exception as e:
    logging.error(f'''Error----> {e} <--- in 3rd section ''')
    pass
