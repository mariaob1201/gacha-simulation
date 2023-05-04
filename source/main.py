import streamlit as st
import pandas as pd
import logging
import requests
from auxiliar_functions import *
import plotly.express as px
from scipy.stats import binom



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

st.sidebar.markdown("## 2. Reward Types")
Amazing_p = st.sidebar.slider('Amazing reward chance (on 100 opportunities)',
                              min_value=0.00, max_value=10.00, value=0.50, step=.01, key=1)
Amazing = Amazing_p / 100

Regular_p = st.sidebar.slider('Regular reward chance (on 100 opportunities)', min_value=0.00, max_value=100.00,
                              value=2.0, step=.01, key=11)
Regular = Regular_p / 100
###################### REWARDS
st.sidebar.markdown("## 2.1 Rewards Chances - Amazing")
MBT3_p = st.sidebar.slider('Mystery Box Tier 3', min_value=0.00, max_value=1.00, value=0.7, step=.01, key=13292)
p8_p = st.sidebar.slider('8x8 Plot', min_value=0.00, max_value=1.00 - MBT3_p, value=0.2, step=.01, key=13229)
p16_p = st.sidebar.slider('16x16 Plot', min_value=0.00, max_value=1.00 - (MBT3_p + p8_p), value=0.09, step=.01,
                          key=139292)

st.sidebar.markdown("## 2.2 Rewards Chances - Regular")
MBT1 = st.sidebar.slider('Mystery Box Tier 1', min_value=0.00, max_value=1.00, value=1 / 3, step=.01, key=1232)
Recipe = st.sidebar.slider('Recipe', min_value=0.00, max_value=1.00 - MBT1, value=1 / 3, step=.01, key=1223)

st.sidebar.markdown("## 2.2 Rewards Chances - Poor")
smp_p = st.sidebar.slider('Small Material Pack', min_value=0.00, max_value=1.00, value=1 / 6, step=.01, key=12321)
srp_p = st.sidebar.slider('Small Resource Pile', min_value=0.00, max_value=1.00 - smp_p, value=1 / 6, step=.01,
                          key=12123)
mmp_p = st.sidebar.slider('Medium Material Pack', min_value=0.00, max_value=1.00 - (smp_p + srp_p), value=1 / 6,
                          step=.01, key=121234)
mrp_p = st.sidebar.slider('Medium Resource Pile', min_value=0.00, max_value=1.00 - (smp_p + srp_p + mmp_p),
                          value=1 / 6, step=.01, key=121123)
bmp_p = st.sidebar.slider('Bountiful Material Pack', min_value=0.00, max_value=1.00 - (smp_p + srp_p + mmp_p + mrp_p),
                          value=1 / 6, step=.01, key=1211233)

p = (1 - MBT3_p) / N
probabilities_for_amazing = [MBT3_p, p * p_8, p * p_16, p * p_32]

all_rewards = {'Amazing': {'Types': ['Mystery Box Tier 3', 'Plot 8*8', 'Plot 16*16', 'Plot 32*32'],
                           'Probabilities': [MBT3_p, p8_p, p16_p, 1 - (MBT3_p + p8_p + p16_p)]},
               'Regular': {'Types': ['Mystery Box Tier 1', 'Recipe', 'Mystery Box Tier 2'],
                           'Probabilities': [MBT1, Recipe, 1 - (MBT1 + Recipe)]},
               'Poor': {'Types': ['Small Material Pack', 'Medium Material Pack', 'Bountiful Material Pack',
                                  'Small Resource Pile', 'Medium Resource Pile', 'Bountiful Resource Pile'],
                        'Probabilities': [smp_p, mmp_p, bmp_p, srp_p, mrp_p,
                                          1.00 - (smp_p + srp_p + mmp_p + bmp_p + mrp_p)]}}


##################################################################################################################
#################################MASTER CONVERSION FUNCTION ######################################################
##################################################################################################################







st.title(":blue[Gacha Rolls Dynamics]")

st.title("1. Master Conversion Function")

st.write(f'''The equivalence between Mana and hard currencies can be changed by control usage.''')

usd_spent = st.number_input('USD player Spent (AVG)', step=1., value=100., format="%.2f", key=12112)
rolls_by_usd_price = st.number_input('One roll equals USD', step=1., value=2.5, format="%.2f", key=122)
# sysBP = st.number_input(label=“systolic blood pressure”,step=1.,format="%.2f")
one_roll_mana_price = st.number_input('One roll equals Mana Units', step=1., value=10.0, format="%.2f", key=12221)


eth_rate = ether_to_usd('USD')
conversionfunction = CurrenciesConversion(eth_rate, rolls_by_usd_price, one_roll_mana_price)



st.write(f''':green[Equivalence between currencies:]

    - REAL: The Ethereum corresponds to {eth_rate} USD at this moment.
    - IN GAME: One Roll price is {rolls_by_usd_price} USD or {one_roll_mana_price} Mana units (can change by control).''')

conv_fun, rolls = conversionfunction.master_conversion_function(usd_spent)


st.title("2. Reward Types")

fr = {
    'categories': ['Amazing', 'Regular', 'Poor'],
    'probabilities': [Amazing, Regular, 1 - (Regular + Amazing)]
}

dynamics = RollOut(rolls)
new, plots_earned = dynamics.complete_dynamics(fr, all_rewards)
plots_8 = plots_earned[1]
plots_16 = plots_earned[2]
plots_32 = plots_earned[3]

chart_data = pd.DataFrame(
    {'Rewards_Types': ['Poor', 'Regular', 'Amazing'],
    'Probabilities':[Amazing, Regular, 1-(Amazing+Regular)],
     'Number of rewards after rolls': [len(new['Poor']), len(new['Regular']), len(new['Amazing'])]
    })

n = rolls * n_players

if rolls > 0:
    st.subheader('''2.1 Example (one player) ''')
    st.write(f'''At the begining, we have three types of rewards: Amazing, Regular and Poor by a given probabilities. In this example case, :green[One single player Paid {rolls} rolls], then the given rewards are distributed as follows:''')
    st.dataframe(chart_data)

    plots_earn = {'8x8': plots_8, '16x16': plots_16, '32x32': plots_32}

    rtype = 'Amazing'
    st.caption(f''':moneybag: :moneybag: :moneybag: :red[{rtype}] :moneybag: :moneybag: :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new,rtype, plots_earned, plot_base_price, plots_8, plots_16, plots_32, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no Amazing rewards.''')


    rtype = 'Regular'
    st.caption(f''':moneybag: :moneybag: :red[{rtype}] :moneybag: :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new,rtype, plots_earned, plot_base_price, plots_8, plots_16, plots_32, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no Regular rewards.''')


    rtype = 'Poor'
    st.caption(f''':moneybag: :red[{rtype}] :moneybag:''')
    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] Rewards probabilities:''')
    st.dataframe(chart_data)
    if len(new[rtype]) > 0:
        function(new, rtype, plots_earned, plot_base_price, plots_8, plots_16, plots_32, rolls, reserve_multiplier,conv_fun, plots_earn)

    else:
        st.write('''In this example were no Poor rewards.''')


########################################################################################################################
################################2.2 Probability Distribution Function ##################################################
########################################################################################################################

st.subheader(f" 2.2 Probability Distribution Function")
st.write(f'''The chance for each reward class is (by controls):  

        - Amazing: {str(Amazing * 100) + '%'} probability
    - Regular: {str(Regular * 100) + '%'} probability
    - Poor: {str((1 - (Amazing + Regular)) * 100) + '%'} probability (as complement)''')


if n > 0:
    binomial_pt = BinomialDistributionFunction(n)
    st.write(f'''By {n_players} player{'s' if n_players>1 else ''}, {rolls} rolls each of them (:green[{n} total rolls]), we have.''')
    if Amazing > 0:
        binomial_pt.binomial_plot(Amazing, f'''Amazing Reward - Chances on {n} rolls''')

    if Regular > 0:
        binomial_pt.binomial_plot(Regular, f'''Regular Reward - Chances on {n} rolls''')

    if 1 - (Amazing + Regular) > 0:
        binomial_pt.binomial_plot(1 - (Amazing + Regular), f'''Poor Reward - Chances on {n} rolls''')

else:
    st.write(f'''ZERO rolls''')

########################################################################################################################
####################### Amazing Rewards - Probability Distribution Function on plots ###################################
########################################################################################################################

try:
    st.title("3. Amazing Rewards - Probability Distribution Function")
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