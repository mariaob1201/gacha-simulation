import streamlit as st
import pandas as pd
import logging
import requests
from auxiliar_functions import *
import plotly.express as px
from scipy.stats import binom

url = "https://min-api.cryptocompare.com/data/price?fsym=ETH&tsyms=USD"

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

# st.sidebar.markdown("## 1. Master Conversion Formula")
# usd_spent = st.sidebar.slider('Player Spent (USD)', min_value=0.00, max_value=1000.00, value=100.00, step=.01, key=111)
###################### REWARDS
st.sidebar.markdown("## 1. Number of players")
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
def ether_to_usd(url):
    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json"},
        )
        data = response.json()
        USD = data['USD']

    except Exception as e:
        logging.error(f'''In ether to usd function {e}''')
        USD = None

    return USD


eth_rate = ether_to_usd(url)


def master_conversion_function(usd_spent, rolls_by_usd_price, one_roll_mana_price, eth_rate):
    conversion = {'USD': usd_spent,
                  'ETH': (usd_spent / eth_rate),
                  'ROLLS': int(usd_spent / rolls_by_usd_price),
                  'Mana': one_roll_mana_price * int(usd_spent / rolls_by_usd_price)}

    return conversion


st.title(":blue[Gacha Rolls Dynamics]")

st.title("1. Master Conversion Function")

st.write(f'''The equivalence between Mana and hard currencies can be changed by control usage.''')

usd_spent = st.number_input('USD player Spent (AVG)', step=1., value=100., format="%.2f", key=12112)
rolls_by_usd_price = st.number_input('One roll equals USD', step=1., value=2.5, format="%.2f", key=122)
# sysBP = st.number_input(label=“systolic blood pressure”,step=1.,format="%.2f")
one_roll_mana_price = st.number_input('One roll equals Mana Units', step=1., value=10.0, format="%.2f", key=12221)

st.write(f''':green[Equivalence between currencies:]

    - REAL: The Ethereum corresponds to {eth_rate} USD at this moment.
    - IN GAME: Roll price is {rolls_by_usd_price} USD, or {one_roll_mana_price} Mana units (can change by control). ''')

conv_fun = master_conversion_function(usd_spent, rolls_by_usd_price, one_roll_mana_price, eth_rate)
df = pd.DataFrame(conv_fun, index=[0])
st.dataframe(df)

st.title("2. Reward Types")
st.subheader(f" 2.0 Explanation")
st.write(f'''The chance for each reward class is (by controls):  

        - Amazing: {str(Amazing * 100) + '%'} probability
    - Regular: {str(Regular * 100) + '%'} probability
    - Poor: {str((1 - (Amazing + Regular)) * 100) + '%'} probability (as complement)''')

fr = {
    'categories': ['Amazing', 'Regular', 'Poor'],
    'probabilities': [Amazing, Regular, 1 - (Regular + Amazing)]
}

new, plots_earned = complete_dynamics(conv_fun['ROLLS'], fr, all_rewards)
plots_8 = plots_earned[1]
plots_16 = plots_earned[2]
plots_32 = plots_earned[3]

chart_data = pd.DataFrame(
    {'Rewards_Types': ['Poor', 'Regular', 'Amazing'],
     'Quantity': [len(new['Poor']), len(new['Regular']), len(new['Amazing'])]})

n = conv_fun['ROLLS'] * n_players

if n > 0:
    def binomial_plot(n, p, title):

        # defining list of r values
        r_values = list(range(n + 1))
        # list of pmf values
        dist = [binom.pmf(r, n, p) for r in r_values]
        mean = n * p
        variance = n * p * (1 - p)
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

    st.write(f'''By {n_players} player{'s' if n_players>1 else ''}, {conv_fun['ROLLS']} rolls each of them (:green[{n} total rolls]), we have.''')
    if Amazing > 0:
        binomial_plot(n, Amazing, f'''Amazing Reward - Chances on {n} rolls''')

    if Regular > 0:
        binomial_plot(n, Regular, f'''Regular Reward - Chances on {n} rolls''')

    if 1 - (Amazing + Regular) > 0:
        binomial_plot(n, 1 - (Amazing + Regular), f'''Poor Reward - Chances on {n} rolls''')

########################################################################################################################
########################################################################################################################
########################################################################################################################

if conv_fun['ROLLS'] > 0:
    st.subheader('''2.1 Example one player ''')
    st.write(f''':green[One player Paid {conv_fun['ROLLS']} rolls], then the rewards types are distributed as follows:''')
    st.dataframe(chart_data)

    plots_earn = {'8x8': plots_8, '16x16': plots_16, '32x32': plots_32}

    rtype = 'Amazing'
    st.caption(f''':moneybag: :moneybag: :moneybag: :red[{rtype}] :moneybag: :moneybag: :moneybag:''')
    if len(new[rtype]) > 0:
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
                     title=f"""{rtype} Rewards""",
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


    else:
        st.write('''In this example were no Amazing rewards.''')

    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] probabilities:''')
    st.dataframe(chart_data)

    rtype = 'Regular'
    st.caption(f''':moneybag: :moneybag: :red[{rtype}] :moneybag: :moneybag:''')
    if len(new[rtype]) > 0:

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
        # st.dataframe(df)
        fig = px.bar(df, x="Rewards", y="Quantity", text="Quantity",
                     title=f"""{rtype} Rewards""",
                     height=400
                     )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    else:
        st.write('''In this example were no Regular rewards.''')

    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] probabilities:''')
    st.dataframe(chart_data)

    rtype = 'Poor'
    st.caption(f''':moneybag: :red[{rtype}] :moneybag:''')
    if len(new[rtype]) > 0:

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
        # st.dataframe(df)
        fig = px.bar(df, x="Rewards", y="Quantity", text="Quantity",
                     title=f"""{rtype} Rewards""",
                     height=400
                     )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    else:
        st.write('''In this example were no Poor rewards.''')

    chart_data = pd.DataFrame(all_rewards[rtype])
    st.write(f''':green[{rtype} Type] probabilities:''')
    st.dataframe(chart_data)

########################################################################################################################
########################################################################################################################
########################################################################################################################

try:
    st.title("3. Amazing Rewards - Probability Distribution Function")

    st.subheader("3.1 Explanation")
    st.write(f'''At the beginning, there are {N} plots as rewards distributed as follows:

            - 8x8: {round(100 * p_8 / N)} % ({p_8} plots)
    - 16x16: {round(100 * p_16 / N)} % ({p_16} plots)
    - 32x32: {round(100 * p_32 / N)} % ({p_32} plots)
once one is giving as a reward, the collection decreases (no replacement) and then its probability changes.''')

    total_spent = n_players * usd_spent
    tolls_per_tspent = master_conversion_function(total_spent, rolls_by_usd_price, one_roll_mana_price, eth_rate)
    dyn, plots_earn = complete_dynamics(tolls_per_tspent['ROLLS'], fr, all_rewards)
    plots_earned_by_rolls = plots_earn[0]

    ether = "{:.2f}".format(tolls_per_tspent['ETH'])

    st.write(
        f'''(3 Plot Chances - Control) {n_players} players, each of them spending {usd_spent} USD on average means:''')

    if plots_earned_by_rolls > 0:

        st.write(f''':green[Players Earnings:]

        - {tolls_per_tspent['ROLLS']} rolls (In average {int(tolls_per_tspent['ROLLS'] / n_players)} per player)
    - {tolls_per_tspent['Mana']} Mana Units 
    - {plots_earned_by_rolls} plots ({human_format(480 * ((plots_earn[1] * reserve_multiplier['8x8']) + (plots_earn[2] * reserve_multiplier['16x16']) + (plots_earn[-1] * reserve_multiplier['32x32'])))} USD) as a reward as follows:
        - 8x8: {plots_earn[1]} plots ({480 * plots_earn[1] * reserve_multiplier['8x8']} USD)
        - 16x16: {plots_earn[2]} plots ({480 * plots_earn[2] * reserve_multiplier['16x16']} USD)
        - 32x32: {plots_earn[3]} plots ({480 * plots_earn[3] * reserve_multiplier['32x32']} USD) ''')

        st.write(f''':green[Runiverse incomes:] (Control 1 conversion rates)

            - {human_format(total_spent)} USD
    - {ether} ETHER''')

        K = plots_earn[0]
        if plots_earn[0] > 0:
            st.subheader(f"3.2 Probability to have plot types as a reward for the given number of plots {(K)}.")
            st.write(
                f''':green[We use an hyper geometric distribution function to our dynamics.] Probabilities are given on {K} events.''')

            ps = '8x8'
            n = p_8
            mean = n * K / N
            variance = n * K * (N - n) * (N - K) / ((N * N * (N - 1)))
            std = np.sqrt(variance)
            hypergeom_plot(N, n, K, ps, mean, std)
            st.write(f''':green[{ps} plots] After {K} events Statistics are:

                - The mean is having {"{:.2f}".format(mean)} plots of that type
    - 95% Confidence interval: [{"{:.2f}".format(max(0, mean - 2 * abs(std)))}, {"{:.2f}".format(mean + 2 * abs(std))}] plots of that type
    - Variance {"{:.2f}".format(variance)}''')

            ps = '16x16'
            n = p_16
            mean = n * K / N
            variance = n * K * (N - n) * (N - K) / ((N * N * (N - 1)))
            std = np.sqrt(variance)
            hypergeom_plot(N, n, K, ps, mean, std)
            st.write(f''':green[{ps} plots] After {K} events Statistics are:

                    - The mean is having {"{:.2f}".format(mean)} plots of that type
    - 95% Confidence interval: [{"{:.2f}".format(max(0, mean - 2 * abs(std)))}, {"{:.2f}".format(mean + 2 * abs(std))}] plots of that type
    - Variance {"{:.2f}".format(variance)}''')

            ps = '32x32'
            n = p_32
            mean = n * K / N
            variance = n * K * (N - n) * (N - K) / ((N * N * (N - 1)))
            std = np.sqrt(variance)
            hypergeom_plot(N, n, K, ps, mean, std)
            st.write(f''':green[{ps} plots] After {K} events Statistics are:

                    - The mean is having {"{:.2f}".format(mean)} plots of that type
    - 95% Confidence interval: [{"{:.2f}".format(max(0, mean - 2 * abs(std)))}, {"{:.2f}".format(mean + 2 * abs(std))}] plots of that type
    - Variance {"{:.2f}".format(variance)}''')

    else:
        st.write(''':red[No plots were given by this spent and number of players]''')

except Exception as e:
    logging.error(f'''Error {e} in 3rd section ''')
    pass