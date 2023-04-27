import streamlit as st
import pandas as pd
import logging
import requests
from auxiliar_functions import *
import plotly.express as px

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


st.sidebar.markdown("## 1. Master Conversion Formula")
usd_spent = st.sidebar.slider('Player Spent (USD)', min_value=0.00, max_value=1000.00, value=100.00, step=.01, key=111)
rolls_by_usd_price = st.sidebar.slider('One roll equals USD', min_value=0.01, max_value=50.00, value=1.15, step=.01,
                                       key=122)
one_roll_mana_price = st.sidebar.slider('One roll equals Mana Units', min_value=0.01, max_value=100.00, value=2.50,
                                        step=.01, key=121)

st.sidebar.markdown("## 2. First Roll Probabilities")
Amazing = st.sidebar.slider('Amazing reward chance', min_value=0.00, max_value=1.00, value=0.01, step=.01, key=1)
Regular = st.sidebar.slider('Regular reward chance', min_value=0.00, max_value=1.00 - Amazing, value=0.1, step=.01,
                            key=11)

###################### REWARDS
st.sidebar.markdown("## 2.1 Rewards Chances - Amazing")
MBT3_p = st.sidebar.slider('Mystery Box Tier 3', min_value=0.00, max_value=1.00, value=0.4, step=.01, key=1322)

st.sidebar.markdown("## 2.2 Rewards Chances - Regular")
MBT1 = st.sidebar.slider('Mystery Box Tier 1', min_value=0.00, max_value=1.00, value=0.33, step=.01, key=1232)
Recipe = st.sidebar.slider('Recipe', min_value=0.00, max_value=1.00 - MBT1, value=0.33, step=.01, key=1223)

st.sidebar.markdown("## 2.2 Rewards Chances - Poor")
smp_p = st.sidebar.slider('Small Material Pack', min_value=0.00, max_value=1.00, value=1 / 6, step=.01, key=12321)
srp_p = st.sidebar.slider('Small Resource Pile', min_value=0.00, max_value=1.00 - smp_p, value=1 / 6, step=.01,
                          key=12123)
mmp_p = st.sidebar.slider('Medium Material Pack', min_value=0.00, max_value=1.00 - (smp_p + srp_p), value=1 / 6,
                          step=.01, key=121234)
bmp_p = st.sidebar.slider('Bountiful Material Pack', min_value=0.00, max_value=1.00 - (smp_p + srp_p + mmp_p),
                          value=1 / 6, step=.01, key=1211233)
mrp_p = st.sidebar.slider('Medium Resource Pile', min_value=0.00, max_value=1.00 - (smp_p + srp_p + mmp_p + bmp_p),
                          value=1 / 6, step=.01, key=121123)

p = (1 - MBT3_p) / N
probabilities_for_amazing = [MBT3_p, p * p_8, p * p_16, p * p_32]

all_rewards = {'Amazing': {'Types': ['Mystery Box Tier 3', 'Plot 8*8', 'Plot 16*16', 'Plot 32*32'],
                           'Probabilities': probabilities_for_amazing},
               'Regular': {'Types': ['Mystery Box Tier 1', 'Recipe', 'Mystery Box Tier 2'],
                           'Probabilities': [MBT1, Recipe, 1 - (MBT1 + Recipe)]},
               'Poor': {'Types': ['Small Material Pack', 'Small Resource Pile', 'Medium Material Pack',
                                  'Bountiful Material Pack', 'Medium Resource Pile', 'Bountiful Resource Pile'],
                        'Probabilities': [smp_p, srp_p, mmp_p, bmp_p, mrp_p,
                                          1.00 - (smp_p + srp_p + mmp_p + bmp_p + mrp_p)]}}

###################### REWARDS
st.sidebar.markdown("## 3 Plot Chances")
n_players = st.sidebar.slider('Number of players', min_value=0, max_value=1000, value=100, step=1, key=121117711123)
player_spent = st.sidebar.slider('USD spent per player (avg)', min_value=0, max_value=1000, value=100, step=1, key=77)


##################################################################################################################
##################################################################################################################
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


st.title("Gacha Rolls Dynamics")
st.write(f'''We define a master conversion function between one roll cost in USD, then in Ethereum and finally by Mana. The equivalence between Mana and hard currencies can be changed by control usage. 
Finally we run a probability function based again on defined probabilities per reward.''')

st.subheader("1. Master Conversion Function")
st.write(f'''Equivalence between currencies:

    - REAL: The Ethereum corresponds to {eth_rate} USD at this moment.
    - IN GAME: Roll price is {rolls_by_usd_price} USD, or {one_roll_mana_price} Mana units (can change by control). ''')

conv_fun = master_conversion_function(usd_spent, rolls_by_usd_price, one_roll_mana_price, eth_rate)
df = pd.DataFrame(conv_fun, index=[0])
st.dataframe(df)


st.subheader("2. First Roll Probabilities")

st.write(f'''The chance for each reward class is (by controls):  
        
        - Amazing: {str(int(100 * Amazing)) + '/100'} probability
    - Regular: {str(int(100 * Regular)) + '/100'} probability
    - Poor: {str(100 - 100 * (Amazing + Regular))} probability (as complement)''')

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

st.write(f''':blue[In {conv_fun['ROLLS']} runs] the rewards are:''')
st.dataframe(chart_data)

########################################################################################################################
########################################################################################################################
########################################################################################################################

if conv_fun['ROLLS'] > 0:
    st.subheader(" 2.1 Rewards Types on First Roll")
    plots_earn = {'8x8': plots_8, '16x16': plots_16, '32x32': plots_32}

    rtype = 'Amazing'
    if len(new[rtype]) > 0:
        # st.write(f''':blue[{rtype} Rewards]''')

        dict1 = {}
        for i in new[rtype]:
            if i not in dict1.keys():
                dict1[i] = 1
            else:
                dict1[i] += 1

        dict_f = {'Rewards': dict1.keys(), 'Quantity': dict1.values()}

        df = pd.DataFrame(dict_f)
        # st.dataframe(df)
        fig = px.bar(df, x="Rewards", y="Quantity",
                     title=f"""{rtype} Rewards""",
                     height=400
                     )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        if plots_earned[0] > 0:
            plots_earns = (plots_8 * plot_base_price * reserve_multiplier['8x8']) + (
                        plots_16 * plot_base_price * reserve_multiplier['16x16']) + (
                                      plots_32 * plot_base_price * reserve_multiplier['32x32'])
            runi_earns = conv_fun['USD'] - plots_earns
            st.write(f''':green[Player earns {int(plots_earned[0])} plots] as follows: {plots_earn}. That means in USD: 
            
                - Player earns by plots: {human_format(plots_earns)} USD 
    - Runiverse {'earns' if runi_earns > 0 else 'loses'} about {human_format(runi_earns)} USD''')


    rtype = 'Regular'
    if len(new[rtype]) > 0:
        # st.write(f''':blue[{rtype} Rewards]''')

        dict1 = {}
        for i in new[rtype]:
            if i not in dict1.keys():
                dict1[i] = 1
            else:
                dict1[i] += 1

        dict_f = {'Rewards': dict1.keys(), 'Quantity': dict1.values()}

        df = pd.DataFrame(dict_f)
        # st.dataframe(df)
        fig = px.bar(df, x="Rewards", y="Quantity",
                     title=f"""{rtype} Rewards""",
                     height=400
                     )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    rtype = 'Poor'
    if len(new[rtype]) > 0:
        # st.write(f''':blue[{rtype} Rewards]''')

        dict1 = {}
        for i in new[rtype]:
            if i not in dict1.keys():
                dict1[i] = 1
            else:
                dict1[i] += 1

        dict_f = {'Rewards': dict1.keys(), 'Quantity': dict1.values()}

        df = pd.DataFrame(dict_f)
        # st.dataframe(df)
        fig = px.bar(df, x="Rewards", y="Quantity",
                     title=f"""{rtype} Rewards""",
                     height=400
                     )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

########################################################################################################################
########################################################################################################################
########################################################################################################################

try:
    from scipy.special import comb

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


    st.subheader("3. Probabilities to have a Plot as Reward")

    st.subheader("3.1 Increase the number of players")
    st.write(f'''At the beginning, there are {N} plots, once the Reward is Amazing Type and is not a Mystery Box T3, each plot has the following chance to appear, once one is giving as a reward, the collection decreases and then its probability:
    
            - 8x8: {round(100 * p_8 / N)} % ({p_8} plots)
    - 16x16: {round(100 * p_16 / N)} % ({p_16} plots)
    - 32x32: {round(100 * p_32 / N)} % ({p_32} plots)''')

    total_spent = n_players*player_spent
    tolls_per_tspent = master_conversion_function(total_spent, rolls_by_usd_price, one_roll_mana_price, eth_rate)
    dyn, plots_earn = complete_dynamics(tolls_per_tspent['ROLLS'], fr, all_rewards)
    plots_earned_by_rolls = plots_earn[0]

    ether = "{:.2f}".format(tolls_per_tspent['ETH'])
    if plots_earned_by_rolls > 0:
        st.write(f''':blue[BISONIC incomes:] 
        
            - {human_format(total_spent)} USD
    - {ether} ETHER
    - {tolls_per_tspent['Mana']} Mana Units
    ''')
        st.write(f''':blue[Players Earnings:]
    
        - {tolls_per_tspent['ROLLS']} rolls ({tolls_per_tspent['ROLLS']/n_players} per player)
    - {plots_earned_by_rolls} plots ({human_format((plots_earn[1]*480*reserve_multiplier['8x8'])+(plots_earn[1]*480*reserve_multiplier['16x16'])+(plots_earn[1]*480*reserve_multiplier['32x32']))} USD) as a reward as follows:
        - 8x8: {plots_earn[1]} plots
        - 16x16: {plots_earn[2]} plots
        - 32x32: {plots_earn[3]} plots ''')

        st.subheader("3.2 Probability to have plots as a reward.")
        st.write(
            f''':blue[We use an hyper geometric distribution function to our dynamics.] 
            Taking 8x8 plot size asn an example, with collection size {p_8} on {N} total collection, we draw {plots_earn[1]}, then the probability to chose exactly k (x-axis) events is represented by the scatter plot.''')

        if plots_earn[1] > 0:
            ps = '8x8'
            A = 6000
            hypergeom_plot2(N, A, plots_earn[1], ps)

        if plots_earn[2] > 0:
            ps = '16x16'
            A = 1800
            hypergeom_plot2(N, A, plots_earn[2], ps)

        if plots_earn[3] > 0:
            ps = '32x32'
            A = 88
            hypergeom_plot2(N, A, plots_earn[3], ps)

except Exception as e:
    logging.error(f'''Error {e} in 3rd section''')
    pass