import streamlit as st


reserve_multiplier = {'8x8': 1,
                          '16x16': 2.75,
                          '32x32': 41.67,
                          '64x64': 66.67,
                          '128x128': 666.67}

plot_prices = 480

st.sidebar.markdown("## 2.1 Rewards Chances - Amazing")
Amazing = st.sidebar.slider('Mystery Box Tier 3', min_value=0.00, max_value=1.00, value=0.4, step=.01)

st.sidebar.markdown("## 2.2 Rewards Chances - Regular")
MBT1 = st.sidebar.slider('Mystery Box Tier 1', min_value=0.00, max_value=1.00, value=0.4, step=.01)
Recipe = st.sidebar.slider('Recipe', min_value=0.00, max_value=1.00-MBT1, value=0.4, step=.01)

###################### REWARDS
start_p = Amazing
p = (1 - start_p) / (reserve_multiplier['8x8'] + reserve_multiplier['16x16'] + reserve_multiplier['32x32'])
probabilities = [start_p, p*reserve_multiplier['32x32'], p*reserve_multiplier['16x16'], p*reserve_multiplier['8x8']]

all_rewards = {'Amazing': {'Types':['Mystery Box Tier 3', 'Plot 8*8', 'Plot 16*16', 'Plot 32*32'],
                           'Probabilities': probabilities},
               'Regular': {'Types': ['Mystery Box Tier 1', 'Recipe', 'Mystery Box Tier 2'],
                           'Probabilities': [MBT1, Recipe, 1-(MBT1+Recipe)]},
               'Poor': {'Types': ['Small Material Pack', 'Small Resource Pile', 'Medium Material Pack',
               'Bountiful Material Pack', 'Medium Resource Pile', 'Bountiful Resource Pile'], 'Probabilities': [1/6]*6}}
######################