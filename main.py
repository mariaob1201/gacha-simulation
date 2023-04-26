import numpy as np
from auxiliar_functions import *


# Use np.random.choice to throw the die once and record the outcome
def first_roll_out_dynamics(N, die, probabilities):
    # np.random.seed(1)
    initiall_roll = np.random.choice(die, size=N, p=probabilities)
    # print("The Outcome of the throw is: {}".format(initiall_roll))
    return initiall_roll


# Use np.random.choice to throw the die once and record the outcome
def second_roll_out_dynamics(frou):
    # np.random.seed(0)

    if 'Poor' in frou:
        die = ['Small Material Pack', 'Small Resource Pile', 'Medium Material Pack',
               'Bountiful Material Pack', 'Medium Resource Pile', 'Bountiful Resource Pile']
        probabilities = [1 / len(die)] * len(die)

    elif 'Regular' in frou:
        die = ['Mystery Box Tier 1', 'Recipe', 'Mystery Box Tier 2']
        probabilities = [1 / len(die)] * len(die)

    else:
        die = ['Mystery Box Tier 3', 'Plot 8*8', 'Plot 16*16', 'Plot 32*32']
        start_p = .75
        p = (1 - start_p) / (reserve_multiplier['8x8'] + reserve_multiplier['16x16'] + reserve_multiplier['32x32'])
        probabilities = [start_p, p * reserve_multiplier['32x32'], p * reserve_multiplier['16x16'],
                         p * reserve_multiplier['8x8']]

    sec = np.random.choice(die, size=1, p=probabilities)

    return sec


def complete_dynamics(N, fr, spent_usd):
    plots = 0
    plots_8 = 0
    plots_16 = 0
    plots_32 = 0

    first_rou = first_roll_out_dynamics(N, fr['categories'], fr['probabilities'])

    salida = []
    new = {}
    new['Poor'] = []
    new['Regular'] = []
    new['Amazing'] = []

    for i in first_rou:
        secnd = second_roll_out_dynamics(i)
        # print('*************',i,'---->', secnd[0])
        salida.append(secnd[0])
        new[i].append(secnd[0])
        if 'Plot' in secnd[0]:
            plots += 1
        if '8*8' in secnd[0]:
            plots_8 += 1
        if '16*16' in secnd[0]:
            plots_16 += 1
        if '32*32' in secnd[0]:
            plots_32 += 1

    print(f'''In {N} runs you have {plots} plots given as a rewards: \n 
   --> 8*8 size: {plots_8},
   --> 16*16 size:{plots_16}, 
   --> 32*32 size: {plots_32} \n
   In dolars, by those plots, player earn {human_format((plots_8 * plot_prices * reserve_multiplier['8x8']) + (plots_16 * plot_prices * reserve_multiplier['16x16']) + (plots_32 * plot_prices * reserve_multiplier['32x32']))} USD \n
   Runiverse earns about {human_format(spent_usd - ((plots_8 * plot_prices * reserve_multiplier['8x8']) + (plots_16 * plot_prices * reserve_multiplier['16x16']) + (plots_32 * plot_prices * reserve_multiplier['32x32'])))} USD''')

    return salida, new

def pricing_usd_to_mana(N_usd, exc_rate, rate_type, fluctuation=None):
    depreciation = 0.01
    if 'Flexible' in rate_type:
        depreciation=0.01
    elif 'Fixed' in rate_type:
        depreciation=0.01
    elif 'Managed' in rate_type:
        depreciation=0.01
    N_mana = N_usd * exc_rate
    print('Mana amount ', int(N_mana), 'after ',N_usd,' USD')
    return N_mana

def mana_to_rolls(am_mana, unit_price):
    rolls = am_mana*unit_price
    print(f'''The player did {int(rolls)} rolls''')
    return rolls


usd_spent = 480 #usd

# 7 mana units by in 3 usd
usd_unit_bprice = 3.7
rew_per_usd_base_price = 11
exc_rate = (rew_per_usd_base_price/usd_unit_bprice)

manatorolls_unit_price = 2.1/10 #each mana unit gets .1 rolls

mana = pricing_usd_to_mana(usd_spent, exc_rate, 'Fixed')

rolls = mana_to_rolls(mana, manatorolls_unit_price)

fr = {
    'categories': ['Amazing','Regular','Poor'],
    'probabilities':[1.5/100, .9/10, 1-((1.5/100) + (.9/10))]
}
salida, new = complete_dynamics(int(rolls), fr, usd_spent)
print(f'''The player spent {human_format(usd_spent)} USD ''')

sett_fin = []
for k in new.keys():
  sett = {}
  for i in new[k]:
    if i in sett.keys():
      sett[i]+=1
    else:
      sett[i]=1

  sett_fin.append(sett)
  print('About --->', k, '<---', sett)

for k in new.keys():
  print(k, len(new[k]))

