reserve_multiplier = {'8x8': 1,
                          '16x16': 2.75,
                          '32x32': 41.67,
                          '64x64': 66.67,
                          '128x128': 666.67}

plot_prices = 480

def human_format(num: object) -> object:
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])