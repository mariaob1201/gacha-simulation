# gacha-probabilities-dashboard

## Overview

This code is developed in order to answer business requirements such as:
1) Master conversion function between hard currencies and in-game currencies
2) Modeled the reward assignation by roll up, given probabilities on each one. We use a binomial distribution function.
3) Mathematicaly modeled the plots rewards by an hypergeometric distribution function.
The final dashboard is hosted on AWS EC2 by Streamlit and can be seen here: http://3.83.101.89:8501/

## Prerequisities

Before you begin, ensure you have met the following requirements:

* You have a _Windows/Linux/Mac_ machine running [Python 3.6+](https://www.python.org/).
* You have installed the latest versions of [`pip`](https://pip.pypa.io/en/stable/installing/) and [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/) or `conda` ([Anaconda](https://www.anaconda.com/distribution/)).


## Setup

To install the dependencies, you can simply follow this steps.

Clone the project repository:
```bash
git clone https://github.com/bisonic-official/gacha-probabilities-dashboard
```

To create and activate the virtual environment, follow these steps:

**Using `conda`**

```bash
$ conda create -n streamlit python=3.7

# Activate the virtual environment:
$ conda activate streamlit

# To deactivate (when you're done):
(streamlit)$ conda deactivate
```

**Using `virtualenv`**

```bash
# In this case I'm supposing that your latest python3 version is 3.7
$ virtualenv streamlit --python=python3

# Activate the virtual environment:
$ source streamlit/bin/activate

# To deactivate (when you're done):
(streamlit)$ deactivate
```

To install the requirements using `pip`, once the virtual environment is active:
```bash
(streamlit)$ pip install -r requirements.txt
```

#### Running the script

Finally, if you want to run the main script:
```bash
(streamlit)$ streamlit run source/main.py
```

#### Extend code!

Please feel free to use this repo as a template to extend code!
