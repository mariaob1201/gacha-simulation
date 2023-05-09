import unittest
from source.auxiliar_functions import ether_to_usd, HypergeometricDistributionFunction, human_format, RollOut, BinomialDistributionFunction

from unittest.mock import MagicMock, patch


fr_s = [{
    'categories': ['Amazing', 'Regular', 'Poor'],
    'probabilities': [1, 0, 0]
},
    {
        'categories': ['Amazing', 'Regular', 'Poor'],
        'probabilities': [0, 1, 0]
    },
{
    'categories': ['Amazing', 'Regular', 'Poor'],
    'probabilities': [0, 0, 1]
}]

all_rewards_set = {'Amazing': {'Types': ['Mystery Box Tier 3', 'Plot 8*8', 'Plot 16*16', 'Plot 32*32'],
                           'Probabilities': [1,0,0,0]},
               'Regular': {'Types': ['Mystery Box Tier 1', 'Recipe', 'Mystery Box Tier 2'],
                           'Probabilities': [1,0,0]},
               'Poor': {'Types': ['Small Material Pack', 'Medium Material Pack', 'Bountiful Material Pack',
                                  'Small Resource Pile', 'Medium Resource Pile', 'Bountiful Resource Pile'],
                        'Probabilities': [1,0,0,0,0,0]}}


class TestRolls(unittest.TestCase):

    def test_human_format(self):
        x = human_format(5000)
        expected = "5K"
        assert x == expected, "x should be '5K'"

    def test_first_roll_out(self):
        rolls_out = RollOut(1)
        inputs = [{'array': ['test1'],
                    'probabilities':[1]},
                  {'array': ['test1', 'test2'],
                   'probabilities': [0, 1]},
                  {'array': ['test1', 'test2','test3'],
                   'probabilities': [0, 0, 1]},
                  {'array': ['test1', 'test2','test3', 'test4'],
                   'probabilities': [0,0,0,1]},
                  {'array': ['test1', 'test2', 'test3', 'test4','test5'],
                   'probabilities': [.5, .5, .1, 0,0]}
                  ]
        k=0
        for i in inputs:
            input_set = i
            expected = i['array'][k]
            if k<4:
                self.assertEqual(rolls_out.first_roll_out_dynamics(input_set['array'], input_set['probabilities'], True),
                                 [expected], f"Should be [{expected}]")
            else:
                with self.assertRaises(TypeError, msg="Input should sum 1:"):
                    rolls_out.first_roll_out_dynamics(input_set['array'], input_set['probabilities'], True)

            k += 1


    def test_complete_dynamics_ou1(self):
        rolls_out = RollOut(1)
        k=0
        for fr in fr_s:
            if k<1:
                idx = 'Amazing'
                rew = ['Mystery Box Tier 3']
            else:
                if k < 2:
                    idx = 'Regular'
                    rew = ['Mystery Box Tier 1']
                else:
                    idx = 'Poor'
                    rew = ['Small Material Pack']

            expected = all_rewards_set[idx]
            self.assertEqual(rolls_out.complete_dynamics(fr, all_rewards_set)[0][idx], rew, f"Should be {rew}")
            k+=1
    def test_complete_dynamics_ou2(self):
        rolls_out = RollOut(2)
        fr = fr_s[0]
        expected = [0, 0, 0, 0]

        self.assertEqual(rolls_out.complete_dynamics(fr, all_rewards_set)[-1], expected, "Should be [0, 0, 0, 0]")

        self.assertEqual(len(rolls_out.complete_dynamics(fr, all_rewards_set)[-1]), 4, "Should have len 4")

class TestProbabilityDistributionFunction(unittest.TestCase):
    def test_hypergeom_pmf(self):
        A=3
        N=5
        draws=1
        expected_one_draw_prob = A/N
        hyperg = HypergeometricDistributionFunction(A, N)
        self.assertEqual(hyperg.hypergeom_pmf(N, A, draws, 1), expected_one_draw_prob, f"Should be {expected_one_draw_prob}")

    def test_binomial_distribution(self):
        n_s=[100,10,55]
        p_s=[1,.5,.25]
        for i in range(0,2):
            binom = BinomialDistributionFunction(n_s[i])
            self.assertEqual(
                binom.binomial_distribution(p_s[i])['binomial_mean'], n_s[i] * p_s[i], f"Should be {n_s[i] * p_s[i]}")

            self.assertEqual(binom.binomial_distribution(p_s[i])['binomial_variance'], n_s[i] * p_s[i] * (1 - p_s[i]),
                             f"Should be {n_s[i] * p_s[i] * (1 - p_s[i])}")


class TestConversionFunction(unittest.TestCase):

    @patch('source.auxiliar_functions.requests')
    def test_ether_to_usd(self, mock_requests):
        # mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'USD': 1981.00
        }

        # specify the return value of the get() method
        mock_requests.get.return_value = mock_response

        # call the ether_to_usd and test if the USD is 1981.00
        self.assertEqual(ether_to_usd('USD'), 1981.00)

if __name__ == '__main__':
    unittest.main()