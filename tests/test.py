import unittest
from unittest.mock import MagicMock, patch
from source.auxiliar_functions import ether_to_usd, HypergeometricDistributionFunction, human_format, RollOut, BinomialDistributionFunction


categories = ['Amazing', 'Regular', 'Poor']
probabs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


all_rewards_set1 = {'Amazing': {'Types': ['Mystery Box Tier 3', 'Plot 8*8', 'Plot 16*16', 'Plot 32*32'],
                           'Probabilities': [0,1,0,0]},
               'Regular': {'Types': ['Mystery Box Tier 1', 'Recipe', 'Mystery Box Tier 2'],
                           'Probabilities': [1,0,0]},
               'Poor': {'Types': ['Small Material Pack', 'Medium Material Pack', 'Bountiful Material Pack',
                                  'Small Resource Pile', 'Medium Resource Pile', 'Bountiful Resource Pile'],
                        'Probabilities': [1,0,0,0,0,0]}}


all_rewards_set = {'Plots': {'Types': ['Plot 8*8', 'Plot 16*16', 'Plot 32*32', 'Plot 64x64'],
                           'Probabilities': [1,0,0,0,0]},
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
                           'Probabilities': [1,0,0,0,0,0,0,0,0,0,0]},
               'NMysteryB': {'Types': ['Material Pack 1', 'Material Pack 2', 'Material Pack 3',
                                  'Resource Pack 1', 'Resource Pack 2', 'Resource Pack 3','Recipe (3)'],
                        'Probabilities': [1,0,0,0,0,0,0]}}

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
                self.assertEqual(
                    rolls_out.first_roll_out_dynamics(input_set['array'], input_set['probabilities'], True),
                                 [expected], f"Should be [{expected}]")
            else:
                with self.assertRaises(TypeError, msg="Input should sum 1:"):
                    rolls_out.first_roll_out_dynamics(input_set['array'], input_set['probabilities'], True)

            k += 1

    def test_complete_dynamics_ou1(self):
        rolls_out = RollOut(1)
        k=0
        for fr in probabs:
            if k<1:
                idx = 'Amazing'
                rew = ['Plot 8*8']
            else:
                if k < 2:
                    idx = 'Regular'
                    rew = ['Familiar Bronze Mystery Box (5)']
                else:
                    idx = 'Poor'
                    rew = ['Material Pack 1']

            expected = all_rewards_set[idx]
            self.assertEqual(
                rolls_out.complete_dynamics(
                    dict(categories=categories, probabilities=fr), all_rewards_set)[0][idx], rew, f"Should be {rew}")
            k+=1

    def test_complete_dynamics_ou2(self):
        rolls_out = RollOut(2)

        fr = {'categories': categories,
              'probabilities': probabs[0]}
        expected = [2, 2, 0, 0]

        self.assertEqual(rolls_out.complete_dynamics(fr, all_rewards_set)[-1], expected, "Should be [2, 2, 0, 0]")

        self.assertEqual(len(rolls_out.complete_dynamics(fr, all_rewards_set)[-1]), 4, "Should have len 4")


class TestProbabilityDistributionFunction(unittest.TestCase):
    def test_hypergeom_pmf(self):
        A=3
        N=5
        draws=1
        expected_one_draw_prob = A/N
        hyperg = HypergeometricDistributionFunction(A, N)
        self.assertEqual(
            hyperg.hypergeom_pmf(N, A, draws, 1), expected_one_draw_prob, f"Should be {expected_one_draw_prob}")

    def test_binomial_distribution(self):
        n_s=[100,10,55]
        p_s=[1,.5,.25]
        for i in range(0,2):
            binom = BinomialDistributionFunction(n_s[i])
            self.assertEqual(
                binom.binomial_distribution(p_s[i])['binomial_mean'], n_s[i] * p_s[i], f"Should be {n_s[i] * p_s[i]}")

            self.assertEqual(
                binom.binomial_distribution(p_s[i])['binomial_variance'], n_s[i] * p_s[i] * (1 - p_s[i]),
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