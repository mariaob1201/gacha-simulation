import unittest
import source.auxiliar_functions as auxiliar_functions2
import set
import pytest

from unittest.mock import MagicMock, patch

class TestClass(unittest.TestCase):

    def test_human_format(self):
        x = auxiliar_functions2.human_format(5000)
        expected = "5K"
        assert x == expected, "x should be '5K'"

    def test_first_roll_out(self):
        rolls_out = auxiliar_functions2.RollOut(1)
        input_set = {'array': ['test1','test2'],
                    'probabilities':[1,0]}
        expected = ['test1']
        self.assertEqual(rolls_out.first_roll_out_dynamics(input_set['array'], input_set['probabilities'], True), expected, "Should be 'test1'")

    def test_complete_dynamics_ou1(self):
        rolls_out = auxiliar_functions2.RollOut(1)
        expected = ['Plot 8*8']
        self.assertEqual(rolls_out.complete_dynamics(set.fr, set.all_rewards_set)[0]['Amazing'], expected, f"Should be ['Plot 8*8']")

    def test_complete_dynamics_ou2(self):
        rolls_out = auxiliar_functions2.RollOut(2)
        expected = [2, 2, 0, 0]
        self.assertEqual(rolls_out.complete_dynamics(set.fr, set.all_rewards_set)[-1], expected, "Should be [2, 2, 0, 0]")

    def test_complete_dynamics_ou2_len(self):
        rolls_out = auxiliar_functions2.RollOut(2)
        expected = 4
        self.assertEqual(len(rolls_out.complete_dynamics(set.fr, set.all_rewards_set)[-1]), expected, "Should have len 4")

    def test_hypergeom_pmf(self):
        A=3
        N=5
        draws=1
        expected_one_draw_prob = A/N
        hyperg = auxiliar_functions2.HypergeometricDistributionFunction(A, N)
        self.assertEqual(hyperg.hypergeom_pmf(N, A, draws, 1), expected_one_draw_prob, f"Should be {expected_one_draw_prob}")


class TestConversionFunction(unittest.TestCase):

    @patch('auxiliar_functions2.requests')
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
        self.assertEqual(auxiliar_functions2.ether_to_usd('USD'), 1981.00)

if __name__ == '__main__':
    unittest.main()
