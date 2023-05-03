import unittest
import source.auxiliar_functions as auxiliar_functions2
import set
import pytest


class TestClass(unittest.TestCase):

    def test_human_format(self):
        x = auxiliar_functions2.human_format(5000)
        assert x == "5K", "x should be '5K'"

    def test_first_roll_out(self):
        rolls_out = auxiliar_functions2.RollOut(1)
        self.assertEqual(rolls_out.first_roll_out_dynamics(['test1','test2'], [1,0], True), ['test1'], "Should be 'test1'")

    def test_complete_dynamics_ou1(self):
        rolls_out = auxiliar_functions2.RollOut(1)
        self.assertEqual(rolls_out.complete_dynamics(set.fr, set.all_rewards_set)[0]['Amazing'], ['Plot 8*8'], "Should be ['Plot 8*8']")

    def test_complete_dynamics_ou2(self):
        rolls_out = auxiliar_functions2.RollOut(2)
        self.assertEqual(rolls_out.complete_dynamics(set.fr, set.all_rewards_set)[-1], [2, 2, 0, 0], "Should be [2, 2, 0, 0]")

    def test_complete_dynamics_ou2_len(self):
        rolls_out = auxiliar_functions2.RollOut(2)
        self.assertEqual(len(rolls_out.complete_dynamics(set.fr, set.all_rewards_set)[-1]), 4, "Should have len 4")

    def test_hypergeom_pmf(self):
        A=3
        N=5
        draws=1
        one_draw_prob = A/N
        hyperg = auxiliar_functions2.HypergeometricDistributionFunction(A, N)
        self.assertEqual(hyperg.hypergeom_pmf(N, A, draws, 1), one_draw_prob, f"Should be {one_draw_prob}")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 5, "Should be 5")

    @pytest.fixture()
    def fake_ether_info(self):
        """Fixture that returns a static exchange rate data."""
        return {'USD': 1910.99}

    def test_retrieve_weather_using_mocks(mocker):
        """Given a city name, test that a HTML report about the weather is generated
        correctly."""
        # Creates a fake requests response object
        fake_resp = mocker.Mock()
        # Mock the json method to return the static weather data
        fake_resp.json = mocker.Mock(return_value=self.fake_ether_info)
        # Mock the status code
        fake_resp.status_code = HTTPStatus.OK

        mocker.patch("ether_exch_rate.requests.get", return_value=self.fake_ether_info)
        exch_r = CurrenciesConversion(fake_resp, 1,1)
        er_info = exch_r.ether_to_usd(fake_resp)
        assert er_info == object.from_dict(self.fake_ether_info)


if __name__ == '__main__':
    unittest.main()
