"""Input faults must not silently propagate into forecasts or unsafe backups."""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from gridwatch.data import WEATHER, ZONES
from gridwatch.quality import fit_quality_bounds, guarded_forecast


class QualityContract(unittest.TestCase):
    def setUp(self):
        columns = WEATHER + ZONES + [f"{zone}_lag_1" for zone in ZONES]
        self.training = pd.DataFrame({name: np.arange(10., 30.) for name in columns})
        target = np.column_stack([np.arange(100., 120.) + offset for offset in [0, 10, 20]])
        self.model = LinearRegression().fit(self.training, target)
        self.bounds = fit_quality_bounds(self.training)

    def test_missing_weather_uses_available_current_measurement(self):
        packet = self.training.iloc[[2]].copy()
        packet["temperature"] = np.nan
        forecast, methods, reasons = guarded_forecast(self.model, packet, self.bounds)
        np.testing.assert_array_equal(forecast[0], packet[ZONES].iloc[0])
        self.assertTrue(all(method == "Persistence: current" for method in methods[0]))
        self.assertIn("temperature", reasons[0])

    def test_spiked_current_is_not_reused_as_its_own_backup(self):
        packet = self.training.iloc[[2]].copy()
        packet["zone_1"] *= 100
        forecast, methods, _ = guarded_forecast(self.model, packet, self.bounds)
        self.assertEqual(forecast[0, 0], packet.zone_1_lag_1.iloc[0])
        self.assertEqual(methods[0, 0], "Persistence: 10 minutes earlier")

    def test_unavailable_backups_produce_no_invented_forecast(self):
        packet = self.training.iloc[[2]].copy()
        packet[ZONES + [f"{zone}_lag_1" for zone in ZONES]] = np.nan
        forecast, methods, _ = guarded_forecast(self.model, packet, self.bounds)
        self.assertTrue(np.isnan(forecast).all())
        self.assertTrue((methods == "Unavailable").all())

    def test_clean_packet_matches_the_model(self):
        packet = self.training.iloc[[2]].copy()
        forecast, methods, _ = guarded_forecast(self.model, packet, self.bounds)
        np.testing.assert_allclose(forecast, self.model.predict(packet))
        self.assertTrue((methods == "Linear model").all())

    def test_envelope_uses_training_only_and_rejects_wrong_column_order(self):
        self.assertEqual(self.bounds.loc["zone_1", "upper"], 29 + .25 * 19)
        with self.assertRaises(ValueError):
            guarded_forecast(self.model, self.training.iloc[[0], ::-1], self.bounds)


if __name__ == "__main__":
    unittest.main()
