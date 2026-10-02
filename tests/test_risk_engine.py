import unittest
from unittest.mock import patch

import pandas as pd

from src.inference import risk_engine


class RiskEngineTests(unittest.TestCase):
    def test_risk_classification_thresholds(self):
        self.assertEqual(risk_engine.calculate_risk_level({"negative_rate": 0.51}), "HIGH")
        self.assertEqual(risk_engine.calculate_risk_level({"late_delivery_rate": 0.081}), "HIGH")
        self.assertEqual(risk_engine.calculate_risk_level({"seller_health_index_v2": 0.29}), "HIGH")
        self.assertEqual(risk_engine.calculate_risk_level({"seller_health_index_v2": 0.4}), "MEDIUM")
        self.assertEqual(risk_engine.calculate_risk_level({"seller_health_index_v2": 0.8}), "LOW")

    def test_probability_prediction_loads_model_on_demand(self):
        class Model:
            def predict_proba(self, frame):
                self.frame = frame
                return [[0.2, 0.8]]

        model = Model()
        frame = pd.DataFrame([{
            "estimated_delivery_days": 10,
            "order_month": 6,
            "order_weekday": 2,
            "total_payment_value": 100.0,
            "avg_installments": 2,
            "total_price": 80.0,
            "total_freight": 20.0,
            "total_items": 1,
            "seller_late_rate": 0.1,
        }])
        with patch.object(risk_engine, "load_xgb_model", return_value=model) as load_model:
            self.assertEqual(risk_engine.predict_late_probability(frame), 0.8)
        load_model.assert_called_once_with()
        self.assertIs(model.frame, frame)

    def test_prediction_rejects_missing_features(self):
        with self.assertRaisesRegex(ValueError, "Missing feature column: estimated_delivery_days"):
            risk_engine.predict_late_probability(pd.DataFrame({"late_delivery_rate": [0.1]}))

    def test_prediction_rejects_single_class_output(self):
        frame = pd.DataFrame([{
            "estimated_delivery_days": 10,
            "order_month": 6,
            "order_weekday": 2,
            "total_payment_value": 100.0,
            "avg_installments": 2,
            "total_price": 80.0,
            "total_freight": 20.0,
            "total_items": 1,
            "seller_late_rate": 0.1,
        }])
        model = type("Model", (), {"predict_proba": lambda self, data: [[1.0]]})()
        with patch.object(risk_engine, "load_xgb_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "both classes"):
                risk_engine.predict_late_probability(frame)


if __name__ == "__main__":
    unittest.main()
