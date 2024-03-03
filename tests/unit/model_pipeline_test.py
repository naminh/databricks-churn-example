import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from churn.model import ModelPipeline


class ModelPipelineTest(unittest.TestCase):

    def test_create_train_pipeline(self):
        @dataclass
        class Example:
            A: int # col names
            B: float
            C: int
            D: float
            E: int
            F: float
            G: int

        X = pd.DataFrame(data=[
            Example(0, np.nan, 0, np.nan, 293, 0.0, 0),
            Example(1, 370, 567, 32, 370, 0.0, 0),
            Example(0, 8, 200, 27, 2, 10.0, 4),
            Example(0, 10, 100, 40, 4, 30.0, 6),
            Example(1, 100, np.nan, np.nan, 100, 0.0, 0),
        ])
        y = np.random.randint(2, size=5)

        model_params = {'n_estimators': 4,
                        'max_depth': 4,
                        'min_samples_leaf': 1,
                        'max_features': 'auto',
                        'random_state': 42}

        pipeline = ModelPipeline.create_baseline_pipeline(model_params=model_params)
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        assert np.array_equal(y_pred, y_pred.astype(bool))