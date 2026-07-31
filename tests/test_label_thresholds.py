import unittest

import numpy as np
import pandas as pd

from generateLabels.median_gain import _generate_median_gain
from generateLabels.median_loss import _generate_median_loss
from generateLabels.thresholds import get_purged_training_labels


class LabelThresholdTests(unittest.TestCase):
    def test_threshold_sample_excludes_validation_test_and_embargo(self):
        values = np.arange(200, dtype=float)
        selected = get_purged_training_labels(
            values,
            rolling_window=10,
            test_length=40,
            val_length=20,
        )
        np.testing.assert_array_equal(selected, np.arange(130, dtype=float))

    def test_gain_and_loss_thresholds_ignore_test_price_changes(self):
        close = np.linspace(100, 180, 260)
        base = pd.DataFrame({'Close': close})
        changed_test = base.copy()
        changed_test.loc[200:, 'Close'] *= np.linspace(1, 3, 60)

        _, base_gain_threshold = _generate_median_gain(
            base,
            'Close',
            rolling_window=5,
            test_length=40,
            val_length=20,
        )
        _, changed_gain_threshold = _generate_median_gain(
            changed_test,
            'Close',
            rolling_window=5,
            test_length=40,
            val_length=20,
        )
        _, base_loss_threshold = _generate_median_loss(
            base,
            'Close',
            rolling_window=5,
            test_length=40,
            val_length=20,
        )
        _, changed_loss_threshold = _generate_median_loss(
            changed_test,
            'Close',
            rolling_window=5,
            test_length=40,
            val_length=20,
        )

        self.assertAlmostEqual(base_gain_threshold, changed_gain_threshold)
        self.assertAlmostEqual(base_loss_threshold, changed_loss_threshold)


if __name__ == '__main__':
    unittest.main()
