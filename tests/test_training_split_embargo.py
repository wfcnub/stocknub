import unittest

import pandas as pd

from trainModels.splits import purge_overlapping_label_periods


class TrainingSplitEmbargoTests(unittest.TestCase):
    def test_horizon_is_removed_at_train_and_validation_boundaries(self):
        data = pd.DataFrame({
            'Date': pd.bdate_range('2024-01-02', periods=30).strftime('%Y-%m-%d'),
        })
        train_mask = data.index < 15
        val_mask = (data.index >= 15) & (data.index < 25)
        train_val_mask = data.index < 25

        purged_train_val, purged_val = purge_overlapping_label_periods(
            data,
            pd.Series(train_val_mask),
            pd.Series(train_mask),
            pd.Series(val_mask),
            'Median Gain 5dd',
        )

        retained_indices = set(data.index[purged_train_val])
        self.assertTrue(set(range(10, 15)).isdisjoint(retained_indices))
        self.assertTrue(set(range(20, 25)).isdisjoint(retained_indices))
        self.assertEqual(set(data.index[purged_val]), set(range(15, 20)))


if __name__ == '__main__':
    unittest.main()
