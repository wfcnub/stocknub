import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from prepareTechnicalIndicators.price_characteristics import (
    calculate_price_characteristics,
)


def _ohlcv_fixture(rows=140):
    rng = np.random.default_rng(11)
    index = pd.bdate_range('2023-01-02', periods=rows, name='Date')
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, rows)))
    open_price = close * (1 + rng.normal(0, 0.003, rows))
    return pd.DataFrame({
        'Open': open_price,
        'High': np.maximum(open_price, close) * 1.01,
        'Low': np.minimum(open_price, close) * 0.99,
        'Close': close,
        'Volume': rng.integers(1_000_000, 4_000_000, rows),
    }, index=index)


class PriceCharacteristicTests(unittest.TestCase):
    def test_features_are_causal(self):
        data = _ohlcv_fixture()
        prefix = calculate_price_characteristics(data.iloc[:100])
        full = calculate_price_characteristics(data).loc[prefix.index]
        assert_frame_equal(prefix, full)

    def test_normalized_ranges_and_finite_output(self):
        result = calculate_price_characteristics(_ohlcv_fixture())
        populated = result.dropna()
        self.assertTrue(populated['Close Location Value'].between(-1, 1).all())
        self.assertTrue(populated['Up Volume Share 20D'].between(0, 1).all())
        self.assertEqual(int(np.isinf(result.select_dtypes('number')).sum().sum()), 0)


if __name__ == '__main__':
    unittest.main()
