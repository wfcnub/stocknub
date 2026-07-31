import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from prepareTechnicalIndicators.all_technical_indicators import (
    MINIMUM_WARMUP_PERIODS,
    generate_all_technical_indicators,
)


def _complete_fixture(rows=640):
    rng = np.random.default_rng(23)
    dates = pd.bdate_range('2021-01-04', periods=rows)
    close = 80 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, rows)))
    open_price = close * (1 + rng.normal(0, 0.003, rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.015, rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.015, rows))
    volume = rng.integers(800_000, 6_000_000, rows).astype(float)
    ohlcv = pd.DataFrame({
        'Date': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    })
    additional = ohlcv.copy()
    additional['Foreign Buy'] = volume * rng.uniform(0.05, 0.50, rows)
    additional['Foreign Sell'] = volume * rng.uniform(0.05, 0.50, rows)
    additional['Non Regular Volume'] = volume * rng.uniform(0, 0.15, rows)
    additional['Non Regular Value'] = additional['Non Regular Volume'] * close
    additional['Non Regular Frequency'] = rng.integers(1, 100, rows)
    return ohlcv, additional


class TechnicalIndicatorIntegrationTests(unittest.TestCase):
    def test_schema_warmup_and_causal_invariance(self):
        ohlcv, additional = _complete_fixture()
        prefix_rows = 580
        prefix = generate_all_technical_indicators(
            ohlcv.iloc[:prefix_rows],
            additional.iloc[:prefix_rows],
        )
        full = generate_all_technical_indicators(ohlcv, additional)
        full_prefix = full.loc[prefix.index]

        self.assertEqual(len(full), len(ohlcv) - MINIMUM_WARMUP_PERIODS)
        self.assertFalse(any('Zig Zag' in column for column in full.columns))
        self.assertIn('ADX Value', full.columns)
        self.assertIn('ATR Percent 14D', full.columns)
        self.assertIn('Foreign Net Intensity 20D', full.columns)
        self.assertIn('Non Regular Volume Share 20D', full.columns)
        self.assertEqual(int(np.isinf(full.select_dtypes('number')).sum().sum()), 0)
        assert_frame_equal(
            prefix,
            full_prefix,
            check_exact=False,
            rtol=1e-11,
            atol=1e-11,
        )


if __name__ == '__main__':
    unittest.main()
