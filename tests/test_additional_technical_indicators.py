import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from prepareTechnicalIndicators.additional_technical_indicators import (
    calculate_additional_technical_indicators,
)


def _flow_fixture(rows=100):
    index = pd.bdate_range('2024-01-02', periods=rows, name='Date')
    volume = pd.Series(np.linspace(1_000, 2_000, rows), index=index)
    close = pd.Series(np.linspace(100, 130, rows), index=index)
    data = pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': volume,
        'Foreign Buy': volume * 0.40,
        'Foreign Sell': volume * 0.20,
        'Non Regular Volume': volume * 0.10,
        'Non Regular Value': volume * 0.10 * close,
        'Non Regular Frequency': np.arange(rows) + 1,
    }, index=index)
    return data


class AdditionalTechnicalIndicatorTests(unittest.TestCase):
    def test_daily_flow_formulas_and_publication_lag(self):
        data = _flow_fixture()
        result = calculate_additional_technical_indicators(data)

        # Row t contains transaction information from t-1.
        self.assertAlmostEqual(result['Foreign Net Intensity 1D'].iloc[1], 0.20)
        self.assertAlmostEqual(result['Foreign Buy Share 1D'].iloc[1], 0.40)
        self.assertAlmostEqual(result['Foreign Sell Share 1D'].iloc[1], 0.20)
        self.assertAlmostEqual(result['Foreign Participation 1D'].iloc[1], 0.30)
        self.assertAlmostEqual(result['Non Regular Volume Share 1D'].iloc[1], 0.10)
        self.assertTrue(result.iloc[0].isna().all())

    def test_features_are_causal(self):
        data = _flow_fixture(100)
        prefix = calculate_additional_technical_indicators(data.iloc[:80])
        full = calculate_additional_technical_indicators(data).loc[prefix.index]
        assert_frame_equal(prefix, full)

    def test_result_has_no_infinite_values(self):
        data = _flow_fixture()
        data.iloc[10, data.columns.get_loc('Volume')] = 0
        result = calculate_additional_technical_indicators(data)
        self.assertEqual(int(np.isinf(result.select_dtypes('number')).sum().sum()), 0)


if __name__ == '__main__':
    unittest.main()
