"""Named single curves should work without collapsing distinct requested series."""
import json
import unittest

import numpy as np

from phase8_geometry_adapter import read_line
from phase8_hybrid_qa import run_line


class LineSeriesResolution(unittest.TestCase):
    def setUp(self):
        self.image = np.full((300, 400, 3), 255, np.uint8)
        mask = np.zeros((300, 400), np.uint8)
        for x in range(60, 341):
            mask[240-x//2, x] = 1
        self.series = [dict(series_id=0, mask=mask, bbox=[60, 70, 281, 141],
                            median_bgr=[20, 20, 200], median_hsv=[0, 200, 200])]
        self.chart = dict(axis=dict(plot_bbox=[40, 20, 360, 260], slope=-.1, intercept=26),
                          x_axis_anchors=[dict(label='2023', center_x=200, confidence=1)],
                          legend={'entries': []}, single_series_hint=False, caption='')

    def test_actual_single_line_does_not_require_hint_caption_or_legend_name(self):
        target = dict(x_label='2023', series='Total loans applied (3MMA)',
                      measurement='value', position='label')
        result = read_line(self.image, self.chart, target, self.series)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['series_resolution']['series_mode'], 'single_series_without_legend')
        self.assertAlmostEqual(result['auto_local_value'], 12, delta=.05)  # Half a pixel on this axis.

    def test_explicit_axis_date_wins_over_first_position(self):
        target = dict(x_label='2023', series='named curve', measurement='value', position='first',
                      region=[150, 150, 850, 850])
        result = read_line(self.image, self.chart, target, self.series)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['grounding']['mode'], 'direct_anchor')
        self.assertAlmostEqual(result['auto_local_value'], 12, delta=.05)  # Half a pixel on this axis.

    def test_visual_region_cannot_override_missing_named_legend_series(self):
        chart = {**self.chart, 'legend': {'entries': [
            dict(legend_index=0, name='Red curve', normalized='redcurve',
                 swatch={'median_bgr': [20, 20, 200]}),
            dict(legend_index=1, name='Blue curve', normalized='bluecurve',
                 swatch={'median_bgr': [200, 20, 20]}),
        ]}}
        target = dict(x_label='2023', series='Blue curve', measurement='value', position='label',
                      region=[150, 150, 850, 850])
        result = read_line(self.image, chart, target, self.series)
        self.assertEqual(result['status'], 'ambiguous_series')

    def test_distinct_requested_names_cannot_become_the_same_detected_curve(self):
        targets = [dict(x_label='2023', series=name, measurement='value', position='label')
                   for name in ('Red curve', 'Post Covid average')]
        cache = {('chart', json.dumps(t, sort_keys=True)):
                 read_line(self.image, self.chart, t, self.series) for t in targets}
        result = run_line(None, self.image, dict(chart_id='chart'),
                          dict(targets=targets, operation='difference'), cache)
        self.assertEqual(result['status'], 'ambiguous_series')
        self.assertEqual(result['status_reason'], 'distinct_requested_series_mapped_to_one_line')


if __name__ == '__main__':
    unittest.main()
