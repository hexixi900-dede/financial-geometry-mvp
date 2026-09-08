"""Regression cases for series selection in grouped and stacked bars."""
import unittest

import numpy as np

from phase8_geometry_adapter import detect_series, match_bar_target, read_bar_height, read_line


class BarSeriesReview(unittest.TestCase):
    def setUp(self):
        self.image = np.full((300, 400, 3), 255, np.uint8)
        self.image[150:250, 80:110] = [150, 50, 20]
        self.image[100:150, 80:110] = [30, 160, 40]
        self.axis = dict(plot_bbox=[40, 20, 360, 260], slope=-.1, intercept=26,
                         ticks=[{'pixel_y': 260}])
        self.legend = {'entries': [
            dict(legend_index=0, name='lower', normalized='lower',
                 swatch={'median_bgr': [150, 50, 20]}),
            dict(legend_index=1, name='upper', normalized='upper',
                 swatch={'median_bgr': [30, 160, 40]}),
        ]}

    def test_named_segment_is_not_replaced_by_column_total(self):
        analysis = dict(axis=self.axis, legend=self.legend)
        # An approximate box includes both stacked segments. The legend must
        # distinguish the requested one before measuring its physical extent.
        results = []
        for name in ('lower', 'upper'):
            target = dict(x_label='category', series=name, measurement='height',
                          region=[190, 320, 290, 850])
            result = read_bar_height(None, self.image, analysis, target)
            self.assertEqual(result['status'], 'success')
            results.append(result['value'])
        self.assertAlmostEqual(results[0], 9.9)
        self.assertAlmostEqual(results[1], 4.9)

    def test_no_legend_preserves_visual_segment_and_total_measurements(self):
        for region, expected in (([190, 320, 290, 510], 4.9),
                                 ([190, 320, 290, 850], 14.9)):
            target = dict(x_label='category', series='visual object',
                          measurement='height', region=region)
            result = read_bar_height(None, self.image, {'axis': self.axis}, target)
            self.assertEqual(result['status'], 'success')
            self.assertAlmostEqual(result['value'], expected)

    def test_grouped_series_share_category_anchor_without_collapsing(self):
        bars = [dict(bar_id=i, detected_x_label='2023', target_x=90+20*i,
                     median_bgr=entry['swatch']['median_bgr'],
                     x_label_tokens=[dict(text='2023', cx=100)])
                for i, entry in enumerate(self.legend['entries'])]
        for i, name in enumerate(('lower', 'upper')):
            result = match_bar_target(dict(x_label='2023', series=name), bars, self.legend)
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['target_bar_ids'], [i])


class LineRegionReview(unittest.TestCase):
    def setUp(self):
        import cv2
        self.image = np.full((300, 400, 3), 255, np.uint8)
        cv2.line(self.image, (60, 70), (340, 230), (20, 20, 20), 2)
        cv2.line(self.image, (60, 230), (340, 70), (20, 20, 20), 2)
        axis = dict(plot_bbox=[40, 20, 360, 260], slope=-.1, intercept=26, ticks=[])
        self.chart = dict(axis=axis, x_axis_anchors=[], legend={'entries': []})
        self.series = detect_series(self.image, axis)
        # This is the real detector behavior for touching same-color strokes.
        self.assertEqual(len(self.series), 1)

    def read(self, region, measurement='value', position='label'):
        target = dict(x_label='upper right branch', series='', position=position,
                      measurement=measurement, region=region)
        return read_line(self.image, self.chart, target, self.series)

    def test_crossing_point_and_endpoint_use_region_pixels(self):
        point = self.read([775, 240, 825, 325])
        self.assertEqual(point['status'], 'success')
        self.assertAlmostEqual(point['auto_local_value'], 17.86, delta=.15)
        endpoint = self.read([830, 210, 870, 270], position='last')
        self.assertEqual(endpoint['status'], 'success')
        self.assertAlmostEqual(endpoint['auto_local_value'], 19, delta=.15)
        self.assertIn('visual_search_region', endpoint)

    def test_crossing_sequence_uses_region_instead_of_averaging_two_lines(self):
        result = self.read([750, 210, 875, 350], measurement='sequence')
        self.assertEqual(result['status'], 'success')
        trace = result['auto_local_value']
        self.assertAlmostEqual(trace['first'], 16.7, delta=.15)
        self.assertAlmostEqual(trace['last'], 19, delta=.15)
        self.assertGreater(trace['last'], trace['first'])

    def test_empty_region_is_not_reported_as_a_measurement(self):
        for measurement, position in (('value', 'label'), ('value', 'last'), ('sequence', 'label')):
            result = self.read([775, 420, 825, 550], measurement=measurement, position=position)
            self.assertEqual(result['status'], 'line_localization_failed')
            self.assertEqual(result['status_reason'], 'no_series_pixels_in_visual_region')

    def test_target_without_region_still_uses_axis_anchor(self):
        import cv2
        image = np.full((300, 400, 3), 255, np.uint8)
        cv2.line(image, (60, 230), (340, 70), (20, 20, 20), 2)
        series = detect_series(image, self.chart['axis'])
        chart = {**self.chart, 'x_axis_anchors': [dict(label='2024', center_x=320, confidence=1)]}
        result = read_line(image, chart, dict(x_label='2024', series='',
                           measurement='value', position='label'), series)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['grounding']['mode'], 'direct_anchor')
        self.assertAlmostEqual(result['auto_local_value'], 17.86, delta=.15)
        self.assertNotIn('visual_search_region', result)


if __name__ == '__main__':
    unittest.main()
