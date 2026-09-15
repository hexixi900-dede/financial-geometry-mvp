"""End-of-series and bar-endpoint requests retain their physical meaning."""
import unittest

import numpy as np

from phase8_geometry_adapter import read_bar_height, read_line


class MeasurementLocatorSemantics(unittest.TestCase):
    def setUp(self):
        self.image = np.full((300, 400, 3), 255, np.uint8)
        mask = np.zeros((300, 400), np.uint8)
        for x in range(60, 341):
            mask[240-x//2, x] = 1
        self.axis = dict(plot_bbox=[40, 20, 360, 260], slope=-.1, intercept=26,
                         ticks=[{'pixel_y': 260}])
        self.chart = dict(axis=self.axis, x_axis_anchors=[], legend={'entries': []})
        self.series = [dict(series_id=0, mask=mask, bbox=[60, 70, 281, 141],
                            median_bgr=[0, 0, 0], median_hsv=[0, 0, 0])]

    def line(self, position, region=None):
        target = dict(series='', x_label='observation', position=position, measurement='value')
        if region is not None:
            target['region'] = region
        return read_line(self.image, self.chart, target, self.series)

    def test_middle_crop_is_not_a_first_or_last_series_point(self):
        for position in ('first', 'last'):
            result = self.line(position, [650, 250, 750, 500])
            self.assertEqual(result['status'], 'line_localization_failed')
            self.assertNotIn('auto_local_value', result)

    def test_true_endpoint_with_or_without_region_uses_same_point(self):
        for region in (None, [830, 210, 870, 270]):
            result = self.line('last', region)
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['auto_local_geometry']['point'], [340, 70])
            self.assertAlmostEqual(result['auto_local_value'], 19)

    def test_corrected_point_region_can_replace_misplaced_axis_anchor(self):
        self.chart['x_axis_anchors'] = [dict(label='2023', center_x=100, confidence=1)]
        target = dict(series='', x_label='2023', position='label', measurement='value',
                      region=[720, 250, 780, 350])
        result = read_line(self.image, self.chart, target, self.series)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['auto_local_geometry']['target_x'], 300)
        self.assertAlmostEqual(result['auto_local_value'], 17, delta=.05)
        self.assertEqual(result['grounding']['previous_grounding']['predicted_target_x'], 100)

    def test_correct_axis_anchor_inside_region_keeps_its_precision(self):
        self.chart['x_axis_anchors'] = [dict(label='2023', center_x=294, confidence=1)]
        target = dict(series='', x_label='2023', position='label', measurement='value',
                      region=[720, 250, 780, 350])
        result = read_line(self.image, self.chart, target, self.series)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['auto_local_geometry']['target_x'], 294)
        self.assertAlmostEqual(result['auto_local_value'], 16.7, delta=.05)
        self.assertEqual(result['grounding']['mode'], 'direct_anchor')

    def bar(self, region, measurement='value'):
        self.image[60:101, 80:110] = [150, 50, 20]
        axis = {**self.axis, 'intercept': 15}
        target = dict(series='', x_label='floating mark', measurement=measurement, region=region)
        return read_bar_height(None, self.image, {'axis': axis}, target)

    def test_floating_bar_region_selects_the_requested_endpoint(self):
        for region, expected_y, expected_value in (([190, 310, 290, 355], 100, 5),
                                                   ([190, 180, 290, 225], 60, 9)):
            result = self.bar(region)
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['point'][1], expected_y)
            self.assertAlmostEqual(result['value'], expected_value)

    def test_whole_bar_value_and_height_keep_their_existing_meaning(self):
        region = [190, 180, 290, 355]
        self.assertAlmostEqual(self.bar(region)['value'], 9)
        self.assertAlmostEqual(self.bar(region, 'height')['value'], 4)


if __name__ == '__main__':
    unittest.main()
