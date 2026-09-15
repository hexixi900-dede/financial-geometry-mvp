import unittest
import numpy as np
from geometry_core import ocr_tokens
from geometry_toolbox import merge_date_anchors
from phase8_geometry_adapter import ground_label
from phase3_line_core import parse_axis_scalar
from phase4_legend import _swatch_in_zone, detect_legend_entries


def token(x,y,text):
    return [[[x-30,y-6],[x+30,y-6],[x+30,y+6],[x-30,y+6]],text,.99]


class ExistingOCRRegressions(unittest.TestCase):
    def test_full_dates_survive_and_bracket_target_without_extra_ocr(self):
        axis=dict(plot_bbox=[50,40,580,347],slope=-5,intercept=1735,token_indices=[])
        tokens=ocr_tokens([token(100,355,'2022/01/01'),token(300,355,'"2022/07/01'),
                           token(400,20,'2023/01/01'),token(300,390,'2024/01/01')])
        anchors=merge_date_anchors(tokens,axis,600,400,[])
        self.assertEqual(len(anchors),2)
        reading=ground_label('2022-04-01',anchors)
        self.assertEqual(reading['status'],'ok')
        self.assertAlmostEqual(reading['predicted_target_x'],200)
        self.assertEqual(ground_label('2022/07/01',anchors)['predicted_target_x'],300)
        self.assertNotEqual(ground_label('2024/01/01',anchors)['status'],'ok')

    def test_three_digit_ocr_year_cannot_poison_following_quarters(self):
        self.assertIsNone(parse_axis_scalar('3Q151'))
        self.assertEqual(parse_axis_scalar('3Q15').value,2015.5)
        self.assertEqual(parse_axis_scalar('1Q2016').value,2016.0)

    def test_black_one_pixel_line_is_a_swatch_but_neighbor_text_is_not(self):
        image=np.full((100,220,3),255,dtype=np.uint8)
        image[60,30:46]=0
        image[55:65,53:91]=85  # Neighbor glyphs should not dominate the color.
        boxes=[(53,55,91,65)]
        swatch=_swatch_in_zone(image,(20,50,95,70),boxes)
        self.assertEqual(swatch['median_bgr'],[0,0,0])
        self.assertEqual(swatch['bbox'],[30,60,46,61])
        image[60,30:46]=255
        self.assertIsNone(_swatch_in_zone(image,(20,50,95,70),boxes))


if __name__=='__main__':unittest.main()
