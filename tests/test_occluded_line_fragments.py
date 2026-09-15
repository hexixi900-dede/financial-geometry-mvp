"""Actual colored pixels split by occlusion; no OCR or benchmark labels."""
import unittest
import cv2
import numpy as np
from phase4_series import detect_line_series_components
from phase8_geometry_adapter import read_line

RED=(32,36,202)

class OccludedTrace(unittest.TestCase):
    def test_short_fragments_form_series_without_filling_occluder(self):
        image=np.full((220,640,3),255,np.uint8)
        axis=dict(plot_bbox=[20,20,620,200],slope=-1,intercept=200)
        # Five visible pieces: every piece is shorter than 24% of plot width.
        for start in (30,150,270,390,510):
            cv2.line(image,(start,90),(start+100,140),RED,3)
        image[:,319:326]=(198,198,198)
        series=[s for s in detect_line_series_components(image,axis)
                if s['median_hsv'][1]>=60]
        self.assertEqual(len(series),1)
        self.assertGreater(series[0]['column_count'],450)
        self.assertFalse(series[0]['mask'][:,319:326].any())
        self.assertFalse(series[0]['mask'][:,135:145].any())
        self.assertTrue(series[0]['mask'][:,550].any())
        chart=dict(axis=axis,x_axis_anchors=[dict(label='2024',center_x=550)],
                   legend={'entries':[]})
        result=read_line(image,chart,dict(x_label='2024',series='Revenue'),series)
        self.assertEqual(result['status'],'success')
        self.assertAlmostEqual(result['auto_local_value'],90,delta=1)

    def test_two_tiny_distant_marks_do_not_become_a_full_trace(self):
        image=np.full((220,640,3),255,np.uint8)
        axis=dict(plot_bbox=[20,20,620,200])
        cv2.line(image,(30,60),(40,65),RED,3)
        cv2.line(image,(600,80),(610,85),RED,3)
        self.assertEqual(detect_line_series_components(image,axis),[])

if __name__=='__main__':unittest.main()
