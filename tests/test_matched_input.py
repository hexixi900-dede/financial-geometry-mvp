import tempfile
import unittest
from pathlib import Path

from prepare_matched_input import matched_input
from vlm_semantic_planner import messages
from vlm_evidence_answerer import build_prompt


class MatchedInput(unittest.TestCase):
    def test_raw_image_and_requested_unit_reach_both_vlm_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            png=Path(directory)/'raw.png';png.touch()
            src=dict(sample_id='1',question='How many subscribers?',question_type='numerical',options='',
                     caption='extra description unavailable to Raw',image_path='/tmp/geometry.jpg')
            raw={**src,'image_path':str(png),'image_sha256':'raw-hash','metadata':{'unit':'million'}}
            aligned=matched_input(src,raw)
            self.assertEqual(aligned['image_path'],'/tmp/geometry.jpg')
            self.assertEqual(aligned['caption'],'')
            msg=messages(aligned,200704,802816)
            self.assertEqual(msg[1]['content'][0]['image'],png.as_uri())
            self.assertEqual(msg[1]['content'][0]['max_pixels'],802816)
            for prompt in (msg[1]['content'][1]['text'],build_prompt(aligned,{})):
                self.assertIn('Requested answer unit: million',prompt)
                self.assertNotIn(src['caption'],prompt)
            with self.assertRaisesRegex(ValueError,'question mismatch'):
                matched_input(src,{**raw,'question':'different question'})


if __name__=='__main__':unittest.main()
