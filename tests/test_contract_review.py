import unittest
from vlm_semantic_planner import normalize_plan
from vlm_evidence_answerer import build_prompt
from phase9_plan_contract import expression

class ContractReview(unittest.TestCase):
    def test_invalid_target_never_changes_calculation_indices(self):
        plan=dict(chart_type='bar',y_axis_count=1,has_direct_value_labels=False,operation='evidence',
                  targets=[{},dict(x_label='A'),dict(x_label='B')],
                  calculations=[dict(name='A',expression={'target':1})])
        with self.assertRaisesRegex(ValueError,'target_missing_location'):
            normalize_plan(plan,'Read A')
        plan['targets']=[None,dict(x_label='A'),dict(x_label='B')]
        with self.assertRaisesRegex(ValueError,'target_not_object'):
            normalize_plan(plan,'Read A')

    def test_question_type_is_explicit_in_backend_prompt(self):
        common=dict(question='Which statements hold?',options='A: one\nB: two\nC: three')
        single=build_prompt({**common,'question_type':'single_choice'}, {})
        multi=build_prompt({**common,'question_type':'multiple_choice'}, {})
        self.assertNotEqual(single,multi)
        self.assertIn('This is SINGLE CHOICE',single)
        self.assertNotIn('This is MULTIPLE CHOICE',single)
        self.assertIn('This is MULTIPLE CHOICE',multi)
        self.assertIn('"answer": ["A", "C"]',multi)

    def test_sequence_extrema_can_feed_multistep_arithmetic(self):
        trace=dict(kind='sequence',min=3,max=12)
        op=lambda name:dict(op=name,args=[{'target':0}])
        self.assertEqual(expression({'op':'difference','args':[op('max'),op('min')]},[trace]),9)
        with self.assertRaisesRegex(ValueError,'scalar'):
            expression(op('mean'),[trace])

if __name__=='__main__':unittest.main()
