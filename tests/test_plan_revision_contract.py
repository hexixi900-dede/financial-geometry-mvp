"""Contract regressions for visual revision and executable numerical plans."""
import copy
import unittest

from phase9_plan_contract import calculate, expression, validate_plan
from vlm_semantic_planner import normalize_plan, SYNTHETIC_PLANNING_EXAMPLES


class PlanRevisionContract(unittest.TestCase):
    def test_normalized_plan_can_be_revised_without_stale_routing(self):
        example = SYNTHETIC_PLANNING_EXAMPLES[0]
        original = normalize_plan(example['plan'], example['question'])
        revised = normalize_plan(original, example['question'])
        self.assertEqual(revised, original)
        self.assertIn('17-Jun-2023', revised['reason'])
        revised['has_direct_value_labels'] = True
        result, reason = validate_plan(revised)
        self.assertEqual(reason, 'direct_value_labels_or_unknown')
        self.assertFalse(result['route_geometry'])
        result = normalize_plan(revised, example['question'])
        self.assertFalse(result['route_geometry'])
        self.assertEqual(result['route_reason'], 'direct_value_labels_or_unknown')

    def test_invalid_index_or_arity_is_repairable_before_measurement(self):
        example = SYNTHETIC_PLANNING_EXAMPLES[0]
        broken = copy.deepcopy(example['plan'])
        expr = broken['calculations'][0]['expression']
        expr['args'][0]['target'] = 99
        result = normalize_plan(broken, example['question'])
        self.assertEqual(result['route_reason'], 'invalid_target_reference')
        self.assertFalse(result['route_geometry'])
        self.assertEqual(len(result['targets']), 2)
        self.assertEqual(result['calculations'][0]['expression']['args'][0]['target'], 99)
        with self.assertRaisesRegex(ValueError, 'invalid_target_reference'):
            expression(expr, [40, 50])
        expr['args'] = [{'target': 0}]
        result = normalize_plan(broken, example['question'])
        self.assertEqual(result['route_reason'], 'invalid_calculation_arity')

    def test_complete_synthetic_measurements_have_the_intended_arithmetic(self):
        comparison, annual = SYNTHETIC_PLANNING_EXAMPLES
        p = normalize_plan(comparison['plan'], comparison['question'])
        # An increase and a relative growth use different argument order.
        self.assertEqual([expression(c['expression'], [40, 50]) for c in p['calculations']], [10, 25])
        p = normalize_plan(annual['plan'], annual['question'])
        self.assertTrue(p['route_geometry'])
        self.assertEqual(len({t['x_label'] for t in p['targets']}), 12)
        self.assertTrue(all(t['position'] == 'label' and 'region' not in t for t in p['targets']))
        self.assertEqual(calculate(p['operation'], list(range(1, 13))), 6.5)


if __name__ == '__main__':
    unittest.main()
