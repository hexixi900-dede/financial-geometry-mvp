"""Shared visual measurement contract and deterministic arithmetic."""
import math

OPERATIONS = {'direct', 'difference', 'growth_rate', 'sum', 'mean', 'min', 'max', 'ratio', 'evidence'}

# Literal schema tokens are not chart locations. Keep the original targets and
# indices intact when rejecting these so a visual plan-repair pass can fix them.
PLACEHOLDER_LABELS = {
    'date', 'month', 'year', 'day', 'category', 'x_label', 'target_date',
    'start_date', 'end_date', 'start_month', 'end_month', 'date/month',
    'month/year', 'category/date', 'date/category',
    'category/date or interval description', '...',
}

def validate_plan(plan, question=''):
    # Routing is derived again after a visual correction, never inherited from
    # the previous version of a plan that may now request a Raw fallback.
    plan = {**plan, 'route_geometry': False, 'targets': [dict(t) for t in plan.get('targets', [])]}
    calculations = plan.get('calculations', [])
    if calculations is None: calculations = []
    if not isinstance(calculations,list) or any(not isinstance(c,dict) or not isinstance(c.get('expression'),dict) for c in calculations):
        return plan, 'invalid_calculations'
    plan['calculations'] = calculations
    targets = plan['targets']
    for calculation in calculations:
        error = expression_error(calculation['expression'], len(targets))
        if error:
            return plan, error
    op = plan.get('operation')
    if op not in OPERATIONS or not targets:
        return plan, 'missing_measurements'
    if op == 'direct' and len(targets) != 1 or op in {'difference','growth_rate','ratio'} and len(targets) != 2:
        return plan, 'operation_target_count_mismatch'
    for t in targets:
        t.setdefault('x_label', '')
        t.setdefault('series', '')
        t.setdefault('measurement', 'value')
        t.setdefault('position', 'label')
        if str(t['x_label']).strip().lower().strip('<>') in PLACEHOLDER_LABELS:
            return plan, 'placeholder_target_label'
        region = t.get('region')
        if region is not None:
            if not isinstance(region, list) or len(region) != 4 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in region):
                return plan, 'invalid_visual_region'
            if region[0] >= region[2] or region[1] >= region[3]:
                return plan, 'invalid_visual_region'
        if not t['x_label'] and region is None and t['position'] == 'label':
            return plan, 'missing_target_location'
        if t['measurement'] not in {'value','height','sequence'} or t['position'] not in {'label','first','last'}:
            return plan, 'unsupported_measurement'
        if t['measurement'] == 'sequence' and (plan.get('chart_type') != 'line' or op != 'evidence'):
            return plan, 'sequence_requires_line_evidence'
    if plan.get('chart_type') not in {'bar','line'} or plan.get('y_axis_count') != 1:
        return plan, 'unsupported_chart_structure'
    if plan.get('has_direct_value_labels') is not False:
        return plan, 'direct_value_labels_or_unknown'
    plan['route_geometry'] = True
    return plan, 'ok'

def calculate(operation, values):
    if operation == 'evidence':
        return None
    if operation in {'min','max'}:
        # Geometry already computes extrema from all recovered sequence pixels.
        values = [v[operation] if isinstance(v,dict) and v.get('kind')=='sequence' else v for v in values]
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('arithmetic_requires_scalar_measurements')
    if operation == 'direct': return values[0]
    if operation == 'difference': return values[0] - values[1]
    if operation == 'growth_rate': return (values[1] - values[0]) / values[0] * 100
    if operation == 'ratio': return values[0] / values[1]
    if operation == 'sum': return sum(values)
    if operation == 'mean': return sum(values) / len(values)
    if operation == 'min': return min(values)
    if operation == 'max': return max(values)
    raise ValueError('unsupported_operation')

def expression_error(expr, target_count):
    """Check the existing arithmetic contract before spending work on geometry."""
    if not isinstance(expr, dict): return 'invalid_calculation'
    if set(expr) == {'target'}:
        i = expr['target']
        return None if type(i) is int and 0 <= i < target_count else 'invalid_target_reference'
    if set(expr) != {'op', 'args'}: return 'invalid_calculation'
    op = expr.get('op'); args = expr.get('args', [])
    if op not in OPERATIONS - {'evidence'} or not isinstance(args,list) or not args:
        return 'invalid_calculation'
    if op == 'direct' and len(args) != 1 or op in {'difference','ratio','growth_rate'} and len(args) != 2:
        return 'invalid_calculation_arity'
    for arg in args:
        error = expression_error(arg, target_count)
        if error: return error
    return None


def expression(expr, values):
    """Evaluate nested arithmetic over measured targets only, never execute model code."""
    error = expression_error(expr, len(values))
    if error: raise ValueError(error)
    def evaluate(node):
        if 'target' in node: return values[node['target']]
        return calculate(node['op'], [evaluate(arg) for arg in node['args']])
    return evaluate(expr)
