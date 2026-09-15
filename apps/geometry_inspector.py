"""Browse saved runs and actual visual feedback; never start model inference."""
from pathlib import Path
from collections import Counter
import sys, json, hashlib, tempfile
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geometry_visual_feedback import render_measurement_overlay
st.set_page_config(page_title='Geometry QA', layout='wide')

@st.cache_data
def load(path, mtime):
    if not path.exists(): return {}
    out = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line); out[str(r['sample_id'])] = r
        except (ValueError, KeyError): continue
    return out

def read(path): return load(path, path.stat().st_mtime_ns if path.exists() else 0)
def document(path):
    try: return json.loads(path.read_text())
    except (ValueError, OSError): return {}

def runtime_state(state):
    if state.startswith('waiting_for_gpu'): return '等待 GPU 空闲，尚未加载模型'
    if state.startswith(('stopped', 'superseded')): return '已停止的旧实验，仅保留历史结果'
    if state.startswith('prepared') and 'source update' in state: return '代码准备中，尚未排队'
    if state.startswith('prepared'): return '已准备，等待前序实验完成'
    if state.startswith('failed'): return '运行失败，请查看日志'
    if state.startswith('pilot_complete'): return '64 题开发实测已完成，全量尚未启动'
    if state == 'complete': return '本轮推理已完成'
    if state == 'loading_model': return '正在加载模型'
    prefix = '典型案例：' if state.startswith('pilot:') else '评测全集：' if state.startswith('full:') else ''
    for key, value in [('question-level routing', 'VLM 正在判断题目是否需要几何数值'), ('target planning', 'VLM 正在规划目标及其坐标轴'), ('semantic planning', 'VLM 正在看图制定计划'), ('visual_planning', 'VLM 正在看图制定计划'),
                       ('initial CPU geometry', '正在测量'), ('review image', 'VLM 正在查看定位图'),
                       ('corrected CPU', '正在按修正计划重新测量'), ('final evidence', '正在生成最终答案'),
                       ('CPU geometry', '正在测量'), ('evidence answers', '正在生成增强答案')]:
        if key in state: return prefix + value
    return state or '尚未启动'

REASONS = {
    'ambiguous_series': '没有确定目标属于哪条曲线',
    'line_localization_failed': '目标区域内未定位到可测曲线',
    'x_grounding_unrecoverable': '日期／类别未对应到横轴位置',
    'bar_target_unrecoverable': '未定位到目标柱或柱段',
    'targets_not_distinct': '不同目标错误地对应到了同一根柱子',
    'placeholder_target_label': '计划使用了 date 等占位词，没有给出实际日期',
    'invalid_visual_region': '计划给出的定位框无效',
    'invalid_target_reference': '计算引用了不存在的目标，需修正计划',
    'invalid_calculation_arity': '算式缺少操作数，需修正计划',
    'operation_target_count_mismatch': '计算所需的点数与计划不一致',
    'semantic_plan_missing_or_invalid': '前端未生成可用计划',
    'distinct_requested_series_mapped_to_one_line': '不同系列被识别成了同一条线',
    'insufficient_evidence': '后端认为测量证据不足以回答',
    'answer_parse_failed': '模型答复格式无法解析',
    'inference_error': '模型调用失败',
}

def reason_text(m, reply=None):
    code = m.get('status_reason') or m.get('status', '')
    if reply and reply.get('status') != 'success' and m.get('status') == 'success': code = reply['status']
    return REASONS.get(code, '；'.join(REASONS.get(c, c) for c in code.split(' | ')))

def route_state(plan_record, measurement, reply, terminal):
    p = measurement.get('semantic_plan') or plan_record.get('plan') or {}
    if not plan_record and not measurement: return '待运行：还没有前端计划'
    if reply and reply.get('status') == 'success':
        return '修正后采用 Geometry 增强答案' if reply.get('plan_revised') else '采用 Geometry 增强答案'
    if terminal and measurement.get('geometry_entered') is True:
        return '流程最终答复无效，计错；不回填 Raw'
    if p.get('has_direct_value_labels') is True: return '前端分流：目标有直接数值标签，使用 Raw VLM'
    if p and (p.get('chart_type') not in {'bar', 'line'} or p.get('y_axis_count') not in {1, 'unknown', None}):
        return '前端分流：当前测量器不支持该图表结构，使用 Raw VLM'
    if p.get('operation') == 'unsupported': return '前端分流：计划未找到可支持的测量任务，使用 Raw VLM'
    if not terminal:
        return '处理中：初次执行已完成，等待看图反馈／修正' if measurement else '处理中：已有计划，等待测量'
    if reply and reply.get('plan_revised'): return '一次修正后仍缺少可用证据，回退 Raw VLM'
    if reply: return '后端未形成有效增强答案，回退 Raw VLM'
    return '执行失败后回退 Raw VLM（该旧版本没有看图修正）'

runs_root = ROOT/'phase9_evidence'
runs = [p for p in runs_root.iterdir() if p.is_dir() and (p/'full/geometry_inputs.jsonl').exists()]
active = document(runs_root/'current_run.json')
next_run = document(runs_root/'next_run.json')
runs.sort(key=lambda p: (str(p)==active.get('root'), document(p/'release_manifest.json').get('prepared_at',p.stat().st_mtime)), reverse=True)
if not runs: st.info('暂无可浏览的实验'); st.stop()
labels = {'question_router_v4':'题目级分流 v4 · 恢复遮挡后的折线片段', 'question_router_v3':'题目级分流 v3 · 证据来源与日期图例修复', 'question_router_v2':'题目级分流 v2 · 修正判断与计划提示', 'question_router_v1':'题目级分流 v1 · 独立判断＋模块化测量', 'visual_feedback_v2':'v2 · 保留失败回退规则', 'visual_feedback_v3':'v3 · 进入 Geometry 后不回填 Raw', 'visual_feedback_v4':'v4 · 修复嵌套计划读取', 'visual_feedback_v1':'视觉反馈初版', 'matched_input_v1':'输入对齐版', 'unified':'旧版'}
def run_label(p):
    state=(p/'status.txt').read_text().strip() if (p/'status.txt').exists() else ''
    q=document(p/'queue_status.json')
    suffix=('等待前序完成' if q.get('status')=='waiting_for_predecessor' else runtime_state(state))
    return labels.get(p.name,p.name)+'｜'+suffix
if active.get('root'):
    active_path=Path(active['root'])
    st.info('当前执行版本：'+run_label(active_path))
if next_run.get('root') and next_run.get('root')!=active.get('root'):
    st.caption('下一版：'+run_label(Path(next_run['root'])))
run = st.sidebar.selectbox('查看实验版本',runs,format_func=run_label,key='verified_run_version')
st.caption('本页数据目录：'+str(run))
protocol=document(run/'input_protocol.json')
no_raw_fallback=protocol.get('post_geometry_raw_fallback') is False
question_router=protocol.get('method_version','').startswith('question_router_')
state = (run/'status.txt').read_text().strip() if (run/'status.txt').exists() else ''
st.sidebar.caption('运行状态：' + runtime_state(state))
if st.sidebar.button('刷新结果'): st.cache_data.clear(); st.rerun()
# Full inputs are visible immediately. Pilot results overlay them until the cohort is merged.
names = ['geometry_inputs','plans','measurements','measurements_initial','feedback_initial',
         'revision_plans','measurements_revision','feedback_final','replies','gold','routers','outcomes']
logs = {name: {**read(run/'pilot'/(name+'.jsonl')), **read(run/'full'/(name+'.jsonl'))} for name in names}
sources, plans, measurements, replies = (logs[n] for n in ['geometry_inputs','plans','measurements','replies'])
raw_path=Path(protocol.get('raw_predictions','/data/liu_jun/finmme_reproduction/outputs/predictions/qwen25vl7b_direct_full.jsonl'))
raw = read(raw_path)
raw_metrics = read(Path(protocol.get('raw_metrics','/data/liu_jun/finmme_reproduction/outputs/metrics/qwen25vl7b_direct_full_per_sample.jsonl')))
review_notes=document(run/'router_review.json').get('cases',{})
source_audit=document(run/'display_source_audit.json')
summary = document(run/'full/summary.json')
if question_router:
    pilot_summary=document(run/'pilot/summary.json')
    if pilot_summary.get('completed',0)>summary.get('completed',0):summary=pilot_summary
def sample_state(sid):
    if not question_router:
        return route_state(plans.get(sid,{}), measurements.get(sid, logs['measurements_initial'].get(sid,{})),
                           replies.get(sid), sid in measurements)
    decision=logs['routers'].get(sid,{})
    outcome=logs['outcomes'].get(sid,{})
    if outcome.get('route')=='router_error': return '题目级判断格式无效，计错；未使用 Raw'
    if decision.get('needs_geometry') is False: return 'Router 判定不需要：使用 Raw VLM（判断可能有误）'
    if decision.get('needs_geometry') is not True: return '待运行：尚未完成题目级判断'
    if outcome:
        if outcome.get('status')=='success': return 'Router 判定需要：采用本流程最终回答'
        return 'Router 判定需要：最终答复无效，计错；不回填 Raw'
    return 'Router 判定需要：正在规划、测量或生成回答'

if question_router:
    ps=document(run/'pilot/summary.json')
    if ps:
        st.caption(f"64题开发实测：已完成 {ps.get('completed',0)}/64；Raw {ps.get('raw_correct',0)}，系统 {ps.get('hybrid_correct',0)}。开发结果不代表全量成绩。")
with st.expander('全量 FinMME 结果与本轮覆盖', expanded=False):
    st.caption('所有 11,099 题分别判断是否需要几何数值；明确不需要才使用 Raw。测量能力不足不改变分流，未完成结果仅为暂计。' if question_router else '所有 11,099 题统一计分；候选外与初始直答题沿用 Raw，进入 Geometry 后最终答复无效计错，不回填 Raw。未完成结果仅为暂计。' if no_raw_fallback else '所有 11,099 题统一计分；这是保留失败回退 Raw 的旧规则版本。未完成结果仅为暂计。')
    if summary:
        if summary.get('score_provisional') or not summary.get('full_cohort_scope',True):
            st.warning('以下全量分数含未运行题目的 Raw 暂填值，不是完成的 11,099 题实验结果。')
        st.dataframe([{'方法': r['method'], **{k: f'{r[k]*100:.2f}%' for k in ['Single','Multi','Calculation','Overall']}}
                      for r in summary.get('main_table', [])], hide_index=True)
        st.write(f"修对 {summary.get('rescued',0)} 题 · 改错 {summary.get('harmed',0)} 题 · 评分状态：{summary.get('evaluation_status','未完成')}")
    else: st.info('新版尚未形成评分结果；上一轮完整结果可在左侧切换查看。')
    counts = Counter(sample_state(s) for s in sources)
    st.dataframe([{'状态': k, '题数': v} for k,v in counts.items()], hide_index=True)
priority = [s for s in ['43','18','124','11095','7738'] if s in sources]
search = st.sidebar.text_input('搜索题号／题目')
kind = st.sidebar.selectbox('题型', ['全部','single_choice','multiple_choice','numerical'],
                          format_func=lambda k: {'single_choice':'单选','multiple_choice':'多选','numerical':'计算'}.get(k,k))
ids = priority + [s for s in sources if s not in priority]
if kind != '全部': ids = [s for s in ids if sources[s]['question_type'] == kind]
if search: ids = [s for s in ids if search.lower() in (s+' '+sources[s]['question']).lower()]
if not ids: st.info('没有匹配题目'); st.stop()
sid = st.sidebar.selectbox('选择题目', ids, format_func=lambda s: f'{"★ " if s in priority else ""}Q{s} · {sources[s]["question"][:48]}')
src = sources[sid]; reply = replies.get(sid)
terminal = measurements.get(sid, {})
m = terminal or logs['measurements_revision'].get(sid) or logs['measurements_initial'].get(sid, {})
state_text = sample_state(sid)
st.title(f'Q{sid}'); st.write(src['question']); st.info(state_text)
if sid in review_notes:
    st.warning('原图复核：'+review_notes[sid]['finding'])

if m and (m.get('status') != 'success' or reply and reply.get('status') != 'success'):
    st.caption('具体原因：' + reason_text(m, reply))
if src.get('options'): st.code(src['options'], language=None)
if sid in logs['measurements_revision']:
    view = st.radio('查看测量过程', ['修正后', '初次'], horizontal=True)
    if view == '初次': m = logs['measurements_initial'][sid]
plan = m.get('semantic_plan') or plans.get(sid,{}).get('plan',{})
left, right = st.columns([1.25,1])
with left:
    show = st.toggle('显示标定线与定位点', value=True)
    if show and m:
        # Drawing saved coordinates is CPU-only and never changes experiment logs.
        digest = hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()[:20]
        overlay = Path(tempfile.gettempdir())/'geometry_inspector_overlays'/(digest+'.png')
        if not overlay.exists(): render_measurement_overlay(src,m,overlay)
        st.image(str(overlay), use_container_width=True)
    else: st.image(src.get('vlm_image_path') or src['image_path'], use_container_width=True)
    st.caption('橙色虚框：VLM 提出的区域；彩色实线／点：程序实际测量位置；紫线：轴标定；黄框：OCR。')
with right:
    if question_router:
        st.subheader('Router 模型判断（不是已核实标签）')
        decision=logs['routers'].get(sid,{})
        if decision:
            st.write(decision.get('reason') or decision.get('status'))
            st.json({k:decision.get(k) for k in ['evidence_source','needs_geometry','status']},expanded=True)
        else: st.caption('等待独立判断；尚未调用 Planner。')
    st.subheader('VLM 看图后的测量计划')
    if plan: st.json(plan, expanded=True)
    else: st.info('Router 将本题分到 Raw，因此没有生成测量计划；不代表这一判断正确。' if question_router and logs['routers'].get(sid,{}).get('needs_geometry') is False else '尚未生成有效测量计划。')
    st.subheader('测量与 Python 计算')
    values = m.get('geometry_values', [])
    if values:
        st.dataframe([dict(target=t.get('x_label'), series=t.get('series'),
                           mark_type=t.get('mark_type'),axis=t.get('axis_id'),value=json.dumps(v,ensure_ascii=False))
                      for t,v in zip(plan.get('targets',[]),values)], hide_index=True)
        st.write('计算结果：', m.get('numeric_answer')); st.json(m.get('reasoning') or {}, expanded=False)
    else: st.caption('尚无完整测量结果；已有的部分定位仍会画在左图上。')
    st.subheader('实际模型回复与最终答案')
    if reply:
        st.code(reply.get('raw_reply',''), language=None)
        if reply.get('error'): st.caption(reply['error'])
    else: st.caption('本题已由独立判断选择 Raw 直答。' if question_router and logs['outcomes'].get(sid,{}).get('route')=='raw' else '后端尚未完成，本题还没有新版最终答案。')
    raw_answer = raw.get(sid,{}).get('prediction','缓存缺失')
    st.write('Raw VLM 原始预测：', raw_answer)
    if question_router and sid in logs['outcomes']:
        outcome=logs['outcomes'][sid]
        final_answer=raw_answer if outcome['route']=='raw' else outcome.get('prediction') if outcome.get('status')=='success' else '答复无效，计错（未回填 Raw）'
        st.write('当前系统最终答案：',final_answer)
    elif sid in measurements:
        if reply and reply.get('status')=='success': final_answer=reply['prediction']
        elif no_raw_fallback and terminal.get('geometry_entered'): final_answer='答复无效，计错（未回填 Raw）'
        else: final_answer=raw_answer
        st.write('当前系统最终答案：',final_answer)
for title, name in [('题目级判断的实际请求与回复','routers'),('初次看图请求与计划', 'plans'), ('初次定位反馈（包含实际发送的两张图）','feedback_initial'),
                    ('修正后的计划','revision_plans'), ('修正后的最终回复','feedback_final')]:
    if sid in logs[name]:
        with st.expander(title):
            record = logs[name][sid]
            if record.get('raw_reply'): st.code(record['raw_reply'],language=None)
            if record.get('overlay_path') and Path(record['overlay_path']).exists(): st.image(record['overlay_path'])
            st.json(record, expanded=False)
with st.expander('完整增强 Prompt（实际发送）'):
    if reply: st.code(reply.get('prompt',''),language=None)
    else: st.info('尚未发送；不把预览冒充实际请求。')
with st.expander('定位细节与原始失败原因'): st.json(m,expanded=False)
st.subheader('答案对照与来源（仅事后审查）')
gold_record=logs['gold'].get(sid,{})
gold_answer=gold_record.get('reference','记录缺失')
raw_record=raw.get(sid,{})
checks={
    '题号一致': str(raw_record.get('sample_id'))==sid and str(gold_record.get('sample_id'))==sid,
    '题干一致': raw_record.get('question')==src.get('question'),
    '图片记录一致': raw_record.get('image_path')==(src.get('vlm_image_path') or src.get('image_path')),
    'Gold与原始预测记录的reference一致': gold_answer==raw_record.get('reference'),
    'Raw与评分记录一致': raw_record.get('prediction')==raw_metrics.get(sid,{}).get('prediction'),
}
if not all(checks.values()):st.error('发现来源不一致，请展开核对详情；不要据此判断模型对错。')
else:st.caption('本题题号、题干、图片记录、Gold、Raw来源核对一致。Gold是数据集标注，仍可能有标注错误。')
answer_rows=[{'来源':'Gold（数据集标注）','答案':str(gold_answer)},
             {'来源':'Raw（独立基线原始预测）','答案':str(raw_record.get('prediction','缺失'))}]
if question_router and sid in logs['outcomes']:
    o=logs['outcomes'][sid]
    system_answer=raw_record.get('prediction') if o.get('route')=='raw' else o.get('prediction')
    answer_rows.append({'来源':'系统最终答案（'+('采用Raw' if o.get('route')=='raw' else '框架重新作答')+'）','答案':str(system_answer)})
st.dataframe(answer_rows,hide_index=True)
with st.expander('Gold、Raw原始回复与评分来源'):
    st.json({'sample_id':sid,'checks':checks,'gold':gold_record,
             'raw_prediction':raw_record.get('prediction'),'raw_original_response':raw_record.get('raw_response'),
             'raw_official_correct':raw_metrics.get(sid,{}).get('official_code_correct'),
             'raw_source':str(raw_path),'gold_source':str(run/'full/gold.jsonl'),
             'image':src.get('vlm_image_path') or src.get('image_path'),
             'source_audit':source_audit},expanded=False)

st.sidebar.caption(f"已完成 {len(logs['outcomes']) if question_router else len(measurements)}/{len(sources)} 题 · 计划 {len(plans)} · 后端回复 {len(replies)}")
if question_router:
    st.sidebar.caption(f"题目级判断 {len(logs['routers'])} · 需要 {sum(r.get('needs_geometry') is True for r in logs['routers'].values())} · 不需要 {sum(r.get('needs_geometry') is False for r in logs['routers'].values())}")
    st.sidebar.caption('本次先运行 64 题开发实测；全量是否启动以运行状态为准。')
if (run/'input_protocol.json').exists() and document(run/'input_protocol.json').get('method_version', '').startswith('visual_feedback'):
    st.sidebar.caption(f"初测 {len(logs['measurements_initial'])} · 请求修正 {len(logs['revision_plans'])} · 增强答案 {sum(r.get('status')=='success' for r in replies.values())}")
