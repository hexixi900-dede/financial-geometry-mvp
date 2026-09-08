"""Read-only chart/evidence browser. Opening a chart never triggers inference."""
from pathlib import Path
import sys,json
from PIL import Image,ImageDraw
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_evidence_answerer import build_prompt
st.set_page_config(page_title='Geometry QA',layout='wide')

@st.cache_data
def load(path,mtime):
    if not path.exists():return {}
    out={}
    for line in path.read_text().splitlines():
        try:
            r=json.loads(line);out[str(r['sample_id'])]=r
        except (ValueError,KeyError):continue
    return out

def read(path):return load(path,path.stat().st_mtime_ns if path.exists() else 0)
base=ROOT/'phase8_hybrid';new=ROOT/'phase9_evidence'
run_name=st.sidebar.selectbox('运行版本',['统一流程（单选／多选／计算）','上一版'])
unified=ROOT/'phase9_evidence'/'unified'
active=unified/'full' if (unified/'full'/'measurements.jsonl').exists() else unified/'pilot'
sources=read(base/'geometry_inputs.jsonl');plans=read(base/'plans_full.jsonl')
old=read(base/'predictions_final.jsonl');measurements=read(new/'measurements.jsonl');replies=read(new/'replies.jsonl')
gold=read(base/'gold.jsonl')
if run_name.startswith('统一') and (active/'geometry_inputs.jsonl').exists():
    sources=read(active/'geometry_inputs.jsonl');plans=read(active/'plans.jsonl')
    measurements=read(active/'measurements.jsonl');replies=read(active/'replies.jsonl');gold=read(active/'gold.jsonl')
raw=read(Path('/data/liu_jun/finmme_reproduction/outputs/metrics/qwen25vl7b_direct_full_per_sample.jsonl'))
status_file=unified/'status.txt'
if run_name.startswith('统一') and status_file.exists():
    state=status_file.read_text().strip()
    st.sidebar.caption('运行状态：'+('等待 GPU 空闲' if state.startswith('waiting_for_gpu') else state))
if st.sidebar.button('刷新结果'):st.cache_data.clear();st.rerun()
priority=[s for s in ['43','18','124','11095','7738'] if s in sources]
search=st.sidebar.text_input('搜索题号／题目')
ids=priority+[s for s in sources if s not in priority]
kind=st.sidebar.selectbox('题型',['全部','multiple_choice','numerical','single_choice'])
if kind!='全部':ids=[s for s in ids if sources[s]['question_type']==kind]
if search:ids=[s for s in ids if search.lower() in (s+' '+sources[s]['question']).lower()]
if not ids:st.info('没有匹配题目');st.stop()
sid=st.sidebar.selectbox('选择题目',ids,format_func=lambda s:f'{"★ " if s in priority else ""}Q{s} · {sources[s]["question"][:48]}')
src=sources[sid];m=measurements.get(sid,{} if run_name.startswith('统一') else old.get(sid,{}));reply=replies.get(sid)
st.title(f'Q{sid}')
st.write(src['question']);st.caption('当前测量：'+('新版' if sid in measurements else '尚未执行' if run_name.startswith('统一') else '旧缓存，新版待运行')+' · '+m.get('status','待运行')+' · '+src['question_type'])
if src.get('options'):st.code(src['options'],language=None)
left,right=st.columns([1.25,1])
with left:
    im=Image.open(src['image_path']).convert('RGB');overlay=im.copy();draw=ImageDraw.Draw(overlay)
    pipes=m.get('pipeline_audit',[])
    if m.get('bar_audit'):pipes=[m['bar_audit']]
    colors=['#ef4444','#2563eb','#16a34a']
    for i,p in enumerate(pipes):
        color=colors[i%len(colors)];axis=p.get('axis',{});bbox=axis.get('plot_bbox')
        if bbox:draw.rectangle(bbox,outline='#666666',width=2)
        for tick in axis.get('ticks',[]):
            y=tick['pixel_y'];x=tick['bbox'][2]
            draw.line((x,y,bbox[2] if bbox else im.width,y),fill='#ba83bf',width=1)
        grounding=p.get('grounding',{});x=grounding.get('predicted_target_x')
        if x is not None:draw.line((x,0,x,im.height),fill=color,width=2)
        for a in p.get('x_axis_anchors',[]):
            if a.get('bbox'):draw.rectangle(a['bbox'],outline='#e1a328',width=1)
        geo=p.get('auto_local_geometry',{})
        for fit in geo.get('fitted_segments',[]):
            x1,x2=fit['x_min'],fit['x_max'];tx=geo['target_x']
            y1=fit['target_y']+fit['slope']*(x1-tx);y2=fit['target_y']+fit['slope']*(x2-tx)
            draw.line((x1,y1,x2,y2),fill='#16a34a',width=3)
        if len(geo.get('trace_points',[]))>1:draw.line([tuple(p) for p in geo['trace_points']],fill=color,width=2)
        if geo.get('point'):
            x,y=geo['point'];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3)
    for i,t in enumerate(m.get('target_audit',[])):
        if t.get('bbox'):
            x,y,w,h=t['bbox'];draw.rectangle((x,y,x+w,y+h),outline=colors[i%3],width=3)
    plan=m.get('semantic_plan',plans.get(sid,{}).get('plan',{}))
    for i,t in enumerate(plan.get('targets',[])):
        if t.get('region'):
            l,top,r,b=t['region'];box=(l*im.width/1000,top*im.height/1000,r*im.width/1000,b*im.height/1000)
            draw.rectangle(box,outline='#e67e22',width=2);draw.text((box[0],box[1]),f'T{i}',fill='#e67e22')
    show=st.toggle('显示标定线与定位点',value=True)
    st.image(overlay if show else im,use_container_width=True)
    st.caption('紫色：y 轴刻度标定线；黄色：x 轴 OCR；红／蓝：目标；绿色：局部拟合')
with right:
    st.subheader('前端语义计划')
    st.caption('来源：Qwen2.5-VL 看原图后生成的计划缓存；包含题目与全部选项；橙框是 VLM 提出的测量区域。')
    st.json(m.get('semantic_plan',plans.get(sid,{}).get('plan',{})),expanded=True)
    st.subheader('测量与 Python 结果')
    targets=m.get('semantic_plan',{}).get('targets',[])
    st.dataframe([dict(target=t.get('x_label'),series=t.get('series'),value=json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else str(v)) for t,v in zip(targets,m.get('geometry_values',[]))],hide_index=True)
    st.write('计算结果：',m.get('numeric_answer','无'));st.write('公式：',m.get('reasoning',{}).get('formula','无'))
    if m.get('reasoning',{}).get('calculations'):st.json(m['reasoning']['calculations'])
    st.subheader('实际模型回复')
    if reply:
        st.code(reply.get('raw_reply',''),language=None);st.write('最终答案：',reply.get('prediction'));st.write('状态：',reply.get('status'))
    elif m and m.get('status')!='success':st.info('本题保留 Raw VLM 回答：'+m.get('status_reason',m.get('status','测量未完成')))
    else:st.info('后端 VLM 尚未执行，暂时没有新版答案。')
    raw_answer=raw.get(sid,{}).get('prediction','缓存缺失')
    st.write('Raw VLM：',raw_answer)
    if reply and reply.get('status')=='success':st.write('当前系统答案（增强）：',reply.get('prediction'))
    elif reply or m and m.get('status')!='success':st.write('当前系统答案（Raw 回退）：',raw_answer)
    with st.expander('历史 Geometry/Python 答案，仅供对照'):
        st.write(old.get(sid,{}).get('prediction','无'))
with st.expander('前端看图请求与原始回复'):
    st.json(plans.get(sid,{}),expanded=False)
with st.expander('完整增强 Prompt',expanded=False):
    if reply:st.caption('以下为实际发送内容');st.code(reply['prompt'],language=None)
    elif m.get('status')=='success':st.caption('预览，尚未发送');st.code(build_prompt(src,m),language=None)
    else:st.info('尚无可用测量值，不能拼接增强 Prompt。')
with st.expander('定位细节与失败原因'):st.json(m,expanded=False)
with st.expander('Gold 与评分信息（仅事后查看，不进入模型输入）'):st.json(gold.get(sid,{}))
st.sidebar.caption(f'新版测量 {len(measurements)}/{len(sources)} · 后端回复 {len(replies)}')
