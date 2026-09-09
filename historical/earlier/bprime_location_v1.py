"""Frozen post-hoc B-prime location-preservation diagnostic (B' and B'L0 only)."""
from __future__ import annotations
import argparse, collections, csv, hashlib, importlib.util, inspect, itertools, json, math, os, re, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; WORK=ROOT/'work'; OUT=ROOT/'outputs/bprime-location-v1'; REV=ROOT/'outputs/revised'; NEW=ROOT/'outputs/new-project-evaluation-v2'; SEL=ROOT/'outputs/selector-comparison-v1'; AUDIT=ROOT/'outputs/b-location-audit-v1'; ART=Path('/RESEARCH_HOME/gptrace-artifacts')
TARGETS=('freetype__char2svg','poppler__pdfimages','soxmp3__sox','libtiff__tiff2pdf','libtiff__tiffcp','libxml2__libxml2_xml_read_memory_fuzzer','libxml2__xmllint')
KIND=('trace','no_args_trace','asan'); SELECTORS=('S0','Smax')
LEAK_RE=re.compile(r'\bpoc_[A-Za-z0-9_-]+\b|\bMAGMA_[A-Z0-9_]+\b|\bPDF\d{3}\b')
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def shat(s):return hashlib.sha256(s.encode()).hexdigest()
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def jl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x]
def wj(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def wjl(p,r):Path(p).write_text(''.join(canon(x)+'\n' for x in r),encoding='utf-8')
def wc(p,r):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  z=csv.DictWriter(f,fieldnames=list(r[0]));z.writeheader();z.writerows(r)
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def inv(p):return {'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write_new(p,s):
 if Path(p).exists():raise ValueError('would_overwrite:'+str(p))
 Path(p).write_text(s,encoding='utf-8')
def records():
 a=mod('audit_bprime_source',WORK/'audit_b_location_v1.py'); rs,_=a.input_records(); return rs,a
def bprime(text,name,scan,baseline):
 # Protect source-coordinate tokens during the pre-existing generic hygiene pass.
 loc=scan(text); pieces=[]; last=0
 for i,x in enumerate(loc):
  pieces += [text[last:x['span'][0]],f'__BPRIMELOC{i}__'];last=x['span'][1]
 pieces.append(text[last:]); red=baseline.redact(''.join(pieces),name)
 for i,x in enumerate(loc):
  suffix=':'+str(x['line'])+(':'+str(x['column']) if x['column'] is not None else '')
  red=red.replace(f'__BPRIMELOC{i}__',x['basename']+suffix)
 return red,loc
def old_rows():
 d={r['report_id']:r for r in jl(REV/'encoder-inputs-v4.jsonl')};n={r['report_id']:r for r in jl(NEW/'encoder-inputs.jsonl')}; return d,n
def freeze():
 if OUT.exists() and any(OUT.iterdir()):raise ValueError('nonempty output')
 OUT.mkdir(parents=True,exist_ok=True); rs,_=records(); d,n=old_rows()
 files=[WORK/'bprime_location_v1.py',WORK/'audit_b_location_v1.py',WORK/'prepare_encoder_inputs.py',REV/'encoder-inputs-v4.jsonl',REV/'bge-control-eval-v1-vectors.npz',REV/'bge-control-eval-v1-manifest.json',NEW/'encoder-inputs.jsonl',NEW/'vectors.npz',NEW/'encoding-manifest.json',SEL/'predictions.jsonl',NEW/'predictions.jsonl',AUDIT/'validation.json']
 if json.loads((AUDIT/'validation.json').read_text())['status']!='passed':raise ValueError('b_audit_not_passed')
 ids=sorted(r['report_id'] for r in rs)
 if set(ids)!=set(d)|set(n) or len(ids)!=326:raise ValueError('frozen_id_mismatch')
 protocol={'schema_version':'bprime-location-v1','status':'post_hoc_observed_data_diagnostic_not_original_gptrace_reproduction','conditions':['B','BL0','Bprime','BprimeL0','LOC'],'new_embeddings':['Bprime base trace/no_args_trace/asan only'],'Bprime_rule':'structurally recognized report source coordinate -> basename:line[:column]; generic redaction handles all other text','source_coordinate_failure':'leave to existing generic redaction and count','basename_collision':'count only; do not switch to relative path','unchanged':['BGE revision/tokenizer/chunking/normalization/fusion L weight/candidates/S0/Smax/noise/scoring'],'cohorts':{'fallback_all':'all frozen IDs; L0 fallback policy','I_L0_common':'all IDs with stored L0 (verified equal to full set)'},'labels_before_selection':False,'implementation_sha256':sha(WORK/'bprime_location_v1.py')}
 wj(OUT/'protocol.json',protocol);wj(OUT/'input-freeze.json',{'ids':ids,'target_counts':dict(collections.Counter(r['target'] for r in rs)),'inputs':{str(x.relative_to(ROOT)):inv(x) for x in files},'raw_inventory_before':a_inventory(rs),'labels_loaded':False})
def a_inventory(rs):
 return hashlib.sha256(canon([(r['report_id'],sha(r['report']),sha(r['trace'])) for r in rs]).encode()).hexdigest()
def prepare():
 f=json.loads((OUT/'input-freeze.json').read_text()); rs,a=records(); base=mod('bprime_base',WORK/'prepare_encoder_inputs.py'); pre=mod('bprime_pre',ART/'gptrace/src/gptrace/preprocessing.py'); d,n=old_rows(); rows=[]; audit=[]; collisions=collections.defaultdict(set)
 for r in rs:
  old=(d if r['report_id'] in d else n)[r['report_id']]; args=a.make_args(argparse,r['trace'],r['report']); gp=pre.read_trace_and_asan(args,r['trace']); tx={};hs={}
  for k in KIND:
   v,loc=bprime(gp[k],r['report_name'],a.scan_source_locations,base);tx[k]=v;hs[k]=shat(v)
   for x in loc:collisions[x['basename']].add(x['path_normalized'])
   audit.append({'report_id':r['report_id'],'target':r['target'],'text_kind':k,'recognized_coordinates':len(loc),'old_hash':old['text_hashes'][k],'bprime_hash':hs[k],'changed':old['texts'][k]!=v,'leak_marker':bool(LEAK_RE.search(v))})
  rows.append({'schema_version':'bprime-input-v1','report_id':r['report_id'],'target':r['target'],'texts':tx,'text_hashes':hs,'L_hash':old['text_hashes'].get('L'),'L_text':old['texts'].get('L')})
 if any(x['leak_marker'] for x in audit):raise ValueError('hygiene_leak')
 if a_inventory(rs)!=f['raw_inventory_before']:raise ValueError('raw_changed_during_prepare')
 wjl(OUT/'bprime-base-inputs.jsonl',[{'report_id':r['report_id'],'target':r['target'],'texts':r['texts'],'text_hashes':r['text_hashes']} for r in rows]);wjl(OUT/'bprime-inputs.jsonl',rows);wc(OUT/'coordinate-transform-audit.csv',audit)
 wj(OUT/'bprime-preparation-manifest.json',{'records':len(rows),'changed_texts':sum(x['changed'] for x in audit),'recognized_coordinate_occurrences':sum(x['recognized_coordinates'] for x in audit),'unrecognized_source_like_tokens':'reported by prior B audit; no restoration attempted','basename_collision_count':sum(len(v)>1 for v in collisions.values()),'basename_collisions':{k:len(v) for k,v in collisions.items() if len(v)>1},'L_hashes_unchanged':all(r['L_hash'] for r in rows),'raw_inventory_after':a_inventory(rs)})
def call_encoder():
 os.environ['HF_HUB_OFFLINE']='1';e=mod('bprime_encoder',WORK/'encode_bge_pilot.py');e.OUTPUT=OUT; old=sys.argv;sys.argv=[str(e.__file__),'--input','bprime-base-inputs.jsonl','--vectors-output','bprime-base-vectors.npz','--manifest-output','bprime-encoding-manifest.json']
 try:e.main()
 finally:sys.argv=old
def loadnp(p):
 with np.load(p,allow_pickle=False) as z:return {str(k):v for k,v in zip(z['hashes'].tolist(),z['vectors'])}
def encode():
 call_encoder(); rows=jl(OUT/'bprime-inputs.jsonl'); newv=loadnp(OUT/'bprime-base-vectors.npz'); oldv={**loadnp(REV/'bge-control-eval-v1-vectors.npz'),**loadnp(NEW/'vectors.npz')}; need={h for r in rows for h in [*r['text_hashes'].values(),r['L_hash']] if h}; missing=need-set(newv)-set(oldv)
 if missing:raise ValueError('missing_vector:'+str(len(missing)))
 merged={h:(newv[h] if h in newv else oldv[h]) for h in sorted(need)};np.savez_compressed(OUT/'vectors.npz',hashes=np.array(sorted(merged)),vectors=np.stack([merged[x] for x in sorted(merged)]).astype(np.float32))
 em=json.loads((OUT/'bprime-encoding-manifest.json').read_text()); tokens={**json.loads((REV/'bge-control-eval-v1-manifest.json').read_text())['tokens_by_hash'],**json.loads((NEW/'encoding-manifest.json').read_text())['tokens_by_hash'],**em['tokens_by_hash']}
 wj(OUT/'encoding-manifest.json',{'base_encoding_manifest_sha256':sha(OUT/'bprime-encoding-manifest.json'),'vectors_sha256':sha(OUT/'vectors.npz'),'tokens_by_hash':{h:tokens[h] for h in need},'new_unique_texts':em['unique_texts'],'new_elapsed_seconds':em['elapsed_seconds'],'chunking':em['chunking'],'revision':em['revision'],'model':em['model']})
def candidate():
 cl=mod('bprime_cluster',WORK/'cluster_control_pilot_v1.py'); co=mod('bprime_selector',WORK/'compare_selectors_v1.py'); rows=jl(OUT/'bprime-inputs.jsonl'); vec=loadnp(OUT/'vectors.npz'); reps={}
 for r in rows:
  bs=[cl.unit(vec[r['text_hashes'][k]]) for k in KIND];b=cl.unit(np.mean(bs,axis=0));l=cl.unit(vec[r['L_hash']])
  reps[r['report_id']]={'target':r['target'],'Bprime':b,'BprimeL0':cl.unit(b+l),'tokens':r['text_hashes'],'L_hash':r['L_hash']}
 cand=[];mat=[];s0=[]; computed={}
 for target in TARGETS:
  ids=sorted(k for k,v in reps.items() if v['target']==target)
  for cond in ('Bprime','BprimeL0'):
   x=np.stack([reps[i][cond] for i in ids]).astype(np.float64);mh=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(); items,sweep=cl.enumerate_sweep(x,OUT/'hdbscan-cache'/mh); primary,_,trace=cl.select_partitions(items)
   key='fallback_all/'+target+'/'+cond
   mat.append({'key':key,'target':target,'condition':cond,'report_ids':ids,'matrix_sha256':mh,'sweep':sweep,'primary':{k:primary[k] for k in ('evaluation_index','epsilon','dbcv','persistence','reported_cluster_count','partition_sha256')},'selection_trace':trace})
   for q in items:cand.append({'key':key,'target':target,'condition':cond,'matrix_sha256':mh,**q,'primary_selected':q['evaluation_index']==primary['evaluation_index']})
   for i,z in zip(ids,primary['partition']):s0.append({'key':key,'target':target,'condition':cond,'report_id':i,'cluster':int(z)})
 wjl(OUT/'candidates.jsonl',cand);wjl(OUT/'matrices.jsonl',mat);wjl(OUT/'s0-predictions.jsonl',s0)
 # selection is label-free and freezes both selectors.
 grouped=collections.defaultdict(list)
 for x in cand:grouped[x['key']].append(x)
 pred=[];sel=[]
 for m in mat:
  for name,choice in [('S0',next(x for x in grouped[m['key']] if x['primary_selected'])),('Smax',co.choose_smax(grouped[m['key']])[0])]:
   if choice is None:raise ValueError('smax_unavailable')
   sel.append({'key':m['key'],'target':m['target'],'condition':m['condition'],'selector':name,'evaluation_index':choice['evaluation_index'],'epsilon':choice['epsilon'],'dbcv':choice['dbcv'],'persistence':choice['persistence'],'partition_sha256':choice['partition_sha256']})
   for i,z in zip(m['report_ids'],choice['partition']):pred.append({'key':m['key'],'target':m['target'],'condition':m['condition'],'selector':name,'report_id':i,'cluster':int(z)})
 wjl(OUT/'selections.jsonl',sel);wjl(OUT/'predictions.jsonl',pred);wj(OUT/'selection-freeze.json',{'labels_loaded':False,'inputs':{x:sha(OUT/x) for x in ('candidates.jsonl','matrices.jsonl','s0-predictions.jsonl')},'outputs':{x:sha(OUT/x) for x in ('selections.jsonl','predictions.jsonl')},'selector_signatures':{'S0':'cluster.choose_unsupervised','Smax':list(inspect.signature(co.choose_smax).parameters)}})
def score():
 sc=mod('bprime_score',WORK/'score_control_pilot_v1.py'); metric,_=sc.load_metric_module(ART/'gptrace',json.loads((REV/'control-protocol-v1.json').read_text())['current_development_snapshot']['gptrace']['ground_truth_analysis.py_sha256']); labels={x['report_id']:x['label'] for x in jl(REV/'evaluation-labels-v4.jsonl')};labels.update({x['report_id']:x['label'] for x in jl(NEW/'labels.jsonl')}); mats=jl(OUT/'matrices.jsonl'); pred=collections.defaultdict(dict)
 for x in jl(OUT/'predictions.jsonl'):pred[(x['key'],x['selector'])][x['report_id']]=x['cluster']
 inp={x['report_id']:x for x in jl(OUT/'bprime-inputs.jsonl')}; vec=loadnp(OUT/'vectors.npz')
 def unit(v): return v/np.linalg.norm(v)
 rows=[]
 for m in mats:
  ids=m['report_ids'];truth=[labels[i] for i in ids]
  values=[]
  for rid in ids:
   r=inp[rid]; b=unit(np.mean([unit(vec[r['text_hashes'][k]]) for k in KIND],axis=0)); values.append(b if m['condition']=='Bprime' else unit(b+unit(vec[r['L_hash']])))
  ranking=sc.ranking_metrics(np.stack(values).astype(np.float64),truth)
  for s in SELECTORS:
   a=[pred[(m['key'],s)][i] for i in ids];q=sc.primary_metrics(metric,a,truth);rows.append({'row_type':'target','target':m['target'],'condition':m['condition'],'selector':s,'reports':len(ids),'true_bugs':len(set(truth)),'predicted_clusters':len(set(a)),'F':q['f_measure'],'ARI':q['adjusted_rand_index'],'AP':ranking['pairwise_average_precision_global'],'anchor_mAP':ranking['bug_balanced_anchor_map'],'purity':q['purity'],'inverse_purity':q['inverse_purity'],'epsilon':next(x['epsilon'] for x in jl(OUT/'selections.jsonl') if x['key']==m['key'] and x['selector']==s),'dbcv':next(x['dbcv'] for x in jl(OUT/'selections.jsonl') if x['key']==m['key'] and x['selector']==s)})
 wc(OUT/'scores.csv',rows)
 wj(OUT/'score-manifest.json',{'labels_joined_after_selection_freeze':True,'labels_sha256':hashlib.sha256(canon(labels).encode()).hexdigest(),'score_rows':len(rows)})
def validate():
 f=json.loads((OUT/'input-freeze.json').read_text());rs,_=records(); checks={'raw_inventory_unchanged':a_inventory(rs)==f['raw_inventory_before'],'b_audit_passed':json.loads((AUDIT/'validation.json').read_text())['status']=='passed','selection_precedes_labels':json.loads((OUT/'selection-freeze.json').read_text())['labels_loaded'] is False,'L0_full_common':all(r['L_hash'] for r in jl(OUT/'bprime-inputs.jsonl')),'all_new_base_inputs_have_no_obvious_marker':not any(LEAK_RE.search(v) for r in jl(OUT/'bprime-inputs.jsonl') for v in r['texts'].values())}
 wj(OUT/'validation.json',{'checks':checks,'status':'passed' if all(checks.values()) else 'failed'}); write_new(OUT/'HANDOFF.md','# Handoff\n\nB′ location-preservation post-hoc diagnostic completed; see REPORT.md and validation.json.\n');write_new(OUT/'FAILURE_LOG.md','# Failure log\n\nNo execution failure. This is observed-data post-hoc analysis, not an independent evaluation.\n')
 s=list(csv.DictReader((OUT/'scores.csv').open(encoding='utf-8-sig'))); lines=['# B′ 위치 보존 진단','', 'B′는 GPTrace 출력에서 구조적으로 인식된 보고서 source coordinate만 `basename:line[:column]`으로 보존한 변형이다. B′는 원 GPTrace 재현이 아니며, 326건 모두 이미 관측한 자료다.','','| target | condition | selector | n | F | ARI | clusters |','|---|---|---|---:|---:|---:|---:|']
 for r in s:lines.append(f"| {r['target']} | {r['condition']} | {r['selector']} | {r['reports']} | {float(r['F']):.5f} | {float(r['ARI']):.5f} | {r['predicted_clusters']} |")
 lines+=['','B/BL0/LOC의 동결 결과와 직접 대비하는 재집계는 review-followup-v1/FINAL_REPORT.md에서 target·selector별로 병기한다. B′L0−B′가 L0의 위치 정보 외 추가 효과인지 여부는 이 비교만으로 일반화하지 않는다.']
 write_new(OUT/'REPORT.md','\n'.join(lines)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('step',choices=('freeze','prepare','encode','candidate','score','validate'));x=p.parse_args();{'freeze':freeze,'prepare':prepare,'encode':encode,'candidate':candidate,'score':score,'validate':validate}[x.step]()
if __name__=='__main__':main()
