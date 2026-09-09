"""Portable saved-vector reproduction. No model, network, or crash execution."""
from __future__ import annotations
import argparse,ast,collections,csv,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from numeric import unit
from frozen_selectors import choose_s0,choose_smax

def readj(p):return json.loads(p.read_text(encoding='utf8'))
def readjl(p):
 opener=gzip.open if p.suffix=='.gz' else open
 with opener(p,'rt',encoding='utf-8-sig') as f:return [json.loads(l) for l in f if l.strip()]
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ap(y,s):
 y=np.asarray(y);s=np.asarray(s);order=np.argsort(-s,kind='stable');y=y[order];s=s[order]
 ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(y)[ends]
 return float(np.sum(np.diff(np.r_[0,tp])/sum(y)*(tp/(ends+1))))
def measure(y,p):
 bs=collections.Counter(y);cs=collections.Counter(p);joint=collections.Counter(zip(y,p));n=len(y)
 f=sum(nb/n*max(2*joint[b,c]/(nb+nc) for c,nc in cs.items()) for b,nb in bs.items())
 lost=[b for b in bs if all(any(b2!=b and joint[b2,c]>0 for b2 in bs) for c in cs if joint[b,c]>0)]
 comb=lambda k:k*(k-1)/2
 a=sum(comb(v) for v in joint.values());b=sum(comb(v) for v in bs.values());c=sum(comb(v) for v in cs.values());expected=b*c/comb(n);den=(b+c)/2-expected
 ari=(a-expected)/den if den else 1.0
 return f,ari,lost
def canonical(ids,partition):
 groups=collections.defaultdict(list)
 for rid,label in zip(ids,partition):groups[label].append(rid)
 return sorted(sorted(g) for g in groups.values())

def error_counts(y,p):
 clusters=collections.defaultdict(set);bugs=collections.defaultdict(set)
 for bug,cluster in zip(y,p):clusters[cluster].add(bug);bugs[bug].add(cluster)
 return sum(len(v)>1 for v in bugs.values()),sum(len(v)>1 for v in clusters.values())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('recomputed'));args=parser.parse_args()
 out=args.output.resolve()
 if out==ROOT or ROOT/'data'==out or ROOT/'data' in out.parents:raise ValueError('Output must not overwrite the packaged data')
 out.mkdir(parents=True,exist_ok=True)
 def dump(n,v):(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 def table(n,rows):
  with (out/n).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 checks=[]
 def check(name,ok):
  if not ok:raise AssertionError(name)
  checks.append(name)
 manifest=readj(ROOT/'SHA256.json')
 for path,digest in manifest.items():check('sha256:'+path,sha(ROOT/path)==digest)
 data=ROOT/'data';reports={r['report_id']:r for r in readj(data/'reports.json')};inputs=readj(data/'components.json')
 with np.load(data/'bge-vectors.npz',allow_pickle=False) as z:vec=dict(zip(z['hashes'].tolist(),z['vectors']))
 fused={}
 for r in inputs:
  c=r['component_text_sha256'];a,b,d,l,e=(unit(vec[c[k]]) for k in ('trace','no_args_trace','asan','L','D4'))
  base=unit(np.mean([a,b,d],axis=0));fused[r['condition'],r['report_id']]={'B':base,'BL':unit(base+l),'BD4':unit(base+e),'BLD4':unit(base+l+e)}
 matrices=readjl(data/'matrices.jsonl.gz');scores=readcsv(data/'scores.csv');aps={};distances={};ms={m['key']:m for m in matrices}
 for m in matrices:
  X=np.stack([fused[m['condition'],rid][m['representation']] for rid in m['report_ids']]);check('matrix:'+m['key'],hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()==m['matrix_sha256'])
  sq=np.sum(X*X,axis=1);dist=np.sqrt(np.maximum(-2*(X@X.T)+sq[:,None]+sq[None,:],0));np.fill_diagonal(dist,0)
  ix,jx=np.triu_indices(len(X),1);y=np.array([reports[m['report_ids'][i]]['label']==reports[m['report_ids'][j]]['label'] for i,j in zip(ix,jx)])
  aps[m['key']]=ap(y,-dist[ix,jx]);distances[m['key']]=dist
  expected=next(r for r in scores if all(r[k]==m[k] for k in ('condition','target','representation')))
  check('AP:'+m['key'],abs(aps[m['key']]-float(expected['pairwise_average_precision']))<1e-10)
 cand=collections.defaultdict(list)
 for c in readjl(data/'candidates.jsonl.gz'):cand[c['key']].append(c)
 selections=readjl(data/'selections.jsonl.gz');smap={(s['key'],s['selector']):s for s in selections};recomputed=[]
 for r in scores:
  key='/'.join(r[k] for k in ('condition','target','representation'));s=smap[key,r['selector']]
  got,_=(choose_s0 if r['selector']=='S0' else choose_smax)(cand[key]);check('selector:'+key+'/'+r['selector'],got['evaluation_index']==s['evaluation_index'] and got['partition_sha256']==s['partition_sha256'])
  f,ari,lost=measure([reports[rid]['label'] for rid in s['report_ids']],s['partition'])
  over,under=error_counts([reports[rid]['label'] for rid in s['report_ids']],s['partition'])
  check('over_under:'+key+'/'+r['selector'],over==int(r['num_overcount']) and under==int(r['num_undercount']))
  check('F:'+key+'/'+r['selector'],abs(f-float(r['f_measure']))<5.1e-6);check('ARI:'+key+'/'+r['selector'],abs(ari-float(r['adjusted_rand_index']))<1e-10);check('lost:'+key+'/'+r['selector'],len(lost)==int(r['num_completely_lost']))
  recomputed.append({**r,'recomputed_AP':aps[key],'recomputed_F':f,'recomputed_ARI':ari,'lost_labels':','.join(sorted(lost))})
 table('all-primary-and-L-sweep-scores.csv',recomputed)
 # Grid scores are stored rounded to five decimals, matching the published evaluator.
 gs=readcsv(data/'grid-scores.csv');gsel=readjl(data/'grid-selections.jsonl.gz');gm={(s['key'],str(s['min_cluster_size']),str(s['min_samples'])):s for s in gsel}
 for r in gs:
  key='/'.join(r[k] for k in ('condition','target','representation'));s=gm[key,r['min_cluster_size'],r['min_samples']]
  f,ari,lost=measure([reports[i]['label'] for i in s['report_ids']],s['partition'])
  over,under=error_counts([reports[i]['label'] for i in s['report_ids']],s['partition'])
  check('grid_over_under:'+key+'/'+r['min_cluster_size']+'/'+r['min_samples'],over==int(r['num_overcount']) and under==int(r['num_undercount']))
  check('grid:'+key+'/'+r['min_cluster_size']+'/'+r['min_samples'],abs(f-float(r['f_measure']))<5.1e-6 and abs(ari-float(r['adjusted_rand_index']))<1e-10 and len(lost)==int(r['num_completely_lost']))
 target_names=sorted({r['target'] for r in reports.values()});dataset=[]
 for t in target_names:
  rr=[r for r in reports.values() if r['target']==t];counts=collections.Counter(r['label'] for r in rr);pos=sum(n*(n-1)//2 for n in counts.values());dataset.append({'target':t,'reports':len(rr),'bugs':len(counts),'positive_pairs':pos,'negative_pairs':len(rr)*(len(rr)-1)//2-pos})
 table('table-1.csv',dataset);dump('table-2.json',{'encoding':readj(data/'encoding-manifest.json'),'full_runtime':'requirements-full.txt','hdbscan':{'min_cluster_size':2,'min_samples':1,'metric':'euclidean','allow_single_cluster':True,'cluster_selection_method':'eom','STEPS':100}})
 primary=[r for r in recomputed if r['condition']=='L_f3_W5'];macro=[]
 for condition in sorted({r['condition'] for r in recomputed}):
  row={'condition':condition}
  for rep in ('B','BL','BD4','BLD4'):row[rep]=float(np.mean([r['recomputed_AP'] for r in recomputed if r['condition']==condition and r['representation']==rep and r['selector']=='S0']))
  macro.append(row)
 table('table-3.csv',macro)
 table('table-4.csv',[{'target':t,**{rep:next(r['recomputed_AP'] for r in primary if r['target']==t and r['representation']==rep) for rep in ('B','BL','BD4','BLD4')}} for t in target_names])
 table('table-5.csv',[{'representation':rep,**{sel+'_'+metric:float(np.mean([float(r[col]) for r in primary if r['representation']==rep and r['selector']==sel])) for sel in ('S0','Smax') for metric,col in (('F','f_measure'),('ARI','adjusted_rand_index'))}} for rep in ('B','BL','BD4','BLD4')])
 table5a=[]
 for t in target_names:
  for sel in ('S0','Smax'):
   v={rep:next(float(r['f_measure']) for r in primary if r['target']==t and r['selector']==sel and r['representation']==rep) for rep in ('B','BL','BLD4')};table5a.append({'target':t,'selector':sel,**v,'BL_minus_B':v['BL']-v['B'],'BLD_minus_B':v['BLD4']-v['B']})
 table('table-5a.csv',table5a);table('table-5b.csv',[r for r in primary if r['target']=='soxmp3__sox'])
 for t in target_names:
  a=smap['L_f3_W5/'+t+'/BL','S0'];b=smap['L_f3_W5/'+t+'/BLD4','S0'];check('BL_BLD_same_S0:'+t,canonical(a['report_ids'],a['partition'])==canonical(b['report_ids'],b['partition']))
 depth=readj(data/'depth-selection.json');table('table-6.csv',[{'depth':d,**v} for d,v in depth['depth_results'].items()])
 grid=[r for r in readcsv(data/'grid-score-summary.csv') if r['condition']=='L_f3_W5'];table('table-7.csv',grid)
 # Recheck all 320 published grid macro rows from the stored per-target metric precision.
 for row in readcsv(data/'grid-score-summary.csv'):
  matches=[r for r in gs if all(r[k]==row[k] for k in ('condition','representation','min_cluster_size','min_samples'))]
  check('grid_macro:'+str(tuple(row[k] for k in ('condition','representation','min_cluster_size','min_samples'))),len(matches)==3 and abs(np.mean([float(r['f_measure']) for r in matches])-float(row['macro_f_measure']))<1e-10)
 groups=collections.defaultdict(dict)
 for r in grid:groups[r['min_cluster_size'],r['min_samples']][r['representation']]=float(r['macro_f_measure'])
 def direction(a,b):return 'increase' if a>b+1e-9 else 'decrease' if a<b-1e-9 else 'tie'
 grid_counts={comp:dict(collections.Counter(direction(v['BLD4'],v[rep]) for v in groups.values())) for comp,rep in (('BLD_vs_BL','BL'),('BLD_vs_B','B'))};dump('grid-contrasts.json',grid_counts)
 baselines=[]
 for target in target_names:
  rr=[r for r in reports.values() if r['target']==target];y=[r['label'] for r in rr]
  for kind in ('source_key','endpoint_status'):
   pred=[json.dumps(r[kind],sort_keys=True) for r in rr];f,ari,lost=measure(y,pred);truth=[];score=[]
   for i in range(len(rr)):
    for j in range(i+1,len(rr)):truth.append(y[i]==y[j]);score.append(int(pred[i]==pred[j]))
   baselines.append({'target':target,'baseline':'source_key' if kind=='source_key' else 'status','clusters':len(set(pred)),'F':f,'AP':ap(truth,score),'lost_bugs':len(lost)})
 for r in baselines:
  expected=next(x for x in readcsv(data/'expected-deterministic-baselines.csv') if x['target']==r['target'] and x['baseline']==r['baseline']);check('baseline:'+r['target']+'/'+r['baseline'],all(abs(r[k]-float(expected[k]))<1e-12 for k in ('F','AP','lost_bugs','clusters')))
 table('table-8.csv',baselines)
 key='L_f3_W5/freetype__char2svg/B';ids=ms[key]['report_ids'];dist=distances[key];pairs=[]
 for i,a in enumerate(ids):
  for j in range(i+1,len(ids)):
   b=ids[j];pairs.append({'a':a,'b':b,'positive':reports[a]['label']==reports[b]['label'],'distance':float(dist[i,j]),'status_different':reports[a]['endpoint_status']!=reports[b]['endpoint_status']})
 positives=[r for r in pairs if r['positive']];negatives=[r for r in pairs if not r['positive']];comparison=collections.Counter();loss=0
 for p in positives:
  bad=[n for n in negatives if n['distance']<=p['distance']];p['opposite_violations']=len(bad);loss+=len(bad)/sum(r['distance']<=p['distance'] for r in pairs)/len(positives)
  for n in bad:
   preference=int(not p['status_different'])-int(not n['status_different']);comparison['favorable' if preference>0 else 'adverse' if preference<0 else 'neutral']+=1;comparison['ties' if n['distance']==p['distance'] else 'strict']+=1
 for n in negatives:n['opposite_violations']=sum(n['distance']<=p['distance'] for p in positives)
 table9=[]
 for name,rr in (('positive_all',positives),('positive_affected',[r for r in positives if r['opposite_violations']]),('negative_all',negatives),('negative_affected',[r for r in negatives if r['opposite_violations']])):
  table9.append({'set':name,'pairs':len(rr),'different_status':sum(r['status_different'] for r in rr),'different_status_rate':sum(r['status_different'] for r in rr)/len(rr)})
 expected=readj(data/'expected-v6-diagnostics.json')
 for r in table9:check('rank_status:'+r['set'],all(r[k]==expected[r['set']][k] for k in ('pairs','different_status','different_status_rate')))
 check('ranking_comparisons',dict(comparison)==expected['rank_comparisons']);check('AP_loss',abs(loss-expected['AP_loss'])<1e-12)
 table('table-9.csv',table9);table('freetype-pairs.csv',pairs);dump('freetype-ranking.json',{'AP':1-loss,'AP_loss':loss,'rank_comparisons':dict(comparison)})
 table('appendix-A-recorded-controls.csv',readcsv(data/'structural-controls.csv'))
 result={'status':'passed','checks':len(checks),'matrices':len(matrices),'selected_partitions':len(selections),'grid_partitions':len(gs),'tables':'1-9, 5a, 5b; Appendix A recorded results','fresh_encoder_or_HDBSCAN_run':False,'checks_passed':checks}
 dump('validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks_passed'},indent=2))
if __name__=='__main__':main()
