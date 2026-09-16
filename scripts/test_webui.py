import urllib.request,urllib.error,json,time,os
from pathlib import Path
op=urllib.request.build_opener(urllib.request.ProxyHandler({}));base=os.environ.get('KVMEM_UI_TEST_URL','http://127.0.0.1:18400').rstrip('/')
for i in range(120):
 try:op.open(base+'/health',timeout=1).close();break
 except OSError:time.sleep(1)
else:raise RuntimeError('startup timeout')
props=json.load(op.open(base+'/props'));assert isinstance(props['modalities']['vision'],bool);limit=props['kvmem']['generation_limit'];assert limit>=128
payload={'messages':[{'role':'user','content':'What is 2+3? Answer with the number only.'}],'temperature':0,'reasoning_budget_tokens':0,'max_tokens':64}
def post(body):
 req=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 try:
  with op.open(req,timeout=90) as r:return r.status,r.read().decode()
 except urllib.error.HTTPError as e:return e.code,e.read().decode()
rows=[]
for value in [0,-2,limit+1,1.5,None,18446744073709551615]:
 code,body=post(payload|{'max_tokens':value});assert code==400,(value,code,body);rows.append({'invalid_max_tokens':value,'status':code})
for override in [{'max_tokens':64},{'max_tokens':-1},{'stream':True}]:
 code,raw=post(payload|override);assert code==200,raw
 if override.get('stream'):
  events=[json.loads(l[6:]) for l in raw.splitlines() if l.startswith('data: ') and l!='data: [DONE]'];assert 'data: [DONE]' in raw
  assert ''.join(e.get('choices',[{}])[0].get('delta',{}).get('content','') for e in events if e.get('choices'))=='5',raw
  result=next(e for e in events if 'usage' in e)
 else:
  result=json.loads(raw);assert result['choices'][0]['message']['content'].strip()=='5',result
 assert result['timings']['predicted_n']==result['usage']['completion_tokens'];assert result['timings']['predicted_ms']>0
 rows.append({'override':override,'result':result})
assert json.load(op.open(base+'/v1/models'))['data'][0]['status']['value']=='loaded'
for path in ['/slots','/tools','/v1/streams/lookup','/missing-api']:
 try:op.open(base+path);raise AssertionError(path)
 except urllib.error.HTTPError as e:assert e.code==404
Path('logs/webui').mkdir(parents=True,exist_ok=True)
Path('logs/webui/api-results.json').write_text(json.dumps({'props':props,'tests':rows},indent=2)+'\n');print('API checks passed')
