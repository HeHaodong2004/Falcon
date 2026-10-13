import json,pathlib,subprocess,time,threading,xml.etree.ElementTree as ET
from collections import deque
class Monitor:
 def __init__(self,dest,pids):
  self.dest,self.pids=dest,pids;self.done=threading.Event();self.samples=deque(maxlen=2);self.thread=threading.Thread(target=self.run,daemon=True)
 def start(self):self.thread.start()
 def stop(self):self.done.set();self.thread.join(5)
 def run(self):
  with self.dest.open('w') as f:
   while not self.done.is_set():
    row=dict(time=time.time(),stat=pathlib.Path('/proc/stat').read_text(),loadavg=pathlib.Path('/proc/loadavg').read_text(),meminfo=pathlib.Path('/proc/meminfo').read_text(),pressure={n:(pathlib.Path('/proc/pressure')/n).read_text() for n in ['cpu','memory','io']},workers=[])
    for pid in list(self.pids):
     try:
      p=pathlib.Path('/proc')/str(pid)
      row['workers'].append(dict(pid=pid,stat=(p/'stat').read_text(),status=(p/'status').read_text(),io=(p/'io').read_text(),smaps_rollup=(p/'smaps_rollup').read_text()))
     except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    try:
     tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,stderr=subprocess.DEVNULL))
     row['gpus']=[dict(id=g.get('id'),utilization={x.tag:x.text for x in g.find('utilization')},memory={x.tag:x.text for x in g.find('fb_memory_usage')},processes=[{x.tag:x.text for x in pr} for pr in g.findall('.//process_info')]) for g in tree.findall('gpu')]
    except Exception as e:row['gpu_error']=str(e)
    self.samples.append(row);f.write(json.dumps(row)+'\n');f.flush();self.done.wait(2)
