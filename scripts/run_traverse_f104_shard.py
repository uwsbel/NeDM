#!/usr/bin/env python3
"""Weighted static F104 shards: no shared lock, claim state, or quota writer.

Task ownership is index modulo sum(shard_weights) within each shard's cumulative
weight interval. One process owns a shard; its threads share an in-memory state.
The external monitor creates STOP_CLAIMS at the validated global quota. Existing
claims finish normally. Source/physics/episode verification reuse frozen v1.
"""
from __future__ import annotations
import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback


def load_base():
    path=Path(__file__).with_name('run_traverse_f104_queue.py')
    if not path.is_file():
        # The control entrypoint may live outside the immutable native source;
        # its wrapper supplies that source's scripts directory through PYTHONPATH.
        return importlib.import_module('run_traverse_f104_queue')
    spec=importlib.util.spec_from_file_location('f104_queue_frozen_base',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


BASE=load_base()


def ownership(contract,index):
    count=contract['shard_count']
    BASE.require(isinstance(count,int) and count>0,'Positive integer shard_count required')
    weights=contract.get('shard_weights',[1]*count)
    BASE.require(len(weights)==count and all(isinstance(w,int) and not isinstance(w,bool) and w>0 for w in weights),'Require one positive integer weight per shard')
    edges=[];total=0
    for w in weights:total+=w;edges.append(total)
    return bisect.bisect_right(edges,index%total)


class Shard:
    def __init__(self,c,tasks,index,workers):
        BASE.require(0<=index<c['shard_count'],'Shard index out of range')
        self.c,self.tasks,self.index=c,tasks,index
        self.root=Path(c['campaign_root']);self.directory=self.root/'shards'/f'shard_{index:05d}'
        self.indices=[i for i in range(len(tasks)) if ownership(c,i)==index]
        self.lock=threading.Lock();self.stop=threading.Event()
        self.state={'schema':'f104_static_shard_state_v2','contract_sha256':c['_contract_sha256'],
            'task_sha256':c['task_sha256'],'shard_index':index,'shard_count':c['shard_count'],
            'shard_weights':c.get('shard_weights',[1]*c['shard_count']),
            'ownership':'Task index modulo sum(weights), cumulative half-open interval',
            'assigned_task_count':len(self.indices),'next_local_index':0,'claimed_count':0,
            'completed_count':0,'completed_frames':0,'completed_seconds':0.,'completed_by_split':{},
            'failures':0,'consecutive_errors':0,'inflight':{},'workers':workers,
            'node':socket.gethostname(),'job_id':os.environ.get('SLURM_JOB_ID'),
            'pid':os.getpid(),'status':'binding','created_unix_s':time.time()}

    def initialize(self):
        self.directory.parent.mkdir(parents=True,exist_ok=True)
        # This is only a duplicate-invocation guard. Disjoint ownership is pure
        # arithmetic and never depends on a cross-node lock or shared counter.
        self.directory.mkdir(exist_ok=False)
        (self.directory/'completed_ledger.jsonl').touch(exist_ok=False)
        self.save()

    def save(self):
        self.state['updated_unix_s']=time.time()
        BASE.atomic_json(self.directory/'state.json',self.state)

    def claim(self):
        with self.lock:
            if self.stop.is_set() or (self.root/'STOP_CLAIMS').exists():
                self.state['stop_reason']='STOP_CLAIMS observed' if (self.root/'STOP_CLAIMS').exists() else self.state.get('stop_reason','Local fatal error')
                return None
            cursor=self.state['next_local_index']
            if cursor>=len(self.indices):
                self.state['stop_reason']='Assigned static shard exhausted';return None
            index=self.indices[cursor];task=self.tasks[index]
            self.state['next_local_index']+=1;self.state['claimed_count']+=1
            self.state['inflight'][task['id']]={'index':index,'claimed_unix_s':time.time()}
            self.save();return index,task

    def complete(self,index,task,result):
        with self.lock:
            BASE.require(self.state['inflight'][task['id']]['index']==index,'Completion does not own local claim')
            entry={**result,'sequence':self.state['completed_count']+1,'id':task['id'],'index':index,
                'group_id':task['group_id'],'split':task['split'],'shard_index':self.index,
                'completed_unix_s':time.time(),'contract_sha256':self.c['_contract_sha256']}
            BASE.append(self.directory/'completed_ledger.jsonl',entry)
            del self.state['inflight'][task['id']]
            self.state['completed_count']+=1;self.state['completed_frames']+=result['validated_frames']
            self.state['completed_seconds']=self.state['completed_frames']*.05
            split=self.state['completed_by_split'].setdefault(task['split'],{'count':0,'frames':0,'seconds':0.})
            split['count']+=1;split['frames']+=result['validated_frames'];split['seconds']=split['frames']*.05
            self.state['consecutive_errors']=0;self.save()

    def fail(self,index,task,error):
        with self.lock:
            BASE.append(self.directory/'failures.jsonl',{'id':task['id'],'index':index,'error':error,
                'failed_unix_s':time.time(),'counted_frames':0,'shard_index':self.index})
            del self.state['inflight'][task['id']]
            self.state['failures']+=1;self.state['consecutive_errors']+=1
            if self.state['consecutive_errors']>=5:
                self.state['stop_reason']='Five consecutive implementation/process/validation errors';self.stop.set()
            self.save()


def run(c,tasks,index,workers):
    BASE.require(workers>0,'Positive worker count required')
    BASE.require(c.get('worker_sha256')==BASE.sha(BASE.__file__),'Frozen base worker SHA missing/mismatched')
    BASE.checked(__file__,c['shard_worker_sha256'])
    shard=Shard(c,tasks,index,workers);shard.initialize()
    try:
        env,verify,runtime=BASE.bind_node(c,shard.directory)
        shard.state['status']='running';shard.save()
        attempts=int(c.get('max_attempts',1));BASE.require(attempts>=1,'max_attempts must be positive')
        def worker():
            while not shard.stop.is_set():
                try:claim=shard.claim()
                except BaseException:shard.stop.set();raise
                if claim is None:return
                task_index,task=claim;error=None
                for attempt in range(attempts):
                    out=shard.root/'runs'/(task['id'] if attempt==0 else task['id']+f'__retry_{attempt:03d}')
                    committing=False
                    try:
                        BASE.require(not out.exists(),f'Existing output retained; refusing reuse: {out}');out.mkdir(parents=True)
                        for key in ('case','route'):BASE.checked(task[key],task[key+'_sha256'])
                        cmd=[sys.executable,'-P','-u',c['collector_path'],'--source-root',c['source_root'],
                            '--source-manifest-sha256',c['source_manifest_sha256'],'--case',task['case'],
                            '--case-sha256',task['case_sha256'],'--route',task['route'],'--route-sha256',task['route_sha256'],
                            '--out',str(out),'--chrono-data',c['chrono_data'],'--horizon-s',str(c.get('horizon',c.get('horizon_s',120.)))]
                        BASE.atomic_json(out/'queue_launch.json',{'task':task,'index':task_index,'attempt':attempt,
                            'shard_index':index,'argv':cmd,'contract_sha256':c['_contract_sha256'],'launched_unix_s':time.time()})
                        with (out/'collector.log').open('w') as log:
                            process=subprocess.run(cmd,env=env,cwd=c['source_root'],stdout=log,stderr=subprocess.STDOUT,
                                timeout=float(c.get('episode_wall_timeout_s',1800.)),check=False)
                        BASE.require(process.returncode==0,f'Collector exited {process.returncode}: {out}/collector.log')
                        result=BASE.verify_episode(out,task,c,verify,runtime)
                        committing=True;shard.complete(task_index,task,result);error=None;break
                    except Exception as exc:
                        if committing:shard.stop.set();raise
                        error=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))
                        BASE.atomic_json(shard.directory/f'error_{task_index:06d}_{attempt:03d}.json',{'id':task['id'],'out':str(out),'error':error,'counted_frames':0})
                if error is not None:shard.fail(task_index,task,error)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures=[executor.submit(worker) for _ in range(workers)]
            for f in futures:f.result()
        shard.state['status']='failed' if shard.stop.is_set() else 'complete'
    except BaseException:
        shard.stop.set();shard.state['status']='failed';shard.state['fatal_error']=traceback.format_exc();raise
    finally:
        shard.state['finished_unix_s']=time.time();shard.save()
    print(json.dumps(shard.state))
    return int(shard.state['status']=='failed')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--contract',type=Path,required=True)
    ap.add_argument('--shard-index',type=int,required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();c=BASE.load_contract(args.contract);tasks=BASE.load_tasks(c)
    ownership(c,0)
    raise SystemExit(run(c,tasks,args.shard_index,args.workers))


if __name__=='__main__':main()
