"""Bounded setup correction. Preserve the live parent model and all frozen code.

QANet exposes conv_body_net/parsing_net as methods. Measure its real nonnested
registered children instead. No model inputs, outputs or learned calls change.
"""
import argparse
import fcntl
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
import unittest
from pathlib import Path

from common import PROJECT, WORKSPACE, atomic, install_shared_paths, load, now, sha, verify_sources
import worker

QANET = 'QANet_CIHP_HRNetW48_512x384'
REVISION = 'QANET_NATIVE_METHOD_TIMER_2'


class CorrectedTimer(worker.AdapterTimer):
    def __init__(self, model, torch):
        if model.family != 'QANet':
            super().__init__(model, torch)
            return
        self.torch, self.intervals, self.handles = torch, [], []
        modules = list(model.model.children())
        if not modules:
            raise ValueError('Expected registered QANet child modules')
        for module in modules:
            self.handles.append(module.register_forward_pre_hook(self.before))
            self.handles.append(module.register_forward_hook(self.after))


class MethodAcceptance(unittest.TestCase):
    def test_actual_method_style_dispatch(self):
        import torch
        from types import SimpleNamespace
        class MethodModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.Conv_Body = torch.nn.Identity()
                self.Parsing = torch.nn.ReLU()
            def conv_body_net(self, x):
                return self.Conv_Body(x)
            def parsing_net(self, x):
                return self.Parsing(x)
        model = MethodModel()
        self.assertFalse(isinstance(model.conv_body_net,torch.nn.Module))
        cuda = SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
                               max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
        timer = CorrectedTimer(SimpleNamespace(family='QANet',model=model),SimpleNamespace(cuda=cuda))
        output, timing = timer.measure(lambda:model.parsing_net(model.conv_body_net(torch.tensor([[-3.,5.]]))))
        self.assertEqual(output.tolist(),[[0.,5.]])
        self.assertEqual(timing['module_calls'],2)


def alive(pid):
    path = Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split(') ')[1][0] != 'Z'


def prepare(parent):
    from acceptance import Acceptance
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (Acceptance,MethodAcceptance)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError('Correction CPU acceptance failed')
    old = load(parent/'CONFIG.json')
    verify_sources(old)
    failure = load(parent/QANET/'STATE.json')
    if failure['model_ROI_API_calls'] != 0 or failure['setup_failures'] != 1:
        raise ValueError('This correction is limited to the zero-call QANet setup failure')
    current = load(parent/'CURRENT_WORKER.json')
    if current['model_id'] != 'M2FP_R101_CIHP' or current['timing_round'] is not None:
        raise ValueError('Expected only the already-running first unaffected parent model')
    root = PROJECT/'outputs'/(old['run_id']+'-timer-r2')
    root.mkdir(exist_ok=False)
    for name in ('MANIFEST.json','GT','GT_BINDINGS.json','detector','crops','RESOURCE_PLAN.json','environments'):
        (root/name).symlink_to(parent/name)
    mid = current['model_id']
    (root/mid).symlink_to(parent/mid, target_is_directory=True)
    config = dict(old,run_id=root.name,root=str(root),created_utc=now(),
                  parent_segment=str(parent),parent_config_sha256=sha(parent/'CONFIG.json'),
                  preserved_parent_model=mid,preserved_parent_worker_pid=current['pid'],
                  setup_correction=REVISION,
                  reuse_compatibility='same 600 IDs, crops, weights, native adapters, GT, mapping, scorer and ownership; only QANet timer attachment changes',
                  bounded_correction='one new QANet setup attempt, parent had zero model calls; no OOM/model retries')
    config['source_hashes'] = dict(old['source_hashes'],**{str(Path(__file__).relative_to(WORKSPACE)):sha(__file__)})
    atomic(root/'CONFIG.json',config)
    atomic(root/'CPU_ACCEPTANCE.json',dict(status='PASS',test_cases=result.testsRun,failures=0,
           config_sha256=sha(root/'CONFIG.json'),protocol_sha256=config['protocol_sha256'],revision=REVISION))
    # Stop only the parent queue. Its GPU child is in a separate process session.
    state = load(parent/'BATCH_STATE.json')
    controller_pid = state['pid']
    command = (Path('/proc')/str(controller_pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(parent/'CONFIG.json') not in command or 'batch.py' not in command:
        raise ValueError('Parent controller identity changed')
    os.kill(controller_pid,signal.SIGSTOP)
    check = load(parent/'CURRENT_WORKER.json')
    if check['pid'] != current['pid']:
        os.kill(controller_pid,signal.SIGCONT)
        raise ValueError('Parent queue advanced before handover, inspect before proceeding')
    if os.getpgid(current['pid']) == os.getpgid(controller_pid):
        os.kill(controller_pid,signal.SIGCONT)
        raise ValueError('Worker process is not detached from controller')
    atomic(parent/'CONTROLLED_HANDOVER.json',dict(at_utc=now(),previous_state=state,
           reason='QANet timer setup attachment error before all model calls',next_segment=str(root),
           preserved_worker=current,stop_parent_queue_only=True,model_calls_repeated=0))
    os.kill(controller_pid,signal.SIGKILL)
    state.update(status='CONTROLLER_STOPPED_PARENT_M2FP_FINISHING',ended_utc=now(),next_segment=str(root))
    atomic(parent/'BATCH_STATE.json',state)
    atomic(root/'CURRENT_WORKER.json',dict(current,origin='LIVE_PARENT_PRESERVED'))
    atomic(root/'BATCH_STATE.json',dict(status='QUEUED_WAITING_PARENT_MODEL_COMPLETION',parent_segment=str(parent),
           parent_worker_pid=current['pid'],models={mid:'RUNNING_PARENT'},at_utc=now()))
    atomic(PROJECT/'outputs/CURRENT.json',dict(root=str(root),run_id=root.name,config_sha256=sha(root/'CONFIG.json')))
    print(root/'CONFIG.json')


def controller(config_path):
    install_shared_paths()
    from batch import score
    config = load(config_path)
    root, parent = Path(config['root']),Path(config['parent_segment'])
    lock = (root.parent/'CONTROLLER.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources(config)
    manifest = load(root/'MANIFEST.json')
    mid_parent = config['preserved_parent_model']
    state = dict(status='QUEUED_WAITING_PARENT_MODEL_COMPLETION',pid=os.getpid(),started_utc=now(),
                 models={mid_parent:'RUNNING_PARENT'},parent_segment=str(parent),automatic_retry=False)
    atomic(root/'BATCH_STATE.json',state)
    wait_start = time.monotonic()
    while alive(config['preserved_parent_worker_pid']):
        if time.monotonic()-wait_start > 24*3600:
            raise RuntimeError('Parent handover wait exceeded 24 hours, no new GPU process started')
        time.sleep(10)
    state['parent_wait_seconds'] = time.monotonic()-wait_start
    parent_state = load(parent/mid_parent/'STATE.json')
    state['models'][mid_parent] = parent_state['status']
    state['status'] = 'QUALITY_BATCH_RUNNING'
    atomic(root/'BATCH_STATE.json',state)

    def launch(mid,round_number=None):
        command = [str(WORKSPACE/config['models'][mid]['environment']/'bin/python'),str(Path(__file__)),
                   '--worker','--config',str(config_path),'--model',mid]
        label = mid
        if round_number is not None:
            command += ['--timing-round',str(round_number)]
            label += '-timing-'+str(round_number)
        log_path = root/'logs'/(label+'.log')
        log_path.parent.mkdir(exist_ok=True)
        env = dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        env.setdefault('CUDA_VISIBLE_DEVICES','0')
        with log_path.open('xb') as log:
            child = subprocess.Popen(command,cwd=WORKSPACE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(root/'CURRENT_WORKER.json',dict(model_id=mid,pid=child.pid,command=command,
                   timing_round=round_number,log=str(log_path),started_utc=now()))
            try:
                code = child.wait(timeout=24*3600+300)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                raise
        output = root/mid if round_number is None else root/'timing'/('round-'+str(round_number))/mid
        observed = load(output/'STATE.json') if (output/'STATE.json').exists() else dict(status='SETUP_FAILED')
        atomic(output/'CONTROLLER_OBSERVATION.json',dict(returncode=code,observed=observed,at_utc=now()))
        return observed['status'] if code==0 else 'STOPPED_REQUIRES_RECONCILIATION'

    for mid in config['models']:
        if mid == mid_parent:
            continue
        state['current_model'] = mid
        atomic(root/'BATCH_STATE.json',state)
        state['models'][mid] = launch(mid)
        atomic(root/'BATCH_STATE.json',state)
    state['status'] = 'SAVED_MASK_SCORING'
    atomic(root/'BATCH_STATE.json',state)
    results = {}
    for mid in config['models']:
        source_root = parent if mid==mid_parent else root
        try:
            results[mid] = score(source_root,load(source_root/'CONFIG.json'),mid,manifest)
        except Exception:
            results[mid] = dict(model_id=mid,status='SCORING_CONTRACT_FAILURE',primary=None,traceback=traceback.format_exc())
        results[mid]['source_segment'] = str(source_root)
        results[mid]['timer_revision'] = REVISION if mid==QANET else 'ORIGINAL_MODULE_TIMER'
    atomic(root/'RESULT_INDEX.json',results)
    state['status'],state['timing'] = 'DEDICATED_CLEAN_TIMING',{}
    atomic(root/'BATCH_STATE.json',state)
    roster = list(config['models'])
    for number in range(1,4):
        for mid in roster[number-1:]+roster[:number-1]:
            if results[mid].get('status')=='COMPLETE_AD4_DIAGNOSTIC':
                state['timing'][str(number)+'/'+mid] = launch(mid,number)
                atomic(root/'BATCH_STATE.json',state)
    state.update(status='STOPPED_AFTER_APPROVED_NINE',ended_utc=now(),next_experiment=False,
                 completed_models=sum(r.get('status')=='COMPLETE_AD4_DIAGNOSTIC' for r in results.values()))
    atomic(root/'BATCH_STATE.json',state)
    from report import report
    from gallery import gallery
    report(root)
    gallery(root)
    atomic(root/'BATCH_COMPLETION.json',state)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--timing-round',type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.worker:
        worker.AdapterTimer = CorrectedTimer
        sys.exit(worker.run(args.config,args.model,args.timing_round))
    else:
        try: controller(args.config)
        except BaseException:
            config = load(args.config)
            atomic(Path(config['root'])/'CONTROLLER_FAILURE.json',dict(at_utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
            raise
