"""Detached sequential quality, saved-mask scoring, then isolated timing rounds."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import PROJECT, WORKSPACE, archive, atomic, install_shared_paths, load, now, sha, verify_sources
from evaluation import aggregate, compose, project, score_frame


def score(root, config, mid, manifest):
    from common_view.packets import read_packet
    directory = root / mid
    expected = manifest['expected_ids']
    completion = load(directory / 'COMPLETION.json') if (directory / 'COMPLETION.json').exists() else {}
    if not completion.get('strict_complete'):
        from reconcile import reconcile
        checked = reconcile(root,mid)
        state = load(directory/'STATE.json') if (directory/'STATE.json').exists() else {}
        result = dict(model_id=mid, primary=None, full_frame_primary=None, status='INCOMPLETE',
                      reason='Exact prediction/ROI completion not accepted',
                      membership=checked['image_membership'],ROI_membership=checked['ROI_membership'],
                      setup_failures=state.get('setup_failures'),VAL_pipeline_failures=state.get('failed'),
                      invalid_artifacts=len(checked['invalid']),partial_artifacts=len(checked['partial']),
                      orphan_artifacts=len(checked['orphan']))
        atomic(directory / 'METRICS.json', result)
        return result
    mapping = config['protocol']['mapping']['SAPIENS29' if config['models'][mid]['family'] == 'Sapiens2' else 'CIHP20']
    rows = []
    checkpoint = directory / 'scoring'
    checkpoint.mkdir(exist_ok=True)
    for frame in manifest['frames']:
        iid = frame['image_id']
        path = checkpoint / (iid + '.json')
        gt_path, gb_path = root / 'GT' / (iid + '.npz'), root / 'GT' / (iid + '.json')
        if sha(gt_path) != frame['gt_npz_sha256'] or sha(gb_path) != frame['gt_binding_sha256']:
            raise ValueError('GT binding mismatch')
        metadata, packet = read_packet(directory / 'packets' / iid,
            dict(config_sha256=sha(root / 'CONFIG.json'), image_id=iid,
                 detector_sha256=frame['detector_sha256'], input_sha256=frame['image_sha256']))
        if len(metadata['ROI_references']) != len(frame['jobs']):
            raise ValueError('ROI packet count mismatch')
        for reference, job in zip(metadata['ROI_references'], frame['jobs']):
            parent = directory / 'roi_packets' / job['job_id']
            if reference['job_id'] != job['job_id'] or sha(parent / 'packet.json') != reference['metadata_sha256']:
                raise ValueError('ROI metadata reference mismatch')
            if sha(parent / 'prediction.npz') != reference['packet_sha256']:
                raise ValueError('ROI NPZ reference mismatch')
        if sha(root / 'detector' / (iid + '.npz')) != frame['detector_sha256']:
            raise ValueError('Frozen detector cache mismatch')
        cache = archive(root / 'detector' / (iid + '.npz'))
        for name, key in [('person_extents', 'extents'), ('person_scores', 'scores'),
                          ('bboxes_xyxy', 'boxes'), ('person_ids', 'ids')]:
            if not np.array_equal(packet[name], cache[key]):
                raise ValueError('Ownership/identity mismatch')
        native = compose(packet['person_semantic_masks'],cache['scores'],cache['boxes'],cache['ids'],(frame['height'],frame['width']))
        if not np.array_equal(native,packet['composed_native_mask']) or not np.array_equal(project(native,mapping),packet['common_mask']):
            raise ValueError('Saved native composition/projection mismatch')
        binding = dict(config_sha256=sha(root / 'CONFIG.json'), packet_sha256=metadata['packet_sha256'],
                       GT_sha256=frame['gt_npz_sha256'])
        if path.exists():
            row = load(path)
            if row['bindings'] != binding:
                raise ValueError('CPU resume checkpoint binding mismatch')
        else:
            row = dict(image_id=iid, bindings=binding,
                       **score_frame(archive(gt_path), load(gb_path), frame,
                                     packet['person_semantic_masks'], cache, mapping))
            atomic(path, row)
        rows.append(row)
        atomic(directory / 'SCORING_STATE.json', dict(status='SCORING', scored_images=len(rows), expected=600))
    result = dict(model_id=mid, status='COMPLETE_AD4_DIAGNOSTIC', protocol_sha256=config['protocol_sha256'],
                  config_sha256=sha(root / 'CONFIG.json'), **aggregate(rows, expected))
    atomic(directory / 'METRICS.json', result)
    atomic(directory / 'SCORING_STATE.json', dict(status='COMPLETE', scored_images=600, at_utc=now()))
    return result


def controller(config_path):
    install_shared_paths()
    config = load(config_path)
    root = Path(config['root'])
    lock = (root.parent / 'CONTROLLER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    acceptance = load(root / 'CPU_ACCEPTANCE.json')
    if acceptance['status'] != 'PASS' or acceptance['config_sha256'] != sha(config_path):
        raise ValueError('CPU acceptance missing or stale')
    verify_sources(config)
    manifest = load(root / 'MANIFEST.json')
    state = dict(status='QUALITY_BATCH_RUNNING', pid=os.getpid(), started_utc=now(), models={},
                 config_sha256=sha(config_path), quality_analysis_during_inference=False,
                 sequential_workers=True, automatic_retry=False)

    def launch(mid, round_number=None):
        env = os.environ.copy()
        env.setdefault('CUDA_VISIBLE_DEVICES', '0')
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        spec = config['models'][mid]
        command = [str(WORKSPACE / spec['environment'] / 'bin/python'), str(PROJECT / 'src/worker.py'),
                   '--config', str(config_path), '--model', mid]
        label = mid
        if round_number is not None:
            command += ['--timing-round', str(round_number)]
            label += '-timing-' + str(round_number)
        log_path = root / 'logs' / (label + '.log')
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic(root / 'CURRENT_WORKER.json', dict(model_id=mid, timing_round=round_number,
                   pid=child.pid, command=command, log=str(log_path), started_utc=now()))
            try:
                returncode = child.wait(timeout=config['protocol']['resources']['max_model_hours'] * 3600 + 300)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                returncode = -signal.SIGTERM
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                raise
        path = root / mid if round_number is None else root / 'timing' / ('round-' + str(round_number)) / mid
        observed = load(path / 'STATE.json') if (path / 'STATE.json').exists() else dict(status='SETUP_FAILED')
        atomic(path / 'CONTROLLER_OBSERVATION.json', dict(returncode=returncode, recorded=observed,
               at_utc=now(), requires_reconciliation=returncode != 0))
        return observed['status'] if returncode == 0 else 'STOPPED_REQUIRES_RECONCILIATION'

    atomic(root / 'BATCH_STATE.json', state)
    for mid in config['models']:
        state['current_model'] = mid
        atomic(root / 'BATCH_STATE.json', state)
        state['models'][mid] = launch(mid)
        atomic(root / 'BATCH_STATE.json', state)
    # Quality summaries are computed after every approved quality attempt stops.
    state['status'] = 'SAVED_MASK_SCORING'
    atomic(root / 'BATCH_STATE.json', state)
    results = {}
    for mid in config['models']:
        try:
            results[mid] = score(root, config, mid, manifest)
        except Exception as error:
            import traceback
            results[mid] = dict(model_id=mid, primary=None, status='SCORING_CONTRACT_FAILURE',
                               error=str(error), traceback=traceback.format_exc())
            atomic(root / mid / 'METRICS.json', results[mid])
    atomic(root / 'RESULT_INDEX.json', results)
    state['status'] = 'DEDICATED_CLEAN_TIMING'
    state['timing'] = {}
    atomic(root / 'BATCH_STATE.json', state)
    roster = list(config['models'])
    for round_number in range(1, 4):
        order = roster[round_number - 1:] + roster[:round_number - 1]
        for mid in order:
            if results[mid].get('status') != 'COMPLETE_AD4_DIAGNOSTIC':
                continue
            key = str(round_number) + '/' + mid
            state['timing'][key] = launch(mid, round_number)
            atomic(root / 'BATCH_STATE.json', state)
    state.update(status='STOPPED_AFTER_APPROVED_NINE', ended_utc=now(), next_experiment=False,
                 completed_models=sum(r.get('status') == 'COMPLETE_AD4_DIAGNOSTIC' for r in results.values()))
    atomic(root / 'BATCH_STATE.json', state)
    from report import report
    report(root)
    from gallery import gallery
    gallery(root)
    atomic(root / 'BATCH_COMPLETION.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--score-model')
    args = parser.parse_args()
    if args.score_model:
        install_shared_paths()
        cfg = load(args.config)
        score(Path(cfg['root']), cfg, args.score_model, load(Path(cfg['root']) / 'MANIFEST.json'))
    else:
        try:
            controller(args.config)
        except BaseException:
            import traceback
            cfg = load(args.config)
            root = Path(cfg['root'])
            failure = dict(status='INTERRUPTED_CONTROLLER_REQUIRES_RECONCILIATION',
                           at_utc=now(),pid=os.getpid(),traceback=traceback.format_exc(),automatic_retry=False)
            atomic(root/'CONTROLLER_FAILURE.json',failure)
            raise
