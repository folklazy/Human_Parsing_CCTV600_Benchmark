"""Controlled native reuse/continuation and the shared-reference CCTV600 batch."""
import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from common import PROJECT, WORKSPACE, archive, atomic, event, install_shared_paths, load, membership, now, npz, sha, verify_sources

def proven_empty_parsing(error):
    if error.args != ('parsing',):
        return False
    tb = error.__traceback__
    while tb:
        frame = tb.tb_frame
        if frame.f_code.co_filename.endswith('/cihp_benchmark/adapters.py'):
            prediction = frame.f_locals.get('pred')
            return prediction is not None and len(prediction) == 0
        tb = tb.tb_next
    return False

def release_unused_cache(observed, threshold, cuda, query):
    """Release only unused allocator memory before the unchanged physical guard."""
    if observed['free_MiB'] >= threshold or not cuda.is_initialized():
        return (observed, None)
    unused = max(0, cuda.memory_reserved() - cuda.memory_allocated())
    if not unused:
        return (observed, None)
    started = time.perf_counter()
    cuda.synchronize()
    cuda.empty_cache()
    cuda.synchronize()
    return (query(), dict(unused_cache_bytes=unused, elapsed_ms=(time.perf_counter() - started) * 1000))

def collector(config_path, mid, timing_round=None):
    import torch
    install_shared_paths()
    import worker
    import controlled_roi_models
    from correct_qanet import CorrectedTimer
    Original = controlled_roi_models.CropModel
    config = load(config_path)
    root = Path(config['root'])

    class ReferenceCropModel(Original):

        def infer(self, rgb, image_path, designated_box):
            try:
                return super().infer(rgb, image_path, designated_box)
            except KeyError as error:
                if self.family != 'RP-R-CNN' or not proven_empty_parsing(error):
                    raise
                h, w = rgb.shape[:2]
                return (np.zeros((h, w), np.uint8), np.empty((0, h, w), np.uint8), dict(selection='NATIVE_ASSOCIATION_MISS', native_empty_BoxList_confirmed=True, implementation_revision='RPR_EMPTY_NATIVE_OUTPUT_1', native_boxes=[], native_scores=[], native_selected_index=None, native_geometry=[dict(whole_frame=True, native_empty=True)]))
    controlled_roi_models.CropModel = ReferenceCropModel
    worker.AdapterTimer = CorrectedTimer
    original_gpu = worker.gpu

    def gpu_with_cache_guard():
        observed = original_gpu()
        threshold = config['protocol']['resources']['minimum_pre_forward_free_MiB']
        if config['models'][mid]['family'] == 'CPI-Parser':
            after, released = release_unused_cache(observed, threshold, torch.cuda, original_gpu)
            if released:
                event(root / 'CACHE_GUARD_EVENTS.jsonl', event='RELEASE_OWN_UNUSED_CACHE_BEFORE_GUARD', before=observed, after=after, model_retry=False, **released)
                return after
        return observed
    worker.gpu = gpu_with_cache_guard
    return worker.run(config_path, mid, timing_round)

def provider_roots(root):
    parent = Path(load(root / 'CONFIG.json')['parent_segment'])
    return {mid: parent / mid for mid in load(root / 'CONFIG.json')['models']}

def reconcile_model(root, mid):
    install_shared_paths()
    from common_view.packets import read_packet
    config = load(root / 'CONFIG.json')
    manifest = load(root / 'MANIFEST.json')
    source = provider_roots(root)[mid]
    accepted = {}
    invalid = []
    jobs = {j['job_id']: j for f in manifest['frames'] for j in f['jobs']}
    for job_id, job in jobs.items():
        packet = source / 'roi_packets' / job_id
        if not packet.exists():
            continue
        try:
            meta, arrays = read_packet(packet)
            for key, value in dict(model_id=mid, job_id=job_id, crop_sha256=job['crop_sha256'], checkpoint_sha256=config['models'][mid]['checkpoint_sha256']).items():
                if meta.get(key) != value:
                    raise ValueError('Parent ROI binding mismatch ' + key)
            native = arrays['native_crop_mask']
            classes = 29 if config['models'][mid]['family'] == 'Sapiens2' else 20
            if native.dtype != np.uint8 or list(native.shape) != job['crop_hw'] or np.any(native >= classes):
                raise ValueError('Parent native mask contract')
            accepted[job_id] = dict(packet=str(packet), packet_sha256=meta['packet_sha256'], metadata_sha256=sha(packet / 'packet.json'), source_config_sha256=meta['config_sha256'])
        except Exception as error:
            invalid.append(dict(job_id=job_id, error=str(error), path=str(packet)))
    known = set(jobs)
    orphan = [str(p) for p in (source / 'roi_packets').glob('*') if p.is_dir() and (not p.name.startswith('.')) and (p.name not in known)]
    partial = [str(p) for p in source.glob('**/.partial-*')]
    result = dict(model_id=mid, source=str(source), accepted=accepted, pending=[j for j in jobs if j not in accepted], invalid=invalid, orphan=orphan, partial=partial, at_utc=now())
    atomic(root / 'reconciliation' / (mid + '.json'), result)
    return result

def collection_config(root, mid, reconciled, timing=False):
    config = load(root / 'CONFIG.json')
    parent = Path(config['parent_segment'])
    original = load(parent / 'CONFIG.json')
    manifest = load(root / 'MANIFEST.json')
    directory = root / ('timing_collection' if timing else 'collections') / mid
    directory.mkdir(parents=True, exist_ok=False)
    pending = set(reconciled['pending']) if not timing else {j['job_id'] for f in manifest['frames'] for j in f['jobs']}
    frames = []
    for frame in manifest['frames']:
        source_cache = archive(root / 'detector' / (frame['image_id'] + '.npz'))
        indices = [i for i, j in enumerate(frame['jobs']) if j['job_id'] in pending]
        cache = {k: v[indices] for k, v in source_cache.items()}
        path = directory / 'detector' / (frame['image_id'] + '.npz')
        npz(path, **cache)
        frames.append(dict(frame, jobs=[frame['jobs'][i] for i in indices], detector_sha256=sha(path)))
    submanifest = dict(manifest, frames=frames, ROI_jobs=len(pending))
    atomic(directory / 'MANIFEST.json', submanifest)
    native_protocol = dict(original['protocol'], protocol='CCTV600_NATIVE_PENDING_COLLECTION_V1', role='NATIVE_COLLECTION_ONLY_NOT_CANONICAL_REFERENCE_SCORES')
    native = dict(original, root=str(directory), run_id=directory.name, protocol=native_protocol, protocol_sha256=sha(root / 'CONFIG.json'), manifest_sha256=sha(directory / 'MANIFEST.json'), source_hashes=config['source_hashes'], role='pending-only native collector', parent_reference_config_sha256=sha(root / 'CONFIG.json'))
    atomic(directory / 'CONFIG.json', native)
    atomic(directory / 'LAUNCH.json', load(root / 'LAUNCH.json'))
    return directory / 'CONFIG.json'

def execute(root, mid, config_path, round_number=None):
    config = load(root / 'CONFIG.json')
    command = [str(WORKSPACE / config['models'][mid]['environment'] / 'bin/python'), str(Path(__file__)), '--worker', '--root', str(config_path), '--model', mid]
    label = mid
    if round_number is not None:
        command += ['--timing-round', str(round_number)]
        label += '-timing-' + str(round_number)
    log_path = root / 'logs' / (label + '.log')
    log_path.parent.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    with log_path.open('xb') as log:
        child = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(root / 'CURRENT_WORKER.json', dict(model_id=mid, pid=child.pid, command=command, log=str(log_path), timing_round=round_number, config=str(config_path), at_utc=now()))
        try:
            returncode = child.wait(timeout=24 * 3600 + 300)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    return returncode

def assemble(root, mid, reconciled, collection):
    from common_view.packets import read_packet, write_packet
    from controlled_roi_core import place
    config = load(root / 'CONFIG.json')
    manifest = load(root / 'MANIFEST.json')
    refs = dict(reconciled['accepted'])
    if collection:
        for frame in manifest['frames']:
            for job in frame['jobs']:
                packet = collection / mid / 'roi_packets' / job['job_id']
                if not packet.exists():
                    continue
                meta, arrays = read_packet(packet, dict(model_id=mid, job_id=job['job_id'], checkpoint_sha256=config['models'][mid]['checkpoint_sha256'], crop_sha256=job['crop_sha256']))
                classes = 29 if config['models'][mid]['family'] == 'Sapiens2' else 20
                native = arrays['native_crop_mask']
                if native.dtype != np.uint8 or list(native.shape) != job['crop_hw'] or np.any(native >= classes):
                    raise ValueError('New native ROI contract failure')
                refs[job['job_id']] = dict(packet=str(packet), packet_sha256=meta['packet_sha256'], metadata_sha256=sha(packet / 'packet.json'), source_config_sha256=meta['config_sha256'])
    expected = [j['job_id'] for f in manifest['frames'] for j in f['jobs']]
    checked = membership(expected, refs)
    output = root / 'models' / mid
    output.mkdir(parents=True, exist_ok=False)
    if not checked['strict_complete'] or reconciled['invalid']:
        result = dict(model_id=mid, status='INCOMPLETE', primary=None, ROI_membership=checked, invalid=len(reconciled['invalid']), historical_failures_preserved=True)
        atomic(output / 'COMPLETION.json', result)
        return result
    index = {}
    classes = 29 if config['models'][mid]['family'] == 'Sapiens2' else 20
    parent = Path(reconciled['source'])
    for frame in manifest['frames']:
        iid = frame['image_id']
        cache = archive(root / 'detector' / (iid + '.npz'))
        old = parent / 'packets' / iid
        new = collection / mid / 'packets' / iid if collection else None
        candidates = [p for p in (old, new) if p is not None and p.exists()]
        chosen = None
        for path in candidates:
            meta, arrays = read_packet(path)
            if meta.get('model_id') != mid or meta.get('image_id') != iid or meta.get('input_sha256') != frame['image_sha256']:
                raise ValueError('Frame model/input identity mismatch')
            if meta.get('checkpoint_sha256') != config['models'][mid]['checkpoint_sha256']:
                raise ValueError('Frame checkpoint identity mismatch')
            if all((np.array_equal(arrays[name], cache[key]) for name, key in (('person_extents', 'extents'), ('person_scores', 'scores'), ('bboxes_xyxy', 'boxes'), ('person_ids', 'ids')))):
                chosen = path
                break
        if chosen is None:
            planes = []
            for number, job in enumerate(frame['jobs']):
                meta, arrays = read_packet(refs[job['job_id']]['packet'])
                raw, owned = place(arrays['native_crop_mask'], job['crop_xyxy'], cache['extents'][number], classes)
                planes.append(owned)
            height, width = (frame['height'], frame['width'])
            chosen = write_packet(output / 'assembled_packets', iid, dict(person_semantic_masks=np.asarray(planes, np.uint8).reshape(-1, height, width), person_extents=cache['extents'], person_scores=cache['scores'], bboxes_xyxy=cache['boxes'], person_ids=cache['ids']), dict(model_id=mid, image_id=iid, native_classes=classes, config_sha256=sha(root / 'CONFIG.json'), checkpoint_sha256=config['models'][mid]['checkpoint_sha256'], detector_sha256=frame['detector_sha256'], input_sha256=frame['image_sha256'], ROI_references=[refs[j['job_id']] for j in frame['jobs']], GT_used_in_prediction=False, origin='assembled accepted native crops'), {})
        meta, arrays = read_packet(chosen)
        if arrays['person_semantic_masks'].shape != cache['extents'].shape or arrays['person_semantic_masks'].dtype != np.uint8 or np.any(arrays['person_semantic_masks'] >= classes):
            raise ValueError('Assembled native plane contract')
        if np.any(arrays['person_semantic_masks'][~arrays['person_extents']]):
            raise ValueError('Ownership changed during assembly')
        index[iid] = dict(packet=str(chosen), packet_sha256=meta['packet_sha256'], metadata_sha256=sha(chosen / 'packet.json'))
    atomic(output / 'ASSEMBLED_INDEX.json', dict(frames=index, ROI_references=refs, reused_ROIs=len(reconciled['accepted']), new_ROIs=len(expected) - len(reconciled['accepted']), input_membership_sha256=config['input_membership_sha256']))
    result = dict(model_id=mid, status='PREDICTIONS_COMPLETE', membership=membership(manifest['expected_ids'], index), ROI_membership=checked, reused_ROIs=len(reconciled['accepted']), new_ROIs=len(expected) - len(reconciled['accepted']), invalid=0, primary=None, scoring='WAITING_BATCH_END_REFERENCE_SCORING')
    atomic(output / 'COMPLETION.json', result)
    return result

def score(root, mid):
    from common_view.packets import read_packet
    from reference_parts600 import score_frame, aggregate
    config = load(root / 'CONFIG.json')
    manifest = load(root / 'MANIFEST.json')
    output = root / 'models' / mid
    completion = load(output / 'COMPLETION.json')
    if completion['status'] != 'PREDICTIONS_COMPLETE':
        result = dict(completion, Macro4_local=None)
        atomic(output / 'METRICS.json', result)
        return result
    assembly = load(output / 'ASSEMBLED_INDEX.json')
    rows = []
    space = 'SAPIENS2_DOME29_PRED' if config['models'][mid]['family'] == 'Sapiens2' else 'CIHP_GT_AND_NATIVE_PRED'
    for iid in manifest['expected_ids']:
        gt_path = root / 'GT' / (iid + '.npz')
        binding = load(root / 'GT' / (iid + '.json'))
        ref = assembly['frames'][iid]
        if sha(gt_path) != config['GT_packet_bindings'][iid]:
            raise ValueError('Reference GT changed')
        meta, arrays = read_packet(ref['packet'])
        if meta['packet_sha256'] != ref['packet_sha256'] or sha(Path(ref['packet']) / 'packet.json') != ref['metadata_sha256']:
            raise ValueError('Assembled packet identity changed')
        checkpoint = output / 'scoring' / (iid + '.json')
        identity = dict(packet_sha256=ref['packet_sha256'], GT_sha256=binding['npz_sha256'], config_sha256=sha(root / 'CONFIG.json'))
        if checkpoint.exists():
            row = load(checkpoint)
            if row['bindings'] != identity:
                raise ValueError('Resume scoring binding changed')
        else:
            row = dict(image_id=iid, bindings=identity, **score_frame(archive(gt_path), binding['matches'], arrays['person_semantic_masks'], space, config['mapping']))
            atomic(checkpoint, row)
        rows.append(row)
    result = dict(model_id=mid, status='COMPLETE', protocol_sha256=config['protocol_sha256'], **aggregate(rows, manifest['expected_ids']))
    atomic(output / 'METRICS.json', result)
    return result

def controller(root):
    import fcntl
    from correct_qanet import alive
    install_shared_paths()
    config = load(root / 'CONFIG.json')
    verify_sources(config)
    if load(root / 'CPU_ACCEPTANCE.json')['config_sha256'] != sha(root / 'CONFIG.json'):
        raise ValueError('Reference CPU acceptance is stale')
    lock = (PROJECT / 'outputs/CONTROLLER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    current = load(root / 'HANDOVER.json')['current_worker']
    state = dict(status='WAITING_PRESERVED_NATIVE_WORKER', pid=os.getpid(), started_utc=now(), models={}, preserved_worker=current, quality_analysis_during_inference=False)
    atomic(root / 'BATCH_STATE.json', state)
    wait = time.monotonic()
    while alive(current['pid']):
        if time.monotonic() - wait > 24 * 3600:
            raise RuntimeError('Parent collection handover exceeded 24h')
        time.sleep(10)
    state['queue_wait_seconds'] = time.monotonic() - wait
    state['status'] = 'QUALITY_COLLECTION'
    for mid in config['models']:
        state['current_model'] = mid
        atomic(root / 'BATCH_STATE.json', state)
        try:
            reconciled = reconcile_model(root, mid)
            collection = None
            if not reconciled['invalid'] and reconciled['pending']:
                cp = collection_config(root, mid, reconciled)
                collection = cp.parent
                execute(root, mid, cp)
            state['models'][mid] = assemble(root, mid, reconciled, collection)
        except Exception:
            output = root / 'models' / mid
            output.mkdir(parents=True, exist_ok=True)
            state['models'][mid] = dict(model_id=mid, status='INVALID_CONTRACT', primary=None,
                                        error=traceback.format_exc())
            atomic(output / 'COMPLETION.json', state['models'][mid])
        atomic(root / 'BATCH_STATE.json', state)
    state['status'] = 'REFERENCE_SCORING'
    atomic(root / 'BATCH_STATE.json', state)
    results = {}
    for mid in config['models']:
        try:
            results[mid] = score(root, mid)
        except Exception:
            results[mid] = dict(model_id=mid, status='SCORING_FAILED', Macro4_local=None, error=traceback.format_exc())
            atomic(root / 'models' / mid / 'METRICS.json', results[mid])
    atomic(root / 'RESULT_INDEX.json', results)
    state['status'] = 'DEDICATED_TIMING'
    state['timing'] = {}
    atomic(root / 'BATCH_STATE.json', state)
    timing_configs = {mid: collection_config(root, mid, {}, timing=True) for mid in config['models'] if results[mid].get('status') == 'COMPLETE'}
    roster = list(config['models'])
    for number in range(1, 4):
        for mid in roster[number - 1:] + roster[:number - 1]:
            if mid not in timing_configs:
                continue
            state['timing'][str(number) + '/' + mid] = execute(root, mid, timing_configs[mid], number)
            atomic(root / 'BATCH_STATE.json', state)
    state.update(status='STOPPED_AFTER_APPROVED_NINE', ended_utc=now(), next_experiment=False, complete_models=sum((r.get('status') == 'COMPLETE' for r in results.values())))
    atomic(root / 'BATCH_STATE.json', state)
    from reference_parts600 import report, gallery
    report(root)
    gallery(root)
    atomic(root / 'BATCH_COMPLETION.json', state)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--timing-round', type=int)
    args = parser.parse_args()
    if args.worker:
        sys.exit(collector(args.root, args.model, args.timing_round))
    try:
        controller(args.root)
    except BaseException:
        atomic(args.root / 'CONTROLLER_FAILURE.json', dict(at_utc=now(), status='REQUIRES_RECONCILIATION', error=traceback.format_exc()))
        raise
