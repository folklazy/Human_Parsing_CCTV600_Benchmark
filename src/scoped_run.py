"""Fresh nine-parser CCTV600 run with current GT and partial-person reporting."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from common import PROJECT, WORKSPACE, archive, atomic, install_shared_paths, load, membership, now, npz, sha, verify_sources
import reference_collection as collection
import reference_parts600 as reference
import scoped_metrics


def assert_idle():
    """Use the existing controller lock and inspect only benchmark processes."""
    with (PROJECT / 'outputs/CONTROLLER.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            args = path.read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        ours = any(a.endswith(('/src/reference_collection.py', '/src/scoped_run.py', '/src/worker.py', '/src/correct_qanet.py'))
                   and str(PROJECT) in a for a in args)
        if ours and ('--worker' in args or '--controller' in args):
            raise RuntimeError(f'Existing benchmark process {path.parent.name}: {shlex.join(args[:-1])}')


def global_guard(root, resources):
    if shutil.disk_usage(root).free < resources['minimum_free_disk_GiB'] * 2**30:
        raise SystemExit('GLOBAL_MINIMUM_FREE_DISK_GUARD')
    used = int(subprocess.check_output(['du', '-sb', str(root)], text=True).split()[0])
    if used > resources['max_run_GiB'] * 2**30:
        raise SystemExit('GLOBAL_RUN_STORAGE_GUARD')


def gt_frame(frame, document):
    """Rebuild matching inputs from current GT, independent of YOLO inputs."""
    people, excluded, ignore = [], [], []
    width, height = document['imageWidth'], document['imageHeight']
    for shape in document['shapes']:
        if shape['label'] not in ('bbox', 'ignore'):
            continue
        xs, ys = zip(*shape['points'])
        source = [min(xs), min(ys), max(xs), max(ys)]
        clipped = [max(0, source[0]), max(0, source[1]), min(width, source[2]), min(height, source[3])]
        if shape['label'] == 'ignore':
            ignore.append(clipped)
            continue
        long_side = max(source[2] - source[0], source[3] - source[1])
        person = dict(person_id=shape['group_id'], bbox=clipped, source_bbox=source, long_side_px=long_side)
        if long_side >= 50:
            people.append(person)
        else:
            excluded.append(person)
            ignore.append(clipped)
    return dict(frame, people=people, excluded_people=excluded, ignore=ignore)


def verify_models(config):
    """Verify actual files, sources and existing environments before launch."""
    checked = set()
    environments = {}
    for mid, model in config['models'].items():
        for key in ('checkpoint', 'native_config'):
            path = WORKSPACE / model[key]
            expected = model[key + '_sha256']
            if (str(path), expected) not in checked:
                if sha(path) != expected:
                    raise ValueError(f'{mid}: {key} changed')
                checked.add((str(path), expected))
        head = subprocess.check_output(['git', '-C', str(WORKSPACE / model['source']), 'rev-parse', 'HEAD'], text=True).strip()
        if head != model['source_commit']:
            raise ValueError('Native source HEAD changed: ' + mid)
        env = model['environment']
        if env not in environments:
            command = [str(WORKSPACE / env / 'bin/python'), '-c',
                       "import hashlib,importlib.metadata as m,json,torch; "
                       "p='\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name']))+'\\n'; "
                       "print(json.dumps(dict(package_freeze_sha256=hashlib.sha256(p.encode()).hexdigest(),"
                       "cuda_available=torch.cuda.is_available(),torch=torch.__version__,CUDA=torch.version.cuda)))"]
            result = subprocess.check_output(command, text=True)
            environments[env] = json.loads(result.strip().splitlines()[-1])
        observed = environments[env]
        if not observed['cuda_available'] or observed['package_freeze_sha256'] != config['verified_models'][mid]['package_freeze_sha256']:
            raise ValueError('Model environment binding changed: ' + mid)
    return environments


def prepare(base):
    assert_idle()
    old = load(base / 'CONFIG.json')
    verify_sources(old)
    data_root = WORKSPACE.parent / 'CCTV_Human_Parsing_600.local'
    current = load(data_root / 'CURRENT.json')
    delivery = load(current['dataset_metadata'])
    if delivery['annotation_coverage']['complete_person_inventory']:
        raise ValueError('Expected the confirmed partial-person dataset contract')
    manifest_path = Path(current['manifest'])
    if sha(manifest_path) != current['manifest_sha256']:
        raise ValueError('Dataset manifest identity changed')
    rows = list(csv.DictReader(manifest_path.open()))
    logical = [[r['image_id'], r['image_sha256'], r['annotation_sha256']] for r in rows]
    digest = hashlib.sha256(json.dumps(logical, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    previous = load(base / 'MANIFEST.json')
    by_id = {r['image_id']: r for r in rows}
    if digest != current['membership_sha256'] or len(rows) != len(by_id) or set(by_id) != set(previous['expected_ids']) or len(rows) != 600:
        raise ValueError('Exact dataset population/binding mismatch')
    root = PROJECT / 'outputs' / ('cctv600-fresh-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root.mkdir(parents=True)
    atomic(root / 'PREPARATION.json', dict(status='PREPARING_GT', pid=os.getpid(), started_utc=now(), completed_images=0))
    # These are frozen input/cache assets. No previous parser prediction is read.
    for name in ('detector', 'crops', 'environments', 'RESOURCE_PLAN.json'):
        (root / name).symlink_to(base / name)
    native_parent = Path(old['old_collector_config']).parent if 'old_collector_config' in old else Path(old['parent_segment'])
    summary, support = Counter(), Counter()
    frames = []
    for number, frame in enumerate(previous['frames'], 1):
        row = by_id[frame['image_id']]
        if frame['image_sha256'] != row['image_sha256'] or sha(row['image_path']) != row['image_sha256']:
            raise ValueError('Image input changed: ' + row['image_id'])
        annotation = Path(row['annotation_path'])
        if sha(annotation) != row['annotation_sha256']:
            raise ValueError('Annotation snapshot changed: ' + row['image_id'])
        document = load(annotation)
        if [document['imageWidth'], document['imageHeight']] != [frame['width'], frame['height']]:
            raise ValueError('Original-grid GT dimensions changed')
        updated = gt_frame(dict(frame, image_path=row['image_path'], annotation_path=str(annotation),
                                annotation_sha256=row['annotation_sha256']), document)
        arrays, binding = reference.gt_export(document, [])
        cache_path = root / 'detector' / (row['image_id'] + '.npz')
        if sha(cache_path) != frame['detector_sha256']:
            raise ValueError('Detector cache changed')
        cache = archive(cache_path)
        if len(updated['jobs']) != len(cache['ids']):
            raise ValueError('Detector ROI job count changed')
        for job in updated['jobs']:
            if sha(job['crop_path']) != job['crop_sha256']:
                raise ValueError('Frozen RGB crop changed')
        binding['matches'] = reference.bbox_matches(updated, cache)
        gt_path = root / 'GT' / (row['image_id'] + '.npz')
        npz(gt_path, **arrays)
        binding.update(npz_sha256=sha(gt_path), annotation_sha256=row['annotation_sha256'], excluded_shape_indices=[])
        atomic(gt_path.with_suffix('.json'), binding)
        updated.update(gt_npz_sha256=binding['npz_sha256'], gt_binding_sha256=sha(gt_path.with_suffix('.json')))
        frames.append(updated)
        summary.update(images=1, GT_people=binding['eligible_GT_people'], VOID_pixels=binding['shared_VOID_pixels'])
        summary.update(binding['eligible_parts'])
        support.update(binding['GT_support'])
        if number % 20 == 0:
            atomic(root / 'PREPARATION.json', dict(status='PREPARING_GT', pid=os.getpid(), completed_images=number, expected_images=600))
    manifest = dict(previous, frames=frames)
    atomic(root / 'MANIFEST.json', manifest)
    gt_summary = dict(summary, GT_support=dict(support), target_ID_names={'1': 'Face_local', '2': 'Hair', '3': 'Upper', '4': 'Lower'})
    atomic(root / 'GT_BINDINGS.json', gt_summary)
    approved = delivery['shared_reference_GT']
    if summary['GT_people'] != approved['eligible_people'] or {str(t): summary[str(t)] for t in range(1, 5)} != approved['eligible_parts_after_shared_domain']:
        raise ValueError('GT preflight denominators changed')
    if dict(support) != approved['GT_support_after_shared_domain']:
        raise ValueError('GT preflight positive pixel support changed')
    protocol = copy.deepcopy(old['protocol'])
    protocol.update(protocol='CCTV600_ANNOTATED_PERSON_PARTS_FRESH', status='FROZEN_BEFORE_NEW_INFERENCE',
                    parser_prediction_reuse=0, dataset_membership_sha256=digest,
                    cross_confusion='positive GT person-part pixels; rows Hair/Face_local/Upper/Lower, columns targets plus prediction OTHER',
                    missed_parts='eligible positive GT person-parts with TP=0, including missed GT people',
                    full_scene_person_AP_precision_F1=None,
                    unmatched_predictions='diagnostics; partial-person inventory prevents confirmed global FP status')
    protocol['GT']['valid'] = 'original grid except shared explicit ignore, under50 GT boxes, class/owner conflicts and degenerate geometry; fresh GT export, no historical shape-index exclusions'
    atomic(root / 'PROTOCOL.json', protocol)
    sources = dict(old['source_hashes'])
    for path in (Path(__file__).resolve(), Path(scoped_metrics.__file__).resolve(), PROJECT / 'src/scoped_acceptance.py'):
        sources[str(path.relative_to(WORKSPACE))] = sha(path)
    config = dict(old, root=str(root), run_id=root.name, source_parent=str(base), parent_segment=str(native_parent),
                  protocol=protocol, protocol_sha256=sha(root / 'PROTOCOL.json'), source_hashes=sources,
                  manifest_sha256=sha(root / 'MANIFEST.json'), input_membership_sha256=digest,
                  GT_bindings_sha256=sha(root / 'GT_BINDINGS.json'),
                  GT_packet_bindings={f['image_id']: f['gt_npz_sha256'] for f in frames},
                  parser_prediction_reuse=0, fresh_inference=True,
                  orchestration_amendment='all nine parser quality predictions collected fresh; current GT; scoped reporting; unchanged model inference/resource/timing policy')
    atomic(root / 'CONFIG.json', config)
    atomic(root / 'HANDOVER.json', dict(parent_root=str(base), current_worker=dict(pid=0), fresh_parser_run=True))
    for mid in config['models']:
        (root / 'providers' / mid / 'roi_packets').mkdir(parents=True)
    atomic(root / 'PREPARATION.json', dict(status='VERIFYING_MODEL_BINDINGS', pid=os.getpid(), completed_images=600))
    environments = verify_models(config)
    atomic(root / 'MODEL_READINESS.json', dict(status='PASS', observed_environments=environments, at_utc=now()))
    from scoped_acceptance import acceptance
    acceptance()
    atomic(root / 'CPU_ACCEPTANCE.json', dict(status='PASS', config_sha256=sha(root / 'CONFIG.json'),
                                              protocol_sha256=config['protocol_sha256'], model_calls=0, at_utc=now()))
    atomic(root / 'SCIENTIFIC_FREEZE.json', dict(config_sha256=sha(root / 'CONFIG.json'), source_hashes=sources,
                                                parser_prediction_reuse=0, protocol_sha256=config['protocol_sha256'], at_utc=now()))
    global_guard(root, config['resources'])
    atomic(root / 'PREPARATION.json', dict(status='READY_TO_LAUNCH', completed_images=600, pid=os.getpid(), at_utc=now()))
    return root


def native_config(root, mid, reconciled, timing=False):
    path = ORIGINAL_CONFIG(root, mid, reconciled, timing)
    current = load(root / 'CONFIG.json')
    native = load(path)
    native['input_membership_sha256'] = current['input_membership_sha256']
    native['GT_bindings_sha256'] = current['GT_bindings_sha256']
    native['GT_packet_bindings'] = current['GT_packet_bindings']
    native['parser_prediction_reuse'] = 0
    native['protocol']['dataset_membership_sha256'] = current['input_membership_sha256']
    native['protocol']['GT'] = dict(role='GT_NOT_USED_FOR_INFERENCE', canonical_GT_config_sha256=sha(root / 'CONFIG.json'))
    native['protocol']['metrics'] = dict(role='NATIVE_COLLECTION_ONLY', canonical_scorer='scoped_metrics.py')
    atomic(path, native)
    return path


def execute(root, mid, config_path, round_number=None):
    config = load(root / 'CONFIG.json')
    command = [str(WORKSPACE / config['models'][mid]['environment'] / 'bin/python'),
               str(Path(collection.__file__)), '--worker', '--root', str(config_path), '--model', mid]
    label = mid
    if round_number is not None:
        command += ['--timing-round', str(round_number)]
        label += '-timing-' + str(round_number)
    path = root / 'logs' / (label + '.log')
    path.parent.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    global_guard(root, config['resources'])
    with path.open('xb') as log:
        child = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(root / 'CURRENT_WORKER.json', dict(model_id=mid, pid=child.pid, command=command,
                                                log=str(path), timing_round=round_number, config=str(config_path), at_utc=now()))
        started = time.monotonic()
        try:
            while True:
                try:
                    returncode = child.wait(timeout=20)
                    global_guard(root, config['resources'])
                    return returncode
                except subprocess.TimeoutExpired:
                    global_guard(root, config['resources'])
                    if time.monotonic() - started > config['resources']['max_model_hours'] * 3600 + 300:
                        raise SystemExit('GLOBAL_MODEL_EXECUTION_TIME_GUARD')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise


ORIGINAL_CONFIG = collection.collection_config
ORIGINAL_ASSEMBLE = collection.assemble


def assemble(root, mid, reconciled, directory):
    result = ORIGINAL_ASSEMBLE(root, mid, reconciled, directory)
    manifest = load(root / 'MANIFEST.json')
    output = directory / mid if directory else root / 'providers' / mid
    state = load(output / 'STATE.json') if (output / 'STATE.json').exists() else {}
    frames = [p.parent.name for p in (output / 'packets').glob('*/COMMIT.json')]
    rois = [p.parent.name for p in (output / 'roi_packets').glob('*/COMMIT.json')]
    jobs = [j['job_id'] for f in manifest['frames'] for j in f['jobs']]
    result['image_membership'] = membership(manifest['expected_ids'], frames)
    result['ROI_membership'] = membership(jobs, rois)
    result['failed'] = state.get('failed', 0)
    result['invalid'] = max(result.get('invalid', 0), state.get('invalid', 0))
    result['setup_failures'] = state.get('setup_failures', 0)
    result['partial_packet_paths'] = [str(p) for p in output.glob('**/.partial-*')]
    result['native_state'] = state.get('status')
    result['parser_prediction_reuse'] = 0
    if (not result['image_membership']['strict_complete'] or not result['ROI_membership']['strict_complete']
            or result['failed'] or result['invalid'] or result['setup_failures'] or result['partial_packet_paths']
            or state.get('status') != 'PREDICTIONS_COMPLETE'):
        result.update(status='INCOMPLETE', primary=None)
    atomic(root / 'models' / mid / 'COMPLETION.json', result)
    return result


def controller(root):
    try:
        collection.provider_roots = lambda directory: {mid: directory / 'providers' / mid for mid in load(directory / 'CONFIG.json')['models']}
        collection.collection_config = native_config
        collection.execute = execute
        collection.assemble = assemble
        collection.score = scoped_metrics.score
        reference.report = scoped_metrics.report
        collection.controller(root)
    except BaseException:
        atomic(root / 'CONTROLLER_FAILURE.json', dict(status='STOPPED_REQUIRES_RECONCILIATION', at_utc=now(), error=traceback.format_exc()))
        if (root / 'BATCH_STATE.json').exists():
            state = load(root / 'BATCH_STATE.json')
            state.update(status='STOPPED_REQUIRES_RECONCILIATION', ended_utc=now())
            atomic(root / 'BATCH_STATE.json', state)
        raise


def launch(root):
    assert_idle()
    session = 'cctv600_fresh_' + root.name.split('-')[-1]
    command = [str(Path(sys.executable).absolute()), '-u', str(Path(__file__).resolve()), '--controller', str(root)]
    log = root / 'controller.log'
    commit = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
    record = dict(run_id=root.name, command=command, launch_commit=commit, tmux_session=session,
                  controller_log=str(log), at_utc=now(), config_sha256=sha(root / 'CONFIG.json'),
                  parser_prediction_reuse=0)
    atomic(root / 'LAUNCH.json', record)
    archive_index = PROJECT / 'outputs/ARCHIVED_RUNS.json'
    history = load(archive_index) if archive_index.exists() else []
    history.append(dict(previous_current=load(PROJECT / 'outputs/CURRENT.json'), superseded_at_utc=now(),
                        reason='user requested fresh parsing predictions for all nine models', successor=str(root)))
    atomic(archive_index, history)
    atomic(PROJECT / 'outputs/CURRENT.json', dict(root=str(root), run_id=root.name, fresh_parser_predictions=True))
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(WORKSPACE),
                    'exec ' + shlex.join(command) + ' > ' + shlex.quote(str(log)) + ' 2>&1'], check=True)
    subprocess.run(['tmux', 'set-option', '-t', session, 'remain-on-exit', 'on'], check=True)
    identity = subprocess.check_output(['tmux', 'display-message', '-p', '-t', session,
                                        '#{session_id} #{pane_pid}'], text=True).strip()
    record['session_and_process'] = identity
    atomic(root / 'LAUNCH.json', record)
    print('Run', root.name, '\ntmux', session, '\nSession/PID', identity, '\nLog', log, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    install_shared_paths()
    if args.self_test:
        from scoped_acceptance import acceptance
        acceptance()
    elif args.controller:
        controller(args.controller)
    elif args.status:
        try:
            while True:
                root = Path(load(PROJECT / 'outputs/CURRENT.json')['root'])
                reference.progress(root)
                if (root / 'CONTROLLER_FAILURE.json').exists():
                    print('Controller stopped. Evidence', root / 'CONTROLLER_FAILURE.json')
                if not args.watch:
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    elif args.start or args.prepare_only:
        with (PROJECT / 'outputs/LAUNCH.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            base = args.base or Path(load(PROJECT / 'outputs/CURRENT.json')['root'])
            root = prepare(base)
            print('Prepared', root, flush=True)
            if args.start:
                launch(root)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
