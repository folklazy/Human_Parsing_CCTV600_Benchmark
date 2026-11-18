"""Controlled completion of the interrupted 1B observation in the fresh batch."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from common import PROJECT, WORKSPACE, archive, atomic, event, install_shared_paths, load, membership, now, sha, verify_sources
import reference_collection as collection
import reference_parts600 as reference
import scoped_metrics
import scoped_run

MODEL = 'Sapiens2_1b'
ORIGINAL_RUNTIME = scoped_metrics.runtime


def stable_size(path, runner=subprocess.run, attempts=3):
    """Require a successful scan, retrying only an atomic-rename ENOENT race."""
    for attempt in range(attempts):
        result = runner(['du', '-sb', str(path)], capture_output=True, text=True,
                        env=dict(os.environ, LC_ALL='C'))
        if result.returncode == 0:
            return int(result.stdout.split()[0])
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        transient = result.returncode == 1 and lines and all(line.endswith('No such file or directory') for line in lines)
        if not transient or attempt + 1 == attempts:
            raise RuntimeError('Storage inspection failed: ' + result.stderr.strip())
        time.sleep(.05)
    raise AssertionError('Bounded scan loop exhausted')


def guard(root, resources):
    if shutil.disk_usage(root).free < resources['minimum_free_disk_GiB'] * 2**30:
        raise SystemExit('GLOBAL_MINIMUM_FREE_DISK_GUARD')
    parent_bytes = load(root / 'CONFIG.json')['continuation']['parent_retained_bytes']
    used = stable_size(root) + parent_bytes
    if used > resources['max_run_GiB'] * 2**30:
        raise SystemExit('GLOBAL_RUN_STORAGE_GUARD')


def acceptance():
    calls = []

    def temporary_race(*args, **kwargs):
        calls.append(True)
        if len(calls) < 3:
            return subprocess.CompletedProcess(args[0], 1, '10 folder\n', "du: cannot access 'folder/.partial-x': No such file or directory\n")
        return subprocess.CompletedProcess(args[0], 0, '1234 folder\n', '')

    assert stable_size(Path('synthetic'), runner=temporary_race) == 1234 and len(calls) == 3
    for diagnostic in ('Permission denied', 'Input/output error'):
        calls.clear()

        def permanent_error(*args, **kwargs):
            calls.append(True)
            return subprocess.CompletedProcess(args[0], 1, '10 folder\n', diagnostic)

        try:
            stable_size(Path('synthetic'), runner=permanent_error)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Storage errors must propagate')
        assert len(calls) == 1
    calls.clear()

    def repeated_race(*args, **kwargs):
        calls.append(True)
        return subprocess.CompletedProcess(args[0], 1, '10 folder\n', "du: cannot access 'x': No such file or directory")

    try:
        stable_size(Path('synthetic'), runner=repeated_race)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Repeated failures must stop at the bound')
    assert len(calls) == 3
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        atomic(root / 'QUALITY_TIMING_OBSERVATIONS.json', {
            'old': dict(timing=dict(adapter_ms=10, peak_allocated_MiB=100, peak_reserved_MiB=200)),
            'new': dict(timing=dict(adapter_ms=20, peak_allocated_MiB=110, peak_reserved_MiB=220)),
        })
        frames = [dict(image_id='one', jobs=[dict(job_id='old'), dict(job_id='new')]),
                  dict(image_id='empty', jobs=[])]
        value, images = continued_runtime(root, root / 'collections' / MODEL / MODEL, frames)
        assert value['mean_ms_per_ROI'] == 15 and value['mean_s_per_image'] == .015
        assert value['total_s'] == .03 and value['peak_allocated_MiB'] == 110
        assert value['application_from_cached_inputs_total_s'] is None and images[1]['parsing_work_s'] == 0
        atomic(root / 'QUALITY_TIMING_OBSERVATIONS.json', {})
        assert continued_runtime(root, root / 'collections' / MODEL / MODEL, frames)[0]['total_s'] is None
        atomic(root / 'CONFIG.json', dict(continuation=dict(parent_retained_bytes=79 * 2**30)))
        limits = dict(minimum_free_disk_GiB=50, max_run_GiB=80)
        with patch.object(shutil, 'disk_usage', return_value=SimpleNamespace(free=50 * 2**30)):
            with patch.dict(globals(), stable_size=lambda path: 2**30):
                guard(root, limits)
            with patch.dict(globals(), stable_size=lambda path: 2**30 + 1):
                try:
                    guard(root, limits)
                except SystemExit as error:
                    assert str(error) == 'GLOBAL_RUN_STORAGE_GUARD'
                else:
                    raise AssertionError('Storage limit was relaxed')
        with patch.object(shutil, 'disk_usage', return_value=SimpleNamespace(free=50 * 2**30 - 1)):
            try:
                guard(root, limits)
            except SystemExit as error:
                assert str(error) == 'GLOBAL_MINIMUM_FREE_DISK_GUARD'
            else:
                raise AssertionError('Free-disk guard was relaxed')
    print('Storage race CPU acceptance PASS: successful rescan, permanent errors and bounded attempts', flush=True)


def prepare(parent):
    from common_view.packets import read_packet
    scoped_run.assert_idle()
    original = load(parent / 'CONFIG.json')
    verify_sources(original)
    current = load(WORKSPACE.parent / 'CCTV_Human_Parsing_600.local/CURRENT.json')
    if original['input_membership_sha256'] != current['membership_sha256']:
        raise ValueError('Current GT no longer matches the parent experiment')
    parent_results = load(parent / 'RESULT_INDEX.json')
    for mid, result in parent_results.items():
        if mid == MODEL:
            continue
        completion = load(parent / 'models' / mid / 'COMPLETION.json')
        if result['status'] != 'COMPLETE' or not result['strict_complete'] or result['completed'] != 600:
            raise ValueError('A previously completed model requires separate review: ' + mid)
        if not completion['ROI_membership']['strict_complete'] or result['protocol_sha256'] != original['protocol_sha256']:
            raise ValueError('Parent metric/protocol membership mismatch')
        for number in range(1, 4):
            timed = parent / 'timing_collection' / mid / 'timing' / f'round-{number}' / mid / 'COMPLETION.json'
            if not load(timed)['strict_complete']:
                raise ValueError('Parent timing round is incomplete: ' + str(timed))
    root = PROJECT / 'outputs' / ('cctv600-1b-continuation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root.mkdir()
    for name in ('MANIFEST.json', 'GT', 'GT_BINDINGS.json', 'detector', 'crops', 'environments', 'RESOURCE_PLAN.json'):
        (root / name).symlink_to(parent / name)
    config = copy.deepcopy(original)
    config.update(root=str(root), run_id=root.name, source_parent=str(parent),
                  parser_prediction_reuse_from_pre_experiment=0,
                  continuation=dict(parent_root=str(parent), parent_config_sha256=sha(parent / 'CONFIG.json'),
                                    parent_retained_bytes=stable_size(parent), quality_model_scope=[MODEL],
                                    reason='Resource scanner ENOENT during atomic packet rename stopped the worker',
                                    effective_orchestration='successful du scan required, at most three retries for ENOENT only'))
    public_report = PROJECT / 'reports/CCTV600_RESULTS_TH.md'
    config['continuation']['public_report_before_sha256'] = sha(public_report) if public_report.exists() else None
    config['source_hashes'][str(Path(__file__).resolve().relative_to(WORKSPACE))] = sha(__file__)
    atomic(root / 'CONFIG.json', config)
    for mid in config['models']:
        provider = root / 'providers' / mid
        provider.parent.mkdir(exist_ok=True)
        provider.symlink_to(parent / 'collections' / mid / mid)
        if mid != MODEL:
            output = root / 'models' / mid
            output.mkdir(parents=True)
            for name in ('COMPLETION.json', 'ASSEMBLED_INDEX.json', 'METRICS.json'):
                shutil.copy2(parent / 'models' / mid / name, output / name)
            for directory in ('collections', 'timing_collection'):
                target = root / directory / mid
                target.parent.mkdir(exist_ok=True)
                target.symlink_to(parent / directory / mid)
    collection.provider_roots = lambda directory: {mid: directory / 'providers' / mid for mid in config['models']}
    reconciled = collection.reconcile_model(root, MODEL)
    if reconciled['invalid'] or reconciled['orphan'] or reconciled['partial']:
        raise ValueError('Parent packets have invalid/orphan/partial evidence requiring review')
    parent_native_config = parent / 'collections' / MODEL / 'CONFIG.json'
    source_config_sha = sha(parent_native_config)
    jobs = {j['job_id']: j for frame in load(root / 'MANIFEST.json')['frames'] for j in frame['jobs']}
    for job_id, ref in reconciled['accepted'].items():
        packet = Path(ref['packet']).resolve()
        meta, arrays = read_packet(packet, dict(config_sha256=source_config_sha, precision='FP32', GT_used_in_prediction=False))
        if meta['crop_xyxy'] != jobs[job_id]['crop_xyxy'] or meta['native_classes'] != 29:
            raise ValueError('Parent crop/native label contract changed')
        ref['packet'] = str(packet)
    atomic(root / 'reconciliation' / (MODEL + '.json'), reconciled)
    compatibility = dict(accepted_current_experiment_ROIs=len(reconciled['accepted']), pending_ROIs=len(reconciled['pending']),
                         inherited_model_METRICS_sha256={mid: sha(parent / 'models' / mid / 'METRICS.json')
                                                        for mid in config['models'] if mid != MODEL},
                         unchanged_scientific_bindings=['model', 'checkpoint', 'native_config', 'crop', 'GT', 'ownership', 'precision', 'mapping', 'scorer'],
                         failed_call_history_preserved=str(parent / 'collections' / MODEL / MODEL / 'FAILURE.json'))
    atomic(root / 'CONTINUATION_MANIFEST.json', compatibility)
    acceptance()
    config['continuation'].update(accepted_ROIs=len(reconciled['accepted']), pending_ROIs=len(reconciled['pending']))
    config['parser_prediction_reuse'] = len(reconciled['accepted'])
    config['fresh_inference'] = False
    atomic(root / 'CONFIG.json', config)
    model_readiness = scoped_run.verify_models(dict(config, models={MODEL: config['models'][MODEL]}))
    atomic(root / 'MODEL_READINESS.json', dict(status='PASS', environments=model_readiness, model_scope=[MODEL]))
    atomic(root / 'CPU_ACCEPTANCE.json', dict(status='PASS', model_calls=0, config_sha256=sha(root / 'CONFIG.json'),
                                              inherited_acceptance_sha256=sha(parent / 'CPU_ACCEPTANCE.json')))
    atomic(root / 'SCIENTIFIC_FREEZE.json', dict(config_sha256=sha(root / 'CONFIG.json'),
                                                protocol_sha256=config['protocol_sha256'], source_hashes=config['source_hashes']))
    return root


def timing_observations(root):
    index = load(root / 'models' / MODEL / 'ASSEMBLED_INDEX.json')['ROI_references']
    ledgers = {}
    images = {}
    observations = {}
    for job_id, ref in index.items():
        packet = Path(ref['packet']).resolve()
        directory = packet.parents[1]
        iid = load(packet / 'packet.json')['image_id']
        key = str(directory / 'image_records' / (iid + '.json'))
        if key not in images:
            images[key] = load(key) if Path(key).exists() else {}
        matches = [r for r in images[key].get('records', []) if r['job_id'] == job_id]
        if matches:
            observations[job_id] = dict(timing=matches[0]['timing'], source_image_record=key)
            continue
        path = str(directory / 'ATTEMPTS.jsonl')
        if path not in ledgers:
            finished = {}
            for line in Path(path).read_text().splitlines():
                row = json.loads(line)
                if row.get('event') == 'CALL_FINISHED' and row.get('phase') == 'MEASURED_ROI':
                    finished.setdefault(row['job_id'], []).append(row['timing'])
            ledgers[path] = finished
        values = ledgers[path].get(job_id, [])
        observations[job_id] = dict(timing=values[0] if len(values) == 1 else None,
                                     source_ledger=path, missing_reason=None if len(values) == 1 else 'NO_UNAMBIGUOUS_FINISHED_CALL')
    atomic(root / 'QUALITY_TIMING_OBSERVATIONS.json', observations)


def continued_runtime(root, directory, frames):
    if directory != root / 'collections' / MODEL / MODEL:
        return ORIGINAL_RUNTIME(directory, frames)
    observations = load(root / 'QUALITY_TIMING_OBSERVATIONS.json')
    per_image = []
    total = 0.0
    measured = 0
    peaks = {'peak_allocated_MiB': None, 'peak_reserved_MiB': None}
    for frame in frames:
        timings = [observations.get(j['job_id'], {}).get('timing') for j in frame['jobs']]
        complete = all(t is not None and t.get('adapter_ms') is not None for t in timings)
        value = sum(t['adapter_ms'] for t in timings) if complete else None
        if complete:
            total += value
            measured += len(timings)
            for timing in timings:
                for key in peaks:
                    if timing.get(key) is not None:
                        peaks[key] = max(peaks[key] or 0, timing[key])
        per_image.append(dict(image_id=frame['image_id'], ROI_count=len(frame['jobs']),
                              timed_ROIs=len(timings) if complete else 0,
                              parsing_work_s=value / 1000 if complete else None))
    complete_images = sum(row['parsing_work_s'] is not None for row in per_image)
    complete = complete_images == len(frames)
    result = dict(complete=complete, images_with_complete_timer=complete_images, expected_images=len(frames),
                  observed_ROIs=measured, mean_ms_per_ROI=total / measured if complete and measured else None,
                  mean_s_per_image=total / (1000 * len(frames)) if complete and frames else None,
                  total_s=total / 1000 if complete else None, observed_totals_ms=dict(adapter_ms=total), **peaks,
                  application_from_cached_inputs_total_s=None,
                  component_limitation='quality adapter intervals assembled from accepted ROI observations across two processes; interrupted-frame application/I-O totals unavailable',
                  peak_scope='maximum observed process peak across accepted ROI calls in both quality segments')
    return result, per_image


def publish(root):
    destination = PROJECT / 'reports/CCTV600_RESULTS_TH.md'
    expected = load(root / 'CONFIG.json')['continuation']['public_report_before_sha256']
    actual = sha(destination) if destination.exists() else None
    if actual != expected:
        raise ValueError('Public report was edited after continuation preparation')
    destination.parent.mkdir(exist_ok=True)
    shutil.copy2(root / 'BATCH_REPORT_TH.md', destination)
    subprocess.run(['git', '-C', str(PROJECT), 'add', 'reports/CCTV600_RESULTS_TH.md'], check=True)
    subprocess.run(['git', '-C', str(PROJECT), '-c', 'user.name=Dataset Workspace',
                    '-c', 'user.email=dataset-workspace@localhost', 'commit', '-m',
                    'Publish CCTV600 parsing results after controlled 1B completion', '--', 'reports/CCTV600_RESULTS_TH.md'], check=True)
    subprocess.run(['git', '-C', str(PROJECT), 'push', 'origin', 'HEAD'], check=True)


def controller(root):
    lock = (PROJECT / 'outputs/CONTROLLER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = load(root / 'CONFIG.json')
    verify_sources(config)
    if load(root / 'CPU_ACCEPTANCE.json')['config_sha256'] != sha(root / 'CONFIG.json'):
        raise ValueError('Continuation acceptance identity changed')
    scoped_run.global_guard = guard
    state = dict(status='QUALITY_COLLECTION', pid=os.getpid(), current_model=MODEL, started_utc=now(),
                 models={}, quality_analysis_during_inference=False, continuation_parent=config['source_parent'])
    atomic(root / 'BATCH_STATE.json', state)
    try:
        rec = load(root / 'reconciliation' / (MODEL + '.json'))
        cp = scoped_run.native_config(root, MODEL, rec)
        state['worker_returncode'] = scoped_run.execute(root, MODEL, cp)
        atomic(root / 'BATCH_STATE.json', state)
        result = collection.assemble(root, MODEL, rec, cp.parent)
        # Canonical membership includes accepted parent packets plus new calls.
        worker_state = load(cp.parent / MODEL / 'STATE.json')
        result.update(image_membership=result.get('membership', {}), failed=worker_state['failed'],
                      invalid=worker_state['invalid'], setup_failures=worker_state['setup_failures'],
                      historical_interruption_preserved=True, accepted_parent_ROIs=len(rec['accepted']),
                      parent_failure_class='ORCHESTRATION_RESOURCE_INSPECTION_FAILURE')
        if state['worker_returncode'] or worker_state['failed'] or worker_state['invalid']:
            result.update(status='INCOMPLETE', primary=None)
        atomic(root / 'models' / MODEL / 'COMPLETION.json', result)
        state['models'][MODEL] = result
        state['status'] = 'REFERENCE_SCORING'
        atomic(root / 'BATCH_STATE.json', state)
        results = {mid: load(root / 'models' / mid / 'METRICS.json') for mid in config['models'] if mid != MODEL}
        results[MODEL] = scoped_metrics.score(root, MODEL)
        # Restore frozen roster order, including the previously accepted results.
        results = {mid: results[mid] for mid in config['models']}
        atomic(root / 'RESULT_INDEX.json', results)
        state['timing'] = {}
        if results[MODEL].get('status') == 'COMPLETE':
            timing_observations(root)
            state['status'] = 'DEDICATED_TIMING'
            atomic(root / 'BATCH_STATE.json', state)
            timing_config = scoped_run.native_config(root, MODEL, {}, timing=True)
            for number in range(1, 4):
                state['timing'][str(number) + '/' + MODEL] = scoped_run.execute(root, MODEL, timing_config, number)
                atomic(root / 'BATCH_STATE.json', state)
        else:
            atomic(root / 'QUALITY_TIMING_OBSERVATIONS.json', {})
        scoped_metrics.runtime = lambda directory, frames: continued_runtime(root, directory, frames)
        scoped_metrics.report(root)
        with (root / 'BATCH_REPORT_TH.md').open('a') as stream:
            stream.write('\nSapiens2 1B quality timing รวม adapter intervals ของ accepted ROI observations จากสอง segments '
                         'เวลาของเฟรมที่ถูกขัดจังหวะด้าน application/I-O ให้ null ส่วน dedicated timing ใช้ process ใหม่สามรอบ '
                         'raw interruption และ storage-scanner failure เก็บใน parent records\n')
        reference.gallery(root)
        state.update(status='STOPPED_AFTER_1B_CONTINUATION', ended_utc=now(),
                     complete_models=sum(r.get('status') == 'COMPLETE' for r in results.values()))
        atomic(root / 'BATCH_STATE.json', state)
        atomic(root / 'BATCH_COMPLETION.json', state)
        try:
            publish(root)
            atomic(root / 'PUBLICATION.json', dict(status='PUSHED', at_utc=now(),
                commit=subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()))
        except Exception:
            atomic(root / 'PUBLICATION.json', dict(status='REQUIRES_PUBLICATION_CHECK', at_utc=now(), error=traceback.format_exc()))
    except BaseException:
        state.update(status='STOPPED_REQUIRES_RECONCILIATION', ended_utc=now())
        atomic(root / 'BATCH_STATE.json', state)
        atomic(root / 'CONTROLLER_FAILURE.json', dict(at_utc=now(), error=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    install_shared_paths()
    if args.self_test:
        acceptance()
    elif args.controller:
        controller(args.controller)
    elif args.start:
        with (PROJECT / 'outputs/LAUNCH.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parent = Path(load(PROJECT / 'outputs/CURRENT.json')['root'])
            root = prepare(parent)
            # Use the established detached launcher with this controller entrypoint.
            original_file = scoped_run.__file__
            try:
                scoped_run.__file__ = __file__
                scoped_run.launch(root)
            finally:
                scoped_run.__file__ = original_file
            launch_record = load(root / 'LAUNCH.json')
            launch_record.update(parser_prediction_reuse=load(root / 'CONFIG.json')['parser_prediction_reuse'],
                                 quality_model_scope=[MODEL], continuation_parent=str(parent))
            atomic(root / 'LAUNCH.json', launch_record)
            current = load(PROJECT / 'outputs/CURRENT.json')
            current.update(continuation_parent=str(parent), quality_model_scope=[MODEL])
            atomic(PROJECT / 'outputs/CURRENT.json', current)
            history = load(PROJECT / 'outputs/ARCHIVED_RUNS.json')
            if history[-1]['successor'] == str(root):
                history[-1]['reason'] = 'Controlled 1B continuation after storage-inspection race; eight accepted models retained'
                atomic(PROJECT / 'outputs/ARCHIVED_RUNS.json', history)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
