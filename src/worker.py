"""One model process with immutable ROI/frame packets and bounded GPU calls."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from common import WORKSPACE, archive, atomic, event, install_shared_paths, load, now, sha, verify_sources
from evaluation import compose, project


def gpu():
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    free, utilization = [int(x.strip()) for x in query.split(',')]
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    return dict(free_MiB=free, utilization=utilization,
                external_PIDs=[int(x) for x in rows if x.strip().isdigit() and int(x) != os.getpid()])


class AdapterTimer:
    """Time the executed top-level module boundaries without modifying outputs."""
    def __init__(self, model, torch):
        self.torch, self.intervals, self.handles = torch, [], []
        if model.family == 'QANet':
            modules = [model.model.conv_body_net, model.model.parsing_net]
        elif model.family == 'RP-R-CNN':
            # Native im_detect_bbox/parsing call box_net/parsing_net methods directly.
            # Their nonnested top-level child modules execute instead of root.forward.
            modules = list(model.model.children())
        else:
            modules = [model.model]
        for module in modules:
            self.handles.append(module.register_forward_pre_hook(self.before))
            self.handles.append(module.register_forward_hook(self.after))

    def before(self, module, inputs):
        self.torch.cuda.synchronize()
        self.started = time.perf_counter()

    def after(self, module, inputs, output):
        self.torch.cuda.synchronize()
        self.intervals.append((self.started, time.perf_counter()))

    def measure(self, call):
        self.intervals = []
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = call()
        self.torch.cuda.synchronize()
        ended = time.perf_counter()
        if not self.intervals:
            raise ValueError('Expected actual model module calls were not observed')
        forward = sum(b - a for a, b in self.intervals)
        before = self.intervals[0][0] - started
        after = ended - self.intervals[-1][1]
        timing = dict(adapter_ms=(ended - started) * 1000,
                      pre_module_ms=before * 1000, module_call_ms=forward * 1000,
                      post_module_ms=after * 1000,
                      inter_module_ms=max(0., ended - started - before - after - forward) * 1000,
                      peak_allocated_MiB=self.torch.cuda.max_memory_allocated() / 2**20,
                      peak_reserved_MiB=self.torch.cuda.max_memory_reserved() / 2**20,
                      module_calls=len(self.intervals))
        return result, timing


def run(config_path, mid, timing_round=None):
    import torch
    install_shared_paths()
    from controlled_roi_models import CropModel
    from controlled_roi_core import place
    from common_view.packets import read_packet, write_packet

    config = load(config_path)
    root, spec = Path(config['root']), config['models'][mid]
    protocol = config['protocol']
    manifest = load(root / 'MANIFEST.json')
    selected_frames = manifest['frames'] if timing_round is None else [f for f in manifest['frames'] if f['image_id'] in manifest['timing_ids']]
    call_budget = sum(len(f['jobs']) for f in selected_frames) + (1 if timing_round is None else protocol['timing']['warmup_calls_per_round'])
    if sha(root / 'MANIFEST.json') != config['manifest_sha256']:
        raise ValueError('Manifest drift')
    output = root / mid if timing_round is None else root / 'timing' / ('round-' + str(timing_round)) / mid
    output.mkdir(parents=True, exist_ok=False)
    lock = (root.parent / 'GPU_WORKER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    state = dict(model_id=mid, status='INITIALIZING', pid=os.getpid(), started_utc=now(),
                 complete_frames=0, complete_ROIs=0, model_ROI_API_calls=0,
                 warmup_calls=0, failed=0, invalid=0, setup_failures=0, primary=None,
                 expected_frames=600 if timing_round is None else len(manifest['timing_ids']))
    state['model_call_budget'] = call_budget
    ledger = output / 'ATTEMPTS.jsonl'
    resource = protocol['resources']
    config_hash = sha(config_path)

    def save():
        atomic(output / 'STATE.json', state)

    def guard():
        if shutil.disk_usage(root).free < resource['minimum_free_disk_GiB'] * 2**30:
            raise RuntimeError('MINIMUM_FREE_DISK_GUARD')
        if time.monotonic() - started > resource['max_model_hours'] * 3600:
            raise RuntimeError('MAXIMUM_MODEL_EXECUTION_TIME_GUARD')
        observation = gpu()
        if observation['free_MiB'] < resource['minimum_pre_forward_free_MiB']:
            raise RuntimeError('PRE_FORWARD_FREE_VRAM_GUARD')
        if timing_round is not None and observation['external_PIDs']:
            raise RuntimeError('TIMING_CONTAMINATED_EXTERNAL_PID')
        return observation

    def interrupted(signum, frame):
        raise InterruptedError('Received signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        verify_sources(config)
        initial = guard()
        if initial['free_MiB'] < resource['admission_free_MiB'][mid]:
            raise RuntimeError('PRELOAD_FREE_VRAM_GUARD')
        if timing_round is not None and initial['utilization'] > 5:
            raise RuntimeError('TIMING_GPU_NOT_IDLE_AT_ADMISSION')
        if sha(WORKSPACE / spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('Checkpoint binding mismatch')
        if sha(WORKSPACE / spec['native_config']) != spec['native_config_sha256']:
            raise ValueError('Native configuration binding mismatch')
        head = subprocess.check_output(['git','-C',str(WORKSPACE/spec['source']),'rev-parse','HEAD'],text=True).strip()
        if head != spec['source_commit']:
            raise ValueError('Native source HEAD mismatch')
        packages = '\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions() if d.metadata['Name'])) + '\n'
        if hashlib.sha256(packages.encode()).hexdigest() != config['verified_models'][mid]['package_freeze_sha256']:
            raise ValueError('Model environment package binding changed')
        torch.set_num_threads(2)
        torch.manual_seed(protocol['seed'])
        np.random.seed(protocol['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        sys.path.insert(0, str(WORKSPACE / spec['source']))
        model = CropModel(WORKSPACE, spec, output)
        timer = AdapterTimer(model, torch)
        atomic(output / 'ENVIRONMENT.json', dict(python=sys.version, executable=sys.executable,
            torch=torch.__version__, CUDA=torch.version.cuda, GPU=torch.cuda.get_device_name(0),
            numpy=np.__version__, FP32=True, AMP=False, TF32=False,
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), admission=initial,
            source_bindings_sha256=sha(config_path), launch_commit=load(root/'LAUNCH.json')['launch_commit']))

        def infer(rgb, path, box, job_id, phase):
            observation = guard()
            if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or model.model.training:
                raise ValueError('FP32/no-TF32/eval execution contract changed')
            if state['model_ROI_API_calls'] >= call_budget:
                raise RuntimeError('MODEL_API_CALL_BUDGET_EXHAUSTED')
            event(ledger, event='CALL_STARTED', job_id=job_id, phase=phase, resource=observation)
            state['model_ROI_API_calls'] += 1
            state['current_job'] = job_id
            state['current_phase'] = phase
            save()
            with torch.no_grad():
                result, timing = timer.measure(lambda: model.infer(rgb, path, box))
            if timing_round is not None:
                guard()
            event(ledger, event='CALL_FINISHED', job_id=job_id, phase=phase, timing=timing)
            return result, timing

        warm = config['warmup']
        warm_path = WORKSPACE / warm['image']
        if sha(warm_path) != warm['png_sha256']:
            raise ValueError('TRAIN warmup source changed')
        warm_rgb = np.asarray(Image.open(warm_path).convert('RGB'))
        warmups = 1 if timing_round is None else protocol['timing']['warmup_calls_per_round']
        for index in range(warmups):
            infer(warm_rgb, warm_path, warm['designated_box'], 'TRAIN0000930-' + str(index), 'TRAIN_WARMUP')
            state['warmup_calls'] += 1
        state['status'] = 'QUALITY_RUNNING' if timing_round is None else 'CLEAN_TIMING_RUNNING'
        save()
        frames = manifest['frames']
        if timing_round is not None:
            by_id = {f['image_id']: f for f in frames}
            frames = [by_id[iid] for iid in manifest['timing_ids']]
        packet_ids = []
        retained_bytes = 0
        model_limit = min(resource['max_model_GiB'], resource['max_run_GiB']) * 2**30
        for frame in frames:
            iid = frame['image_id']
            state['current_phase'] = 'VAL_FRAME'
            frame_start = time.perf_counter()
            # Measure source decode too, despite cached lossless crops being model inputs.
            if sha(frame['image_path']) != frame['image_sha256']:
                raise ValueError('Input image drift')
            with Image.open(frame['image_path']) as original:
                original.convert('RGB').load()
            read_end = time.perf_counter()
            cache_path = root / 'detector' / (iid + '.npz')
            if sha(cache_path) != frame['detector_sha256']:
                raise ValueError('Detector cache drift')
            cache = archive(cache_path)
            detector_end = time.perf_counter()
            owned, references, records = [], [], []
            height, width = frame['height'], frame['width']
            for j, job in enumerate(frame['jobs']):
                crop_start = time.perf_counter()
                path = Path(job['crop_path'])
                if sha(path) != job['crop_sha256']:
                    raise ValueError('Crop binding changed')
                rgb = np.asarray(Image.open(path).convert('RGB'))
                if list(rgb.shape[:2]) != job['crop_hw']:
                    raise ValueError('RGB crop geometry mismatch')
                crop_end = time.perf_counter()
                x0, y0, _, _ = job['crop_xyxy']
                box = job['bbox_xyxy']
                local = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
                (native, candidates, details), timing = infer(rgb, path, local, job['job_id'], 'MEASURED_ROI')
                ownership_start = time.perf_counter()
                raw, clipped = place(native, job['crop_xyxy'], cache['extents'][j], model.classes)
                owned.append(clipped)
                timing['ownership_ms'] = (time.perf_counter() - ownership_start) * 1000
                timing['crop_read_ms'] = (crop_end - crop_start) * 1000
                if timing_round is None:
                    packet_start = time.perf_counter()
                    roi_path = write_packet(output / 'roi_packets', job['job_id'],
                        dict(native_crop_mask=native, native_candidate_masks=candidates),
                        dict(model_id=mid, job_id=job['job_id'], image_id=iid, person_id=job['person_id'],
                             crop_xyxy=job['crop_xyxy'], crop_sha256=job['crop_sha256'],
                             checkpoint_sha256=spec['checkpoint_sha256'], config_sha256=config_hash,
                             native_classes=model.classes, native_geometry_and_selection=details,
                             precision='FP32', GT_used_in_prediction=False), {})
                    metadata, readback = read_packet(roi_path, dict(config_sha256=config_hash))
                    if not np.array_equal(native, readback['native_crop_mask']):
                        raise ValueError('Atomic native ROI readback failed')
                    timing['roi_packet_write_and_validation_ms'] = (time.perf_counter() - packet_start) * 1000
                    references.append(dict(job_id=job['job_id'], packet_sha256=metadata['packet_sha256'],
                                           metadata_sha256=sha(roi_path / 'packet.json')))
                    retained_bytes += sum(p.stat().st_size for p in roi_path.iterdir())
                records.append(dict(job_id=job['job_id'], timing=timing, selection=details['selection']))
                state['complete_ROIs'] += 1
                save()
                del raw, native, candidates
            packet_started = time.perf_counter()
            planes = np.asarray(owned, np.uint8).reshape(-1, height, width)
            composed_native = compose(planes, cache['scores'], cache['boxes'], cache['ids'], (height, width))
            mapping = protocol['mapping']['SAPIENS29' if model.classes == 29 else 'CIHP20']
            common_mask = project(composed_native, mapping)
            projection_ended = time.perf_counter()
            if timing_round is None:
                packet_path = write_packet(output / 'packets', iid,
                    dict(person_semantic_masks=planes, composed_native_mask=composed_native, common_mask=common_mask,
                         person_extents=cache['extents'], person_scores=cache['scores'],
                         bboxes_xyxy=cache['boxes'], person_ids=cache['ids']),
                    dict(model_id=mid, image_id=iid, original_wh=[width, height], input_sha256=frame['image_sha256'],
                         config_sha256=config_hash, protocol_sha256=config['protocol_sha256'],
                         checkpoint_sha256=spec['checkpoint_sha256'], detector_sha256=frame['detector_sha256'],
                         ROI_references=references, native_classes=model.classes,
                         GT_used_in_prediction=False, precision='FP32'), {})
                metadata, arrays = read_packet(packet_path, dict(config_sha256=config_hash))
                if not np.array_equal(arrays['person_extents'], cache['extents']):
                    raise ValueError('Detector ownership changed')
                if np.any(arrays['person_semantic_masks'][~arrays['person_extents']]):
                    raise ValueError('Parsing escaped declared ownership')
                retained_bytes += sum(p.stat().st_size for p in packet_path.iterdir())
            packet_ended = time.perf_counter()
            image_record = dict(image_id=iid, ROI_calls=len(records), records=records,
                source_read_ms=(read_end - frame_start) * 1000,
                cached_detector_read_ms=(detector_end - read_end) * 1000,
                parsing_work_ms=sum(r['timing']['adapter_ms'] for r in records),
                composition_projection_ms=(projection_ended - packet_started) * 1000,
                frame_packet_write_and_validation_ms=(packet_ended - projection_ended) * 1000,
                application_from_cached_inputs_ms=(packet_ended - frame_start) * 1000,
                includes_new_YOLO_inference=False, resource_after=gpu())
            atomic(output / 'image_records' / (iid + '.json'), image_record)
            if retained_bytes > model_limit:
                raise RuntimeError('RETAINED_MODEL_STORAGE_GUARD')
            if state['complete_frames'] % 10 == 0:
                actual_bytes = int(subprocess.check_output(['du','-sb',str(root)],text=True).split()[0])
                if actual_bytes > resource['max_run_GiB'] * 2**30:
                    raise RuntimeError('TOTAL_RUN_STORAGE_GUARD')
            packet_ids.append(iid)
            state['complete_frames'] += 1
            state['latest_image'] = iid
            save()
            event(ledger, event='FRAME_COMMITTED', image_id=iid, ROI_calls=len(records))
            print(mid, state['complete_frames'], '/', state['expected_frames'], iid, flush=True)
            del owned, cache, planes, composed_native, common_mask
        from common import membership
        expected = manifest['expected_ids'] if timing_round is None else manifest['timing_ids']
        checked = membership(expected, packet_ids)
        if not checked['strict_complete']:
            raise ValueError('Exact population membership failure')
        if state['model_ROI_API_calls'] != call_budget:
            raise ValueError('Unexpected model API call count')
        state.update(status='PREDICTIONS_COMPLETE' if timing_round is None else 'TIMING_COMPLETE', ended_utc=now())
        atomic(output / 'COMPLETION.json', dict(**checked, failed=0, invalid=0, primary=None,
               prediction_status=state['status'], config_sha256=config_hash,
               retained_bytes=retained_bytes, model_calls=state['model_ROI_API_calls'],
               warmup_calls=state['warmup_calls'], scoring_status='BATCH_END_SCORING_ONLY'))
    except BaseException as error:
        is_val = state.get('current_phase') in ('MEASURED_ROI','VAL_FRAME')
        state.update(status='STOPPED_INCOMPLETE', ended_utc=now(), failed=int(is_val),
                     setup_failures=int(not is_val), invalid=int(is_val and isinstance(error,ValueError)),
                     error=traceback.format_exc())
        atomic(output / 'FAILURE.json', state)
        event(ledger, event='ATTEMPT_STOPPED', traceback=state['error'], automatic_retry=False)
        print(state['error'], flush=True)
    finally:
        save()
    return 0 if state['status'] in ('PREDICTIONS_COMPLETE', 'TIMING_COMPLETE') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--timing-round', type=int)
    args = parser.parse_args()
    sys.exit(run(args.config, args.model, args.timing_round))
