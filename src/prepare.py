"""Freeze CCTV600 inputs and reuse the accepted detector without model calls."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import PROJECT, WORKSPACE, archive, atomic, install_shared_paths, load, now, npz, sha
from evaluation import derive_gt


def prepare(data_root, dispositions_path):
    install_shared_paths()
    from cctv600.packets import read_packet
    from pycocotools import mask as coco_mask
    from sapiens_input_variants import detector_crop

    protocol_path = PROJECT / 'configs/PROTOCOL.json'
    protocol = load(protocol_path)
    parent = WORKSPACE / 'CCTV_Human_Parsing_Benchmark/outputs/cctv-controlled-roi-20261101T115653Z/CONFIG.json'
    previous = load(parent)
    detector_root = WORKSPACE / 'YOLO_Instance_Segmentation_CCTV600_Scaling_Study'
    detector_run = Path(load(detector_root / 'outputs/CURRENT.json')['run_dir'])
    detector = load(detector_run / 'PREPARED.json')
    detector_packets = WORKSPACE / 'YOLO_Large_Seg_CCTV600_Benchmark/outputs' / detector['run_id'] / 'yolo26x-seg'
    audit = load(detector_packets / 'PACKET_AUDIT.json')
    if audit['state'] != 'PASS' or audit['membership']['completed'] != 600:
        raise ValueError('Accepted detector packet audit required')
    rows = sorted(csv.DictReader((data_root / 'manifest.csv').open()), key=lambda r: r['image_id'])
    dispositions = load(dispositions_path)
    if dispositions['status'] != 'REVIEWED_SHARED_QA_VOID' or len(dispositions['shapes']) != 1:
        raise ValueError('Reviewed GT disposition required before preparation')
    logical = [[r['image_id'], r['image_sha256'], r['annotation_sha256']] for r in rows]
    digest = hashlib.sha256(json.dumps(logical, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if digest != protocol['dataset_membership_sha256'] or len({r['image_id'] for r in rows}) != 600:
        raise ValueError('Dataset membership mismatch')
    run_id = 'parsing600-' + now().replace('-', '').replace(':', '').replace('.', '').replace('+0000', 'Z')
    root = PROJECT / 'outputs' / run_id
    root.mkdir(parents=True, exist_ok=False)
    atomic(root / 'PREPARATION.json', dict(status='PREPARING', pid=__import__('os').getpid(), started_utc=now()))
    source_hashes = {}
    for relative, expected in previous['source_hashes'].items():
        if relative.endswith(('controlled_roi.py', 'benchmark.py')):
            continue
        actual = sha(WORKSPACE / relative)
        if actual != expected:
            raise ValueError('Shared scientific adapter changed since accepted ROI run: ' + relative)
        source_hashes[relative] = actual
    for relative in ('src', 'configs'):
        for path in (PROJECT / relative).rglob('*'):
            if path.suffix in ('.py', '.json'):
                source_hashes[str(path.relative_to(WORKSPACE))] = sha(path)
    identity_fields = ('model_id','family','checkpoint','checkpoint_sha256','checkpoint_bytes',
                       'source','source_commit','native_config','native_config_sha256',
                       'environment','native_semantic_classes')
    models = {mid: dict({key: value for key, value in spec.items() if key in identity_fields},
                       route=protocol['route']['name']) for mid,spec in previous['models'].items()}
    verified = {}
    environments = {}
    # These shims take precedence over native source on sys.path.
    shim_roots = [WORKSPACE / '.venvs/parser-build-tools' / name for name in ('QANet', 'RP-R-CNN', 'm2fp_ops')]
    shim_roots.append(WORKSPACE / 'Human_Parsing_MHPv2_Study/provenance/RUN_MANIFESTS/implementation-handoff-20261015T063548Z/native_smoke/scripts')
    for directory in shim_roots:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.so'):
                source_hashes[str(path.relative_to(WORKSPACE))] = sha(path)
    # Verify each distinct weight once, plus source HEAD and native config.
    for mid, spec in models.items():
        if sha(WORKSPACE / spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('Checkpoint mismatch: ' + mid)
        if sha(WORKSPACE / spec['native_config']) != spec['native_config_sha256']:
            raise ValueError('Native config mismatch: ' + mid)
        revision = subprocess.check_output(['git', '-C', str(WORKSPACE / spec['source']), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != spec['source_commit']:
            raise ValueError('Native source HEAD mismatch: ' + mid)
        # Bind actual tracked files, including local modifications and config dependencies.
        listing = subprocess.check_output(['git', '-C', str(WORKSPACE / spec['source']), 'ls-files', '-z'])
        for name in listing.decode().split('\0'):
            path = WORKSPACE / spec['source'] / name
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.json'):
                source_hashes[str(path.relative_to(WORKSPACE))] = sha(path)
        python = WORKSPACE / spec['environment'] / 'bin/python'
        if spec['environment'] not in environments:
            probe = subprocess.check_output([str(python), '-c',
                "import json,sys,torch,numpy; print(json.dumps(dict(python=sys.version,executable=sys.executable,torch=torch.__version__,CUDA=torch.version.cuda,numpy=numpy.__version__,cuda_available=torch.cuda.is_available())))"], text=True)
            environment = json.loads(probe.strip().splitlines()[-1])
            freeze = subprocess.check_output([str(python), '-c',
                "import importlib.metadata as m; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name'])))"], text=True)
            env_path = root / 'environments' / (Path(spec['environment']).name + '.txt')
            env_path.parent.mkdir(exist_ok=True)
            env_path.write_text(freeze)
            environments[spec['environment']] = dict(environment=environment, package_freeze_sha256=sha(env_path))
        environment = environments[spec['environment']]['environment']
        if not environment['cuda_available']:
            raise ValueError('Environment CUDA unavailable: ' + mid)
        verified[mid] = dict(**environments[spec['environment']],
                             checkpoint_sha256=spec['checkpoint_sha256'], source_commit=revision)
    frames = []
    roi_count = raw_bytes = 0
    gt_summary = dict(valid_pixels=0, conflict_pixels=0, QA_excluded_shapes=0,
                      support=[0] * 5, eligible_person_parts=[0] * 4, eligible_boxes=0,
                      identical_GT_box_pairs=0, identical_detector_box_pairs=0)
    for number, (row, frame) in enumerate(zip(rows, detector['frames'])):
        iid = row['image_id']
        if iid != frame['image_id'] or row['image_sha256'] != frame['image_sha256']:
            raise ValueError('Detector population mismatch')
        for key in ('image', 'annotation'):
            if sha(row[key + '_path']) != row[key + '_sha256']:
                raise ValueError('Dataset source drift: ' + iid)
        annotation = load(row['annotation_path'])
        decisions = [d for d in dispositions['shapes'] if d['annotation_sha256'] == row['annotation_sha256']]
        excluded = [d['shape_index'] for d in decisions]
        for decision in decisions:
            shape = annotation['shapes'][decision['shape_index']]
            if shape['label'] != decision['source_label'] or shape['group_id'] != decision['source_owner']:
                raise ValueError('Prior visual QA disposition binding mismatch')
        gt, binding = derive_gt(annotation, excluded)
        npz(root / 'GT' / (iid + '.npz'), **gt)
        binding.update(npz_sha256=sha(root / 'GT' / (iid + '.npz')),
                       annotation_sha256=row['annotation_sha256'], excluded_shape_indices=excluded)
        atomic(root / 'GT' / (iid + '.json'), binding)
        gt_summary['valid_pixels'] += binding['valid_pixels']
        gt_summary['conflict_pixels'] += binding['conflict_pixels']
        gt_summary['QA_excluded_shapes'] += len(excluded)
        gt_summary['support'] = [a + b for a, b in zip(gt_summary['support'], binding['support'])]
        gt_summary['eligible_person_parts'] = [a + sum(p['support'][c] > 0 for p in binding['persons'])
                                             for c, a in enumerate(gt_summary['eligible_person_parts'])]
        gt_summary['eligible_boxes'] += len(frame['people'])
        from collections import Counter
        gt_summary['identical_GT_box_pairs'] += sum(n*(n-1)//2 for n in Counter(tuple(p['bbox']) for p in frame['people']).values())
        packet_path = detector_packets / 'predictions' / (iid + '.json.gz')
        if sha(packet_path) != audit['packet_sha256'][iid]:
            raise ValueError('Detector packet changed since accepted audit')
        record = read_packet(packet_path, frame, audit['binding'])
        retained = [p for p in record['predictions'] if p['confidence'] >= .25]
        gt_summary['identical_detector_box_pairs'] += sum(n*(n-1)//2 for n in Counter(tuple(p['bbox_xyxy']) for p in retained).values())
        rgb = np.asarray(Image.open(row['image_path']).convert('RGB'))
        if rgb.shape[:2] != (int(row['height']), int(row['width'])):
            raise ValueError('Source RGB geometry')
        extents, jobs = [], []
        for prediction in retained:
            rle = dict(prediction['rle'], counts=prediction['rle']['counts'].encode('ascii'))
            extent = coco_mask.decode(rle).astype(bool)
            crop, box = detector_crop(rgb, prediction['bbox_xyxy'], extent, context=.1)
            job_id = iid + '-person-' + str(prediction['person_id'])
            image_path = root / 'crops' / (job_id + '.png')
            image_path.parent.mkdir(exist_ok=True)
            Image.fromarray(crop).save(image_path)
            jobs.append(dict(job_id=job_id, person_id=prediction['person_id'],
                             crop_xyxy=box, crop_hw=list(crop.shape[:2]),
                             crop_path=str(image_path), crop_sha256=sha(image_path),
                             bbox_xyxy=prediction['bbox_xyxy'], confidence=prediction['confidence']))
            extents.append(extent)
            raw_bytes += crop.shape[0] * crop.shape[1] * 3
        npz(root / 'detector' / (iid + '.npz'),
            extents=np.asarray(extents, bool).reshape(-1, *rgb.shape[:2]),
            scores=np.array([p['confidence'] for p in retained], np.float32),
            boxes=np.array([p['bbox_xyxy'] for p in retained], np.float64).reshape(-1, 4),
            ids=np.array([p['person_id'] for p in retained], np.int32))
        raw_bytes += len(jobs) * rgb.shape[0] * rgb.shape[1]
        frames.append(dict(frame, jobs=jobs, detector_sha256=sha(root / 'detector' / (iid + '.npz')),
                           gt_npz_sha256=binding['npz_sha256'], gt_binding_sha256=sha(root / 'GT' / (iid + '.json')),
                           parent_detector_packet_sha256=audit['packet_sha256'][iid]))
        roi_count += len(jobs)
        if (number + 1) % 50 == 0:
            atomic(root / 'PREPARATION.json', dict(status='PREPARING', prepared_frames=number + 1, ROIs=roi_count))
            print('Prepared', number + 1, 'frames,', roi_count, 'ROIs', flush=True)
    manifest = dict(frames=frames, expected_ids=[f['image_id'] for f in frames], ROI_jobs=roi_count,
                    timing_ids=detector['sample'], visual_ids=detector['visual_sample'],
                    detector_parent=str(detector_run), detector_audit_sha256=sha(detector_packets / 'PACKET_AUDIT.json'))
    atomic(root / 'MANIFEST.json', manifest)
    atomic(root / 'GT_BINDINGS.json', gt_summary)
    if gt_summary['QA_excluded_shapes'] != len(dispositions['shapes']):
        raise ValueError('Not every reviewed QA disposition matched the frozen dataset')
    config = dict(run_id=run_id, created_utc=now(), root=str(root), protocol=protocol,
                  protocol_sha256=sha(protocol_path), manifest_sha256=sha(root / 'MANIFEST.json'),
                  GT_bindings_sha256=sha(root / 'GT_BINDINGS.json'), models=models, warmup=previous['warmup'],
                  source_hashes=source_hashes, verified_models=verified,
                  input_membership_sha256=digest, reused_detector_images=600, parser_prediction_reuse=0,
                  parent_ROI_adapter_config_sha256=sha(parent), estimated_uncompressed_arrays_per_model=raw_bytes,
                  QA_dispositions_sha256=sha(dispositions_path),
                  estimates_exclude_variable_native_candidate_counts=True,
                  amendment='New CCTV600 population and annotation-supported domain. Existing ROI model behavior preserved. No full-frame semantic certification.')
    atomic(root / 'CONFIG.json', config)
    atomic(root / 'PREPARATION.json', dict(status='PREPARED_NOT_LAUNCHED', frames=600, ROI_jobs=roi_count, at_utc=now()))
    atomic(PROJECT / 'outputs/CURRENT.json', dict(run_id=run_id, root=str(root), config_sha256=sha(root / 'CONFIG.json')))
    print(json.dumps(dict(run_id=run_id, root=str(root), ROI_jobs=roi_count, GT=gt_summary)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--QA-dispositions', dest='dispositions', type=Path, required=True)
    existing = set((PROJECT/'outputs').glob('parsing600-*'))
    try:
        args = parser.parse_args()
        prepare(args.data_root,args.dispositions)
    except BaseException:
        import traceback
        for directory in set((PROJECT/'outputs').glob('parsing600-*')) - existing:
            atomic(directory/'PREPARATION_FAILURE.json',dict(status='PREPARATION_FAILED',
                   traceback=traceback.format_exc(), model_calls=0, at_utc=now()))
        raise
