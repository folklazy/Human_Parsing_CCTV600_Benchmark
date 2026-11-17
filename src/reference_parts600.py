"""CCTV600 adapter for the existing shared person-part and Face-local scorer."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from common import PROJECT, WORKSPACE, archive, atomic, install_shared_paths, load, membership, now, npz, sha
from evaluation import raster, degenerate
install_shared_paths()
from common_view.reference_parts import counts, summarize
from common_view.core import project
from cctv600.scoring import fixed
TARGETS = {'hair': 2, 'face': 1, 'upper': 3, 'lower': 4}

def gt_export(annotation, excluded):
    shape = (annotation['imageHeight'], annotation['imageWidth'])
    wh = shape[::-1]
    valid = np.ones(shape, bool)
    owner = np.zeros(shape, np.int32)
    category = np.zeros(shape, np.uint8)
    conflict = np.zeros(shape, bool)
    people = {}
    for s in annotation['shapes']:
        if s['label'] == 'ignore':
            valid &= ~raster(s, wh)
        elif s['label'] == 'bbox':
            xs, ys = zip(*s['points'])
            if max(max(xs) - min(xs), max(ys) - min(ys)) >= 50:
                people[s['group_id']] = [min(xs), min(ys), max(xs), max(ys)]
            else:
                valid &= ~raster(s, wh)
    parts = {}
    for i, s in enumerate(annotation['shapes']):
        if s['label'] not in TARGETS:
            continue
        mask = raster(s, wh)
        if i in excluded or degenerate(s) or s['group_id'] not in people:
            valid &= ~mask
            continue
        pid, target = (s['group_id'], TARGETS[s['label']])
        key = (pid, target)
        parts[key] = parts.get(key, np.zeros(shape, bool)) | mask
        conflict |= mask & (category > 0) & ((category != target) | (owner != pid))
        free = mask & (category == 0)
        category[free], owner[free] = (target, pid)
    valid &= ~conflict
    keys = sorted((k for k, v in parts.items() if np.any(v & valid)))
    arrays = dict(valid=valid, part_masks=np.asarray([parts[k] for k in keys], bool).reshape(-1, *shape), part_owner_ids=np.array([k[0] for k in keys], np.int32), part_target_ids=np.array([k[1] for k in keys], np.uint8))
    binding = dict(eligible_GT_people=len(people), eligible_parts=dict(Counter((str(k[1]) for k in keys))), GT_support={str(t): sum((int((parts[k] & valid).sum()) for k in keys if k[1] == t)) for t in (1, 2, 3, 4)}, valid_pixels=int(valid.sum()), shared_VOID_pixels=int((~valid).sum()), conflict_pixels=int(conflict.sum()))
    return (arrays, binding)

def bbox_matches(frame, cache):
    predictions = [dict(confidence=float(s), bbox_xyxy=b.tolist()) for s, b in zip(cache['scores'], cache['boxes'])]
    order = sorted(range(len(predictions)), key=lambda i: (-predictions[i]['confidence'], *predictions[i]['bbox_xyxy']))
    result = {}
    duplicates = sum((n * (n - 1) // 2 for n in Counter((tuple(p['bbox']) for p in frame['people'])).values()))
    for threshold in (0.5, 0.75):
        m = fixed(frame, predictions, threshold=threshold)
        result[str(threshold)] = dict(by_GT={str(frame['people'][r['gt_index']]['person_id']): order[r['prediction_index']] for r in m['matches']}, TP=m['tp'], FP=m['fp'], FN=m['fn'], GT=len(frame['people']), predictions=m['tp'] + m['fp'], all_predictions=len(predictions), ignored_predictions=m['ignored_predictions'], ambiguous=int(duplicates > 0), identical_GT_geometry_groups=duplicates)
    return result

def score_frame(gt, matching, planes, space, spec):
    shape = gt['valid'].shape
    if planes.ndim != 3 or planes.shape[1:] != shape:
        raise ValueError('Original-grid native planes required')
    mapped = project(planes, space, spec)
    selected = matching['0.5']['by_GT']
    empty = np.zeros(shape, bool)
    rows = []
    for pid, target, positive in zip(gt['part_owner_ids'], gt['part_target_ids'], gt['part_masks']):
        index = selected.get(str(int(pid)))
        predicted = mapped[index] == int(target) if index is not None else empty
        row = counts(positive, predicted, gt['valid'], local=int(target) == 1)
        if row is not None:
            rows.append(dict(gt_id=int(pid), target=int(target), unmatched=index is None, **row))
    return dict(parts=rows, localization={t: {k: v for k, v in r.items() if k != 'by_GT'} for t, r in matching.items()})

def aggregate(rows, expected):
    result = summarize(rows, expected)
    means = [result['targets'][t]['mean_part_IoU'] for t in ('Hair', 'Face_local', 'Upper', 'Lower')]
    result['Macro4_local'] = float(np.mean(means)) if result['strict_complete'] and all((v is not None for v in means)) else None
    for target, r in result['targets'].items():
        r['TN'] = sum((p['TN'] for frame in rows for p in frame['parts'] if p['target'] == {'Face_local': 1, 'Hair': 2, 'Upper': 3, 'Lower': 4}[target]))
        r['binary_confusion_target_other'] = [[r['TP'], r['FN']], [r['FP'], r['TN']]]
    for threshold in result['localization']:
        loc = result['localization'][threshold]
        loc['ignored_predictions'] = sum((f['localization'][threshold]['ignored_predictions'] for f in rows))
        loc['all_predictions'] = sum((f['localization'][threshold]['all_predictions'] for f in rows))
        loc['precision'] = loc['TP'] / (loc['TP'] + loc['FP']) if loc['TP'] + loc['FP'] else None
        loc['recall'] = loc['TP'] / (loc['TP'] + loc['FN']) if loc['TP'] + loc['FN'] else None
    return result

def prepare(root):
    handover = load(root / 'HANDOVER.json')
    parent = Path(handover['parent_root'])
    old = load(parent / 'CONFIG.json')
    manifest = load(parent / 'MANIFEST.json')
    for name in ('detector', 'crops', 'environments', 'RESOURCE_PLAN.json'):
        (root / name).symlink_to(parent / name)
    (root / 'MANIFEST.json').write_bytes((parent / 'MANIFEST.json').read_bytes())
    spec_path = WORKSPACE / 'Human_Parsing_Benchmark/protocol/CIHP_SAPIENS_COMMON_VIEW_REVISION2.json'
    spec = load(spec_path)
    summary = Counter()
    support = Counter()
    for frame in manifest['frames']:
        iid = frame['image_id']
        if sha(frame['annotation_path']) != frame['annotation_sha256']:
            raise ValueError('Accepted annotation changed')
        previous = load(parent / 'GT' / (iid + '.json'))
        gt, binding = gt_export(load(frame['annotation_path']), previous['excluded_shape_indices'])
        cache = archive(root / 'detector' / (iid + '.npz'))
        binding['matches'] = bbox_matches(frame, cache)
        npz(root / 'GT' / (iid + '.npz'), **gt)
        binding['npz_sha256'] = sha(root / 'GT' / (iid + '.npz'))
        binding['annotation_sha256'] = frame['annotation_sha256']
        atomic(root / 'GT' / (iid + '.json'), binding)
        summary['images'] += 1
        summary['GT_people'] += binding['eligible_GT_people']
        summary['VOID_pixels'] += binding['shared_VOID_pixels']
        for t, n in binding['eligible_parts'].items():
            summary[t] += n
        support.update(binding['GT_support'])
    gt_summary = dict(summary, GT_support=dict(support), target_ID_names={'1': 'Face_local', '2': 'Hair', '3': 'Upper', '4': 'Lower'})
    atomic(root / 'GT_BINDINGS.json', gt_summary)
    old_counts = load(parent / 'GT_BINDINGS.json')['eligible_person_parts']
    if [summary[str(t)] for t in (2, 1, 3, 4)] != old_counts:
        raise ValueError('Positive GT membership changed unexpectedly')
    protocol_path = PROJECT / 'configs/SHARED_REFERENCE_V1.json'
    sources = dict(old['source_hashes'])
    for path in (Path(__file__), PROJECT / 'src/reference_collection.py', protocol_path, WORKSPACE / 'Human_Parsing_Benchmark/src/common_view/reference_parts.py', WORKSPACE / 'Human_Parsing_Benchmark/protocol/SHARED_C3_FACE_LOCAL_V1.json', spec_path):
        sources[str(path.relative_to(WORKSPACE))] = sha(path)
    config = dict(root=str(root), run_id=root.name, parent_segment=str(parent), protocol=load(protocol_path), protocol_sha256=sha(protocol_path), source_hashes=sources, mapping=spec, mapping_sha256=sha(spec_path), GT_bindings_sha256=sha(root / 'GT_BINDINGS.json'), GT_packet_bindings={f['image_id']: load(root / 'GT' / (f['image_id'] + '.json'))['npz_sha256'] for f in manifest['frames']}, mapping_path=str(spec_path), old_collector_config=str(parent / 'CONFIG.json'), models=old['models'], verified_models=old['verified_models'], warmup=old['warmup'], input_membership_sha256=old['input_membership_sha256'], manifest_sha256=sha(root / 'MANIFEST.json'), resources=old['protocol']['resources'], parent_config_sha256=sha(parent / 'CONFIG.json'))
    atomic(root / 'CONFIG.json', config)
    atomic(root / 'PREPARATION.json', dict(status='PREPARED_REFERENCE_PARTS', at_utc=now(), GT=gt_summary))
    return config

def acceptance():
    """Independent literal expectations for reference scoring and recovery guards."""
    valid = np.ones((2, 5), bool)
    truth = np.zeros((2, 5), bool)
    truth[0, 0:2] = True
    pred = truth.copy()
    pred[1, 4] = True
    r = counts(truth, pred, valid)
    assert (r['TP'], r['FP'], r['FN'], r['TN']) == (2, 1, 0, 7)
    assert abs(r['IoU'] - 2 / 3) < 1e-12 and abs(r['Dice'] - 0.8) < 1e-12
    face = np.zeros((3, 5), bool)
    face[0, 0] = True
    face[1, 2] = True
    candidate = np.zeros_like(face)
    candidate[:2, :3] = True
    candidate[2, 4] = True
    r = counts(face, candidate, np.ones_like(face), local=True)
    assert r['bbox_xyxy'] == [0, 0, 3, 2] and (r['TP'], r['FP'], r['FN']) == (2, 4, 0)
    assert abs(r['IoU'] - 1 / 3) < 1e-12 and r['ignored_prediction_pixels'] == 1
    r = counts(truth, np.zeros_like(truth), valid)
    assert r['IoU'] == 0 and r['FN'] == 2
    r = counts(truth, pred, np.zeros_like(valid))
    assert r is None
    a = counts(np.array([[True]]), np.array([[False]]), np.array([[True]]))
    b = counts(np.ones((1, 9), bool), np.ones((1, 9), bool), np.ones((1, 9), bool))
    assert (a['IoU'] + b['IoU']) / 2 == 0.5
    assert (a['TP'] + b['TP']) / (a['union'] + b['union']) == 0.9
    frame = dict(people=[dict(person_id=7, bbox=[0, 0, 2, 2])], ignore=[[0, 0, 5, 5]])
    cache = dict(scores=np.array([0.9, 0.8]), boxes=np.array([[0, 0, 2, 2], [3, 3, 4, 4]]))
    m = bbox_matches(frame, cache)
    assert (m['0.5']['TP'], m['0.5']['FP'], m['0.5']['ignored_predictions']) == (1, 0, 1)
    frame['people'][0]['person_id'] = 88
    renamed = bbox_matches(frame, cache)
    assert renamed['0.5']['by_GT'] == {'88': 0}
    empty = bbox_matches(frame, dict(scores=np.array([]), boxes=np.empty((0, 4))))
    assert empty['0.5']['FN'] == 1
    assert np.array_equal(project(np.array([2, 13, 5, 7, 9, 12, 6], np.uint8), 'CIHP_GT_AND_NATIVE_PRED', load(WORKSPACE / 'Human_Parsing_Benchmark/protocol/CIHP_SAPIENS_COMMON_VIEW_REVISION2.json')), np.array([2, 1, 3, 3, 4, 4, 5]))
    for bad in (np.array([255], np.uint8), np.array([1.2])):
        try:
            project(bad, 'CIHP_GT_AND_NATIVE_PRED', load(WORKSPACE / 'Human_Parsing_Benchmark/protocol/CIHP_SAPIENS_COMMON_VIEW_REVISION2.json'))
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown prediction ID must fail')
    void = valid.copy()
    void[1, 4] = False
    assert counts(truth, pred, void)['IoU'] == 1.0
    annotation = dict(imageWidth=64, imageHeight=64, shapes=[dict(label='bbox', group_id=1, shape_type='rectangle', points=[[0, 0], [63, 63]]), dict(label='hair', group_id=1, shape_type='rectangle', points=[[1, 1], [2, 2]]), dict(label='hair', group_id=1, shape_type='rectangle', points=[[4, 1], [5, 2]]), dict(label='face', group_id=1, shape_type='rectangle', points=[[2, 1], [3, 2]])])
    exported, binding = gt_export(annotation, [])
    assert exported['valid'][0, 0] and (not exported['valid'][1, 2])
    assert binding['eligible_parts'] == {'1': 1, '2': 1}
    assert binding['GT_support']['2'] == 6 and binding['GT_support']['1'] == 2
    from reference_collection import proven_empty_parsing
    namespace = {}
    exec(compile('def simulated(pred):\n    raise KeyError("parsing")\n', str(WORKSPACE / 'Human_Parsing_Benchmark/src/cihp_benchmark/adapters.py'), 'exec'), namespace)
    for value, expected in (([], True), ([1], False)):
        try:
            namespace['simulated'](value)
        except KeyError as error:
            assert proven_empty_parsing(error) == expected
    assert not proven_empty_parsing(KeyError('other_field'))
    from reference_collection import release_unused_cache
    from types import SimpleNamespace
    released = []
    fake = SimpleNamespace(is_initialized=lambda: True, memory_reserved=lambda: 1000, memory_allocated=lambda: 100, empty_cache=lambda: released.append(True), synchronize=lambda: None)
    after, record = release_unused_cache(dict(free_MiB=100), 1024, fake, lambda: dict(free_MiB=2000))
    assert after['free_MiB'] == 2000 and record['unused_cache_bytes'] == 900 and (len(released) == 1)
    release_unused_cache(dict(free_MiB=2000), 1024, fake, lambda: None)
    assert len(released) == 1
    gt = dict(valid=np.ones((1, 4), bool), part_masks=np.eye(4, dtype=bool).reshape(4, 1, 4), part_owner_ids=np.ones(4, np.int32), part_target_ids=np.array([2, 1, 3, 4], np.uint8))
    match = {t: dict(by_GT={'1': 0}, TP=1, FP=0, FN=0, GT=1, predictions=1, all_predictions=1, ignored_predictions=0, ambiguous=0, identical_GT_geometry_groups=0) for t in ('0.5', '0.75')}
    spec = load(WORKSPACE / 'Human_Parsing_Benchmark/protocol/CIHP_SAPIENS_COMMON_VIEW_REVISION2.json')
    from common_view.packets import write_packet, read_packet
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        packet = write_packet(Path(temp), 'synthetic', dict(person_semantic_masks=np.array([[[2, 13, 5, 9]]], np.uint8)), dict(image_id='synthetic'), {})
        before = sha(packet / 'prediction.npz')
        meta, data = read_packet(packet)
        result = aggregate([dict(image_id='synthetic', **score_frame(gt, match, data['person_semantic_masks'], 'CIHP_GT_AND_NATIVE_PRED', spec))], ['synthetic'])
        assert result['Macro4_local'] == 1.0 and result['primary_mIoU3_part_mean'] == 1.0
        assert all((result['targets'][t]['F1_Dice'] == 1.0 for t in ('Face_local', 'Hair', 'Upper', 'Lower')))
        assert before == sha(packet / 'prediction.npz')
        assert aggregate([], ['missing'])['Macro4_local'] is None
        atomic(Path(temp) / 'CONFIG.json', dict(models={'synthetic': {}}))
        atomic(Path(temp) / 'RESULT_INDEX.json', dict(synthetic=dict(result, status='COMPLETE')))
        atomic(Path(temp) / 'GT_BINDINGS.json', dict(GT_people=1))
        report(Path(temp))
        text = (Path(temp) / 'BATCH_REPORT_TH.md').read_text()
        assert 'Macro4-local' in text and 'mIoU3 (part-mean)' in text and ('Face-local' in text)
        assert '100.000' in text
    print('Reference CPU fixtures PASS', flush=True)

def runtime(root, mid):
    import csv
    output = root / 'models' / mid
    if not (output / 'ASSEMBLED_INDEX.json').exists():
        return dict(complete=False)
    assembly = load(output / 'ASSEMBLED_INDEX.json')
    by_ledger = {}
    for ref in assembly['ROI_references'].values():
        path = Path(ref['packet']).parents[1] / 'ATTEMPTS.jsonl'
        if str(path) in by_ledger:
            continue
        entries = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get('event') == 'CALL_FINISHED' and row.get('phase') == 'MEASURED_ROI':
                    entries[row['job_id']] = row['timing']
        by_ledger[str(path)] = entries
    manifest = load(root / 'MANIFEST.json')
    rows = []
    total = 0.0
    observed = 0
    missing = 0
    for frame in manifest['frames']:
        values = []
        for job in frame['jobs']:
            ref = assembly['ROI_references'][job['job_id']]
            path = Path(ref['packet']).parents[1] / 'ATTEMPTS.jsonl'
            timing = by_ledger[str(path)].get(job['job_id'])
            if timing and timing.get('adapter_ms') is not None:
                values.append(timing['adapter_ms'])
                observed += 1
                total += timing['adapter_ms']
        complete = len(values) == len(frame['jobs'])
        if not complete:
            missing += 1
        rows.append(dict(image_id=frame['image_id'], ROI_count=len(frame['jobs']), timed_ROIs=len(values), parsing_work_s=sum(values) / 1000 if complete else None))
    with (output / 'PER_IMAGE_RUNTIME.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clean = 0
    for number in range(1, 4):
        p = root / 'timing_collection' / mid / 'timing' / ('round-' + str(number)) / mid / 'COMPLETION.json'
        if p.exists() and load(p).get('strict_complete'):
            clean += 1
    return dict(complete=missing == 0, images_with_complete_timer=len(rows) - missing, observed_ROIs=observed, missing_images=missing, mean_ms_per_ROI=total / observed if not missing and observed else None, mean_s_per_image=total / (1000 * len(rows)) if not missing else None, total_s=total / 1000 if not missing else None, observed_total_s=total / 1000, clean_rounds=clean)

def report(root):
    import csv
    config = load(root / 'CONFIG.json')
    results = load(root / 'RESULT_INDEX.json')
    gt = load(root / 'GT_BINDINGS.json')
    target_order = ['Hair', 'Face_local', 'Upper', 'Lower']

    def pct(v):
        return '—' if v is None else f'{100 * v:.3f}'
    text = ['# ผล Human Parsing บน CCTV600 ตามวิธี shared reference เดิม', '', f'ตรวจปิดรอบ {now()} UTC', '', 'ใช้ Mean-part IoU, mIoU3 (part-mean), Face-local และ Macro4-local ตามงาน CIHP/CCTV shared reference เดิม CCTV600 ใช้ bbox matching ที่ IoU 0.50 สำหรับคะแนนรายส่วน เพราะ GT ที่ส่งมอบเป็น bbox และ masks สี่ส่วน Hair/Upper/Lower ใช้ prediction ของคนที่จับคู่บน valid grid ทั้งภาพ ส่วน Face-local ใช้กรอบ GT Face พอดี ตัวหารคือ GT person-parts ที่มีคำตอบ คนที่พลาดได้ศูนย์', '', 'ผลประเมินอ้างขอบของ positive part masks ที่ผู้จัดทำตรวจรับ พื้นที่นอกส่วนของเจ้าของนั้นใช้คิด FP ในงานรายส่วนนั้น GT unassigned คงสถานะใน source โดยไม่มีการรับรอง background ทั้งฉากจากการคำนวณนี้', '', f'ประชากร 600 ภาพและ GT bbox ที่ผ่านเกณฑ์ {gt['GT_people']} รายการ GT parts Hair/Face/Upper/Lower เท่ากับ {gt.get('2', 0)}/{gt.get('1', 0)}/{gt.get('3', 0)}/{gt.get('4', 0)}', '', '| Model | Hair IoU % | Face-local IoU % | Upper IoU % | Lower IoU % | mIoU3 (part-mean) % | Macro4-local % | Status |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for mid, r in results.items():
        targets = r.get('targets', {})
        cells = [pct(targets.get(t, {}).get('mean_part_IoU')) for t in target_order]
        text.append('| ' + ' | '.join([mid] + cells + [pct(r.get('primary_mIoU3_part_mean')), pct(r.get('Macro4_local')), r['status']]) + ' |')
    text += ['', '## สูตรและการอ่านคะแนน', '', '- Mean-part IoU คำนวณ IoU ของแต่ละ GT person-part แล้วเฉลี่ยภายในคลาส แต่ละส่วนมีน้ำหนักเท่ากัน', '- Macro4-local เฉลี่ยค่า Mean-part IoU ของ Hair/Face-local/Upper/Lower คลาสละ 25%', '- mIoU3 (part-mean) เฉลี่ย Hair/Upper/Lower คลาสละหนึ่งในสาม', '- IoU = TP/(TP+FP+FN) โดยพิกเซลทายเกินขอบ GT ของส่วนนั้นเป็น FP และส่วนที่ทายขาดเป็น FN บน valid domain ของงานนั้น', '- Pooled Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1/Dice = 2TP/(2TP+FP+FN) รวม counts ของทุก GT part ภายในคลาสก่อนหาร', '- Pooled IoU รวม counts ก่อนหาร ส่วน Mean-part IoU เฉลี่ยอัตราส่วนรายส่วน ความต่างของขนาดส่วนจึงมีผลต่างกัน', '- null แสดงเป็น — พร้อมสถานะ ผลไม่ครบไม่ใช้ subset เป็นคะแนนชุดเต็ม ตัวหารศูนย์ยังเป็น null', '', '## Counts และ F1 รายคลาส', '', '| Model | Target | GT parts | TP px | FP px | FN px | TN px | Precision % | Recall % | F1/Dice % | Pooled IoU % |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    exported = []
    for mid, r in results.items():
        for target in target_order:
            if target not in r.get('targets', {}):
                continue
            t = r['targets'][target]
            text.append('| ' + ' | '.join([mid, target, str(t['GT_parts'])] + [str(t[k]) for k in ('TP', 'FP', 'FN', 'TN')] + [pct(t[k]) for k in ('precision', 'recall', 'F1_Dice', 'pooled_IoU')]) + ' |')
            exported.append(dict(model=mid, target=target, **{k: v for k, v in t.items() if not isinstance(v, (dict, list))}))
    for mid, r in results.items():
        if 'targets' not in r:
            continue
        text += ['', f'### Confusion matrix ของ {mid}', '', 'แต่ละคลาสแสดง binary matrix [[TP,FN],[FP,TN]] โดยแถวเป็น GT target/other และคอลัมน์เป็น prediction target/other หน่วยพิกเซลสะสมจากงาน GT person-part จึงมีการนับพื้นที่ valid แยกตามคนและส่วน', '', '| Target | TP | FN | FP | TN |', '| --- | --- | --- | --- | --- |']
        for target in target_order:
            t = r['targets'][target]
            text.append('| ' + ' | '.join([target] + [str(t[k]) for k in ('TP', 'FN', 'FP', 'TN')]) + ' |')
    text += ['', '## Person matching', '', '| Model | Bbox IoU | TP | FP | FN | Ignored predictions | Precision % | Recall % | F1 % |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for mid, r in results.items():
        for threshold, t in r.get('localization', {}).items():
            text.append('| ' + ' | '.join([mid, threshold] + [str(t[k]) for k in ('TP', 'FP', 'FN', 'ignored_predictions')] + [pct(t[k]) for k in ('precision', 'recall', 'F1')]) + ' |')
    text += ['', '## เวลาประมวลผล parsing', '', 'รวม adapter intervals ของ accepted ROI jobs ตาม source packet และ ledger เมื่อมี continuation เวลาแต่ละภาพบวกจาก ROI ของภาพนั้น ภาพไม่มี ROI มี parsing work ศูนย์ timer ที่ขาดให้ null เวลาเหล่านี้ครอบคลุม native classes ทั้งหมดและแต่ละ API มีขอบเขตต่างกัน', '', '| Model | Mean ms/ROI | Mean s/image | Total parsing s | Complete image timers | Clean timing rounds |', '| --- | --- | --- | --- | --- | --- |']
    timers = {}
    for mid in config['models']:
        t = runtime(root, mid)
        timers[mid] = t

        def number(k):
            return '—' if t.get(k) is None else f'{t[k]:.3f}'
        text.append('| ' + ' | '.join([mid, number('mean_ms_per_ROI'), number('mean_s_per_image'), number('total_s'), str(t.get('images_with_complete_timer', 0)) + '/600', str(t.get('clean_rounds', 0)) + '/3']) + ' |')
    atomic(root / 'TIMING_SUMMARY.json', timers)
    if exported:
        with (root / 'TARGET_RESULTS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
            writer.writeheader()
            writer.writerows(exported)
    text += ['', '## ขอบเขตและประวัติ', '', 'Sapiens ใช้ Face_Neck พร้อม facial subparts เป็น face candidate กรอบ Face-local ใช้เหมือนกันทุกโมเดล ผล Face-local วัด agreement ภายในกรอบ GT Face ส่วน Upper/Lower คง garment mapping และ Dress/Apparel เป็น OTHER ข้อมูลยังไม่มี cross-zone tags ครบ จึงมี taxonomy limitation กำกับ', '', 'ผลเปรียบเทียบ checkpoint พร้อม application configuration และ predicted YOLO ownership person matching ของ CCTV600 ใช้ bbox โดยประกาศต่างจาก GT-mask matching ของงานอ้างอิง ไม่มีข้อสรุป architecture-only, statistical significance หรือความแม่นยำสีเสื้อผ้าจากผลชุดนี้', '', 'ประวัติ RPR parsing-field error และ CPI pre-forward resource-guard stop เก็บใน parent segment ผลปิดรอบครบอ้าง assembled accepted view พร้อม source hashes และการแก้ output/resource handling ที่ประกาศไว้ เวลา quality เป็น diagnostics ส่วน clean timing ต้องอ่านขอบเขต API ก่อนใช้เปรียบเทียบ']
    (root / 'BATCH_REPORT_TH.md').write_text('\n'.join(text) + '\n')

def gallery(root):
    """Render the original fixed sample from canonical native packets only."""
    import math
    from PIL import Image
    from gallery import panel
    from common_view.packets import read_packet
    from common_view.core import compose_native
    config = load(root / 'CONFIG.json')
    manifest = load(root / 'MANIFEST.json')
    frames = {f['image_id']: f for f in manifest['frames']}
    output = root / 'gallery'
    output.mkdir(exist_ok=True)
    indices = {}
    for mid in config['models']:
        path = root / 'models' / mid / 'ASSEMBLED_INDEX.json'
        indices[mid] = load(path)['frames'] if path.exists() else {}
    entries = []
    display_ids = np.array([0, 2, 1, 3, 4, 0], np.uint8)
    for iid in manifest['visual_ids']:
        frame = frames[iid]
        rgb = np.asarray(Image.open(frame['image_path']).convert('RGB'))
        gt = archive(root / 'GT' / (iid + '.npz'))
        semantic = np.zeros(gt['valid'].shape, np.uint8)
        for mask, target in zip(gt['part_masks'], gt['part_target_ids']):
            semantic[mask & gt['valid']] = display_ids[int(target)]
        panels = [panel(rgb, None, 'Original RGB'), panel(rgb, semantic, 'GT reference parts')]
        sources = {}
        for mid in config['models']:
            ref = indices[mid].get(iid)
            if ref:
                meta, arrays = read_packet(ref['packet'])
                native = compose_native(arrays['person_semantic_masks'], arrays['person_scores'], semantic.shape)
                space = 'SAPIENS2_DOME29_PRED' if config['models'][mid]['family'] == 'Sapiens2' else 'CIHP_GT_AND_NATIVE_PRED'
                mapped = display_ids[project(native, space, config['mapping'])]
                panels.append(panel(rgb, mapped, mid))
                sources[mid] = ref
            else:
                panels.append(panel(rgb, None, mid + ' [incomplete]'))
                sources[mid] = None
        canvas = Image.new('RGB', (1200, 260 * math.ceil(len(panels) / 3)), '#e8eef3')
        for i, value in enumerate(panels):
            canvas.paste(value, ((i % 3) * 400, (i // 3) * 260))
        filename = iid + '.png'
        canvas.save(output / filename)
        entries.append(dict(image_id=iid, file=filename, sha256=sha(output / filename), models=sources))
    atomic(output / 'MANIFEST.json', dict(protocol_sha256=config['protocol_sha256'], panels=entries,
        selection='original frozen sixteen samples', display_composition_only=True, public_upload=False))
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>CCTV600 fixed sample</h1>' +
        ''.join('<h2>' + e['image_id'] + '</h2><img style="max-width:100%" src="' + e['file'] + '">' for e in entries))


def progress(root):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    config = load(root / 'CONFIG.json')
    state = load(root / 'BATCH_STATE.json') if (root / 'BATCH_STATE.json').exists() else dict(status='PREPARING')
    print('\nCCTV600 shared reference |', datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(timespec='seconds'))
    print(root.name, '|', state['status'])
    current = load(root / 'CURRENT_WORKER.json') if (root / 'CURRENT_WORKER.json').exists() else {}
    total = load(root / 'MANIFEST.json')['ROI_jobs']
    for mid in config['models']:
        rec = root / 'reconciliation' / (mid + '.json')
        reused = len(load(rec)['accepted']) if rec.exists() else 0
        collector = root / 'collections' / mid / mid
        native = load(collector / 'STATE.json') if (collector / 'STATE.json').exists() else {}
        count = native.get('complete_ROIs', 0)
        completion = root / 'models' / mid / 'COMPLETION.json'
        status = load(completion)['status'] if completion.exists() else native.get('status', 'WAITING')
        print(f'{mid:34} {status:24} Native ROIs {reused + count}/{total} reused={reused} new={count}')
    print('Worker', current.get('model_id'), current.get('pid'))
    print('Metrics Mean-part IoU | Face-local | mIoU3 (part-mean) | Macro4-local')
    print('จบรอบเมื่อมี', root / 'BATCH_COMPLETION.json')
    print('รายงาน', root / 'BATCH_REPORT_TH.md')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--acceptance', action='store_true')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if args.acceptance:
        acceptance()
    if args.prepare:
        prepare(args.prepare)
    if args.progress:
        import time
        root = args.root or Path(load(PROJECT / 'outputs/CURRENT.json')['root'])
        try:
            while True:
                progress(root)
                if not args.watch:
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            pass
