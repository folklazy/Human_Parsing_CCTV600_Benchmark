"""Annotated-person-part scoring and reporting, separate from frozen scorers."""
import copy
import csv
import json
from pathlib import Path

import numpy as np

from common import archive, atomic, load, now, sha
import reference_parts600 as reference

TARGET_IDS = (2, 1, 3, 4)
TARGETS = ('Hair', 'Face_local', 'Upper', 'Lower')
PREDICTION_COLUMNS = (*TARGETS, 'OTHER')
COLUMN = np.array([4, 1, 0, 2, 3, 4], np.uint8)


def score_frame(gt, matching, planes, space, spec):
    shape = gt['valid'].shape
    if planes.ndim != 3 or planes.shape[1:] != shape:
        raise ValueError('Original-grid native planes required')
    mapped = reference.project(planes, space, spec)
    selected = matching['0.5']['by_GT']
    empty = np.zeros(shape, np.uint8)
    parts = []
    cross = np.zeros((4, 5), np.int64)
    for pid, target, positive in zip(gt['part_owner_ids'], gt['part_target_ids'], gt['part_masks']):
        target = int(target)
        index = selected.get(str(int(pid)))
        prediction = mapped[index] if index is not None else empty
        value = reference.counts(positive, prediction == target, gt['valid'], local=target == 1)
        if value is None:
            continue
        row = np.bincount(COLUMN[prediction[positive & gt['valid']]], minlength=5)
        if int(row.sum()) != value['GT_pixels']:
            raise ValueError('Cross-class GT support mismatch')
        cross[TARGET_IDS.index(target)] += row
        parts.append(dict(gt_id=int(pid), target=target, unmatched=index is None,
                          entirely_missed=value['TP'] == 0, **value))
    localization = {}
    for threshold, original in matching.items():
        item = {k: v for k, v in original.items() if k != 'by_GT'}
        item['unmatched_predictions'] = item.pop('FP')
        localization[threshold] = item
    return dict(parts=parts, localization=localization, cross_confusion=cross.tolist())


def aggregate(rows, expected):
    # Bridge the frozen aggregator's field name locally. Its detector precision
    # and F1 are replaced with coverage-aware nulls in the canonical result.
    temporary = [copy.deepcopy(row) for row in rows]
    for row in temporary:
        for item in row['localization'].values():
            item['FP'] = item.pop('unmatched_predictions')
    result = reference.aggregate(temporary, expected)
    complete = result['strict_complete']
    cross = sum((np.asarray(r['cross_confusion'], np.int64) for r in rows),
                np.zeros((4, 5), np.int64))
    result['cross_class_confusion'] = dict(
        GT_rows=list(TARGETS), prediction_columns=list(PREDICTION_COLUMNS),
        counts=cross.tolist(), domain='positive GT pixels of each annotated person-part',
        OTHER='prediction background and mapped non-target native labels',
    )
    for number, name in enumerate(TARGETS):
        target = result['targets'][name]
        parts = [p for row in rows for p in row['parts'] if p['target'] == TARGET_IDS[number]]
        missed = sum(p['TP'] == 0 for p in parts)
        target['entirely_missed_GT_parts'] = missed
        target['entirely_missed_fraction'] = missed / len(parts) if complete and parts else None
        if int(cross[number].sum()) != target['GT_pixels'] or int(cross[number, number]) != target['TP']:
            raise ValueError('Binary and cross-class confusion disagree')
    for item in result['localization'].values():
        item['unmatched_predictions'] = item.pop('FP')
        item['annotated_person_recall'] = item.pop('recall')
        item['precision'] = None
        item['F1'] = None
        item['AP'] = None
        item['null_reason'] = 'PARTIAL_PERSON_INVENTORY_NO_CERTIFIED_GLOBAL_NEGATIVE_DOMAIN'
    result['annotation_scope'] = 'annotated person-parts; source zero remains unassigned'
    return result


def score(root, mid):
    from common_view.packets import read_packet
    config = load(root / 'CONFIG.json')
    manifest = load(root / 'MANIFEST.json')
    output = root / 'models' / mid
    completion = load(output / 'COMPLETION.json')
    if completion['status'] != 'PREDICTIONS_COMPLETE':
        result = dict(completion, Macro4_local=None)
        atomic(output / 'METRICS.json', result)
        return result
    index = load(output / 'ASSEMBLED_INDEX.json')['frames']
    space = 'SAPIENS2_DOME29_PRED' if config['models'][mid]['family'] == 'Sapiens2' else 'CIHP_GT_AND_NATIVE_PRED'
    rows = []
    for iid in manifest['expected_ids']:
        gt_path = root / 'GT' / (iid + '.npz')
        gt_binding = load(root / 'GT' / (iid + '.json'))
        ref = index[iid]
        if sha(gt_path) != config['GT_packet_bindings'][iid]:
            raise ValueError('GT binding changed')
        meta, arrays = read_packet(ref['packet'])
        if meta['packet_sha256'] != ref['packet_sha256'] or sha(Path(ref['packet']) / 'packet.json') != ref['metadata_sha256']:
            raise ValueError('Frame packet identity changed')
        identity = dict(packet_sha256=ref['packet_sha256'], GT_sha256=gt_binding['npz_sha256'],
                        config_sha256=sha(root / 'CONFIG.json'))
        checkpoint = output / 'scoring' / (iid + '.json')
        if checkpoint.exists():
            row = load(checkpoint)
            if row['bindings'] != identity:
                raise ValueError('Scoring checkpoint identity mismatch')
        else:
            row = dict(image_id=iid, bindings=identity, **score_frame(
                archive(gt_path), gt_binding['matches'], arrays['person_semantic_masks'], space, config['mapping']))
            atomic(checkpoint, row)
        rows.append(row)
    result = dict(model_id=mid, status='COMPLETE', protocol_sha256=config['protocol_sha256'],
                  **aggregate(rows, manifest['expected_ids']))
    atomic(output / 'METRICS.json', result)
    return result


def runtime(directory, frames):
    """Summarize only persisted frame timers with exact ROI membership."""
    totals = {}
    peaks = {'peak_allocated_MiB': None, 'peak_reserved_MiB': None}
    per_image = []
    observed = 0
    complete = 0
    for frame in frames:
        path = directory / 'image_records' / (frame['image_id'] + '.json')
        value = load(path) if path.exists() else None
        expected = [j['job_id'] for j in frame['jobs']]
        records = value['records'] if value else []
        ids = [r['job_id'] for r in records]
        valid = value is not None and len(ids) == len(set(ids)) and set(ids) == set(expected)
        valid = valid and all(r['timing'].get('adapter_ms') is not None for r in records)
        if valid:
            complete += 1
            observed += len(records)
            for record in records:
                for name, number in record['timing'].items():
                    if name in peaks:
                        peaks[name] = max(peaks[name] or 0, number)
                    elif name.endswith('_ms'):
                        totals[name] = totals.get(name, 0.0) + number
            for name in ('source_read_ms', 'cached_detector_read_ms', 'composition_projection_ms',
                         'frame_packet_write_and_validation_ms', 'application_from_cached_inputs_ms'):
                totals[name] = totals.get(name, 0.0) + value[name]
        per_image.append(dict(image_id=frame['image_id'], ROI_count=len(expected),
                              timed_ROIs=len(records) if valid else 0,
                              parsing_work_s=value['parsing_work_ms'] / 1000 if valid else None))
    all_complete = complete == len(frames)
    total = totals.get('adapter_ms', 0.0)
    return dict(complete=all_complete, images_with_complete_timer=complete, expected_images=len(frames),
                observed_ROIs=observed, mean_ms_per_ROI=total / observed if all_complete and observed else None,
                mean_s_per_image=total / (1000 * len(frames)) if all_complete and frames else None,
                total_s=total / 1000 if all_complete else None, observed_totals_ms=totals,
                peak_scope='process CUDA allocated/reserved during measured ROI calls',
                **peaks), per_image


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root):
    config = load(root / 'CONFIG.json')
    manifest = load(root / 'MANIFEST.json')
    results = load(root / 'RESULT_INDEX.json')
    gt = load(root / 'GT_BINDINGS.json')

    def pct(value):
        return '—' if value is None else f'{100 * value:.3f}'

    def number(value):
        return '—' if value is None else f'{value:.3f}'

    text = ['# ผล Human Parsing บน CCTV600', '', f'ตรวจปิดรอบ {now()} UTC', '',
            'ประชากร600ภาพเป็น annotation ของบุคคลที่เลือก Source ID0 เป็น Unassigned '
            'คะแนนอ้าง GT person-parts ที่มีคำตอบพร้อมขอบเขตอ้างอิง ใช้ predicted YOLO ownership และ bbox matchingที่0.50 '
            'Face-local ใช้กรอบ GT Face พอดีใน scoring ผลนี้เปรียบเทียบ checkpoint พร้อม application configuration', '',
            f"GT peopleผ่านเกณฑ์ {gt['GT_people']} รายการ Hair/Face/Upper/Lower parts "
            f"{gt.get('2', 0)}/{gt.get('1', 0)}/{gt.get('3', 0)}/{gt.get('4', 0)}", '',
            '| Model | Hair IoU % | Face-local IoU % | Upper IoU % | Lower IoU % | mIoU3 part-mean % | Macro4-local % | Status |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for mid in config['models']:
        result = results[mid]
        cells = [pct(result.get('targets', {}).get(t, {}).get('mean_part_IoU')) for t in TARGETS]
        text.append('| ' + ' | '.join([mid, *cells, pct(result.get('primary_mIoU3_part_mean')),
                                      pct(result.get('Macro4_local')), result['status']]) + ' |')
    text += ['', '## ความครบถ้วนของผล', '',
             '| Model | Completed images | Missing images | Duplicate / unexpected | Completed ROIs | Failed / invalid / setup |',
             '| --- | --- | --- | --- | --- | --- |']
    for mid in config['models']:
        completion_path = root / 'models' / mid / 'COMPLETION.json'
        c = load(completion_path) if completion_path.exists() else {}
        images, rois = c.get('image_membership', {}), c.get('ROI_membership', {})
        text.append('| ' + ' | '.join([mid, str(images.get('completed', 0)), str(images.get('missing', 600)),
            f"{images.get('duplicate', 0)} / {images.get('unexpected', 0)}", str(rois.get('completed', 0)),
            f"{c.get('failed', 0)} / {c.get('invalid', 0)} / {c.get('setup_failures', 0)}"]) + ' |')
    text += ['', '## นิยามและวิธีรวมคะแนน', '',
             'IoU = TP/(TP+FP+FN), Precision = TP/(TP+FP), Recall = TP/(TP+FN), '
             'F1/Dice = 2TP/(2TP+FP+FN) พิกเซลนอก reference part ของเจ้าของที่ประเมินเป็น non-target ภายใน shared valid domain '
             'Mean-part IoU เฉลี่ย IoU รายส่วนภายในคลาส ส่วน pooled metrics รวม counts ก่อนหาร '
             'Macro4-local เฉลี่ย Hair/Face-local/Upper/Lower คลาสละ25% mIoU3 part-mean เฉลี่ย Hair/Upper/Lower', '',
             'Missed GT person ใช้ empty prediction และเก็บ positive parts ในตัวหาร ส่วนที่ TP=0 นับเป็นส่วนพลาดทั้งหมด '
             'ส่วนที่ไม่มี valid GT support ยกเว้นจาก denominator ตาม GT bindings '
             'STRICT_COMPLETE ต้องครบ600ภาพและ ROI jobs ตาม manifest คะแนนหลักของระบบที่ไม่ครบเป็น null', '',
             '## Counts, IoU และ F1 รายคลาส', '',
             '| Model | Target | GT parts | TP px | FP px | FN px | TN px | Precision % | Recall % | F1/Dice % | Pooled IoU % | TP=0 parts | TP=0 % |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    exported = []
    for mid, result in results.items():
        for target in TARGETS:
            t = result.get('targets', {}).get(target)
            if not t:
                continue
            cells = [mid, target, str(t['GT_parts']), *[str(t[k]) for k in ('TP', 'FP', 'FN', 'TN')],
                     *[pct(t[k]) for k in ('precision', 'recall', 'F1_Dice', 'pooled_IoU')],
                     str(t['entirely_missed_GT_parts']), pct(t['entirely_missed_fraction'])]
            text.append('| ' + ' | '.join(cells) + ' |')
            exported.append(dict(model=mid, target=target, **{k: v for k, v in t.items() if not isinstance(v, (dict, list))}))
    write_csv(root / 'TARGET_RESULTS.csv', exported)
    for mid, result in results.items():
        if 'targets' not in result:
            continue
        text += ['', f'## Confusion matrix ของ {mid}', '',
                 'Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT '
                 'คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย', '',
                 '| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |', '| --- | --- | --- | --- | --- | --- |']
        for target, row in zip(TARGETS, result['cross_class_confusion']['counts']):
            text.append('| ' + ' | '.join([target, *map(str, row)]) + ' |')
        text += ['', 'Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part '
                 'จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน', '',
                 '| Target | TP | FN | FP | TN |', '| --- | --- | --- | --- | --- |']
        for target in TARGETS:
            t = result['targets'][target]
            text.append('| ' + ' | '.join([target, *[str(t[k]) for k in ('TP', 'FN', 'FP', 'TN')]]) + ' |')
    text += ['', '## การพบคนที่มี GT', '',
             'Recall อ้างเฉพาะคนที่ annotate ส่วน unmatched predictions ยังรวมคนจริงที่ไม่มี GT ได้ '
             'full-scene person Precision/F1/AP เป็น null เพราะยังไม่มี negative domain ที่รับรองครบ', '',
             '| Model | Bbox IoU | TP | FN | Unmatched | Ignored | Annotated-person recall % |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for mid, result in results.items():
        for threshold, t in result.get('localization', {}).items():
            text.append('| ' + ' | '.join([mid, threshold, str(t['TP']), str(t['FN']),
                str(t['unmatched_predictions']), str(t['ignored_predictions']), pct(t['annotated_person_recall'])]) + ' |')
    timers = {}
    text += ['', '## เวลาประมวลผลและ Peak VRAM', '',
             'Adapter intervals ครอบคลุม native classes ทั้งหมด Mean s/image บวกงาน parsing ของ ROI ในแต่ละภาพ '
             'ส่วน application-from-cached-inputs รวม source decode, cache/crop reads, ownership และ packet I/O '
             'ไม่มี new YOLO inference ใน interval นี้ Peak VRAM เป็น CUDA allocated/reserved ของ process '
             'คุณภาพและ clean timing rounds แสดงแยกกันตาม resource checks', '',
             '| Model | Mean ms/ROI | Mean s/image | Total parsing s | Complete image timers | Peak allocated MiB | Peak reserved MiB | Clean rounds |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for mid in config['models']:
        output = root / 'models' / mid
        t, per_image = runtime(root / 'collections' / mid / mid, manifest['frames'])
        write_csv(output / 'PER_IMAGE_RUNTIME.csv', per_image)
        rounds = {}
        frames = [f for f in manifest['frames'] if f['image_id'] in manifest['timing_ids']]
        for n in range(1, 4):
            directory = root / 'timing_collection' / mid / 'timing' / f'round-{n}' / mid
            completion = load(directory / 'COMPLETION.json') if (directory / 'COMPLETION.json').exists() else {}
            measured, _ = runtime(directory, frames)
            rounds[str(n)] = dict(clean_complete=bool(completion.get('strict_complete')), **measured)
        clean = sum(r['clean_complete'] for r in rounds.values())
        timers[mid] = dict(quality=t, dedicated_rounds=rounds)
        text.append('| ' + ' | '.join([mid, number(t['mean_ms_per_ROI']), number(t['mean_s_per_image']),
            number(t['total_s']), f"{t['images_with_complete_timer']}/600", number(t['peak_allocated_MiB']),
            number(t['peak_reserved_MiB']), f'{clean}/3']) + ' |')
    atomic(root / 'TIMING_SUMMARY.json', timers)
    text += ['', '## ขอบเขตและข้อจำกัด', '',
             'Face candidate ของ Sapiens รวม Face_Neck และ facial subparts คะแนน Face-local วัดภายใน GT Face box '
             'Upper/Lower ใช้ garment unions ที่ประกาศ ส่วน Dress/Apparel เป็น OTHER '
             'GT conflicts ใช้ shared VOID และเปิดเผยส่วนที่ถูกยกเว้นใน GT bindings '
             'ข้อมูลมีการใช้ helper-assisted annotation และคนที่เลือก annotate จึงต้องอ้างขอบเขตนี้เมื่อสรุปผล', '',
             'Timing เป็น scoped adapter/application diagnostics ตาม API และสภาพ GPU ที่บันทึก '
             'รายงานนี้มี provenance ของ config, source, dataset, checkpoints และ run records กำกับ '
             'ประวัติ failures ของรอบนี้คงอยู่ใน immutable attempts และ completion records', '']
    (root / 'BATCH_REPORT_TH.md').write_text('\n'.join(text))
