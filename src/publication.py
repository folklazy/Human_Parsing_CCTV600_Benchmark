"""CPU-only publication of accepted aggregates; never imports an inference runner.

Build from a closed private run, or reproduce tables/figures from the public index.
Canonical predictions, frozen configs, scorers and historical evidence are read-only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
TARGETS = ('Hair', 'Face_local', 'Upper', 'Lower')
NAMES = ('QANet', 'M2FP', 'D2FP', 'RP-R-CNN', 'CPI R50', 'CPI ConvNeXt-B',
         'Sapiens2 0.4B', 'Sapiens2 0.8B', 'Sapiens2 1B')
ROSTER = ('QANet_CIHP_HRNetW48_512x384', 'M2FP_R101_CIHP', 'D2FP_R101_CIHP', 'RPR_CIHP',
          'CPI_R50_CIHP', 'CPI_ConvNeXtB_CIHP', 'Sapiens2_0.4b', 'Sapiens2_0.8b', 'Sapiens2_1b')


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def pct(value):
    return '—' if value is None else f'{100 * value:.3f}'


def num(value):
    return '—' if value is None else f'{value:,.3f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                       *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def validate(index):
    """Check sufficient-statistic arithmetic, domains, membership and timing units."""
    assert len(index['models']) == 9
    assert [m['model_id'] for m in index['models']] == list(ROSTER)
    gt = index['GT']
    assert gt['images'] == 600
    for model in index['models']:
        r = model['result']
        assert r['protocol_sha256'] == index['protocol_sha256']
        assert r['strict_complete'] and r['completed'] == r['expected'] == 600
        assert r['missing'] == r['duplicate'] == 0
        for membership in model['membership'].values():
            assert membership['strict_complete']
            assert membership['expected'] == membership['completed']
            assert membership['missing'] == membership['duplicate'] == membership['unexpected'] == 0
        assert model['membership']['ROI']['completed'] == 763
        cross = r['cross_class_confusion']['counts']
        for i, target in enumerate(TARGETS):
            t = r['targets'][target]
            assert t['GT_parts'] == gt['parts'][target]
            assert t['TP'] + t['FN'] == t['GT_pixels'] == gt['support'][target]
            assert t['prediction_pixels'] == t['TP'] + t['FP']
            assert t['union'] == t['TP'] + t['FP'] + t['FN']
            assert t['binary_confusion_target_other'] == [[t['TP'], t['FN']], [t['FP'], t['TN']]]
            assert sum(cross[i]) == t['GT_pixels'] and cross[i][i] == t['TP']
            assert 0 <= t['entirely_missed_GT_parts'] <= t['GT_parts']
            ratios = dict(pooled_IoU=(t['TP'], t['union']),
                          precision=(t['TP'], t['TP'] + t['FP']),
                          recall=(t['TP'], t['GT_pixels']),
                          F1_Dice=(2*t['TP'], 2*t['TP'] + t['FP'] + t['FN']))
            for key, (top, bottom) in ratios.items():
                assert (t[key] is None if bottom == 0 else math.isclose(t[key], top / bottom, abs_tol=1e-12))
        assert math.isclose(r['Macro4_local'], statistics.mean(r['targets'][t]['mean_part_IoU'] for t in TARGETS), abs_tol=1e-12)
        assert math.isclose(r['primary_mIoU3_part_mean'], statistics.mean(r['targets'][t]['mean_part_IoU'] for t in ('Hair', 'Upper', 'Lower')), abs_tol=1e-12)
        for loc in r['localization'].values():
            assert loc['TP'] + loc['FN'] == loc['GT'] == gt['people']
            assert loc['TP'] + loc['unmatched_predictions'] + loc['ignored_predictions'] == loc['all_predictions'] == 763
            assert all(loc[k] is None for k in ('precision', 'F1', 'AP'))
        time = model['timing']
        assert len(time['dedicated_rounds']) == 3
        for label, t in [('quality', time['quality']), *time['dedicated_rounds'].items()]:
            images, rois = (600, 763) if label == 'quality' else (100, 124)
            assert t['complete'] and t['images_with_complete_timer'] == images and t['observed_ROIs'] == rois
            if label != 'quality':
                assert t['clean_complete']
            assert math.isclose(t['mean_s_per_image'] * images, t['total_s'], abs_tol=1e-8)
            assert math.isclose(t['mean_ms_per_ROI'] * rois / 1000, t['total_s'], abs_tol=1e-8)
            components = t['observed_totals_ms']
            keys = ('pre_module_ms', 'module_call_ms', 'inter_module_ms', 'post_module_ms')
            if all(k in components for k in keys):
                assert math.isclose(sum(components[k] for k in keys), components['adapter_ms'], abs_tol=1e-5)
    assert set(index['native_mapping']['CIHP_GT_AND_NATIVE_PRED']) == set(map(str, range(20)))
    assert set(index['native_mapping']['SAPIENS2_DOME29_PRED']) == set(map(str, range(29)))
    # No private image/person identifiers, absolute paths or environment executables.
    public = json.dumps(index)
    assert not any(s in public for s in ('/home/', 'cctv600_000', 'source_image_record', 'source_ledger'))


def collect(root):
    config, manifest = read(root/'CONFIG.json'), read(root/'MANIFEST.json')
    results, timing = read(root/'RESULT_INDEX.json'), read(root/'TIMING_SUMMARY.json')
    finish, gt = read(root/'BATCH_COMPLETION.json'), read(root/'GT_BINDINGS.json')
    assert finish['complete_models'] == 9
    expected = manifest['expected_ids']
    jobs = [j['job_id'] for f in manifest['frames'] for j in f['jobs']]
    assert len(expected) == len(set(expected)) == 600
    assert len(jobs) == len(set(jobs)) == 763
    parent = Path(config['continuation']['parent_root'])
    index = dict(schema='CCTV600_PUBLIC_RESULTS_V1', status='COMPLETE',
                 run_id=root.name, parent_run_id=parent.name,
                 ended_utc=finish['ended_utc'], protocol_sha256=config['protocol_sha256'],
                 dataset_membership_sha256=config['input_membership_sha256'],
                 mapping_sha256=config['mapping_sha256'], protocol=config['protocol'],
                 native_mapping=config['mapping']['mapping'],
                 GT=dict(images=gt['images'], people=gt['GT_people'],
                         parts={t: gt[str(i)] for t, i in zip(TARGETS, (2, 1, 3, 4))},
                         support={t: gt['GT_support'][str(i)] for t, i in zip(TARGETS, (2, 1, 3, 4))},
                         VOID_pixels=gt['VOID_pixels'], zero_ROI_images=sum(not f['jobs'] for f in manifest['frames'])),
                 resources=config['resources'], models=[],
                 provenance=dict(config_sha256=sha(root/'CONFIG.json'), manifest_sha256=sha(root/'MANIFEST.json'),
                                 GT_bindings_sha256=sha(root/'GT_BINDINGS.json'),
                                 result_index_sha256=sha(root/'RESULT_INDEX.json'),
                                 timing_summary_sha256=sha(root/'TIMING_SUMMARY.json'),
                                 completion_sha256=sha(root/'BATCH_COMPLETION.json'),
                                 launch_commits=[read(parent/'LAUNCH.json')['launch_commit'], read(root/'LAUNCH.json')['launch_commit']],
                                 CPU_acceptance=read(parent/'CPU_ACCEPTANCE.json'),
                                 continuation_CPU_acceptance=read(root/'CPU_ACCEPTANCE.json'),
                                 source_hashes={p: h for p, h in config['source_hashes'].items()
                                                if p.startswith(('Human_Parsing_CCTV600_Benchmark/src/', 'CCTV_Human_Parsing_Benchmark/scripts/controlled_roi',
                                                                 'Human_Parsing_Benchmark/src/common_view/reference_parts.py'))}))
    detector = read(Path(manifest['detector_parent'])/'PREPARED.json')
    index['detector'] = dict(model='YOLO26x-Seg', run_id=detector['run_id'],
        checkpoint_sha256=detector['checkpoints']['yolo26x-seg.pt']['sha256'],
        packet_audit_sha256=manifest['detector_audit_sha256'],
        settings={k: detector['config'][k] for k in ('imgsz', 'rect', 'retina_masks', 'ap_confidence_floor',
                                                    'fixed_confidence', 'nms_iou', 'max_detections')})
    for name, (mid, spec) in zip(NAMES, config['models'].items()):
        folder = root/'models'/mid
        result = read(folder/'METRICS.json')
        assert result == results[mid]
        assembled = read(folder/'ASSEMBLED_INDEX.json')
        assert set(assembled['frames']) == set(expected)
        assert set(assembled['ROI_references']) == set(jobs)
        completion = read(folder/'COMPLETION.json')
        assert completion['status'] == 'PREDICTIONS_COMPLETE'
        assert all(completion[k] == 0 for k in ('failed', 'invalid', 'setup_failures'))
        directory = root/'collections'/mid/mid
        details, environment = read(directory/'MODEL_DETAILS.json'), read(directory/'ENVIRONMENT.json')
        from urllib.parse import urlsplit
        upstream = subprocess.check_output(['git', '-C', str(PROJECT.parent/spec['source']), 'remote', 'get-url', 'origin'], text=True).strip()
        if upstream.startswith('git@github.com:'):
            upstream = 'https://github.com/' + upstream.split(':', 1)[1]
        url = urlsplit(upstream)
        assert url.hostname == 'github.com'
        upstream = 'https://github.com' + url.path.removesuffix('.git')
        index['models'].append(dict(model_id=mid, name=name, result=result, timing=timing[mid],
            binding={k: spec[k] for k in ('checkpoint', 'checkpoint_sha256', 'native_config', 'native_config_sha256', 'source', 'source_commit', 'environment', 'route')},
            parameters=details['parameters'], native_classes=29 if spec['family'] == 'Sapiens2' else 20,
            upstream=upstream,
            implementation=details.get('memory_implementation', {}).get('revision', 'BOUND_NATIVE_CROP_ADAPTER'),
            environment={k: environment[k] for k in ('python', 'torch', 'CUDA', 'GPU', 'numpy', 'FP32', 'AMP', 'TF32')},
            package_freeze_sha256=config['verified_models'][mid]['package_freeze_sha256'],
            membership=dict(images=completion['image_membership'], ROI=completion['ROI_membership']),
            provenance=dict(metrics_sha256=sha(folder/'METRICS.json'), assembled_index_sha256=sha(folder/'ASSEMBLED_INDEX.json'),
                            collector_completion_sha256=sha(folder/'COMPLETION.json'),
                            canonical_primary_source='METRICS.json; collector primary:null predates scoring',
                            quality_run_ids=[parent.name, root.name] if mid == 'Sapiens2_1b' else [parent.name])))
    gallery = read(root/'gallery/MANIFEST.json')
    assert gallery['protocol_sha256'] == config['protocol_sha256']
    assert len(gallery['panels']) == 16 and {v['image_id'] for v in gallery['panels']} == set(manifest['visual_ids'])
    for panel in gallery['panels']:
        assert set(panel['models']) == set(config['models']) and all(panel['models'].values())
        assert sha(root/'gallery'/panel['file']) == panel['sha256']
    index['gallery'] = dict(images=16, models_per_image=9, panels_per_image=11,
                            manifest_sha256=sha(root/'gallery/MANIFEST.json'), PNG_hashes_verified=16,
                            selection=gallery['selection'], private=True, display_composition_only=True)
    validate(index)
    return index


def csv_rows(index):
    rows, targets, times = [], [], []
    for m in index['models']:
        r = m['result']
        rows.append(dict(model=m['name'], model_id=m['model_id'], completed=r['completed'], strict_complete=r['strict_complete'],
                         **{t: r['targets'][t]['mean_part_IoU'] for t in TARGETS},
                         mIoU3_part_mean=r['primary_mIoU3_part_mean'], Macro4_local=r['Macro4_local']))
        for t in TARGETS:
            targets.append(dict(model=m['name'], target=t, **{k: v for k, v in r['targets'][t].items() if not isinstance(v, (dict, list))}))
        for label, value in [('quality', m['timing']['quality']), *[(f'dedicated_{k}', v) for k, v in m['timing']['dedicated_rounds'].items()]]:
            times.append(dict(model=m['name'], phase=label, images=value['expected_images'], ROIs=value['observed_ROIs'],
                              mean_ms_per_ROI=value['mean_ms_per_ROI'], mean_s_per_image=value['mean_s_per_image'], total_parsing_s=value['total_s'],
                              peak_allocated_MiB=value['peak_allocated_MiB'], peak_reserved_MiB=value['peak_reserved_MiB'],
                              **{k: value['observed_totals_ms'].get(k) for k in ('pre_module_ms', 'module_call_ms', 'inter_module_ms', 'post_module_ms', 'application_from_cached_inputs_ms')}))
    return {'SUMMARY.csv': rows, 'PER_TARGET.csv': targets, 'TIMING.csv': times}


def export_csv(index, destination):
    for filename, values in csv_rows(index).items():
        with (destination/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)


def error_split(model):
    """Decompose GT-positive FN from saved aggregate counts, not new predictions."""
    result = {}
    for i, target in enumerate(TARGETS):
        row = model['result']['cross_class_confusion']['counts'][i]
        t = model['result']['targets'][target]
        wrong_target = sum(row[:4]) - row[i]
        other = row[4]
        assert other + wrong_target == t['FN']
        matched_zero = t['entirely_missed_GT_parts'] - t['unmatched_GT_parts']
        assert 0 <= matched_zero <= t['GT_parts'] - t['unmatched_GT_parts']
        result[target] = dict(wrong_target_pixels=wrong_target, OTHER_pixels=other,
            OTHER_fraction_of_FN=other/t['FN'] if t['FN'] else None,
            matched_person_zero_overlap_parts=matched_zero)
    return result


def plots(index, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    directory.mkdir(parents=True, exist_ok=True)
    values = np.array([[m['result']['targets'][t]['mean_part_IoU']*100 for t in TARGETS] +
                       [m['result']['Macro4_local']*100] for m in index['models']])
    fig, ax = plt.subplots(figsize=(11, 6.5), layout='constrained')
    image = ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    ax.set(xticks=range(5), xticklabels=['Hair', 'Face-local', 'Upper', 'Lower', 'Macro4-local'],
           yticks=range(9), yticklabels=NAMES,
           title='CCTV600 | Mean-part IoU (%) | 600 images per checkpoint')
    for i in range(9):
        for j in range(5):
            ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', color='white' if values[i,j]>55 else '#142b3f')
    fig.colorbar(image, ax=ax, label='IoU (%)')
    fig.savefig(directory/'QUALITY_OVERVIEW.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(9, 4, figsize=(13, 24), layout='constrained')
    for i, m in enumerate(index['models']):
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            t = m['result']['targets'][target]
            counts = np.array(t['binary_confusion_target_other'])
            percentages = 100 * counts / counts.sum(axis=1, keepdims=True)
            ax.imshow(percentages, cmap='Blues', vmin=0, vmax=100)
            ax.set(xticks=[0,1], xticklabels=['Target','Other'], yticks=[0,1], yticklabels=['Target','Other'])
            ax.set_title(f'{m["name"]} | {target.replace("_", "-")}', fontsize=9)
            for y in range(2):
                for x in range(2):
                    label=(('TP','FN'),('FP','TN'))[y][x]
                    ax.text(x,y,f'{label}\n{counts[y,x]:,}\n{percentages[y,x]:.2f}%',ha='center',va='center',fontsize=7,
                            color='white' if percentages[y,x]>55 else '#122b45')
            if j == 0:
                ax.set_ylabel('GT')
            if i == 8:
                ax.set_xlabel('Prediction')
    fig.suptitle('CCTV600 | Binary confusion per annotated person-part task\nRows: GT; columns: prediction; percentages normalized within each GT row', fontsize=14)
    fig.savefig(directory/'TARGET_CONFUSION.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), layout='constrained')
    for ax, model in zip(axes.flat, index['models']):
        counts = np.asarray(model['result']['cross_class_confusion']['counts'])
        percentages = 100 * counts/counts.sum(axis=1, keepdims=True)
        image = ax.imshow(percentages, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        ax.set(xticks=range(5), xticklabels=['Hair','Face-local','Upper','Lower','OTHER'],
               yticks=range(4), yticklabels=['Hair','Face-local','Upper','Lower'], title=model['name'],
               xlabel='Prediction', ylabel='GT target')
        ax.tick_params(axis='x', labelrotation=25, labelsize=9)
        for y in range(4):
            for x in range(5):
                ax.text(x, y, f'{percentages[y,x]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if percentages[y,x]>55 else '#142b3f')
    fig.colorbar(image, ax=axes.ravel().tolist(), label='% of each GT target row', shrink=.8)
    fig.suptitle('CCTV600 | GT-positive cross-class confusion (%)\nOTHER includes prediction background and recognized non-target classes', fontsize=15)
    fig.savefig(directory/'CROSS_CLASS_CONFUSION.png', dpi=170)
    plt.close(fig)
    # These are annotation display colours, not extracted clothing colours.
    swatches = [('Hair','#8040c0'),('Face proxy','#ffa64d'),('Upper','#009e73'),('Lower','#0072b2')]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="62" viewBox="0 0 760 62">',
           '<rect width="760" height="62" fill="white"/>']
    for i,(name,colour) in enumerate(swatches):
        x = 15 + i*185
        svg += [f'<rect x="{x}" y="16" width="28" height="28" rx="4" fill="{colour}"/>',
                f'<text x="{x+38}" y="35" font-family="sans-serif" font-size="17" fill="#173246">{name}</text>']
    svg.append('</svg>')
    (directory/'TARGET_LEGEND.svg').write_text('\n'.join(svg)+'\n')


def summary_table(index):
    return table(['โมเดล', 'Hair %', 'Face-local %', 'Upper %', 'Lower %', 'mIoU3 %', 'Macro4-local %'],
                 [[m['name'], *[pct(m['result']['targets'][t]['mean_part_IoU']) for t in TARGETS],
                   pct(m['result']['primary_mIoU3_part_mean']), pct(m['result']['Macro4_local'])] for m in index['models']])


def primary_summary(index):
    first, second = sorted(index['models'], key=lambda m: m['result']['Macro4_local'], reverse=True)[:2]
    a, b = first['result']['Macro4_local'], second['result']['Macro4_local']
    return (f"**{first['name']} ได้ Macro4-local สูงสุด {pct(a)}%** ตามด้วย "
            f"**{second['name']} {pct(b)}%** ต่างกัน **{100*(a-b):.3f} จุดเปอร์เซ็นต์**")


def insights(index):
    q, s = index['models'][0], index['models'][-1]
    splits = [error_split(m) for m in index['models']]
    fractions = [r['OTHER_fraction_of_FN'] for split in splits for r in split.values() if r['OTHER_fraction_of_FN'] is not None]
    assert all(r['OTHER_pixels'] >= r['wrong_target_pixels'] for split in splits for r in split.values())
    text = '''## Insight จาก confusion และตัวชี้วัดร่วมกัน

### ความผิดพลาดส่วนใหญ่ลง OTHER แต่ยังระบุสาเหตุจากคลาสอย่างเดียวไม่ได้

'''
    text += f"ใน 36 แถว target-model กลุ่ม OTHER มีจำนวนมากกว่าการทายเป็นอีกสาม target รวมกันทุกแถว คิดเป็น {100*min(fractions):.2f}–{100*max(fractions):.2f}% ของ FN ในแต่ละแถว นี่เป็นสัดส่วนของพิกเซล GT ที่พลาด ไม่ใช่สัดส่วนของทุกพิกเซลในภาพ\n\n"
    text += '''OTHER รวมทั้ง native background และคลาสนอกสี่เป้าหมาย อีกทั้ง prediction ว่างของคนที่จับคู่ไม่ได้ก็ลง OTHER ทั้งส่วนได้ จึงยังสรุปไม่ได้ว่า parser ทายเป็นฉากหลังเองทั้งหมด อาจเกี่ยวกับ YOLO coverage, mask ownership clipping, native instance association หรือ semantic prediction ต้องตรวจ saved masks รายกรณีเพื่อแยกสาเหตุ

### QANet กับ Sapiens2 1B มี trade-off ระหว่างทายเกินกับเก็บ GT

'''
    text += table(['Target','QANet Precision %','QANet Recall %','Sapiens2 1B Precision %','Sapiens2 1B Recall %'],
                  [[t,pct(q['result']['targets'][t]['precision']),pct(q['result']['targets'][t]['recall']),
                    pct(s['result']['targets'][t]['precision']),pct(s['result']['targets'][t]['recall'])] for t in TARGETS])
    text += '''

QANet มี Precision สูงกว่า ส่วน Sapiens2 1B มี Recall สูงกว่าในทั้งสี่คลาสของการเปรียบเทียบคู่นี้ Sapiens2 1B จึงเก็บ GT pixels ได้มากกว่าโดยมีสัดส่วนทายเกินสูงกว่า QANet ไม่ใช่ข้อพิสูจน์ว่าความต่างเกิดจาก architecture หรือ threshold เพียงอย่างเดียว ทั้งคู่ใช้ thresholds ที่ตรึงไว้

Sapiens2 1B มี pooled F1/Dice และ pooled IoU สูงกว่า QANet ทั้งสี่คลาส แต่ QANet ได้ Mean-part IoU ของ Upper/Lower สูงกว่า การเฉลี่ยรายส่วนกับการรวมพิกเซลให้ความสำคัญต่อขนาดส่วนต่างกัน จึงเป็นผลที่อยู่ร่วมกันได้ และยังไม่ยืนยันว่าโมเดลใดชนะทุกคนหรือทุกขนาด

### ค่าเฉลี่ยสูงกว่าไม่ได้แปลว่าไม่มีส่วนที่พลาดทั้งชิ้น

'''
    text += table(['Target','GT parts','จับคู่คนไม่ได้ (ทั้งคู่)','QANet TP=0 หลังจับคู่คนได้','Sapiens2 1B TP=0 หลังจับคู่คนได้'],
                  [[t,q['result']['targets'][t]['GT_parts'],q['result']['targets'][t]['unmatched_GT_parts'],
                    splits[0][t]['matched_person_zero_overlap_parts'],splits[-1][t]['matched_person_zero_overlap_parts']] for t in TARGETS])
    text += '''

ตัวอย่าง Face-local: Sapiens2 1B มีค่าเฉลี่ย IoU สูงกว่า แต่มี Face parts ที่ TP=0 รวม 53 ส่วน เทียบ QANet 48 ส่วน (หลังหักคนที่จับคู่ไม่ได้ เหลือ 19 กับ 14 ส่วน) จึงควรอ่านค่าเฉลี่ยควบคู่จำนวนส่วนที่พลาด ไม่ตีความว่าคะแนนสูงกว่าหมายถึงพบใบหน้าของทุกคนได้ดีกว่า ยังไม่ได้ตรวจว่ากลุ่มที่พลาดเป็นภาพเล็ก/เบลอ/แสงแบบใด

### การจับคู่คนเป็นต้นทุนร่วมของทั้งเก้าโมเดล

'''
    loc = q['result']['localization']['0.5']
    text += f"ที่ bbox IoU 0.50 พบคู่ {loc['TP']}/{loc['GT']} คน และพลาด {loc['FN']} คน GT parts ของคนที่พลาดยังอยู่ในตัวหารด้วยศูนย์ ดังนั้น confusion นี้เป็นผลของ application pipeline ร่วมกับ parser ไม่ใช่คุณภาพ parser ภายใต้ perfect person crops ทั้งหมด ตัวเลขนี้อ้างเฉพาะคนที่ annotate; unmatched predictions ยังจัดเป็น FP ทั้งฉากไม่ได้\n\n"
    text += '''### ความพร้อมของข้อมูลและสิ่งที่ยังไม่รองรับ

| ประเด็น | สถานะหลักฐาน |
| --- | --- |
| Counts/สูตร/ตัวหาร/labels | ตรวจเลขจำนวนเต็มและ ratios กับ canonical statistics แล้ว; binary diagonal ตรงกับ cross-class diagonal |
| ความครบของผล | Exact 600 ภาพ/763 ROI ครบเก้าโมเดล ใช้ scoring domain เดียวกันภายในงานนี้ |
| คุณภาพ annotation | ผู้จัดทำตรวจรับและมี QA/owner/conflict checks แต่ไม่มี inter-annotator agreement หรือ independent expert gold |
| Coverage ทั้งฉาก | ยังมีคนหรือส่วนที่ไม่ annotate; ไม่รับรอง global-scene negatives และ full-scene semantic mIoU |
| Face และเสื้อผ้า | Face-local เป็น proxy ในกรอบ GT; เสื้อผ้าเป็น non-Dress mapping และยังมี taxonomy limitations |
| ข้อสรุปทั่วไป | ไม่มี independent held-out certification/uncertainty study จึงไม่อ้าง statistical significance หรือ superiority บน CCTV ทุกสภาพ |

ชุดผลพร้อมใช้รายงานการเปรียบเทียบแบบ annotated-person-part ตาม protocol นี้ ส่วนคุณภาพข้อมูลสำหรับ benchmark เต็มฉากยังรับรองไม่ได้จากการตรวจ arithmetic หรือ confusion matrix อย่างเดียว ข้อมูลสถิติที่ครบไม่ได้แทนความครบของ annotation

'''
    return text


def report(index):
    closed = datetime.fromisoformat(index['ended_utc']).astimezone(ZoneInfo('Asia/Bangkok')).isoformat(timespec='seconds')
    text = f'''# ผล Human Parsing บน CCTV600

ปิดรอบ **{closed} — Asia/Bangkok** คะแนนและผลทำนายครบ **9 โมเดล × 600 ภาพ** โมเดลละ **763 ROI**; dedicated timing ครบ **27/27 รอบ** บน **NVIDIA Tesla T4**

[ตาราง CSV](../metrics/SUMMARY.csv) · [สถิติรายคลาส](../metrics/PER_TARGET.csv) · [เวลาแต่ละรอบ](../metrics/TIMING.csv) · [Result index และ hashes](../metrics/RESULT_INDEX.json) · [Dataset และคู่มือ annotation](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset)

## ข้อสรุปสำหรับเลือกโมเดล

{primary_summary(index)} เมื่อให้สี่คลาสมีน้ำหนักเท่ากัน ผลรายเป้าหมายอยู่ในตารางด้านล่าง ไม่มีการทดสอบนัยสำคัญทางสถิติหรือ deployment acceptance ในรอบนี้

## คะแนนรายคลาสและค่าเฉลี่ย

{summary_table(index)}

![Mean-part IoU ของเก้าโมเดล](figures/QUALITY_OVERVIEW.png)

ตารางเรียงตาม frozen roster ไม่ใช่เรียงอันดับ แต่ละส่วน GT มีน้ำหนักเท่ากันภายในคลาส Face ใช้ **Face-local**; Upper/Lower เป็นกลุ่ม **non-Dress** ตาม mapping ไม่ใช่ taxonomy ที่เท่ากันทุก checkpoint

'''
    winners = []
    for t in TARGETS:
        order = sorted(index['models'], key=lambda m: m['result']['targets'][t]['mean_part_IoU'], reverse=True)
        a, b = (m['result']['targets'][t]['mean_part_IoU'] for m in order[:2])
        winners.append([t, order[0]['name'], pct(a), order[1]['name'], num(100*(a-b))])
    text += table(['เป้าหมาย', 'สูงสุด', 'Mean-part IoU %', 'อันดับถัดไป', 'ส่วนต่าง (จุดเปอร์เซ็นต์)'], winners)
    pooled = {max(index['models'], key=lambda m: m['result']['targets'][t]['pooled_IoU'])['name'] for t in TARGETS}
    assert pooled == {'Sapiens2 1B'}, 'Review pooled-leader narrative for a changed result population'
    text += '''

เมื่อรวมพิกเซลก่อนหาร (**pooled IoU**) Sapiens2 1B สูงสุดทั้งสี่คลาส ผลนี้ตอบคนละคำถามกับ Mean-part: ส่วนที่มี union ใหญ่มีน้ำหนักมากกว่า ห้ามสลับสองตารางแล้วอธิบายว่าเป็น metric เดียวกัน

## วิธีทดสอบและพื้นที่ที่ประเมิน

ใช้ภาพ 600 ภาพและคำตอบของบุคคลที่เลือก annotate มี GT bbox ผ่านเกณฑ์ด้านยาว ≥50 px จำนวน 769 คน ทุกโมเดลใช้ frozen YOLO26x-Seg cache ชุดเดียวกัน เลือก confidence ≥0.25 แล้วส่ง RGB crop พร้อม context 10% เข้า parser โดยไม่ระบายพื้นหลัง หลังทำนายจึงใช้ YOLO instance mask จัด ownership โมเดลที่คืนหลาย instances เลือกด้วย predicted-box IoU ≥0.50 กับกรอบ YOLO โดยไม่ใช้ GT

คง native preprocessing, resolution, FP32, batch1, eval, no AMP/TF32/extra TTA ของแต่ละ checkpoint ทำนาย native20/native29 ก่อนแปลงคลาส บันทึก masks บนพิกัดภาพต้นฉบับ ทุกระบบใช้ GT/domain/matcher ชุดเดียวกันภายในงานนี้ ไม่ได้ทำให้ input conditioning ภายใน native architectures เหมือนกันทั้งหมด

จับคู่ผลกับ GT ด้วย bbox IoU ≥0.50 เรียง prediction ตาม confidence และ geometry เลือก GT ที่ IoU สูงสุดโดยไม่ซ้ำ กรณี geometry เสมอใช้ลำดับที่ประกาศไว้ ไม่มีการใช้ semantic score เลือกคู่ รายงาน bbox IoU ≥0.75 เป็น person diagnostic เพิ่ม คน GT ที่จับคู่ไม่เจอใช้ prediction ว่างและส่วน GT ยังคงได้ IoU ศูนย์ในตัวหาร

- **Hair/Upper/Lower:** เทียบ prediction ของคนที่จับคู่กับส่วน GT ของคนนั้นบน shared valid grid ทั้งภาพ ทายเกิน reference part เป็น FP ทายขาดเป็น FN ตามขอบอ้างอิงที่ผู้จัดทำยอมรับ
- **Face-local:** ใช้กรอบครอบ raw GT Face พอดี ไม่มี margin ตัดกับ shared valid grid พื้นที่นอกกรอบไม่เข้าสู่คะแนน ใช้ GT เฉพาะ scoring หลัง inference จึงไม่ใช่ face detection โดยไม่มีกรอบ GT
- **Shared VOID:** explicit ignore, คนต่ำกว่า 50 px, class/owner conflicts และ degenerate geometry ใช้ร่วมกันทุกโมเดล Source ID0 ยังคง Unassigned ไม่ใช่คำยืนยัน background ทั้งฉาก ไม่มี historical shape-index exclusions ใน fresh run นี้
- **ไม่มี part annotation:** ไม่สร้าง task ของ owner/class นั้น ไม่อนุมานว่าเป็น absent ส่วนที่ไม่มี valid GT support ยกเว้นตาม GT bindings

'''
    text += table(['Target', 'GT parts ที่ประเมิน', 'GT support (พิกเซล)'],
                  [[t, index['GT']['parts'][t], f"{index['GT']['support'][t]:,}"] for t in TARGETS])
    text += f"\n\nShared VOID รวม {index['GT']['VOID_pixels']:,} พิกเซล หนึ่ง Hair part ถูกยกเว้นทั้งหมดหลัง conflict VOID ส่วน overlaps ที่ผู้จัดทำยืนยันยังเก็บใน source ไม่แก้ polygon เพื่อเพิ่มคะแนน\n"
    text += '''
### ค่าการทดลองและข้อจำกัดทรัพยากร

| รายการ | ค่าที่บันทึกไว้ |
| --- | --- |
| GPU | NVIDIA Tesla T4; precision FP32, batch1, eval, no AMP/TF32/extra TTA |
| Detector cache | YOLO26x-Seg, imgsz 640, rect=False, retina_masks=True, NMS IoU setting 0.70, max_det 1000 |
| Detector confidence | สร้าง cache ด้วย floor 0.001 สำหรับ detector study แล้วใช้ ≥0.25 เป็น parser workload เดียวกันทุกโมเดล |
| Parser input | RGB crop +10% context แต่ละด้าน ขยายให้ครอบ detector mask; native resize/normalization ตาม checkpoint config ที่ผูก hashes |
| Warmup | CIHP TRAIN0000930 หนึ่งภาพต่อ quality process; 10 calls ต่อ dedicated process ไม่เข้าสู่คะแนน CCTV และไม่ใช่ training |
| Quality resource policy | SHARED_GPU_QUALITY_ONLY ตาม guards ไม่ปิด process งานอื่น |
| Guards | free disk ≥50 GiB, retained output ตาม max run/model 80 GiB, max model execution 24 ชั่วโมง, physical free VRAM ก่อน forward ≥1024 MiB และ model-specific admission ตาม config |
| Failure handling | stop affected work, atomic packets, immutable attempt history, zero automatic model/OOM retries; continuation ต้องตรวจ bindings ก่อน reuse |

Image/GT eligibility, valid domains และแหล่ง predictions ยึด current bindings ไม่ใช้ค่าตั้งต้นของ framework ตีความแทน resource/timing records Detector cache ไม่ได้ถูกสร้างใหม่ระหว่างการทดสอบ parser นี้
'''
    text += '''
### Mapping และความหมายของสี่คลาส

| Target | CIHP native IDs | Sapiens native IDs | ขอบเขตคำอ้าง |
| --- | --- | --- | --- |
| Hair | 2 | 4 | ผมตาม reference ที่ annotate |
| Face-local | 13 | 3 และ 24–28 | Sapiens Face_Neck + facial subparts เป็น proxy ภายในกรอบ GT Face ไม่รับรอง strict face-only/neck quality |
| Upper | 5 + 7 | 23 | UpperClothes + Coat เทียบ Upper_Clothing; non-Dress |
| Lower | 9 + 12 | 13 | Pants + Skirt เทียบ Lower_Clothing; non-Dress |
| OTHER | คลาสอื่น รวม Dress 6 | คลาสอื่น รวม Apparel 1 | ไม่มีการแบ่ง dress ด้วยเส้นตัดหรือปรับ mapping ตามคะแนน |

Source GT Hair=1/Face=2/Upper=3/Lower=4 แปลงเป็น shared target Face=1/Hair=2/Upper=3/Lower=4 ก่อนนับ การรวมคลาสใช้ union ของ masks จริง ไม่เฉลี่ย IoU ของ native subclasses Native prediction Background0 คงเป็น0 ส่วน recognized non-target classes map เป็น5=OTHER; คอลัมน์ OTHER ใน cross-confusion รวมทั้ง prediction0และ5 ภายหลัง Source GT0 ยังคงเป็น Unassigned ตาราง native mapping ครบ 0–19/0–28 อยู่ใน result index; unknown prediction IDs หรือ prediction255 เป็น contract failure

## ตัวชี้วัด: สูตร ตัวหาร และสิ่งที่บอก

TP คือพิกเซลทายถูก FP คือพิกเซลทายเกิน และ FN คือ GT ที่ทายขาด ภายใน domain ของ GT person-part นั้น

| ตัวชี้วัด | คำนวณอย่างไร | ใช้อธิบายอะไร |
| --- | --- | --- |
| IoU ของหนึ่งส่วน | TP ÷ (TP+FP+FN) | ความทับกันพร้อมโทษส่วนเกินและส่วนขาด |
| Mean-part IoU | บวก IoU ของแต่ละ GT part แล้วหารจำนวน eligible parts ในคลาส | ให้แต่ละส่วนมีน้ำหนักเท่ากัน รวมคนที่พลาดด้วยศูนย์ |
| Pooled IoU | รวม TP/FP/FN ในคลาสก่อนหาร | ความทับกันจากจำนวนพิกเซลรวม ต่างจากการเฉลี่ยรายส่วน |
| Precision | TP ÷ (TP+FP) จาก counts รวม | พิกเซลที่ทายมา ถูกต้องกี่ส่วน |
| Recall | TP ÷ (TP+FN) จาก counts รวม | เก็บพิกเซล GT ได้ครบแค่ไหน |
| F1/Dice | 2TP ÷ (2TP+FP+FN) | สมดุลการทายเกินและทายขาด; F1 เท่ากับ Dice เมื่อใช้ counts/domain เดียวกัน |
| Macro4-local | เฉลี่ย Mean-part IoU ทั้งสี่คลาส คลาสละ 25% | เกณฑ์เลือกหลักของงานนี้ ไม่ใช่ global semantic mIoU |
| mIoU3 part-mean | เฉลี่ย Hair/Upper/Lower คลาสละหนึ่งในสาม | ผลสามคลาสที่ไม่ใช้ Face-local ไม่ใช่ C3 global เดิม |
| ส่วนที่พลาดทั้งหมด | จำนวน GT parts ที่ TP=0 ÷ eligible parts | รวมทั้งคนที่จับคู่ไม่ได้และคนที่จับคู่ได้แต่ไม่มีพิกเซลถูก |

ตัวอย่างสมมติ: ส่วนเล็กมี IoU 0% ส่วนใหญ่มี IoU 100% จะได้ Mean-part IoU 50% ส่วน pooled ต้องใช้จำนวนพิกเซลจริงจึงยังคำนวณจากสองเปอร์เซ็นต์นี้ไม่ได้ ตัวหารศูนย์แสดง null/— ไม่แทนด้วย 0 คะแนนหลักของผลไม่ครบเป็น null ตาม STRICT_COMPLETE

ตัวอย่างผลจริงของ QANet: Hair มี GT 544 ส่วน ในจำนวนนี้ 106 ส่วนไม่มีพิกเซลทายถูกเลย โดย 62 ส่วนอยู่กับคนที่จับคู่ไม่ได้ และอีก 44 ส่วนจับคู่คนได้แต่ TP=0 กลุ่มหลังอาจมีผลว่าง ผิดคลาส ผิดตำแหน่งหรือ ownership ผิด ต้องดู masks เพื่อแยกสาเหตุ ไม่ใช่เพียงคำว่าขอบ mask ไม่เนียน

## Precision, Recall, F1 และความครบของแต่ละส่วน

'''
    text += table(['โมเดล','Target','Mean-part IoU %','Pooled IoU %','Precision %','Recall %','F1/Dice %','GT parts','TP=0 parts'],
                  [[m['name'],t,*[pct(m['result']['targets'][t][k]) for k in ('mean_part_IoU','pooled_IoU','precision','recall','F1_Dice')],
                    m['result']['targets'][t]['GT_parts'],m['result']['targets'][t]['entirely_missed_GT_parts']]
                   for m in index['models'] for t in TARGETS])
    text += '''

## Confusion matrix และการอ่านความผิดพลาด

![Binary confusion รายคลาสของเก้าโมเดล](figures/TARGET_CONFUSION.png)

แต่ละช่องย่อยเป็น binary matrix **[[TP,FN],[FP,TN]]** แถวเป็น GT คอลัมน์เป็น prediction สีและเปอร์เซ็นต์หารภายในแถว GT: ช่อง TP% ของแถว Target คือ Recall; FN% คือสัดส่วน GT ที่พลาด ไม่ใช่ Precision ของคอลัมน์ จำนวนพิกเซลในภาพเดียวกันอาจถูกนับในหลาย person-part tasks ค่า TN ใหญ่จึงไม่ใช้ accuracy มาสรุปว่า parser แม่นมาก

ตาราง cross-class ด้านล่างใช้เฉพาะ GT positive pixels: อ่านว่า Hair ถูกทายเป็น Face/Upper/Lower/OTHER เท่าไร ไม่ครอบคลุม FP ที่ทายบนพื้นที่นอก positive GT ต้องอ่านคู่กับ binary matrix และ Precision การไม่มีแถว GT OTHER ไม่ได้แปลว่าไม่มีการทายเกิน

![Cross-class confusion บน GT positive pixels](figures/CROSS_CLASS_CONFUSION.png)

ภาพ 4×5 นี้เป็นการแจกแจงผลทำนายแบบมีเงื่อนไขบนสี่ GT targets ที่มีคำตอบ ไม่ใช่ full-scene multiclass confusion พร้อม GT background แถวละ 100% ใช้มาตราสี 0–100 เหมือนกันทั้งเก้าโมเดล ช่องทแยงคือ pooled pixel Recall ของ target ไม่ใช่ Mean-part IoU ส่วนช่องนอกทแยงแยกการทายเป็น target อื่นกับ OTHER ตาราง counts เดิมอยู่ด้านล่างเพื่อย้อนตรวจเปอร์เซ็นต์

'''
    for m in index['models']:
        r = m['result']
        text += f"<details>\n<summary>{m['name']} — counts และ cross-class confusion</summary>\n\n"
        text += table(['GT / Prediction','Hair','Face-local','Upper','Lower','OTHER'],
                      [[t,*row] for t,row in zip(TARGETS,r['cross_class_confusion']['counts'])])
        text += '\n\n' + table(['Target','TP','FP','FN','TN','GT support','Union','จับคู่คนไม่ได้','TP=0 %'],
                    [[t,*[f"{r['targets'][t][k]:,}" for k in ('TP','FP','FN','TN','GT_pixels','union')],
                      r['targets'][t]['unmatched_GT_parts'],pct(r['targets'][t]['entirely_missed_fraction'])] for t in TARGETS])
        text += '\n\n</details>\n\n'
    text += insights(index)
    text += '''## การพบคนและความครบถ้วนของผล

ทุก parser ใช้ frozen YOLO extents/boxes ชุดเดียวกัน ตัวเลขการจับคู่จึงเท่ากัน ไม่ใช่หลักฐานว่าแต่ละ parser มี native instance capability เท่ากัน ข้อมูลมีคนบางส่วนไม่ annotate จึงรายงาน annotated-person recall และ unmatched diagnostics; **full-scene person AP/Precision/F1 เป็น null** ไม่เรียก unmatched ทั้งหมดว่า false positives

'''
    loc = index['models'][0]['result']['localization']
    text += table(['Bbox IoU','GT คน','จับคู่ได้','GT คนที่พลาด','Unmatched predictions','Ignored predictions','Annotated-person recall %'],
                  [[threshold, a['GT'], a['TP'], a['FN'], a['unmatched_predictions'], a['ignored_predictions'], pct(a['annotated_person_recall'])]
                   for threshold,a in loc.items()])
    text += '\n\n' + table(['โมเดล','ภาพที่ตรวจรับ','ROI ที่ตรวจรับ','Missing/duplicate','Failed/invalid/setup ในผลสุดท้าย','STRICT_COMPLETE'],
                [[m['name'],'600/600','763/763','0/0','0/0/0','PASS'] for m in index['models']])
    text += '''

คะแนนหลักอ่านจาก canonical METRICS.json หลัง scoring ช่อง primary:null ใน collector COMPLETION.json เป็นสถานะก่อน scoring ที่เก็บไว้ตามเดิม ไม่ใช้แทนผลคะแนนและไม่เปลี่ยน null เป็น 0 Result index สาธารณะเชื่อม hashes ของทั้ง metrics และ completion เพื่อให้เห็นที่มาของค่าทั้งสอง

## เวลาและทรัพยากร

### เวลารอบเก็บผลคุณภาพ: ครบ 600 ภาพ

Mean ms/ROI = ผลรวม adapter intervals ÷ 763 งาน Mean s/ภาพ = ผลรวมเดียวกันในหน่วยวินาที ÷ 600 ภาพ ภาพศูนย์ ROI มีงาน parsing ศูนย์ ไม่รวมภาพที่ขาด timer เป็นศูนย์แทน รอบนี้มีภาพศูนย์ ROI 112 ภาพ การไม่มี ROI ไม่ยืนยันว่าไม่มีคนจริง

'''
    text += table(['โมเดล','Mean ms/ROI','Mean parsing s/ภาพ','รวม parsing s','ภาพมี timer','Peak allocated MiB','Peak reserved MiB'],
                  [[m['name'],num(m['timing']['quality']['mean_ms_per_ROI']),num(m['timing']['quality']['mean_s_per_image']),
                    num(m['timing']['quality']['total_s']),'600/600',num(m['timing']['quality']['peak_allocated_MiB']),
                    num(m['timing']['quality']['peak_reserved_MiB'])] for m in index['models']])
    text += '''

### Dedicated timing: 100 ภาพเดิม × 3 รอบ

แต่ละรอบมี 124 ROI, TRAIN warmup 10 calls ต่อ process และตรวจไม่มี external GPU compute PID ตาม guards ใช้ค่าเฉลี่ยของ mean รายรอบที่จำนวนงานเท่ากัน ช่วงต่ำสุด–สูงสุดเป็นความต่างระหว่างสามรอบ ไม่ใช่ confidence interval ตารางนี้แยกจาก quality timing ด้านบน

'''
    time_rows, stages, application = [], [], []
    for m in index['models']:
        rounds = list(m['timing']['dedicated_rounds'].values())
        values = [r['mean_ms_per_ROI'] for r in rounds]
        time_rows.append([m['name'], num(statistics.mean(values)), f'{min(values):.3f}–{max(values):.3f}',
                          num(statistics.mean(r['mean_s_per_image'] for r in rounds)), '3/3'])
        q = m['timing']['quality']
        c = q['observed_totals_ms']
        stages.append([m['name'], *[num(c[k]/763) if c.get(k) is not None else '—' for k in
                      ('pre_module_ms','module_call_ms','inter_module_ms','post_module_ms')],num(q['mean_ms_per_ROI'])])
        application.append([m['name'], *[num(c[k]/600) if c.get(k) is not None else '—' for k in
                            ('source_read_ms','cached_detector_read_ms','composition_projection_ms','frame_packet_write_and_validation_ms','application_from_cached_inputs_ms')]])
    text += table(['โมเดล','Dedicated mean ms/ROI','ช่วง mean รายรอบ ms/ROI','Dedicated mean parsing s/ภาพ','Clean rounds'],time_rows)
    text += '''

### ส่วนประกอบ adapter: หน่วย ms/ROI ใน quality run

แต่ละ call ทำนาย native classes ทั้งหมดแล้วจึง map เป็นสี่ส่วน จึงไม่มีเวลาของ Hair/Face/Upper/Lower แยกอิสระ ขอบเขต module API ต่างกันตาม implementation คอลัมน์ “ใน module” อาจรวม native internal detector/postprocessing/export โดยเฉพาะ CPI จึงไม่เรียกว่า backbone-only หรือ clean parser-only latency

'''
    text += table(['โมเดล','ก่อน module','ใน module API','ระหว่าง modules','หลัง module/map-back','Adapter รวม'],stages)
    text += '''

ก่อน/ใน/ระหว่าง/หลัง module เป็นสี่ช่วงที่แยกกันและรวมเป็น adapter; **อย่าบวก adapter ซ้ำกับสี่ช่วงย่อย** จุด synchronize และ hooks เพิ่ม measurement overhead ส่วน ownership, crop reads และ packet writes อยู่ภายนอก adapter

### งานประกอบจาก cached inputs: หน่วย ms/ภาพ ใน quality run

'''
    text += table(['โมเดล','อ่านภาพ','อ่าน YOLO cache','Compose/project','Frame packet write/validate','Application รวมจาก cached inputs'],application)
    text += '''

Application รวมครอบงานอ่านไฟล์/ตรวจ hashes/resource checks/logging/inference/ownership/ส่งออกตามขอบเขต worker ไม่เท่ากับผลบวกเฉพาะคอลัมน์ที่แสดง และไม่บวกซ้ำกับ adapter รวม **ไม่มี YOLO inference ใหม่, CPU scoring หรือการสกัดสีในช่วงนี้** จึงไม่ใช่ end-to-end ระบบเว็บ ส่วนเวลาแยกของ Sapiens2 1B quality มีเพียง adapter intervals ที่ตรวจรับข้ามสอง processes; application/I-O totals ของเฟรมขัดจังหวะยัง null ตารางแสดง — โดยไม่เติมค่าประมาณ สาม dedicated rounds ของ 1B มี timers ครบตาม scope เดียวกับ dedicated rounds ของตัวอื่น

MiB = 2²⁰ bytes; peak allocated คือ tensor memory ที่ PyTorch จัดสรรสูงสุด ส่วน peak reserved รวม CUDA allocator cache ของ process ไม่เท่ากับ nvidia-smi หรือขนาด checkpoint Quality inference ใช้ shared-GPU quality policy ส่วน dedicated timing มี clean-resource guards ขอบเขต API/ROI workload ยังต่างกัน จึงรายงาน diagnostics และไม่จัดอันดับ latency/FPS/VRAM efficiency

## รุ่นโมเดลและ provenance

จำนวน parameters นับจาก loaded model ใน MODEL_DETAILS ไม่อนุมานจากชื่อรุ่น เช่นชื่อ Sapiens2 1B เป็นชื่อ checkpoint; model segmentation ที่โหลดมี parameters รวมตามตาราง ไม่ใช้ขนาด weights ทำนายความต้องการ VRAM

'''
    text += table(['โมเดล','Parameters (ล้าน)','Native classes','Implementation','Source commit'],
                  [[m['name'],num(m['parameters']/1e6),m['native_classes'],m['implementation'],f"`{m['binding']['source_commit'][:12]}`"] for m in index['models']])
    text += '\n\n<details>\n<summary>Checkpoint/config hashes และ environments ของทั้งเก้าโมเดล</summary>\n\n'
    for m in index['models']:
        b, e = m['binding'], m['environment']
        text += f"**{m['name']}** — [source]({m['upstream']}/tree/{b['source_commit']})\n\n"
        text += f"- Checkpoint: `{b['checkpoint']}`; SHA256 `{b['checkpoint_sha256']}`\n"
        text += f"- Native config: `{b['native_config']}`; SHA256 `{b['native_config_sha256']}`\n"
        text += f"- Environment: `{b['environment']}`; Python {e['python'].split()[0]}, PyTorch {e['torch']}, CUDA runtime {e['CUDA']}, NumPy {e['numpy']}, GPU {e['GPU']}\n"
        text += f"- Package-freeze SHA256: `{m['package_freeze_sha256']}`; route `{b['route']}`\n\n"
    text += '</details>\n\n'
    text += f"Run หลัก `{index['parent_run_id']}` และ continuation `{index['run_id']}`; launch commits " + ', '.join(f'`{s}`' for s in index['provenance']['launch_commits']) + '\n\n'
    text += f"Dataset membership SHA256 `{index['dataset_membership_sha256']}`; effective protocol SHA256 `{index['protocol_sha256']}`; mapping SHA256 `{index['mapping_sha256']}` ค่า protocol ที่ใช้จริงอยู่ใน result index แยกจากแม่แบบ config เดิม ไม่มีการแก้ frozen scientific config/source ในงานจัดรายงาน\n"
    text += '''
### การหยุดและทำต่อ

ผลรอบสุดท้ายใช้ parsing observations ของ fresh campaign นี้ ผลก่อน campaign ไม่ถูกนำมาเติมคะแนน First attempt ที่ผู้ใช้กด Ctrl+C เก็บเป็นประวัติ จากนั้นเริ่ม fresh campaign ใหม่ครบเก้าโมเดล 1B ใน campaign นี้หยุดเพราะ controller ตรวจพื้นที่ด้วย du ระหว่าง atomic rename และส่ง SIGTERM ไปยัง worker เป็น orchestration resource-inspection failure ไม่ใช่หลักฐาน OOM

Reconciliation ตรวจรับ 737 parent ROI ของ 1B แล้วทำเฉพาะ 26 ROI ที่เหลือ รวม 763 งาน แปดโมเดลที่ปิดแล้วไม่ inference/rescore ซ้ำเพื่อจัดรายงาน เวลารอบ 1B quality รวม accepted adapter intervals ข้ามสอง processes; warmups และ failed attempts ไม่ถูกปลอมเป็น accepted latency Final failed/invalid=0 บอกสถานะผลที่ตรวจรับ ไม่ได้ลบประวัติ failures

### สิ่งที่ตรวจจริง

- CPU acceptance ก่อน launch ผูก config/protocol hashes: union/OTHER, Face-local, VOID, missed/empty persons, label/geometry, duplicate membership, null และ packet→scorer→reporter รวมถึง immutability/resume ตาม suite เดิม
- Final completion และ assembled indices ตรง exact 600 unique images/763 ROI ต่อระบบ ไม่มี missing/duplicate/unexpected และใช้ canonical METRICS ที่ตรวจรับแล้ว
- งาน publication ตรวจ TP+FN=GT support, union=TP+FP+FN, Precision/Recall/F1/mean arithmetic, binary กับ cross-confusion, denominators/27 clean rounds และหน่วยเวลา จาก sufficient statistics ไม่ full rescore หรือ rehash predictions ซ้ำ
- Gallery ตรวจ membership 16 ภาพเดิม, ครบเก้าโมเดลทุก panel และ PNG hashes ตรง manifest ภาพยังเก็บ private ตามนโยบาย dataset

## ภาพเปรียบเทียบจาก saved predictions

ชุด 16 ภาพล็อกโดย detector manifest ก่อนดูคะแนน parser ทุกภาพมี RGB, GT และผลเก้าโมเดลรวม 11 panels ใช้ชุดเดิมทั้งหมด ไม่เลือกภาพตามคะแนน Gallery แสดงการ compose เพื่อดูภาพ ส่วนคะแนนคำนวณ per-person planes โดยไม่ใช้ภาพ overlay เป็น input ของ scorer GT ที่แสดงไม่ยืนยันพื้นที่นอก annotation ว่าเป็น background

![สีแสดงคลาสใน private gallery](figures/TARGET_LEGEND.svg)

สีม่วง=Hair, ส้ม=Face proxy, เขียว=Upper และน้ำเงิน=Lower ตาม palette ของ gallery พื้นที่อื่นโปร่งใส สีนี้ใช้แสดง label เท่านั้น ไม่ใช่ผลสกัดสีเสื้อผ้า และรอบนี้ไม่ได้ประเมินสี

บน server เปิดไฟล์ `outputs/<run_id>/gallery/index.html` โดยใช้ run_id ใน result index หรือดูตำแหน่งจริงผ่าน `outputs/CURRENT.json` ดาวน์โหลดทั้งโฟลเดอร์ gallery ไปเปิดในเครื่องได้ โดยเก็บ index.html กับ PNGs ในโฟลเดอร์เดียวกัน ภาพจริง/GT/private manifest ไม่เผยแพร่บน public Git; ภาพ heatmap/confusion ในรายงานนี้มีเฉพาะสถิติรวม

## มาตรฐาน แหล่งอ้างอิง และขอบเขตคำอ้าง

| แหล่งอ้างอิง | ใช้ประกอบเรื่องใด | ไม่ได้ยืนยันสิ่งใดให้ protocol นี้ |
| --- | --- | --- |
| [CIHP/PGN — Gong et al., ECCV 2018](https://arxiv.org/abs/1808.00157v1) | แยก semantic parts กับ instance ownership และบริบท parsing-by-detection | ไม่รับรองว่า YOLO-crop evaluation ของเราเป็น official CIHP reproduction |
| [MMSegmentation IoUMetric v1.2.2](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/evaluation/metrics/iou_metric.py) | สูตร intersection/union, Dice, Precision/Recall/F-score และการกรอง ignore ก่อนรวม counts | เป็น official implementation reference ไม่ใช่ paper ที่กำหนด Mean-part/Face-local/Macro4-local ของเรา |
| [scikit-learn confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) | นิยามแถวเป็น GT/คอลัมน์เป็น prediction, label order และการ normalize ตามแถว | ไม่รับรองว่า task-domain/coverage ของ CCTV600 เทียบเท่า full-scene benchmark |
| [NightOwls annotation definitions](https://www.nightowls-dataset.org/examples/) | ตัวอย่างการประกาศเกณฑ์คนขนาดเล็กและ ignore | NightOwls ใช้ความสูง >50 px; ของเราใช้ด้านยาว ≥50 px จึงไม่ใช่เกณฑ์เดียวกัน |
| [Datasheets for Datasets — Gebru et al., 2021](https://arxiv.org/abs/1803.09010v8) | เปิดเผยที่มา องค์ประกอบ กระบวนการสร้าง การใช้และข้อจำกัด | ไม่ใช่การรับรองคุณภาพ annotation หรือ scorer ของชุดนี้ |
| [CCTV90 shared-reference ที่ใช้เป็นแบบ](https://github.com/folklazy/Human_Parsing_Study/blob/29f7030/CCTV_Human_Parsing_Benchmark/reports/CONTROLLED_ROI_FOUR_TARGET_RESULTS_TH.md) | สูตร counts/Mean-part/Face-local และรูปแบบรายงานเดิม | GT/matcher/YOLO cache/timer ข้ามสองงานไม่เหมือนกันทุกประการ |

Face-local, Mean-part IoU และ Macro4-local เป็น **protocol เฉพาะที่ประกาศชัดและใช้สูตรพื้นฐานมาตรฐาน** ไม่ใช่ official benchmark metric ที่ paper รับรอง นิยามนี้สืบจากงานที่เห็นผลก่อนหน้าแล้ว จึงไม่เรียก preregistered/confirmatory protocol ไม่มีการเลือก mapping หรือ threshold ตามคะแนนชุด 600 นี้

ข้อมูลมี selected-person/part coverage, helper-assisted annotation, การยืนยันจากผู้จัดทำและข้อจำกัด contour/overlap ไม่มี independent expert agreement หรือ independent held-out certification สำหรับทุกการใช้ ขอบ positive part masks ที่ผู้จัดทำรับรองเป็นสมมติฐานของงานรายส่วน ถ้าขอบที่วาดไม่ครบ คะแนนจะวัดความตรงกับ reference นั้น ไม่รับรองความถูกต้องทางกายวิภาคทุกพิกเซล

Face_Neck proxy, garment unions และการไม่มี structured Dress/visibility/absence tags ครบ ทำให้ไม่อ้าง strict taxonomy equivalence คะแนนนี้เปรียบเทียบ checkpoint พร้อม application configuration ของ CIHP-trained parsers หก checkpoints และ Sapiens2 segmentation สามขนาดบน CCTV600 ไม่ใช่ architecture-only ranking ไม่ยืนยันความแม่นยำสี, tracking, CCTV ทุกสภาพ หรือการ deploy ข้อกำหนดสิทธิ์ใช้ code/weights รวมข้อจำกัดของ Sapiens2 ยังคงเป็นไปตามเจ้าของโมเดล

## ไฟล์และการทำซ้ำส่วนรายงาน

Public Git มีรายงาน, aggregate CSV/JSON, figures และ source; ภาพ/annotation/weights/venvs/raw ledgers/predictions อยู่ภายนอก ส่วน metrics/RESULT_INDEX.json มี stats แบบไม่ปัดเศษและ hashes ที่เชื่อม canonical results ไม่ใช่ raw log

คะแนนใน CSV/JSON เก็บเป็นสัดส่วน 0–1 แบบไม่ปัดเศษ ส่วนตาราง Markdown แปลงเป็นเปอร์เซ็นต์ เช่น 0.5751689 แสดง 57.517% Counts เป็นจำนวนพิกเซล/ส่วนตามชื่อคอลัมน์ และ TIMING ระบุ ms หรือ s ในชื่อคอลัมน์ ค่า — ใน Markdown และช่องว่างใน CSV หมายถึง null/ไม่มี observation ที่ใช้ได้ ไม่ใช่ 0 การตรวจรับนี้ไม่เพิ่ม annotation หรือให้คะแนนใหม่เพื่อให้เอกสารครบ

```bash
# จากโฟลเดอร์ repository: ตรวจ aggregate arithmetic และไฟล์ส่งมอบ โดยไม่โหลดโมเดล
python3 src/publication.py --check

# สร้าง CSV/กราฟ/ตารางจาก aggregate index เดิมบน CPU
# ต้องมี matplotlib/numpy ใน environment ที่จัดเตรียมไว้; ไม่ต้องมี GPU/weights/dataset
python3 src/publication.py --from-index metrics/RESULT_INDEX.json
```

Inference/scoring ที่ reproduce ถึง predictions ต้องมี private dataset, frozen detector/crops, adapter sources จาก workspace เดิม และ environments/checkpoints ที่ hashes ตรง การ clone repo นี้อย่างเดียวไม่ใช่คำรับรองว่าจะ rerun inference ได้ทันที คำสั่ง publication ข้างต้นไม่เปิด worker และไม่เปลี่ยน native benchmark

'''
    return text


def readme(index):
    return f'''# Human Parsing บน CCTV Human Parsing 600 Dataset

**ปิดรอบครบ 9/9 โมเดล — 600 ภาพและ 763 ROI ต่อโมเดล** พร้อม dedicated timing บน Tesla T4 ครบโมเดลละ 3 รอบ รายงานนี้ใช้คำตอบของบุคคลและส่วนที่มี annotation ไม่ใช่คะแนนเต็มฉาก

[รายงานผลฉบับเต็ม](reports/CCTV600_RESULTS_TH.md) · [คะแนน CSV](metrics/SUMMARY.csv) · [รายคลาส CSV](metrics/PER_TARGET.csv) · [เวลา CSV](metrics/TIMING.csv) · [Result index/hashes](metrics/RESULT_INDEX.json) · [สถานะ](STATUS.md) · [Dataset/คู่มือ annotation](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset)

## ผลหลัก

{primary_summary(index)} Sapiens2 1B สูงสุดด้าน Hair และ Face-local ส่วน QANet สูงสุดด้าน Upper/Lower เมื่อใช้ Mean-part IoU การเลือกนี้อ้างคุณภาพบน reference ชุดนี้ ไม่ใช่ deployment acceptance หรือผลนัยสำคัญทางสถิติ

{summary_table(index)}

![คะแนนรายส่วนของเก้าโมเดล](reports/figures/QUALITY_OVERVIEW.png)

## อ่านคะแนนให้ตรงคำถาม

| ชื่อ | ความหมาย |
| --- | --- |
| Mean-part IoU | คิด IoU ของแต่ละ GT part ก่อนเฉลี่ย แต่ละส่วนมีน้ำหนักเท่ากัน |
| Face-local | คิด IoU ภายในกรอบ GT Face พอดี โดยใช้ Sapiens Face_Neck + facial subparts เป็น proxy |
| Macro4-local | เฉลี่ย Hair/Face-local/Upper/Lower คลาสละ 25% |
| mIoU3 part-mean | เฉลี่ย Hair/Upper/Lower เท่ากัน ไม่ใช่ C3 global เดิม |
| Pooled IoU / Precision / Recall / F1-Dice | รวมพิกเซล TP/FP/FN ก่อนหาร อ่านแยกจาก Mean-part |

GT ที่ใช้จริง: **769 คน; Hair 544 / Face 384 / Upper 745 / Lower 668 ส่วน** Missing parts ไม่เท่ากับ confirmed absent คน GT ที่จับคู่ไม่ได้ยังได้ศูนย์และอยู่ในตัวหาร Source ID0 เป็น Unassigned; full-scene person AP/Precision/F1 และ global semantic background metrics ยังไม่รองรับจาก coverage นี้

## Pipeline และการเปรียบเทียบ

ภาพ → frozen YOLO26x-Seg → RGB person crop พร้อม context 10% → native parser FP32/batch1/no extra TTA → original-grid mask → YOLO mask ownership หลังทำนาย → native-class mapping → GT bbox matching/part scoring

ทุกโมเดลใช้ input population, YOLO cache/crops, GT valid domain และ matcher ร่วมกัน คะแนนอ้าง checkpoint + application configuration การใช้ native20/native29, Face_Neck proxy และ Dress/Apparel เป็น OTHER ไม่ทำให้ taxonomy เท่ากันทั้งหมด ขอบเขต Face-local และเกณฑ์ขนาด ≥50 px เป็นกติกาเฉพาะที่ประกาศไว้

## รายละเอียดที่เปิดดูได้

- [Precision/Recall/F1 และจำนวนส่วนที่พลาด](reports/CCTV600_RESULTS_TH.md#precision-recall-f1-และความครบของแต่ละส่วน) พร้อม counts/support/union และสูตร
- [Confusion matrix](reports/CCTV600_RESULTS_TH.md#confusion-matrix-และการอ่านความผิดพลาด): binary 2×2 ครบ 36 ชุด และ cross-class 4×5 ครบเก้าโมเดล
- [Insight จาก confusion](reports/CCTV600_RESULTS_TH.md#insight-จาก-confusion-และตัวชี้วัดร่วมกัน): OTHER/ทายผิดคลาส, Precision–Recall trade-off และการแยกคนที่จับคู่ไม่ได้จากส่วนที่ parser พลาด
- [เวลาแต่ละขั้นและ Peak VRAM](reports/CCTV600_RESULTS_TH.md#เวลาและทรัพยากร): แยก quality run กับ dedicated timing รวมถึงหน่วยต่อ ROI/ภาพ/ทั้งชุด
- [Model/checkpoint/config/environment bindings](reports/CCTV600_RESULTS_TH.md#รุ่นโมเดลและ-provenance) รวม implementation revisions ของ CPI และ hashes
- [หลักฐานมาตรฐานและข้อจำกัด](reports/CCTV600_RESULTS_TH.md#มาตรฐาน-แหล่งอ้างอิง-และขอบเขตคำอ้าง): paper/official implementation แต่ละแหล่งรองรับเรื่องใด และกติกาใดออกแบบเอง
- [ตรวจเทียบ CCTV90](reports/CCTV600_RESULTS_TH.md#cctv90-comparison): สูตรร่วมกัน แต่ภาพ/GT/matcher/YOLO cache/timer ไม่เหมือนกันทุกประการ

ภาพ RGB/GT/predictions จริงอยู่ใน private gallery ชุด **16 ภาพเดิม × 9 โมเดล** ไม่เลือกตามคะแนน อ่าน [วิธีเปิด gallery](reports/CCTV600_RESULTS_TH.md#ภาพเปรียบเทียบจาก-saved-predictions) ภาพสถิติใน Git ไม่มีภาพ dataset

## ตรวจรายงานและสร้างตารางซ้ำ

```bash
python3 src/publication.py --check
python3 src/publication.py --from-index metrics/RESULT_INDEX.json
```

คำสั่งแรกใช้ Python standard library ส่วนคำสั่งสร้างกราฟต้องมี numpy/matplotlib ใน environment เดิมที่รองรับ เป็นงาน CPU จาก accepted aggregates ไม่มี inference/rescore/download ข้อมูลคนละภาพไม่ขึ้น Git และ null ไม่ถูกแทนด้วย 0 รายละเอียดการรันโมเดลต้องมี private assets และ workspace adapters ตาม bindings ไม่ใช่ติดตั้งจาก requirements แล้ว reproduce ได้ทันที

[Source scorer](src/scoped_metrics.py) · [Native collector](src/reference_collection.py) · [Timer](src/worker.py) · [Publication source](src/publication.py) · [บันทึกงาน](WORK_LOG.md)

แหล่งอ้างอิงหลัก: [CIHP/PGN](https://arxiv.org/abs/1808.00157v1), [MMSegmentation IoUMetric](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/evaluation/metrics/iou_metric.py), [NightOwls](https://www.nightowls-dataset.org/examples/), [Datasheets for Datasets](https://arxiv.org/abs/1803.09010v8) ใช้ประกอบการออกแบบและตรวจสูตร ไม่ใช่การรับรอง protocol/annotation ของชุดนี้จากเจ้าของ paper

สถานะและไฟล์ evidence ที่ระบุ run เก่าเป็น historical snapshots ให้ยึด [result index ปัจจุบัน](metrics/RESULT_INDEX.json) และรายงานหลักสำหรับผลส่งมอบ ประวัติการขัดจังหวะเก็บไว้ ไม่มีการนำ failed attempts มาปิดบังด้วย final failed=0
'''


def check_files(destination, index):
    validate(index)
    import re
    for relative in ('README.md', 'reports/CCTV600_RESULTS_TH.md'):
        path = destination/relative
        text = path.read_text()
        assert '/home/' not in text and summary_table(index) in text
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('http://', 'https://')):
                continue
            filename, _, fragment = target.partition('#')
            linked = path.parent/filename if filename else path
            assert linked.exists(), (relative, target)
            if fragment and linked.suffix == '.md':
                body = linked.read_text()
                anchors = set(re.findall(r'<a id="([^"]+)"', body))
                for heading in re.findall(r'^#+ (.+)$', body, re.M):
                    anchors.add(re.sub(r'[^\w\u0e00-\u0e7f -]', '', heading.lower()).replace(' ', '-'))
                assert fragment in anchors, (relative, target)
        columns = None
        for line in text.splitlines():
            if line.startswith('| '):
                assert line.endswith(' |'), (relative, line)
                assert columns in (None, line.count('|')), (relative, 'inconsistent table width')
                columns = line.count('|')
            else:
                columns = None
    expected = {'SUMMARY.csv':9, 'PER_TARGET.csv':36, 'TIMING.csv':36}
    generated = csv_rows(index)
    for name, count in expected.items():
        with (destination/'metrics'/name).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
        assert rows == [{k: '' if v is None else str(v) for k, v in r.items()} for r in generated[name]], name
    for name in ('QUALITY_OVERVIEW.png', 'TARGET_CONFUSION.png', 'CROSS_CLASS_CONFUSION.png'):
        assert (destination/'reports/figures'/name).read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'


def build(index, destination):
    validate(index)
    index['publication'] = dict(source='src/publication.py', source_sha256=sha(__file__),
        new_model_calls=0, new_saved_mask_rescores=0, statistics_source='accepted canonical aggregates',
        CSV_rows=dict(SUMMARY=9, PER_TARGET=36, TIMING=36),
        checks=['integer/ratio arithmetic', 'membership', '27 timing rounds', 'CSV values', 'local links and anchors'])
    metrics = destination/'metrics'
    metrics.mkdir(parents=True, exist_ok=True)
    dump(metrics/'RESULT_INDEX.json', index)
    export_csv(index, metrics)
    plots(index, destination/'reports/figures')
    path = destination/'reports/CCTV600_RESULTS_TH.md'
    old = path.read_text() if path.exists() else ''
    anchor = '<a id="cctv90-comparison"></a>'
    comparison = old[old.index(anchor):] if anchor in old else ''
    path.write_text(report(index) + comparison)
    (destination/'README.md').write_text(readme(index))
    check_files(destination, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-run', type=Path)
    group.add_argument('--from-index', type=Path)
    group.add_argument('--check', action='store_true')
    parser.add_argument('--destination', type=Path, default=PROJECT)
    args = parser.parse_args()
    if args.check:
        index = read(args.destination/'metrics/RESULT_INDEX.json')
        check_files(args.destination, index)
        print('PASS: canonical aggregate arithmetic, 9 models, 27 timing rounds, CSV row counts and publication links; no model calls')
    else:
        index = collect(args.from_run.resolve()) if args.from_run else read(args.from_index)
        build(index, args.destination)
        print('Published accepted aggregate tables/figures only; no model calls or rescoring')


if __name__ == '__main__':
    main()
