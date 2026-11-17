"""Render canonical statistics only after the batch, preserving JSON nulls."""
import csv
from pathlib import Path

import numpy as np

from common import atomic, load, now


def percent(value):
    return '—' if value is None else f'{value * 100:.3f}'


def report(root):
    config = load(root / 'CONFIG.json')
    results = load(root / 'RESULT_INDEX.json')
    text = ['# ผล Human Parsing บน CCTV Human Parsing 600', '',
            f"ตรวจปิด batch เมื่อ {now()} (UTC)", '',
            'คะแนน AD4 ใช้เฉพาะพิกเซลสี่ส่วนที่ annotate และผ่านกติกา GT ร่วมกัน '
            'พื้นที่ไม่มี annotation เป็น unknown การทำนายเกินเข้า unknown จึงยังประเมิน FP ไม่ได้ '
            'คะแนนเต็มภาพยังเป็น null ทุกระบบ ผลนี้ใช้เปรียบเทียบความสอดคล้องบนพื้นที่ที่มีคำตอบ', '',
            'Face ของ Sapiens รวม Face_Neck และรายละเอียดใบหน้าเป็น proxy '
            'Upper/Lower รวมคลาสเสื้อผ้าตาม mapping โดย Dress/Apparel อยู่ OTHER '
            'GT ไม่มีแท็กเสื้อผ้าชิ้นเดียวครบ จึงยังรับรอง strict garment equivalence ไม่ได้', '',
            '| โมเดล | สถานะ | Hair IoU % | Face proxy IoU % | Upper IoU % | Lower IoU % | AD4 % | C3 % |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for mid, result in results.items():
        primary = result.get('primary')
        cells = [percent(primary['IoU'][c]) for c in range(1, 5)] if primary else ['—'] * 4
        cells += [percent(primary['AD4_mIoU4']), percent(primary['C3_global'])] if primary else ['—'] * 2
        text.append('| ' + ' | '.join([mid, result['status']] + cells) + ' |')
    text += ['', '## ตัวชี้วัด', '',
             'รวม confusion ของภาพครบชุดก่อนหาร โดย TP คือพิกเซลถูกคลาส '
             'FP คือทำนายคลาสนั้นบน GT คลาสอื่นใน valid domain และ FN คือ GT คลาสนั้นแต่ทำนายเป็นอย่างอื่น', '',
             '- IoU = TP / (TP + FP + FN)',
             '- Precision = TP / (TP + FP)',
             '- Recall = TP / (TP + FN)',
             '- F1 หรือ Dice = 2TP / (2TP + FP + FN)',
             '- AD4 คือค่าเฉลี่ย IoU ของสี่ target ส่วน C3 เฉลี่ย Hair, Upper, Lower และยังนับ GT Face เป็น valid non-target',
             '- Q_AD4_GT เฉลี่ยคะแนนราย target จาก GT person-parts ที่มี annotation บน domain ร่วม '
             'คนที่จับคู่ไม่เจอให้ศูนย์ เงื่อนไข bbox IoU 0.50/0.75 แยกรายงาน '
             'ตัวหารมาจาก GT bindings การไม่มี annotation ของส่วนนั้นยังถือเป็น unknown', '',
             '## Precision, Recall, F1 และ support ราย target', '',
             '| โมเดล | Target | Precision % | Recall % | F1/Dice % | GT pixels | Pred pixels ใน domain | Union pixels |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    csv_rows = []
    for mid, result in results.items():
        primary = result.get('primary')
        if not primary:
            continue
        for c, target in enumerate(('Hair', 'Face proxy', 'Upper', 'Lower'), 1):
            row = dict(model=mid, target=target, IoU=primary['IoU'][c], precision=primary['precision'][c],
                       recall=primary['recall'][c], F1_Dice=primary['F1_Dice'][c],
                       GT_pixels=primary['GT_support'][c], prediction_pixels=primary['prediction_support'][c],
                       union=primary['union'][c])
            csv_rows.append(row)
            text.append('| ' + ' | '.join([mid, target, percent(row['precision']), percent(row['recall']),
                percent(row['F1_Dice']), str(row['GT_pixels']), str(row['prediction_pixels']), str(row['union'])]) + ' |')
    for mid, result in results.items():
        primary = result.get('primary')
        if not primary:
            continue
        text += ['', f'### Confusion ของ {mid}', '',
                 'แถวเป็น GT คอลัมน์เป็น prediction จำนวนพิกเซล บน annotation-supported domain', '',
                 '| GT / Prediction | OTHER | Hair | Face | Upper | Lower |', '| --- | --- | --- | --- | --- | --- |']
        for name, values in zip(('OTHER', 'Hair', 'Face', 'Upper', 'Lower'), primary['confusion']):
            text.append('| ' + name + ' | ' + ' | '.join(map(str, values)) + ' |')
    if csv_rows:
        with (root / 'TARGET_RESULTS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    text += ['', '## Person coverage และ semantic ต่อ GT', '',
             '| โมเดล | Box IoU | TP | FP | FN | F1 % | Eligible person-parts Hair/Face/Upper/Lower | Q_AD4_GT % |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for mid, result in results.items():
        for threshold, item in result.get('person', {}).items():
            text.append('| ' + ' | '.join([mid, threshold, str(item['tp']), str(item['fp']), str(item['fn']),
                percent(item['F1']), str(item['eligible_person_parts']), percent(item['Q_AD4_GT'])]) + ' |')
    text += ['', '## เวลา', '',
             'เวลา adapter ต่อ ROI ครอบคลุม native classes ทั้งหมด ใช้ crop RGB เดียวกัน '
             'module-call timer ของบางระบบรวม native postprocessing หรือ file I/O อยู่ภายใน '
             'pre/post module เป็นขอบเขต API ที่วัดจริง ซึ่งยังแยกเป็น clean parser-only ทุกระบบไม่ได้', '',
             '| โมเดล | ภาพมี log | ROI calls | Mean parsing work ms/ภาพ | รวม parsing work s | Clean timing rounds /3 |',
             '| --- | --- | --- | --- | --- | --- |']
    timing_summary = {}
    for mid in config['models']:
        records = [load(path) for path in sorted((root / mid / 'image_records').glob('*.json'))]
        total = sum(r['parsing_work_ms'] for r in records)
        rounds = []
        for number in range(1, 4):
            directory = root / 'timing' / ('round-' + str(number)) / mid
            completion = load(directory / 'COMPLETION.json') if (directory / 'COMPLETION.json').exists() else {}
            if completion.get('strict_complete') and completion.get('prediction_status') == 'TIMING_COMPLETE':
                values = [load(path) for path in (directory / 'image_records').glob('*.json')]
                rounds.append(dict(round=number, images=len(values), ROI_calls=sum(r['ROI_calls'] for r in values),
                                   parsing_work_ms=sum(r['parsing_work_ms'] for r in values)))
        item = dict(observed_images=len(records), ROI_calls=sum(r['ROI_calls'] for r in records),
                    observed_parsing_work_s=total / 1000,
                    mean_parsing_work_ms_per_image=total / 600 if len(records) == 600 else None,
                    total_parsing_work_s=total / 1000 if len(records) == 600 else None,
                    clean_rounds=rounds, speed_ranking_eligible=False)
        stage_keys = ('source_read_ms','cached_detector_read_ms','composition_projection_ms',
                      'frame_packet_write_and_validation_ms','application_from_cached_inputs_ms')
        item['image_stages'] = {}
        for key in stage_keys:
            values = [r[key] for r in records if key in r]
            item['image_stages'][key] = dict(observed=len(values), total_s=sum(values)/1000,
                mean_ms=float(np.mean(values)) if values else None,
                median_ms=float(np.median(values)) if values else None,
                p95_ms=float(np.percentile(values,95)) if values else None)
        item['ROI_stages'] = {}
        for key in ('adapter_ms','pre_module_ms','module_call_ms','post_module_ms','inter_module_ms',
                    'ownership_ms','crop_read_ms','roi_packet_write_and_validation_ms',
                    'peak_allocated_MiB','peak_reserved_MiB'):
            values = [entry['timing'][key] for r in records for entry in r['records'] if key in entry['timing']]
            item['ROI_stages'][key] = dict(observed=len(values), mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                p95=float(np.percentile(values,95)) if values else None,
                maximum=max(values) if values else None)
        timing_summary[mid] = item
        mean = f'{total / 600:.3f}' if len(records) == 600 else '—'
        text.append(f'| {mid} | {len(records)}/600 | {item["ROI_calls"]} | {mean} | {total / 1000:.3f} | {len(rounds)}/3 |')
    atomic(root / 'TIMING_SUMMARY.json', timing_summary)
    text += ['', 'ผลรวมต่อภาพบวก adapter timer ของ ROI ที่เรียกจริง ภาพไม่มี ROI มี parsing work ศูนย์ '
             'เวลาอ่านภาพ, cached YOLO, crop read, ownership และ atomic output แยกอยู่ใน image records '
             'ไม่ได้วัด YOLO inference ใหม่ ค่า cache เดิมอ้างกลับ detector run จึงยังไม่มี live end-to-end speed ranking', '',
             '## ขอบเขตหลักฐาน', '',
             'ผลแสดง checkpoint และ application configuration บน accepted snapshot 600 ภาพ '
             'ยังมีข้อจำกัด annotation completeness, helper exposure, Face_Neck และ Dress/Apparel '
             'ไม่มีข้อสรุปเชิงเหตุจาก architecture ไม่มี uncertainty test หรือผลสีเสื้อผ้าในรอบนี้ '
             'ประวัติ failure อยู่ใน ATTEMPTS/FAILURE และสถานะ final ไม่ลบประวัติ attempt', '',
             'การเพิ่ม GT negatives หรือแก้ annotation ภายหลังต้องสร้าง protocol/GT revision ใหม่ '
             'แล้ว rescore saved masks ของทุกระบบที่ได้รับผลอย่างสอดคล้องกัน']
    (root / 'BATCH_REPORT_TH.md').write_text('\n'.join(text) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    report(parser.parse_args().root)
