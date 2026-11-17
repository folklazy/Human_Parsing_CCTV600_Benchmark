"""Reporting-only metric labels. Canonical records and frozen scorer stay intact."""
import argparse
import os
import re
import time
from pathlib import Path

from common import PROJECT, atomic, load, now, sha

GUIDE = '''## วิธีอ่านชื่อคะแนนในรายงานนี้

**พื้นที่คำนวณของคะแนน semantic ทุกตารางคือพิกเซลที่ GT ระบุ Hair/Face/Upper/Lower และผ่านกติกา ignore/QA** พื้นที่ unassigned ถูกเว้นไว้ การระบายเกินเข้า unassigned จึงยังนับ FP ได้ไม่ครบ

| ชื่อในตาราง | ความหมาย |
| --- | --- |
| IoU รายคลาส | พิกเซลถูกคลาส ÷ พื้นที่ union ของ GT กับ prediction บนพื้นที่คำนวณ |
| Precision รายคลาส | พิกเซลถูกคลาส ÷ พิกเซลที่ทำนายคลาสนั้นบนพื้นที่คำนวณ |
| Recall รายคลาส | พิกเซลถูกคลาส ÷ พิกเซล GT ของคลาสนั้น |
| F1/Dice รายคลาส | 2TP ÷ (2TP+FP+FN) โดยนับเป็นพิกเซล |
| mIoU เฉลี่ย 4 คลาส | เฉลี่ย IoU ของผม ใบหน้า เสื้อผ้าบน เสื้อผ้าล่าง แต่ละคลาสน้ำหนักเท่ากัน |
| mIoU เฉลี่ย 3 คลาส | เฉลี่ย IoU ของผม เสื้อผ้าบน เสื้อผ้าล่าง โดย GT ใบหน้ายังใช้คิด FP ของสามคลาส |
| IoU รายคนเฉลี่ย 4 คลาส | หาค่าเฉลี่ย IoU รายคนภายในแต่ละคลาส แล้วเฉลี่ยสี่คลาสเท่ากัน คนที่จับคู่ไม่เจอได้ 0 สำหรับส่วนที่มี GT |
| เกณฑ์ IoU จับคู่กรอบ | ใช้ IoU ของ bbox ≥0.50 หรือ ≥0.75 เพื่อจับคู่คน ก่อนคำนวณคะแนนรายคน |
| F1 ของการจับคู่คน | คำนวณจาก TP/FP/FN ของกรอบบุคคล ส่วน F1/Dice รายคลาสคำนวณจากพิกเซล |
| เวลางาน parsing รวมทุกคนต่อภาพ | บวกเวลาของ ROI ที่เรียกจริงในภาพนั้น ค่าเฉลี่ยต่อภาพใช้จำนวนภาพเป็นตัวหาร |

ค่าเปอร์เซ็นต์แสดงในช่วง 0–100 รหัส AD4_mIoU4, C3_global และ Q_AD4_GT ใน canonical JSON เชื่อมกับสามค่าเฉลี่ยข้างต้นตามลำดับ

เครื่องหมาย “—” หมายถึงยังไม่มีค่าที่รายงานได้ ต้องอ่านสถานะแถวและ support ประกอบ ค่า null อาจเกิดจากผลไม่ครบ การรอคำนวณ ตัวหารศูนย์ หรือการที่กติกาปัจจุบันยังไม่ประเมินพื้นที่ทั้งภาพ

ผู้จัดทำยืนยันตรวจแก้ dataset ครบ 600 ภาพแล้ว ตารางรหัสรุ่น 1.0.0 ระบุ unassigned_negative_domain เป็น not_certified การเว้นพื้นที่ดังกล่าวเป็นกติกาประเมินของรอบนี้ หลักฐานการขาด annotation ต้องอ้างการตรวจภาพเพิ่มเติม [Dataset specification](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset/blob/d77ad2eec84e3e44b1465abd17031c8f57e1f18f/schema/labels.json)

'''


def numeric_cells(text):
    return [cell.strip() for line in text.splitlines() if line.startswith('|')
            for cell in line.split('|') if re.fullmatch(r'-?\d+(?:\.\d+)?(?:/\d+)?',cell.strip())]


def readable(text):
    # Long phrases first. Raw machine keys remain in the separate reference guide.
    replacements = {
        'COMPLETE_AD4_DIAGNOSTIC': 'ผลครบและคำนวณคะแนนบนพิกเซล GT ที่ระบุคลาสแล้ว',
        'SCORING_CONTRACT_FAILURE': 'ข้อมูลผลประเมินไม่ผ่านการตรวจ',
        'INCOMPLETE': 'ผลไม่ครบตามเกณฑ์',
        'คะแนน AD4': 'คะแนนบนพิกเซล GT ที่ระบุคลาส',
        'Q_AD4_GT': 'IoU รายคนเฉลี่ย 4 คลาส',
        '| AD4 % | C3 % |': '| mIoU เฉลี่ย 4 คลาส % | mIoU เฉลี่ย 3 คลาส % |',
        '- AD4 คือ': '- mIoU เฉลี่ย 4 คลาส คือ',
        'ส่วน C3 เฉลี่ย': 'ส่วน mIoU เฉลี่ย 3 คลาส เฉลี่ย',
        'Face proxy IoU': 'IoU ใบหน้า (Sapiens รวมคอ)',
        'Face proxy': 'ใบหน้า (Sapiens รวมคอ)',
        'Hair IoU': 'IoU ผม',
        'Upper IoU': 'IoU เสื้อผ้าบน',
        'Lower IoU': 'IoU เสื้อผ้าล่าง',
        'Eligible person-parts Hair/Face/Upper/Lower': 'จำนวนคู่คน–ส่วนที่มี GT ผม/หน้า/บน/ล่าง',
        'GT pixels': 'พิกเซล GT',
        'Pred pixels ใน domain': 'พิกเซลที่ทำนายในพื้นที่คำนวณ',
        'Union pixels': 'พิกเซลใน union',
        'Box IoU': 'เกณฑ์ IoU จับคู่กรอบ',
        'Mean parsing work ms/ภาพ': 'เวลางาน parsing เฉลี่ยต่อภาพ ms',
        'รวม parsing work s': 'เวลางาน parsing รวม s',
        'Clean timing rounds /3': 'รอบจับเวลาที่ผ่านเกณฑ์ /3',
        'annotation-supported domain': 'พิกเซลที่ GT ระบุคลาส',
        'annotation-supported': 'เฉพาะพิกเซลที่ GT ระบุคลาส',
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old,new)
    # Explicit full-frame status distinguishes a domain decision from missing files.
    result = result.replace('คะแนนเต็มภาพยังเป็น null ทุกระบบ',
                            'คะแนนพื้นที่ทั้งภาพแสดง “ยังไม่ประเมินตามกติกาปัจจุบัน” ทุกระบบ')
    if numeric_cells(result) != numeric_cells(text):
        raise ValueError('Display revision changed a numeric table cell')
    if result.startswith('# '):
        title, _, body = result.partition('\n')
        return title+'\n\n'+GUIDE+body.lstrip('\n')
    return GUIDE + result


def status_names(text):
    replacements = {
        'คะแนน AD4 ใช้เฉพาะ annotated domain; คะแนนเต็มภาพยัง null':
            'คะแนน 4 คลาสใช้เฉพาะพิกเซลที่ GT ระบุคลาส ส่วนคะแนนทั้งภาพยังไม่ประเมินตามกติกาปัจจุบัน',
        'PREDICTIONS_COMPLETE': 'ผลทำนายครบ',
        'QUALITY_RUNNING': 'กำลังประมวลผล',
        'QUALITY_BATCH_RUNNING': 'กำลังประมวลผลตามคิว',
        'NOT_ATTEMPTED': 'รอคิว',
        'INITIALIZING': 'กำลังเตรียมโมเดล',
        'STOPPED_INCOMPLETE': 'หยุดและผลยังไม่ครบ',
        'SAVED_MASK_SCORING': 'กำลังคำนวณคะแนนจาก masks ที่บันทึกไว้',
        'DEDICATED_CLEAN_TIMING': 'กำลังจับเวลาแยกรอบ',
        'STOPPED_AFTER_APPROVED_NINE': 'จบคิวที่อนุมัติเก้าโมเดลแล้ว',
    }
    for old,new in replacements.items():
        text = text.replace(old,new)
    return text


def status(watch):
    from contextlib import redirect_stdout
    from io import StringIO
    from progress import show
    try:
        while True:
            stream = StringIO()
            with redirect_stdout(stream):
                show()
            print(status_names(stream.getvalue()),end='',flush=True)
            if not watch:
                return
            time.sleep(10)
    except KeyboardInterrupt:
        return


def render(root):
    if not (root/'BATCH_COMPLETION.json').exists():
        raise RuntimeError('Wait for batch completion before rendering the final report')
    path = root/'BATCH_REPORT_TH.md'
    receipt = root/'METRIC_DISPLAY_REVISION.json'
    if receipt.exists():
        if sha(path) != load(receipt)['readable_report_sha256']:
            raise ValueError('Previously rendered report changed, preserve evidence before another revision')
        return
    protected = [root/name for name in ('CONFIG.json','MANIFEST.json','GT_BINDINGS.json',
                                       'RESULT_INDEX.json','TIMING_SUMMARY.json')]
    for mid in load(root/'CONFIG.json')['models']:
        protected += [root/mid/name for name in ('METRICS.json','COMPLETION.json')]
    before = {str(p.relative_to(root)):sha(p) for p in protected if p.exists()}
    old_bytes = path.read_bytes()
    rendered = readable(old_bytes.decode('utf-8'))
    backup = root/'BATCH_REPORT_TH.before_metric_labels.md'
    with backup.open('xb') as stream:
        stream.write(old_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    temporary = path.with_name(path.name+'.display-tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    after = {str(p.relative_to(root)):sha(p) for p in protected if p.exists()}
    if before != after:
        raise ValueError('Canonical records changed during display-only revision')
    atomic(receipt,dict(status='PASS',at_utc=now(),scope='display names and explanation only',
           renderer_sha256=sha(__file__),source_report_sha256=sha(backup),
           readable_report_sha256=sha(path),canonical_small_record_hashes=after,
           table_numeric_cells_preserved=True,model_calls=0,scorer_calls=0,
           dataset_reference_commit='d77ad2eec84e3e44b1465abd17031c8f57e1f18f'))
    print('Readable metric labels applied',path,flush=True)


def follow(root):
    atomic(root/'METRIC_DISPLAY_STATUS.json',dict(status='WAITING_BATCH_COMPLETION',pid=os.getpid(),at_utc=now()))
    while not (root/'BATCH_COMPLETION.json').exists():
        if (root/'CONTROLLER_FAILURE.json').exists():
            atomic(root/'METRIC_DISPLAY_STATUS.json',dict(status='CONTROLLER_FAILURE_REQUIRES_REVIEW',at_utc=now()))
            return
        time.sleep(30)
    render(root)
    atomic(root/'METRIC_DISPLAY_STATUS.json',dict(status='COMPLETE',at_utc=now()))


def self_test():
    import tempfile
    text = ('| โมเดล | AD4 % | C3 % |\n| A | 62.500 | 66.667 |\n'
            '| Q_AD4_GT | 0.5 | — |\nคะแนนเต็มภาพยังเป็น null ทุกระบบ\n'
            '| COMPLETE_AD4_DIAGNOSTIC | 600/600 | 544 |\n')
    result = readable(text)
    assert numeric_cells(result) == numeric_cells(text)
    assert '| mIoU เฉลี่ย 4 คลาส % | mIoU เฉลี่ย 3 คลาส % |' in result
    assert '| IoU รายคนเฉลี่ย 4 คลาส | 0.5 | — |' in result
    assert 'ยังไม่ประเมินตามกติกาปัจจุบัน' in result
    assert 'null' in result and 'AD4_mIoU4' in result
    assert 'not_certified' in result
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        atomic(root/'CONFIG.json',dict(models={'synthetic':{}}))
        atomic(root/'BATCH_COMPLETION.json',dict(status='SYNTHETIC_FIXTURE'))
        atomic(root/'RESULT_INDEX.json',dict(synthetic=dict(primary=None)))
        original_hash = sha(root/'RESULT_INDEX.json')
        (root/'BATCH_REPORT_TH.md').write_text('# Synthetic report\n\n'+text,encoding='utf-8')
        original_report = (root/'BATCH_REPORT_TH.md').read_bytes()
        render(root)
        assert (root/'BATCH_REPORT_TH.md').read_text().startswith('# Synthetic report\n')
        assert (root/'BATCH_REPORT_TH.before_metric_labels.md').read_bytes() == original_report
        assert sha(root/'RESULT_INDEX.json') == original_hash
        first_rendered_hash = sha(root/'BATCH_REPORT_TH.md')
        render(root)
        assert sha(root/'BATCH_REPORT_TH.md') == first_rendered_hash
    assert status_names('QUALITY_RUNNING frames 12/600') == 'กำลังประมวลผล frames 12/600'
    print('PASS display labels, numeric cells, null reasons, canonical immutability, backup, idempotence and dataset reference')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--follow',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--watch-status',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.status or args.watch_status:
        status(args.watch_status)
    else:
        root = args.root or Path(load(PROJECT/'outputs/CURRENT.json')['root'])
        if args.follow:
            follow(root)
        else:
            render(root)
