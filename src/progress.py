"""Read-only status viewer. Ctrl+C only stops this viewer."""
import argparse
import json
import time

from common import PROJECT, load


def show():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    current = load(PROJECT / 'outputs/CURRENT.json')
    from pathlib import Path
    root = Path(current['root'])
    config = load(root / 'CONFIG.json')
    manifest = load(root/'MANIFEST.json')
    batch = load(root / 'BATCH_STATE.json') if (root / 'BATCH_STATE.json').exists() else {'status':'NOT_LAUNCHED'}
    print('\nCCTV600 Parsing |', datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(timespec='seconds'))
    print(current['run_id'], '|', batch['status'])
    for mid in config['models']:
        state = load(root / mid / 'STATE.json') if (root / mid / 'STATE.json').exists() else {'status':'NOT_ATTEMPTED'}
        pid = state.get('pid')
        alive = bool(pid and Path('/proc/' + str(pid)).exists())
        print(f"{mid:34} {state['status']:26} frames {state.get('complete_frames',0):3}/600 "
              f"ROIs {state.get('complete_ROIs',0)}/{manifest['ROI_jobs']} PID {pid} alive={alive}")
    print('คะแนน AD4 ใช้เฉพาะ annotated domain; คะแนนเต็มภาพยัง null')
    print('จบรอบเมื่อมี', root / 'BATCH_COMPLETION.json')
    print('รายงาน', root / 'BATCH_REPORT_TH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            show()
            if not args.watch:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        pass
