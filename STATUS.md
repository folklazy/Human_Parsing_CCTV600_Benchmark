# สถานะ Human Parsing CCTV600

**FRESH QUALITY RUNNING — ตรวจยืนยัน 2026-11-17T16:34:18+07:00**

รอบใหม่ทำ parsing ใหม่ทั้ง9โมเดลบน600ภาพ ใช้ GT ส่งมอบล่าสุดและ frozen YOLO/RGB crop inputs ชุดเดียวกัน Parser prediction reuse0 ผลรอบก่อนเก็บใน archive index และ run records แยกจาก current run

## ตรวจเปิดงานจริง

| รายการ | ผล |
| --- | --- |
| Run | `cctv600-fresh-20261117T092941639648Z` |
| tmux | `cctv600_fresh_20261117T092941639648Z` |
| Controller PID | 3955217 |
| Worker PID ณ snapshot | 3955495 |
| โมเดลแรก | QANet |
| Completed images ณ snapshot | 85/600 |
| Completed ROI calls ณ snapshot | 114/763 |
| Failed / invalid ณ snapshot | 0 / 0 |
| CPU acceptance / model readiness | PASS |
| Native packet / geometry / YOLO ownership / config binding | PASS |
| GPU ที่ใช้ | NVIDIA Tesla T4 |

ค่าความคืบหน้าในตารางเป็น launch snapshot ดูสถานะปัจจุบันด้วยคำสั่งจาก workspace

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/scoped_run.py --status --watch
```

## การทำงานต่ออัตโนมัติ

Controller ไล่โมเดลตาม frozen roster ทีละตัว เก็บ native predictions ใหม่ atomic packets และ failure records คุณภาพคำนวณหลัง collection batch หยุด แล้วดำเนิน dedicated timing ตาม100ภาพ ×3รอบที่อนุญาตไว้ Clean resource checks ยังใช้เกณฑ์เดิม ไม่มี automatic model/OOM retry เมื่อจบหยุดหลังเก้า checkpoints

## GT และขอบเขตคะแนน

GT มี769คนผ่านเกณฑ์และ Hair/Face/Upper/Lower parts544/384/745/668หลัง shared domain Membership SHA256 `256a191352a2f32150fdf2d6a8ddc24845780f5885a8d7b2f785008a505a3706` มีคนบางส่วนไม่มี annotation Source ID0คง Unassigned Person diagnostics ใช้ annotated-person recall และ unmatched predictions ส่วน full-scene person AP/Precision/F1 เป็น null ตาม coverage คะแนนสี่ส่วนใช้ Macro4-local พร้อม IoU/Precision/Recall/F1-Dice และ confusion matrices ที่ประกาศ

## ตำแหน่งหลักฐานและจุดจบ

Current run อ้างจาก `outputs/CURRENT.json` ภายใน run มี `controller.log`, `logs/<model>.log`, `BATCH_STATE.json`, `CURRENT_WORKER.json`, `LAUNCH.json` และ `LAUNCH_VERIFICATION.json` การเปิดโปรแกรม/ปิดบทสนทนาแยกจาก tmux session ที่ยืนยัน PID จริงแล้ว

จบรอบเมื่อมี `BATCH_COMPLETION.json` และ `BATCH_REPORT_TH.md` พร้อม metrics/CSV/TIMING_SUMMARY/gallery หากหยุดผิดปกติอ่าน `CONTROLLER_FAILURE.json` หรือ model failure records คะแนนหลักของระบบที่ไม่ครบเป็น null

[แผนและนิยาม](README.md) · [Dataset](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset)
