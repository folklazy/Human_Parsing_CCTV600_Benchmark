# สถานะ Human Parsing CCTV600

**PREPARING — 17 พฤศจิกายน 2026**

กำลังสร้าง input/GT bindings สำหรับเก้า pretrained checkpoints บน accepted dataset 600 ภาพ ใช้ YOLO26x-Seg cache เดิมครบ 600 IDs ไม่มีการรัน YOLO หรือ training เพิ่ม

CPU acceptance ผ่านชุด synthetic และ actual accepted-packet integration ก่อน model inference โดยใช้ annotation-supported domain ร่วมกัน พื้นที่ unassigned คง unknown คะแนนเต็มภาพเป็น null

Canonical operational status อยู่ใน `outputs/CURRENT.json` และ private run `BATCH_STATE.json` เมื่อจบมี `BATCH_COMPLETION.json` และ `BATCH_REPORT_TH.md` การจบ controller ยังต้องอ่านจำนวนโมเดลที่ครบและ failures

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/progress.py --watch
```

[Protocol และคำอธิบายคะแนน](README.md)
