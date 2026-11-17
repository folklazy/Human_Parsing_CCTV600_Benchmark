# สถานะ Human Parsing CCTV600

**M2FP RUNNING / CONTINUATION QUEUED — 17 พฤศจิกายน 2026**

เตรียม exact 600 IDs และ 763 predicted ROIs ต่อโมเดลแล้ว ใช้ YOLO26x-Seg cache เดิม GT eligible boxes 769 รายการ ส่วนตัวหาร person-parts Hair/Face/Upper/Lower หลัง common VOID เท่ากับ 544/390/740/665

CPU acceptance ของ continuation ผ่าน 17 cases และตรวจ M2FP packet ใหม่ผ่าน hashes, native integer labels, dimensions, ROI map-back, YOLO ownership และการบวก timer พร้อมเปิดภาพ overlay เพื่อตรวจ geometry หนึ่งภาพ ใช้ annotation-supported domain ร่วมกัน พื้นที่ unassigned คง unknown คะแนนเต็มภาพเป็น null

รอบแรก `parsing600-20261117T030309293781Z` พบ QANet timer setup error ก่อน model calls และเก็บ failure ไว้ M2FP PID 3563174 ทำงานต่อใน session ของตัวเอง ส่วน continuation `parsing600-20261117T030309293781Z-timer-r2` รับผล parent และเปิดคิวที่เหลือหลัง M2FP จบ โดยใช้ timer correction `QANET_NATIVE_METHOD_TIMER_2` Native inference, mappings และ scorer ยังคงผูกกับ source hashes เดิม

Resource guards ใช้ตาม protocol GPU มีบริการเว็บร่วมอยู่ จึงเก็บ quality timings เป็น diagnostics Model admission ที่ผ่านไม่รับประกันทุกภาพว่าจะผ่าน และ clean timing ต้องผ่าน admission แยกตามรอบ

Canonical operational status อยู่ใน `outputs/CURRENT.json` และ private run `BATCH_STATE.json` เมื่อจบมี `BATCH_COMPLETION.json` และ `BATCH_REPORT_TH.md` การจบ controller ยังต้องอ่านจำนวนโมเดลที่ครบและ failures

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/progress.py --watch
```

[Protocol และคำอธิบายคะแนน](README.md)

ตรวจเริ่ม continuation เมื่อ 2026-11-17T10:38:18+07:00 Controller PID 3579861 อยู่ใน tmux `parsing_cctv600_r2` และรอ parent M2FP PID 3563174 ซึ่งบันทึกแล้ว 550/600 ภาพ ตัวเลขนี้เป็น launch snapshot ใช้ progress viewer ดูจำนวนปัจจุบัน
