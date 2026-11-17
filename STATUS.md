# สถานะ Human Parsing CCTV600

**QUALITY_BATCH_RUNNING — 17 พฤศจิกายน 2026**

เตรียม exact 600 IDs และ 763 predicted ROIs ต่อโมเดลแล้ว ใช้ YOLO26x-Seg cache เดิม GT eligible boxes 769 รายการ ส่วนตัวหาร person-parts Hair/Face/Upper/Lower หลัง common VOID เท่ากับ 544/390/740/665

CPU acceptance ของ continuation ผ่าน 17 cases และตรวจ M2FP packet ใหม่ผ่าน hashes, native integer labels, dimensions, ROI map-back, YOLO ownership และการบวก timer พร้อมเปิดภาพ overlay เพื่อตรวจ geometry หนึ่งภาพ คะแนนหลักใช้ชื่อ “mIoU เฉลี่ย 4 คลาส — เฉพาะพิกเซลที่ GT ระบุคลาส” พื้นที่ unassigned ถูกเว้นจากคะแนน ส่วนคะแนนพื้นที่ทั้งภาพแสดง “ยังไม่ประเมินตามกติกาปัจจุบัน”

รอบแรก `parsing600-20261117T030309293781Z` พบ QANet timer setup error ก่อน model calls และเก็บ failure ไว้ M2FP PID 3563174 ทำงานต่อใน session ของตัวเอง ส่วน continuation `parsing600-20261117T030309293781Z-timer-r2` รับผล parent และเปิดคิวที่เหลือหลัง M2FP จบ โดยใช้ timer correction `QANET_NATIVE_METHOD_TIMER_2` Native inference, mappings และ scorer ยังคงผูกกับ source hashes เดิม

**ตรวจ handover แล้ว** M2FP มี prediction packets ครบ exact 600 ภาพ, 763 ROIs และ model API calls 764 รวม TRAIN warmup หนึ่งครั้ง Completion missing/duplicate/failed/invalid เป็นศูนย์ ส่วนคะแนนยังรอ saved-mask scoring หลัง quality batch ทั้งเก้าหยุด

QANet ที่แก้ timer แล้วเริ่ม GPU inference สำเร็จ PID 3584087 ตรวจ packet ใหม่ผ่าน native20 integer labels, original-grid geometry, ownership และ actual module timers Controller PID 3579861 ทำคิวต่ออัตโนมัติใน tmux `parsing_cctv600_r2`

Resource guards ใช้ตาม protocol GPU มีบริการเว็บร่วมอยู่ จึงเก็บ quality timings เป็น diagnostics Model admission ที่ผ่านไม่รับประกันทุกภาพว่าจะผ่าน และ clean timing ต้องผ่าน admission แยกตามรอบ

Canonical operational status อยู่ใน `outputs/CURRENT.json` และ private run `BATCH_STATE.json` เมื่อจบมี `BATCH_COMPLETION.json` และ `BATCH_REPORT_TH.md` การจบ controller ยังต้องอ่านจำนวนโมเดลที่ครบและ failures

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/readable_report.py --watch-status
```

[Protocol และคำอธิบายคะแนน](README.md)

ตรวจ dataset repo ทั้ง local และ GitHub ที่ commit `d77ad2e` อีกครั้ง ผู้จัดทำยืนยันตรวจแก้ครบ 600 ภาพ ส่วน schema ระบุรหัส 0 เป็น unassigned และ `unassigned_negative_domain=not_certified` การเลือกเฉพาะพิกเซล GT ที่ระบุคลาสเป็นกติกาของ benchmark รอบนี้ หลักฐานว่าภาพใดขาด annotation ต้องอ้างการตรวจภาพเพิ่มเติม

Launch snapshot เมื่อ 2026-11-17T10:38:18+07:00 เป็นช่วง controller รอ parent M2FP ที่ 550/600 ภาพ จากนั้นเปลี่ยนเข้าสู่ quality batch ตามย่อหน้าด้านบน ใช้ progress viewer ดูจำนวนปัจจุบัน
