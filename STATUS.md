# สถานะ Human Parsing CCTV600

**พักคิวเพื่อใช้ GT revision ที่ผู้จัดทำกำลังตรวจแก้ — 17 พฤศจิกายน 2026**

ผู้จัดทำแก้ annotation 23 ไฟล์จาก dataset รุ่น 1.0.0 การตรวจ dataset รุ่น 1.0.1-rc.1 ผ่านไฟล์และโครงสร้าง 600/600 ภาพ มี part shapes 2,449 ชิ้น และมี class/owner overlap ที่ต้องตรวจต่อ ดู [เอกสาร dataset ปัจจุบัน](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset) ผลคะแนนที่ใช้ GT ใหม่ต้องมี GT bindings, matching และตัวหารที่สร้างจาก revision ใหม่

Controller เดิมหยุดแล้ว Sapiens2 0.4B worker ที่ปล่อยให้เก็บ native outputs ต่อจบเมื่อ 17 พฤศจิกายน 2026 เวลา 13:30:49 น. Asia/Bangkok `COMPLETION.json` ของ collector ระบุ exact 600 IDs, failed/invalid 0 และ state ระบุ 763 ROIs ผล native ของหกโมเดลก่อนหน้ามี assembled completion ครบ 600 ภาพต่อโมเดล ส่วน 0.4B ยังต้องตรวจรวมเข้ากับ continuation view ตามกระบวนการเดิม

ตรวจ process ล่าสุดระหว่างงาน dataset รอบนี้ พบว่า controller/worker ชุดนี้หยุดแล้ว งาน 0.8B/1B และการคำนวณคะแนนยังพักอยู่ การตรวจ dataset รอบนี้มี model calls 0 ผล inference เดิมและ frozen scientific files เก็บตาม bindings เดิม

## บันทึกการเปิด R3 ก่อนพักคิว

ข้อมูลและจำนวน GT ต่อจากนี้เป็น snapshot ของการเปิด R3 ซึ่งใช้ dataset 1.0.0

ผู้ใช้สั่งเริ่มวิธีประเมินใหม่วันที่ 17 พฤศจิกายน 2026 ใช้ Mean-part IoU, Face-local, mIoU3 (part-mean), Macro4-local และ pooled Precision/Recall/F1 ตาม shared-reference run เดิมของ CIHP/CCTV

CCTV600 ใช้ bbox matching ที่ 0.50 สำหรับคะแนนรายส่วน พร้อม person diagnostics ที่ 0.75 เพราะ dataset มี bbox GT และ masks สี่ส่วน ข้อแตกต่างจาก GT-person-mask matching ของงานเดิมระบุใน protocol

ประชากร exact 600 ภาพและ frozen YOLO26x cache 763 ROIs ต่อโมเดล GT people ≥50 px มี 769 รายการ Eligible Hair/Face/Upper/Lower มี 544/390/740/665 person-parts

## ผล native ที่ตรวจแล้วก่อนเปิดรอบใหม่

| Model | Reused ROIs | Pending ROIs |
| --- | --- | --- |
| QANet_CIHP_HRNetW48_512x384 | 763 | 0 |
| M2FP_R101_CIHP | 763 | 0 |
| D2FP_R101_CIHP | 763 | 0 |
| RPR_CIHP | 228 | 535 |
| CPI_R50_CIHP | 2 | 761 |
| CPI_ConvNeXtB_CIHP | 763 | 0 |
| Sapiens2_0.4b | 0 | 763 |
| Sapiens2_0.8b | 0 | 763 |
| Sapiens2_1b | 0 | 763 |

ทุกชุดที่ตรวจมี invalid/partial/orphan เป็นศูนย์ ณ prelaunch snapshot ผลครบสี่โมเดลเก็บแล้ว 600 ภาพต่อโมเดล โดยรอคะแนนของ protocol ใหม่หลัง quality batch จบ

## การแก้ที่ประกาศไว้

- RP-R-CNN รับ empty prediction เฉพาะเมื่อ native traceback ยืนยัน BoxList มีจำนวนศูนย์
- CPI คืนเฉพาะ unused cache ของตัวเองเมื่อ physical free ต่ำกว่า guard แล้วตรวจ threshold เดิม การกระทำนี้เกิดก่อน model call และบันทึก resource event
- ผลจาก failed attempts เดิมเก็บใน parent records และไม่มี automatic model/OOM retry

CPU fixtures และ actual saved-packet integration ของสี่โมเดลผ่านแล้ว Scorer ใช้ counts/summarize เดิม พร้อม source/protocol/GT hashes ที่ตรึงก่อนคำนวณคะแนน CCTV600

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/reference_parts600.py --progress --watch
tmux attach -t parsing_cctv600_r3
```

Current run root อ้างจาก `outputs/CURRENT.json` การจบมี `BATCH_COMPLETION.json` และ `BATCH_REPORT_TH.md` พร้อม canonical metrics/CSV/ภาพชุดเดิม ตารางคะแนนของระบบที่ยังไม่ครบเป็น null พร้อมสาเหตุ

[รายละเอียดและนิยาม](README.md) · [Active protocol](configs/SHARED_REFERENCE_V1.json)

## ตรวจเริ่มงานจริง

เมื่อ 2026-11-17T12:07:21+07:00 Controller PID 3684804 อยู่ใน tmux `parsing_cctv600_r3` RP-R-CNN worker PID 3686707 ทำงานจริงและบันทึก native packets ใหม่แล้ว 228 ROIs ตรวจ packet ของกรณีที่เคย KeyError ผ่าน โดยยืนยัน native empty result และ integer-mask geometry ตาม contract จำนวนนี้เป็น launch snapshot ใช้ progress viewer ดูค่าปัจจุบัน
