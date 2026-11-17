# สถานะ Human Parsing CCTV600

**REFERENCE_PARTS_R3 — พร้อมเปิด continuation ตามวิธี CIHP/CCTV เดิม**

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
