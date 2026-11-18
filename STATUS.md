# สถานะ Human Parsing CCTV600

**Controlled continuation ของ Sapiens2 1B — ตรวจ 2026-11-18T10:27:29+07:00**

แปดโมเดล QANet/M2FP/D2FP/RP-R-CNN/CPI R50/CPI ConvNeXt-B/Sapiens2 0.4B/Sapiens2 0.8B มี canonical metricsครบ600ภาพและ native ROI packetsครบ763งานต่อโมเดล พร้อม clean timingครบสามรอบ ผลเหล่านี้เก็บไว้พร้อม source hashes

## สาเหตุที่1Bหยุด

Controllerตรวจพื้นที่ด้วยduระหว่าง atomic rename ของ ROI packet จึงได้รับ ENOENT และส่งSIGTERMไปยังworker พบ native1B packetsที่ผ่าน reconciliation737ROIs และ complete frame packets579ภาพ ส่วนที่เหลือ26ROIsอยู่ในcontinuationที่เชื่อมกับ fresh experimentเดิม สาเหตุบันทึกเป็น orchestration resource-inspection failure และรักษา raw failure recordsเดิม

ตัวตรวจพื้นที่ใหม่ต้องอ่านได้สำเร็จ ลองซ้ำเฉพาะ ENOENTได้สูงสุดสามครั้งและคงdisk/memory/time guardsตามconfigเดิม CPU regressionผ่านทั้ง filesystem race, permanent errors, retry bound, exact disk thresholds และเวลาROIที่รวมข้ามsegments

## Process และความคืบหน้าที่ตรวจ

| รายการ | ค่า ณ snapshot |
| --- | --- |
| Run | `cctv600-1b-continuation-20261118T032146603433Z` |
| tmux | `cctv600_fresh_20261118T032146603433Z` |
| Controller PID | 301591 |
| Worker PID | 301752 |
| โมเดลที่เปิด | Sapiens2 1B ตัวเดียว |
| Worker state | QUALITY_RUNNING |
| Accepted parent ROIs | 737 |
| New ROIs completed | 26 / 26 |
| Current model calls | 1 / 27 รวมTRAIN warmupหนึ่งcall |

## วิธีดูสถานะ

รันจาก workspace Human-Body-Extraction

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/scoped_run.py --status --watch
```

Tmuxเปิดหน้า statusเป็นค่าเริ่มต้น Ctrl+Cที่หน้านี้หยุดเฉพาะviewer ออกจากtmuxให้งานต่อใช้Ctrl+B แล้วD `outputs/CURRENT.json` ชี้rootปัจจุบัน มีcontroller.log, CURRENT_WORKER.json, BATCH_STATE.json และlogs/Sapiens2_1b.log

## การปิดรอบอัตโนมัติ

เมื่อquality observationsของ1Bรวมได้exact600IDsและ763ROIs จึงคำนวณsaved-mask scoresและเปิดdedicated timingเฉพาะ1Bสามรอบ ผลแปดโมเดลที่ตรวจรับแล้วใช้อ้างอิงตามhashes เมื่อหยุดรอบทั้งหมดจะสร้างBATCH_COMPLETION.json, BATCH_REPORT_TH.md, TARGET_RESULTS.csv, TIMING_SUMMARY.json และgalleryชุด16ภาพเดิม

รายงานส่งมอบจะเผยแพร่ที่ `reports/CCTV600_RESULTS_TH.md` บนGitHub โดยpublication statusอยู่ในPUBLICATION.json หากpublishติดขัดรายงานในrunยังเป็นหลักฐานตรวจต่อได้

เวลาqualityของ1Bบวกadapter intervalsจากaccepted ROI observationsสองprocess Application/I-O totalsของเฟรมที่ขัดจังหวะเป็นnull Timingเป็นscoped diagnosticsพร้อมresource observations ขอบเขตคะแนนอ้างannotated person-parts และsource ID0ยังเป็นUnassigned Person AP/Precision/F1ทั้งภาพเป็นnullตามpartial-person coverage

[แผนและนิยาม](README.md) · [Dataset](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset)

ตรวจยืนยัน packet ใหม่เวลา 2026-11-18T10:30:22+07:00 ผ่าน native29 integer-mask geometry และ current config binding โดยยังเก็บ failure historyของparentไว้ในCONTINUATION_MANIFEST.json
