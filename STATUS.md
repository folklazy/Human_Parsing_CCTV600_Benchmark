# สถานะ Human Parsing CCTV600

**ปิดรอบครบแล้ว — 18 พฤศจิกายน 2026 เวลา 11:12:31 น. (Asia/Bangkok)**

- คะแนนคุณภาพครบ 9/9 โมเดล โมเดลละ 600 ภาพและ 763 ROI, missing/duplicate/invalid เป็นศูนย์ในผลที่ตรวจรับ
- Dedicated timing ผ่าน 3/3 รอบทุกโมเดล รวม 27 รอบ
- [ผลตรวจเทียบ CCTV90/CCTV600](reports/CCTV600_RESULTS_TH.md#cctv90-comparison): สูตรรายส่วนร่วมกัน แต่ population/GT/matcher/detector cache และ timer instrumentation ต่างกัน ห้ามอ่านเป็นการรันซ้ำภายใต้ทุกเงื่อนไขเดิม
- Canonical results: `outputs/cctv600-1b-continuation-20261118T032146603433Z/RESULT_INDEX.json` และ `TIMING_SUMMARY.json`
- รายงาน [CCTV600_RESULTS_TH.md](reports/CCTV600_RESULTS_TH.md) เผยแพร่แล้ว; automatic publication commit `abf0574` ตาม `PUBLICATION.json`
- งาน controller/worker จบแล้ว ตอนตรวจ GPU ไม่มี compute worker ของ benchmark
- Macro4-local สูงสุด Sapiens2 1B 59.184%; QANet 57.517% คะแนนรายส่วนและข้อจำกัดอยู่ในรายงาน

ผลปัจจุบันยังจำกัดเฉพาะ annotated person-parts ตาม dataset contract ไม่รับรอง full-scene person AP/Precision/F1 เก็บประวัติ interruption และ failure ของ parent ไว้ครบ

## ผลส่งมอบบน GitHub

หน้าแรกและ[รายงานหลัก](reports/CCTV600_RESULTS_TH.md) แสดงผลปัจจุบันครบเก้าโมเดล พร้อมภาพคะแนน/confusion 36 ชุด, สูตร/ตัวหาร, quality timing/dedicated timing/เวลาแต่ละขั้น, checkpoint/config/environment และแหล่งอ้างอิง

[metrics/RESULT_INDEX.json](metrics/RESULT_INDEX.json) เป็น normalized derived index ผูก canonical metrics/completion/source hashes และรักษา legacy null ไว้ตามที่มา CSV มี SUMMARY 9 แถว, PER_TARGET 36 แถว และ TIMING 36 แถว ไม่มีภาพ/annotations/raw ledgers ขึ้น Git

Publication ตรวจด้วย CPU จาก saved aggregate เท่านั้น ไม่มี inference/full rescore ใหม่ อ่าน `evidence/READINESS.json` และ `evidence/CPU_ACCEPTANCE.json` เดิมเป็น historical snapshot ของ R3; หลักฐาน fresh campaign และ continuation ที่ใช้กับผลส่งมอบอยู่ใน provenance ของ result index ปัจจุบัน

## บันทึกสถานะก่อนปิดรอบ

ข้อความด้านล่างเป็น snapshot ขณะ continuation กำลังทำงาน ไม่ใช่สถานะปัจจุบัน


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
