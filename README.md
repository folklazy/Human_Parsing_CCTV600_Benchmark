# Human Parsing บน CCTV Human Parsing 600 Dataset

**รอบปัจจุบันใช้วิธี shared reference ของงาน CIHP/CCTV เดิม** ตามคำขอเริ่มรอบใหม่วันที่ 17 พฤศจิกายน 2026 คะแนนหลักเป็น **Macro4-local** พร้อม **mIoU3 (part-mean), Face-local, Precision, Recall และ F1/Dice**

[สถานะ](STATUS.md) · [Protocol ที่ใช้จริง](configs/SHARED_REFERENCE_V1.json) · [Scorer](src/reference_parts600.py) · [Controller/continuation](src/reference_collection.py) · [บันทึกงาน](WORK_LOG.md) · [Dataset CCTV600](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset)

## ข้อมูลและขอบเขตการประเมิน

Dataset ส่งมอบมี 600 ภาพและ annotation ของบุคคลที่เลือก ผู้จัดทำยืนยันว่ามีคนบางส่วนไม่ได้ annotate Source ID0 ใช้ Unassigned คะแนน parsing จะอ้าง GT person-parts ที่มีคำตอบ ส่วน unmatched detections เก็บเป็น diagnostics และ person recall มี denominator เป็นคนที่มี GT การรายงาน full-scene AP/precision/F1 ต้องมี negative domain ที่รับรองครบ

ผลตรวจ GT domain ด้วย CPU ครบ600 ภาพให้ตัวหาร Hair544/Face384/Upper745/Lower668 และคนผ่านเกณฑ์769 รายการ คิว inference ยังพักระหว่างเตรียม GT bindings และ reporter ของรอบใหม่ รายละเอียดที่ต้องแก้อยู่ใน [STATUS](STATUS.md) การตรวจ dataset ครั้งนี้มี model calls0

จำนวน GT และรายละเอียดการเปิด R3 ในหัวข้อถัดไปเก็บเป็นประวัติการทดลอง ส่วนแหล่งข้อมูลและคำตอบปัจจุบันให้ยึด dataset CURRENT.json และ membership hash ที่ประกาศใน STATUS

## ขอบเขตการทดสอบที่ตกลงไว้

ประชากร600ภาพ ใช้ GT ของบุคคลที่เลือก annotate, source ID0 เป็น Unassigned และใช้ YOLO crop/ownership ตาม pipeline ที่ประกาศ แต่ละ parser ทำนาย native classes ทั้งหมดก่อน project ไปยังสี่เป้าหมาย

| ผลส่งมอบ | นิยามที่ต้องใช้ |
| --- | --- |
| IoU รายคลาส | Mean-part IoU พร้อม pooled IoU โดย Face ใช้ Face-local |
| Precision / Recall / F1-Dice รายคลาส | รวม TP/FP/FN ของ GT person-part tasks ก่อนหาร |
| Confusion matrix | GTสี่คลาสไปยัง predictionสี่คลาสและ OTHER บน GT positive domain พร้อม binary matrixรายคลาสสำหรับ TP/FP/FN/TN |
| ค่าเฉลี่ยสี่คลาส | Macro4-local เฉลี่ย Hair/Face-local/Upper/Lower คลาสละ25% พร้อม mIoU3 (part-mean) ตาม protocol |
| ส่วนที่พลาดทั้งหมด | จำนวนและสัดส่วน eligible GT parts ที่ TP=0 รวมกรณีเจ้าของที่จับคู่ไม่ได้ |
| เวลาประมวลผล | Per-ROI, per-image และผลรวม parsing work พร้อม scope ของ interval และ timer completeness |
| Peak VRAM | Peak allocated/reserved ของ process พร้อม resource-policy และ sharing observations |
| ความครบถ้วน | Exact600 IDs ไม่ซ้ำ พร้อม ROI membership, failed/invalid/missing และ STRICT_COMPLETE |
| Person diagnostics | Annotated-person recall และ unmatched predictions ส่วน full-scene AP/Precision/F1 ใช้ null ตาม coverage |

Prediction OTHER ใน cross-class matrix รวมผลทำนายนอกสี่เป้าหมายตาม mapping ที่ประกาศ การนับใน matrix นี้อ้างเฉพาะ GT positive pixels ส่วน binary matrices ใช้ domain ของแต่ละ person-part จึงครอบคลุมการทายเกินขอบ GT ด้วย Source Unassigned มีความหมายตาม dataset

### ขั้นเตรียมเปิดรัน

1. สร้าง GT bindings, matching และ config ของรอบใหม่จาก manifest ส่งมอบ พร้อม source/protocol hashes
2. ปรับ scorer/reporter แยกจาก frozen source ให้รองรับ partial-person diagnostics, cross-class matrix, TP=0 part counts และสรุป peak VRAM ตามขอบเขตข้างต้น
3. ตรวจ CPU acceptance และ reconcile saved native packets ก่อน freeze reused/pending membership
4. ตรวจ GPU/disk/process guards แล้วเปิดคิว sequential ใน tmux เก็บ log และสถานะที่ตรวจได้ หยุดหลังเก้า checkpoints และรอบจับเวลาที่อนุญาต
5. เก็บ progress และ execution diagnostics ระหว่าง inference แล้วประเมินคุณภาพหลัง collection จบ ตรวจ STRICT_COMPLETE และสร้างรายงานจาก saved outputs

สถานะ implementation และผลตรวจจริงให้อ้าง [STATUS](STATUS.md)

## งานเดิมที่ใช้อ้างอิง

ตรวจทั้ง working copy และ publication repository `folklazy/Human_Parsing_Study` ก่อนปรับรอบนี้ แหล่งที่เลือกใช้คือ shared-reference run `shared-c3-face-local-20261108T083533Z` ซึ่งใช้กติกาเดียวกันใน CIHP 5,000 ภาพและ CCTV 90 ภาพ

| แหล่งที่ตรวจ | สิ่งที่นำมาใช้ |
| --- | --- |
| Working/publication copy commit `29f7030` รายงาน CIHP และ CCTV ที่อัปเดต 9 พฤศจิกายน 2026 | Mean-part IoU, mIoU3 (part-mean), Face-local และ Macro4-local |
| `Human_Parsing_Benchmark/src/common_view/reference_parts.py` | ฟังก์ชัน counts และ summarize ที่ใช้คำนวณรายส่วนจริง |
| `Human_Parsing_Benchmark/protocol/SHARED_C3_FACE_LOCAL_V1.json` | Positive person-part eligibility, missed-person penalty และกรอบ Face-local |
| GitHub main snapshot `8505cbf4ca7ba1bec338a33b3f678d5e6c8fb194` | รายงานก่อนหน้าใช้ global C4-P/C3 ใน CIHP และ Macro4/Face-domain ใน CCTV เป็นคนละ revision กับ shared Face-local |

Repository ที่ตรวจคือ [Human Parsing Study](https://github.com/folklazy/Human_Parsing_Study) ส่วน source hashes และ exact reference bindings ของรอบนี้เก็บใน private run config การเลือกรอบอ้างอิงระบุไว้ชัดเพราะชื่อ C4-P, Macro4 และ Face-local ในประวัติใช้วิธีรวมคะแนนและพื้นที่คำนวณต่างกัน

## เป้าหมายและประชากร

เปรียบเทียบเก้า pretrained checkpoints บน accepted dataset 600 ภาพ ใช้ frozen YOLO26x-Seg predictions ชุดเดียวกันและ RGB person crops ของ cohort นี้ โมเดลประกอบด้วย QANet, M2FP, D2FP, RP-R-CNN, CPI R50, CPI ConvNeXt-B และ Sapiens2 segmentation 0.4B/0.8B/1B

GT มี person bbox 774 กรอบ โดย 769 กรอบมีด้านยาวบนภาพต้นฉบับ ≥50 px อีก 5 กรอบเป็นกลุ่มยกเว้น Dataset มี masks ของ Hair/Face/Upper/Lower พร้อม image-local person ID นิยาม source labels และไฟล์ annotation คงตาม dataset release 1.0.0

เมื่อผ่าน common ignore/QA มี GT person-parts ที่ใช้ประเมิน Hair 544, Face 390, Upper 740 และ Lower 665 รายการ จำนวนส่วนเป็นตัวหารรายคลาส คนหนึ่งอาจมีหลายคลาสและหลาย polygon ที่รวมเป็นส่วนเดียว

## Pipeline และผลเดิมที่นำกลับมาใช้

1. อ่าน accepted YOLO26x-Seg cache ของ detector run `cctv600-20261116T073208561699Z` เลือก person confidence ≥0.25
2. ใช้ RGB crop จากภาพต้นฉบับ เพิ่ม context 10% แต่ละด้านและขยายให้ครอบคลุม frozen detector mask
3. Parser รับภาพ RGB พร้อมบริบทรอบคน ใช้ native preprocessing ของ checkpoint นั้นใน FP32, batch 1, eval, ไม่มี AMP/TF32/extra TTA
4. ระบบที่คืนหลาย native instances เลือกผลด้วย predicted-box IoU กับ designated YOLO box ≥0.50 ตามกติกา prediction-only เดิม
5. Map native20/native29 masks กลับ original grid และใช้ YOLO mask จัด ownership หลัง parsing
6. เก็บ native outputs, candidate masks, geometry, hashes และ source references ก่อนประเมินผล

ผล native ที่ bindings ตรงกันนำกลับมาใช้ใน assembled index ของรอบใหม่ IDs ที่ยังขาดประมวลผลใน fresh linked collection แยกจาก parent attempts ทุกผลระบุ model/checkpoint/config/source/environment/input และ packet identity การ reuse ตรวจจาก artifacts จริง

## ตัวชี้วัดที่ใช้

| ชื่อในรายงาน | วิธีคำนวณ | บอกอะไร |
| --- | --- | --- |
| **Hair / Upper / Lower IoU** หรือ Mean-part IoU | คิด IoU ของ GT part แต่ละคน แล้วเฉลี่ยภายในคลาส | คุณภาพของส่วนที่มี GT โดยแต่ละส่วนมีน้ำหนักเท่ากัน |
| **Face-local IoU** | คิด IoU ภายในกรอบพอดีของ GT Face แล้วเฉลี่ยราย GT Face | Agreement ของ face candidate ภายในกรอบอ้างอิงเดียวกันทุกโมเดล |
| **mIoU3 (part-mean)** | เฉลี่ย Mean-part IoU ของ Hair/Upper/Lower เท่ากัน | คะแนนรวมสามคลาสหลัก |
| **Macro4-local** | เฉลี่ย Hair/Face-local/Upper/Lower คลาสละ 25% | คะแนนรวมสี่คลาสหลัก |
| **Pooled IoU** | รวม TP/FP/FN ของทุก GT part ภายในคลาสก่อนเข้าสูตร | คุณภาพจากจำนวนพิกเซลรวม ส่วนใหญ่มีน้ำหนักมากกว่า |
| **Precision / Recall / F1-Dice** | รวม counts ภายในคลาสแล้วหารตามสูตรด้านล่าง | การทายเกิน การเก็บ GT ได้ครบ และสมดุลทั้งสอง |
| **Person Precision / Recall / F1** | นับ TP/FP/FN จาก bbox matching ที่ 0.50/0.75 | Coverage ของ person pipeline ภายใต้ GT bbox และ ignore ที่กำหนด |

ทุกค่าแสดงเป็น % จาก full-precision canonical statistics แล้วจึงปัดเศษ สูตรพื้นฐานคือ

```text
IoU = TP / (TP + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1/Dice = 2TP / (2TP + FP + FN)

mIoU3 (part-mean) = (Hair + Upper + Lower) / 3
Macro4-local = (Hair + Face-local + Upper + Lower) / 4
```

ตัวอย่างสมมติ ส่วน Hair สองคนมี IoU 0% และ 100% จะได้ Mean-part IoU 50% แม้คนที่สองมีพื้นที่ผมมากกว่า ส่วน Pooled IoU ต้องรวมจำนวนพิกเซลจริงก่อนหาร ตารางทั้งสองจึงใช้แหล่ง counts เดียวกันแต่ aggregation ต่างกัน

Confusion matrix รายคลาสเป็น binary matrix `[[TP,FN],[FP,TN]]` แถวเป็น GT target/other คอลัมน์เป็น prediction target/other หน่วยเป็นพิกเซลสะสมจากงาน person-part ของคลาสนั้น การนับพื้นที่ valid ทำแยกตามส่วน จึงต้องอ่าน TN ตามหน่วยงานอ้างอิงนี้

## พื้นที่คำนวณและการจับคู่คน

### Hair, Upper และ Lower

ใช้ prediction ของคนที่จับคู่แล้วเทียบกับ GT part ของเจ้าของคนนั้นบน shared valid grid ทั้งภาพ พิกเซล prediction ที่เกิน GT part เป็น FP ส่วน GT ที่ทำนายขาดเป็น FN งานนี้อ้างขอบเขต positive part mask ที่ผู้จัดทำตรวจรับ

พื้นที่ unassigned คงความหมายใน dataset source ในงาน binary person-part พื้นที่นอก reference part ของเจ้าของนั้นใช้เป็น non-target การตีความนี้ใช้เฉพาะงานส่วนที่มี GT อ้างอิง การรับรอง background ทั้งฉากต้องมีหลักฐานของงาน full-scene แยกกำกับ

### Face-local

สร้างกรอบสี่เหลี่ยมครอบ GT Face พอดี ไม่มี margin และตัดกับ shared valid grid ภายในกรอบ prediction ส่วนเกินยังเป็น FP ส่วนที่ขาดเป็น FN พื้นที่นอกกรอบไม่เข้าสู่คะแนน Face-local กรอบใช้ใน scoring หลัง inference

CIHP parsers ใช้ Face ID 13 ส่วน Sapiens ใช้ Face_Neck และ facial subparts ตาม mapping เดิม ผล Face-local อธิบาย agreement ภายในกรอบอ้างอิง ส่วนคุณภาพ full-frame face-only และ neck ต้องมีข้อกำหนดเพิ่ม

### Person matching ที่เหมาะกับ GT CCTV600

CCTV600 ส่งมอบ bbox และ masks สี่ส่วน รอบนี้จึงใช้ frozen GT-first bbox matcher โดยเรียง prediction ตาม confidence/box geometry แล้วเลือก eligible GT ที่ IoU สูงสุด คะแนนรายส่วนใช้ bbox IoU ≥0.50 และมี person diagnostics ที่ ≥0.75 ด้วย Unmatched prediction ที่ overlap ignore-rectangle union แบบ IoA ≥0.50 ได้สถานะ ignore

งานอ้างอิง CIHP/CCTV90 ใช้ GT-person-mask matching การปรับเป็น bbox matching เป็นข้อแตกต่างที่ประกาศสำหรับ CCTV600 การจับคู่ใช้ geometry และ scores ตามกติกาเดียวกันทุก parser คนที่จับคู่ไม่เจอใช้ prediction ว่าง ส่วนที่มี GT ยังคงอยู่ในตัวหารและมี IoU 0

## GT และ mapping

รวม source components ที่เป็นคลาส/owner เดียวกัน เก็บช่องว่างจากการบดบัง ใช้ common VOID สำหรับ explicit ignore, คนต่ำกว่า 50 px, class/owner conflicts, degenerate shapes และ shape ที่มี visual QA ยืนยัน class error เดิม ข้อกำหนดทั้งหมดใช้ร่วมกันทุกโมเดล

| Target | CIHP native IDs | Sapiens native IDs |
| --- | --- | --- |
| Hair | 2 | 4 |
| Face candidate | 13 | 3 และ 24–28 |
| Upper | 5, 7 | 23 |
| Lower | 9, 12 | 13 |
| OTHER | คลาสที่เหลือ รวม Dress 6 | คลาสที่เหลือ รวม Apparel 1 |

GT source IDs ยังเป็น Hair=1, Face=2, Upper=3, Lower=4 ส่วน shared scorer ใช้ Face=1, Hair=2, Upper=3, Lower=4 การแปลงระบุใน protocol เพื่อป้องกันสลับ Hair/Face การรวม subclasses ทำจาก projected masks ก่อนนับพิกเซล

Dataset ยังมีข้อจำกัดเรื่อง cross-zone garments และคุณภาพขอบ annotation ตาม release QA ผลรอบนี้อธิบาย checkpoint พร้อม application configuration บน reference ที่ผู้จัดทำยอมรับ

## Interruption และการแก้ที่บันทึกไว้

- QANet มี timer setup error ก่อน learned forward ในรอบแรก แก้การจับเวลาที่ registered modules และเก็บประวัติเดิม
- RP-R-CNN เคยหยุดด้วย `KeyError: parsing` การแก้ใหม่รับเป็น empty prediction เฉพาะเมื่อ traceback ยืนยัน native result มีจำนวนคนศูนย์ ผล nonempty ที่ผิด contract ยังทำให้หยุด
- CPI R50 เคยหยุดด้วย `PRE_FORWARD_FREE_VRAM_GUARD` ก่อน forward ของ ROI ถัดไป รอบใหม่อนุญาตให้คืนเฉพาะ unused PyTorch cache ของ process เดิมก่อนตรวจ guard เดิมอีกครั้ง พร้อมบันทึกก่อน/หลังและเวลาที่ใช้

การแก้ CPI คง threshold ของ resource guard, weights, input resolution, precision, batch, deterministic settings และ environment เดิม เมื่อทรัพยากรยังไม่พอหรือเกิด CUDA/model error จะเก็บผลและหยุดโมเดลนั้น ทุก attempt มีประวัติ ไม่มี automatic model/OOM retry

## เวลาและทรัพยากร

| Metric | หน่วยและตัวหาร |
| --- | --- |
| **Mean ms/ROI** | ผลรวม accepted adapter intervals หารจำนวน ROI ที่มี timer |
| **Mean s/image** | ผลรวม parsing work ของ ROI ทุกคน หารจำนวนภาพ 600 ภาพ แล้วแสดงวินาที |
| **Total parsing s** | ผลรวม accepted adapter intervals ของประชากร เมื่อ timer ครบตามเกณฑ์ |
| **Complete image timers** | จำนวนภาพที่มีเวลาครบทุก ROI ภาพไม่มี ROI มี parsing work ศูนย์ |
| **Clean timing rounds** | จำนวนรอบที่ผ่าน admission/contamination checks จากสามรอบ |

เก็บ pre/module/inter-module/post timings, ownership, map-back, output/readback และ peak allocated/reserved ตาม collector เดิม Model forward ทำนาย native classes ทั้งหมด เวลา per-class แยกอิสระไม่ได้ บันทึกการคืน cache เป็น resource housekeeping เพิ่มต่างหาก

ใช้ T4 ที่ตรวจพบจริง Quality inference ใช้ resource guards เดิมและคงงานภายนอกไว้ Dedicated timing ใช้ 100 ภาพ/124 ROIs ต่อรอบ ×3 และ TRAIN warmup 10 calls ต่อรอบ ส่วน quality process ใหม่มี TRAIN warmupหนึ่ง call ขอบเขต native APIs และ native file I/O ยังต่างกัน จึงอธิบายเวลาเป็น scoped diagnostics

## วิธีดูสถานะและผล

รันจาก workspace `Human-Body-Extraction`

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/reference_parts600.py --progress --watch
tmux attach -t parsing_cctv600_r3
```

Progress viewer แสดง reused/new native ROIs และ PID ที่ทำงานจริง กด Ctrl+C ใน viewer เพื่อออก กด Ctrl+B แล้ว D เพื่อ detach tmux

Artifact root ปัจจุบันอยู่ใน `outputs/CURRENT.json` เมื่อจบมี `BATCH_COMPLETION.json` และรายงานหลัก `BATCH_REPORT_TH.md` พร้อม `TARGET_RESULTS.csv`, `TIMING_SUMMARY.json`, รายโมเดล `METRICS.json` และ `PER_IMAGE_RUNTIME.csv` ชุดภาพ 16 ภาพเดิมอยู่ที่ `gallery/index.html` พร้อม manifest ภาพจริงและ raw outputs เก็บแบบ private

STRICT_COMPLETE ต้องมี exact 600 unique image IDs และ native ROI membership ครบทุกงาน คะแนนของระบบที่ยังไม่ครบเป็น null พร้อมจำนวนและเหตุผล ส่วน null ที่เกิดจากตัวหารศูนย์แสดงตาม support โดยมี provenance กำกับ

## สถานะของรอบก่อน

Protocol AD4 และ configs/outputs ก่อน restart เก็บเป็นประวัติใน Git และ parent run records ทางเข้าปัจจุบันใช้ shared-reference protocol ข้างต้น การเลือกวิธีครั้งนี้ทำตามคำขอผู้ใช้หลังมีผล CIHP/CCTV เดิมแล้ว จึงมีเหตุผลจากวิธีอ้างอิงและข้อกำหนด GT กำกับ

Sapiens2 มีข้อกำหนดการใช้ code/weights รวมข้อจำกัด surveillance ใน license การนำผลหรือโมเดลไปใช้งานปลายทางต้องพิจารณาสิทธิ์ของเจ้าของโมเดลตามเงื่อนไขเดิม
