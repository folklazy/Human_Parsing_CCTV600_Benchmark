# Human Parsing บน CCTV Human Parsing 600 Dataset

การทดลองนี้เปรียบเทียบเก้า pretrained checkpoints สำหรับ Hair, Face, Upper และ Lower บนภาพ CCTV 600 ภาพ ใช้ผล YOLO26x-Seg ชุดเดียวเป็นต้นทาง แล้วส่งภาพ RGB crop ให้ parser ทุกตัว GT มี person boxes และ masks สี่ส่วน โดยใช้บุคคลด้านยาวตั้งแต่ 50 px ในคะแนนกลุ่มหลัก

**ชื่อคะแนนหลักที่ใช้แสดงผลคือ “mIoU เฉลี่ย 4 คลาส — เฉพาะพิกเซลที่ GT ระบุคลาส”** ตารางผลต้องระบุพื้นที่คำนวณนี้กำกับ รหัส `AD4_mIoU4` ใน JSON ใช้อ้างผลเดียวกัน ส่วนค่าเฉลี่ยสามคลาสและคะแนนรายคนมีชื่อเต็มในหัวข้อตัวชี้วัดด้านล่าง

[สถานะและคำสั่งดูงาน](STATUS.md) · [ข้อกำหนดที่ใช้จริง](configs/PROTOCOL.json) · [บันทึกงาน](WORK_LOG.md) · [Dataset](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset) · [ผล detector CCTV600](https://github.com/folklazy/YOLO_Instance_Segmentation_CCTV600_Scaling_Study)

## ขอบเขตคะแนนที่ข้อมูลรองรับ

Repo ส่งมอบยืนยันว่าผู้จัดทำตรวจแก้ครบ 600 ภาพแล้ว นิยามของรหัส 0 คือพื้นที่นอก part shapes ซึ่งอาจเป็นฉากหลัง ส่วนอื่นของคน หรือพื้นที่ที่ยังไม่มีคำตอบ ใน schema กำหนด `unassigned_negative_domain = not_certified` และให้แต่ละการทดลองกำหนด valid domain ของตัวเอง ข้อความนี้บอกสถานะการใช้พิกเซลเพื่อคำนวณคะแนน หลักฐานการขาด annotation ต้องพิจารณาภาพและรายการตรวจรับเพิ่มเติม [README ของ dataset รุ่นที่ตรวจ](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset/blob/d77ad2eec84e3e44b1465abd17031c8f57e1f18f/README.md), [ตารางรหัสและกติกา unassigned](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset/blob/d77ad2eec84e3e44b1465abd17031c8f57e1f18f/schema/labels.json)

รอบทดลองนี้เลือกใช้เฉพาะพิกเซล GT ของ Hair/Face/Upper/Lower ที่ผ่านกติกา ignore และข้อสังเกต QA ร่วมกันทุกโมเดล การเลือกนี้เป็นกติกาของ benchmark ที่ตรึงไว้ก่อน inference ชื่อแสดงผลจึงระบุ “เฉพาะพิกเซลที่ GT ระบุคลาส” ทุกคะแนน semantic ใช้พื้นที่คำนวณชุดเดียวกัน และยังประเมินการระบายเกินเข้า unassigned ได้ไม่ครบ รหัสเดิม `AD4` เก็บไว้เชื่อมกับ config/JSON ของรอบที่กำลังรัน

หากยืนยันได้ว่าทุกส่วนที่ต้องประเมินของทุกคนที่เข้าเกณฑ์ได้รับการกำกับครบ และกำหนดพื้นที่ที่เหลือเป็น OTHER พร้อม ignore ที่ชัดเจน จะสร้าง GT/protocol revision สำหรับคะแนนบนพื้นที่ภาพที่ประเมินได้ทั้งหมดได้ โดยใช้ saved native masks ของทุกระบบคำนวณคะแนนตาม revision นั้น การรับรอง negative domain อาศัยความครบถ้วนของสี่ส่วนและกติกา OTHER/ignore

ชุดข้อมูลผ่านการยืนยัน snapshot โดยผู้จัดทำ มีประวัติใช้ annotation helpers และยังไม่มี independent review ทุกพิกเซล รายงานนี้จึงใช้คำว่า accepted snapshot และระบุข้อจำกัดไว้กับผลคะแนน

## โมเดลและ input

| ลำดับ | Checkpoint | Native classes | เส้นทางหลังรับ crop |
| --- | --- | --- | --- |
| 1 | QANet CIHP HRNet-W48 512×384 | 20 | Whole shared crop ผ่าน backbone/parsing head และ letterbox |
| 2 | M2FP R101 CIHP | 20 | Native multi-person parsing แล้วเลือก instance ด้วย designated predicted box |
| 3 | D2FP R101 CIHP | 20 | Native multi-person parsing แล้วเลือก instance ด้วย designated predicted box |
| 4 | RP-R-CNN CIHP | 20 | Native detector/parsing ภายใน crop แล้วเลือก designated instance |
| 5 | CPI-Parser R50 CIHP | 20 | Native detector/parser และ feature-branch scheduling revision 3 |
| 6 | CPI-Parser ConvNeXt-B CIHP | 20 | Native detector/parser และ person-chunk restoration revision 1 |
| 7 | Sapiens2 segmentation 0.4B | 29 | Whole shared crop semantic segmentation และ letterbox |
| 8 | Sapiens2 segmentation 0.8B | 29 | Whole shared crop semantic segmentation และ letterbox |
| 9 | Sapiens2 segmentation 1B | 29 | Whole shared crop semantic segmentation และ letterbox |

Weights, native configs, source HEAD, actual source files, compatibility shims และ environment packages ผูกด้วย hashes ใน private run config ใช้โมเดลที่มีอยู่ใน workspace และ adapter ที่ผ่านงาน controlled ROI เดิม การเปรียบเทียบนี้รวม checkpoint, native preprocessing และ application configuration

โค้ดเรียกโมเดลหลักนำมาจาก `CCTV_Human_Parsing_Benchmark/scripts/controlled_roi_models.py` และ `Human_Parsing_Benchmark/src/cihp_benchmark/` โดยตรวจ hashes ก่อนใช้ Dependencies เหล่านี้อยู่ใน workspace เดียวกัน รายการใน `requirements.txt` อธิบาย environment ที่ต้องใช้ การ clone repository นี้เพียงแห่งเดียวยังต้องจัดเตรียม shared model assets ตาม bindings

### YOLO และ crop

1. อ่าน accepted YOLO26x-Seg packets ของ detector run `cctv600-20261116T073208561699Z` ตรวจ input hash, packet hash และ detector bindings ครบ 600 IDs
2. เลือก person predictions ที่ confidence ≥0.25 โดยเก็บ detector IDs เดิม
3. ตัด RGB จากภาพต้นฉบับ เพิ่ม context 10% แต่ละด้าน และขยาย crop ให้ครอบคลุม frozen YOLO mask ทั้งหมด
4. Parser รับ RGB ที่คงพื้นหลังและคนข้างเคียงในกรอบ ใช้ native preprocessing ตาม config ของตัวเอง
5. ระบบที่คืนหลาย instance ใช้ bbox IoU กับ designated YOLO box ≥0.50 เลือก IoU สูงสุด ตามด้วย native score และ geometry โดยไม่ใช้ GT หรือคะแนน semantic เลือกผล
6. แปลง native hard labels กลับ crop grid แล้ววางบน original image grid ใช้ YOLO instance mask clip หลังทำนายเพื่อกำหนดเจ้าของ
7. Resolve overlap ด้วย YOLO score และ canonical box order โดย compose native labels ทั้งหมดก่อน project เป็นสี่ target กับ OTHER

ภาพไม่มี YOLO ROI เป็น prediction ว่างที่สำเร็จและมี parser calls ศูนย์ คนที่ detector พลาดยังอยู่ใน GT denominator ส่วนคนซ้อนกันยังเป็นข้อจำกัดของทั้ง input crop, designated-instance association และ predicted ownership masks

### สีของช่องภาพและ numerical settings

QANet/M2FP/D2FP ใช้ RGB ตาม config RP-R-CNN รับสะพาน BGR ตาม native API CPI อ่าน lossless PNG ผ่าน MMCV และตรวจ decoder probe ส่วน Sapiens ใช้ RGB crop → BGR CHW bridge → official normalizer หนึ่งครั้งและ mean-color padding

ใช้ FP32, batch 1, eval/inference mode, ไม่เปิด AMP/TF32/extra TTA การเปลี่ยน model/config/precision/resize/threshold หลังเริ่มงานต้องสร้าง revision พร้อมประเมินผลกระทบ การดูคะแนนระหว่างรันไม่มีบทบาทใน readiness

## Ground truth และ mapping

### GT ที่ใช้

`group_id` เป็น owner ภายในภาพ Source annotation ยังคงทุก byte เดิม Derived GT ใช้ Hair=1, Face=2, Upper=3, Lower=4 และ VOID=255

| กรณี | Disposition ร่วมกันทุกโมเดล |
| --- | --- |
| คนด้านยาว ≥50 px | Eligible bbox โดยวัด source floating-point box ก่อน clipping |
| คนต่ำกว่า 50 px | เก็บไว้ใน source และใช้กรอบเป็น shared ignore |
| Explicit ignore | Shared VOID ตาม rasterized source region |
| Parts ของ owner/class เดียวกัน | Union ทุก component โดยเก็บช่องว่างระหว่างชิ้น |
| ต่าง class หรือ owner ทับกัน | Shared VOID เฉพาะพิกเซลที่ขัดกัน |
| Shape ไม่มี geometric area | ไม่ใช้เป็นคำตอบ semantic และ void พิกเซลที่ raster วาดได้ |
| พิกัดออกขอบภาพเล็กน้อย | Raster clip ที่ขอบ canvas โดยไม่แก้ source coordinates |
| Class error ที่มี visual QA เดิมยืนยันแล้วหนึ่ง shape | Shared QA VOID คง source label เดิม และเก็บ exact shape binding แบบ private |
| ไม่มี shape ของ part | Unknown eligibility ของ part นั้น ไม่เติม absent/background |
| พื้นที่ unassigned ทั้งในและนอก bbox | Unknown/VOID |

Source QA เดิมระบุ 5 zero-area part shapes, 4 out-of-frame shapes, class/owner overlaps และ Lower-labelled shape ที่อยู่บริเวณศีรษะหนึ่งรายการ รอบนี้ตรวจภาพ diagnostic รายการหลังซ้ำและ void shape นั้นร่วมกัน ผล valid pixels, support และ eligible person-parts คำนวณจริงใน `GT_BINDINGS.json` ไม่มีการใช้จำนวน bbox คูณสี่แทน denominator

### Native labels → targets

| Target | CIHP20 prediction IDs | Sapiens29 prediction IDs |
| --- | --- | --- |
| Hair | 2 | 4 |
| Face ตาม CCTV600 reference | 13 | 3 และ 24–28 |
| Upper non-Dress | 5, 7 | 23 |
| Lower non-Dress | 9, 12 | 13 |
| OTHER | ทุก ID ที่เหลือ รวม 0 และ Dress 6 | ทุก ID ที่เหลือ รวม 0 และ Apparel 1 |

Sapiens Face_Neck แยก neck ออกด้วย hard labels ไม่ได้ จึงแสดงเป็น Face proxy ซึ่งรวมรายละเอียดใบหน้า ตรงกับ mapping ที่ตรวจไว้ก่อนหน้า Face GT ยึดนิยาม dataset ที่ไม่รวมคอ พื้นที่คอที่เป็น unknown ยังไม่เกิด FP ในคะแนน AD4 ต้องแสดงข้อจำกัดนี้เมื่ออธิบายผล

Source upper/lower ไม่มี cross-zone garment tags ครบ การกำหนด Dress/Apparel เป็น OTHER ยังไม่พิสูจน์ว่า annotation เสื้อผ้าทุกชิ้นตรง taxonomy ของ prediction ห้ามแบ่ง dress ด้วยเส้นตัดหรือใช้ GT เลือก prediction

## ตัวชี้วัดและวิธีอ่านชื่อ

ชื่อ metric ที่แสดงให้ผู้อ่านใช้คำเต็มพร้อมหน่วยและพื้นที่คำนวณ รหัสใน JSON มีไว้ตรวจย้อนกลับ สูตรและกติกาของรอบที่กำลังรันผูกกับ frozen protocol เดิม

| ชื่อที่ใช้ในรายงาน | รหัสเดิมในผลลัพธ์ | หน่วยและหน่วยนับ | วิธีรวมคะแนน |
| --- | --- | --- | --- |
| IoU รายคลาส — เฉพาะพิกเซลที่ GT ระบุคลาส | `IoU` ช่อง Hair/Face/Upper/Lower | % ของพื้นที่พิกเซล | รวม TP/FP/FN จากทุกภาพก่อนหาร แสดงหนึ่งค่าต่อคลาส |
| Precision รายคลาส — เฉพาะพิกเซลที่ GT ระบุคลาส | `precision` | % ของพิกเซลที่ทำนายคลาสนั้นในพื้นที่คำนวณ | TP ÷ (TP+FP) |
| Recall รายคลาส — เฉพาะพิกเซลที่ GT ระบุคลาส | `recall` | % ของพิกเซล GT คลาสนั้น | TP ÷ (TP+FN) |
| F1/Dice รายคลาส — เฉพาะพิกเซลที่ GT ระบุคลาส | `F1_Dice` | % จากจำนวนพิกเซล | 2TP ÷ (2TP+FP+FN) |
| mIoU เฉลี่ย 4 คลาส — เฉพาะพิกเซลที่ GT ระบุคลาส | `AD4_mIoU4`, เดิมแสดง AD4 | % เฉลี่ยสี่คลาส | บวก IoU ของผม ใบหน้า เสื้อผ้าบน เสื้อผ้าล่าง แล้วหาร 4 แต่ละคลาสมีน้ำหนักเท่ากัน |
| mIoU เฉลี่ย 3 คลาส — ผม เสื้อผ้าบน เสื้อผ้าล่าง | `C3_global` | % บนพื้นที่คำนวณเดียวกัน | บวก IoU ของสามคลาสแล้วหาร 3 พิกเซล GT ใบหน้ายังอยู่ในการคิด FP ของสามคลาสนี้ |
| IoU รายคนเฉลี่ย 4 คลาส — จับคู่กรอบที่ IoU ≥0.50 หรือ ≥0.75 | `Q_AD4_GT` ในแต่ละ matching threshold | % เฉลี่ยจาก person-parts ที่มี GT | คำนวณ IoU ของแต่ละคนต่อคลาส เฉลี่ยรายคลาสก่อน แล้วเฉลี่ยสี่คลาสเท่ากัน คนที่จับคู่ไม่เจอได้ 0 สำหรับส่วนที่มี GT |
| Precision/Recall/F1 ของการจับคู่คน | `person` → `precision`, `recall`, `F1` | % จากจำนวนกรอบบุคคล | ใช้ TP/FP/FN ของ bbox matching แต่ละ threshold |

ตัวอย่างสมมติของ mIoU เฉลี่ย 4 คลาส ถ้า IoU ผม=60%, ใบหน้า=40%, เสื้อผ้าบน=80%, เสื้อผ้าล่าง=60% จะได้ `(60+40+80+60)/4 = 60%` พื้นที่ของแต่ละคลาสยังมีผลต่อ IoU ของคลาสนั้น ส่วนขั้นเฉลี่ยสี่คลาสถ่วงแต่ละคลาสเท่ากัน

คะแนนรายคนให้น้ำหนักตาม person-parts ที่มีคำตอบในแต่ละคลาส ส่วน global IoU ให้น้ำหนักตามจำนวนพิกเซลที่รวมในคลาสนั้น ตัวอย่างเช่น ใบหน้าของคนเล็กกับคนใหญ่เป็นสองหน่วยในค่าเฉลี่ยรายคน ขณะที่ global counts รวมพิกเซลใบหน้าของทั้งสองคนก่อนหาร

GT bindings ของรอบนี้มีคู่คน–ส่วนที่ใช้ประเมิน ผม 544, ใบหน้า 390, เสื้อผ้าบน 740 และเสื้อผ้าล่าง 665 คู่ แต่ละคลาสจึงมีตัวหารรายคนของตัวเอง IoU/Precision/Recall/F1 ใช้สูตรจาก TP/FP/FN ส่วนขั้นเฉลี่ยคะแนนรายคนและถ่วงคลาสเท่ากันเป็น aggregation ที่โครงการกำหนด อ่านสูตรที่รันจริงได้ใน [scorer](src/evaluation.py) และ [frozen protocol](configs/PROTOCOL.json)

### คำที่ใช้บอกพื้นที่และสถานะ

| คำ | ความหมายในรอบนี้ |
| --- | --- |
| GT หรือคำตอบอ้างอิง | Annotation snapshot ที่ผู้จัดทำตรวจรับและผูก hashes ไว้ |
| พิกเซลที่ GT ระบุคลาส | พิกเซล Hair/Face/Upper/Lower ที่ผ่าน common ignore/QA เป็นพื้นที่คำนวณของคะแนน semantic รอบนี้ |
| Unassigned หรือพื้นที่ยังไม่กำหนดคลาส | พื้นที่นอก part shapes ตามรหัส 0 ของ dataset อาจประกอบด้วยฉากหลัง ส่วนอื่นของคน และบริเวณที่ยังไม่มีคำตอบ |
| OTHER หรือส่วนอื่นนอกสี่เป้าหมาย | กลุ่มผลทำนายที่รวม native background และคลาสอื่นนอก Hair/Face/Upper/Lower รวม Dress/Apparel ตาม mapping |
| Ignore/VOID หรือพิกเซลยกเว้น | พื้นที่ที่กติกากำหนดให้เว้นจากคะแนน เช่น explicit ignore คนต่ำกว่า 50 px และพิกเซลที่คลาส/เจ้าของขัดกัน |
| GT support | จำนวนพิกเซล GT ของคลาสนั้นที่เข้าสู่การคำนวณ |
| Prediction support | จำนวนพิกเซลที่ทำนายเป็นคลาสนั้นภายในพื้นที่คำนวณเดียวกัน |
| Union | TP+FP+FN ใช้เป็นตัวหารของ IoU |
| มี predictions ครบ | บันทึกผลครบประชากรและผ่าน completion checks ขั้นคำนวณคะแนนมีสถานะแยกกำกับ |

ค่า `null` ต้องแสดงเหตุผลประกอบเสมอ

| เหตุผลใน record | ข้อความแสดงผล |
| --- | --- |
| คะแนนทั้งภาพอยู่นอก domain ที่เลือกไว้ | “ยังไม่ประเมินคะแนนทั้งภาพตามกติกาปัจจุบัน” |
| เก็บ predictions ครบและรอ batch-end scoring | “ผลทำนายครบ รอคำนวณคะแนนหลังจบ batch” |
| ขาดภาพหรือมี invalid/failure ตาม STRICT_COMPLETE | “คะแนนยังใช้ไม่ได้ ผลไม่ครบตามเกณฑ์” พร้อมจำนวนและสาเหตุ |
| ตัวหารของ metric เป็นศูนย์ | “คำนวณค่านี้ไม่ได้ ตัวหารเป็นศูนย์” พร้อม GT/prediction support |

### Global semantic

เก็บ confusion 5×5 โดยแถวเป็น GT และคอลัมน์เป็น prediction รวมจำนวนพิกเซลทุกภาพก่อนหาร GT OTHER support เป็นศูนย์ใน annotation-supported domain ส่วน prediction OTHER บน target GT ยังคงเป็น FN

| Metric | สูตร | ความหมายบน domain นี้ |
| --- | --- | --- |
| IoU ของ target | TP / (TP+FP+FN) | พื้นที่ถูกคลาสเทียบกับพื้นที่ union ที่มีคำตอบ |
| Precision | TP / (TP+FP) | พิกเซลที่ทำนาย target ถูกคลาสเท่าไรบน annotated domain |
| Recall | TP / (TP+FN) | เก็บ target GT ที่มีคำตอบได้เท่าไร |
| F1 หรือ Dice | 2TP / (2TP+FP+FN) | สมดุลพื้นที่ถูกคลาส พื้นที่เกินใน domain และพื้นที่ตกหล่น |
| mIoU เฉลี่ย 4 คลาส — เฉพาะพิกเซลที่ GT ระบุคลาส | Mean IoU ของ Hair/Face/Upper/Lower | ค่าเฉลี่ยสี่ target บน domain ที่มีคำตอบ |
| mIoU เฉลี่ย 3 คลาส — ผม เสื้อผ้าบน เสื้อผ้าล่าง | Mean IoU ของ Hair/Upper/Lower | ตัด Face ออกจากค่าเฉลี่ย แต่ยังใช้ GT Face เป็น valid non-target |

ตัวหารศูนย์ให้ `null` หาก GT support ของ target ใดเป็นศูนย์ทั่วประชากร ค่า mean สี่ target เป็น `null` การ union subclasses คำนวณจาก mapped pixels ก่อนสร้าง confusion

### Person coverage และ semantic ต่อ GT

ใช้ matcher ของ detector CCTV600 ที่ freeze ไว้ แยก bbox IoU 0.50 และ 0.75 เรียง prediction ตาม confidence แล้ว geometry เลือก unmatched eligible GT ที่ IoU สูงสุดก่อนพิจารณา ignore ส่วน unmatched prediction ที่ ignore-rectangle union IoA ≥0.50 ได้สถานะ ignore รายงาน TP/FP/FN และ ignored counts

การตัดสิน tie ไม่ใช้ semantic labels กรอบ geometry ต่างกันคงผลเมื่อเปลี่ยนชื่อ IDs ส่วน identical geometry ใช้ stable source order และนับ ambiguity แยกใน bindings

ต่อ GT person และ target ที่มี positive valid support ใช้ mask ของ prediction ที่จับคู่ได้ ประเมินบน shared valid domain โดยพิกเซล target ของเจ้าของอื่นที่ติดมาเป็น FP คนที่พลาดให้ IoU ศูนย์ ไม่มี annotation ของ part จะไม่เข้าสู่ตัวหาร target นั้น

**IoU รายคนเฉลี่ย 4 คลาส** ใช้รหัส `Q_AD4_GT` ในผลลัพธ์ implementation เฉลี่ย person-part IoU ราย target ก่อน แล้วเฉลี่ยสี่ target เท่ากัน ตัวหารแต่ละ target มาจาก GT person-parts ที่มี positive valid support ซึ่งสร้างจริง รายงานแยกเกณฑ์จับคู่กรอบ IoU ≥0.50 และ ≥0.75 ค่าของ CIHP เดิม `Q_C4_GT` อ้าง matcher และ GT domains ของการทดลองนั้น การอธิบายผล CCTV600 จึงใช้ชื่อตามหน่วยนับและ threshold ของรอบนี้

## เวลา ทรัพยากร และคิว

Quality pass ใช้ GPU worker ครั้งละหนึ่งโมเดล ตรวจ free VRAM ตาม model admission เดิมและขั้นต่ำก่อนแต่ละ forward เก็บเวลาที่ใช้ GPU ร่วมเป็น diagnostics แยกจาก clean timing ตรวจ disk ≥50 GiB, run cap 80 GiB และ execution cap 24 ชั่วโมงต่อโมเดล ไม่มี model/OOM auto-retry หรือ fallback

| เวลาที่เก็บ | ขอบเขต |
| --- | --- |
| Source read | ตรวจ hash และ decode ภาพต้นฉบับ |
| Cached detector read | ตรวจ hash และอ่าน cached boxes/masks/scores |
| Crop read | ตรวจ lossless crop hash และ decode RGB |
| Pre-module | ช่วง adapter entry ถึง module call แรก |
| Module calls | CUDA-synchronized wall duration ของ actual model module calls |
| Inter-module | ช่วงระหว่าง module calls ภายใน adapter |
| Post-module | หลัง module call สุดท้ายจนคืน native selected output |
| Adapter total | ผลรวมสี่ช่วงข้างต้น รวม native internal detector/postprocessing/export ตามแต่ละ API |
| Ownership | Map crop mask กลับ original grid และ clip ด้วย detector extent |
| Composition/project | Resolve native overlaps แล้ว total-label mapping |
| Output/validation | Atomic NPZ/metadata/commit และ readback |
| เวลางาน parsing รวมทุกคนต่อภาพ (ms/ภาพ) | ผลบวก adapter duration ของทุก ROI ในภาพนั้น |
| เวลาต่อภาพจาก YOLO cache จนบันทึกผล (ms/ภาพ) | อ่านภาพและ cache จนเขียน frame packet ครบ โดยอ้าง detector predictions ที่บันทึกแล้ว |

QANet จับเวลา backbone/parsing-head calls ส่วน RP-R-CNN เรียก box_net/parsing_net ตรงจึงใช้ nonnested top-level child modules Native detector/parser APIs บางตัวทำ postprocessing และ file I/O ภายใน module boundary จึงต้องตีความ timer ตามขอบเขตที่ระบุ โมเดลทำนาย native classes ทั้งหมดใน forward เดียว ไม่มี latency ราย target ที่เป็นอิสระกัน

Dedicated timing ใช้ 100 image IDs จาก detector timing manifest เดิม ×3 รอบ ลำดับ roster หมุนตามรอบ Warmup TRAIN 10 calls ต่อรอบ และวัด ROI ตามจำนวนที่แต่ละภาพมีจริง Quality warmup ใช้ TRAIN หนึ่ง call ต่อ model process แยกจาก population 600 ภาพ

Clean round ต้องไม่มี external compute PID และ GPU utilization ≤5% ก่อน load ตรวจ external PID ก่อน/หลัง measured calls หากปนเปื้อนให้เก็บ round เป็น invalid โดยไม่ retry อัตโนมัติ รอบนี้ยังไม่จัดอันดับ clean parser-only หรือ live end-to-end speed จาก timers ต่างขอบเขต ส่วน YOLO inference times อ้างผล detector เดิมแยกต่างหาก

## การรันและติดตาม

รันคำสั่งจาก workspace `Human-Body-Extraction` ใช้ Python ตาม environment โดยตรง

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/readable_report.py --watch-status
tmux attach -t parsing_cctv600_r2
```

กด `Ctrl+B` แล้ว `D` เพื่อออกจาก tmux โดยงานยังทำต่อ กด `Ctrl+C` ใน progress viewer หยุดเฉพาะ viewer ส่วน `Ctrl+C` ในหน้าต่าง controller อาจขัดจังหวะ batch

Private artifact root อยู่ใน `outputs/CURRENT.json` ใช้ `BATCH_STATE.json`, `CURRENT_WORKER.json`, `logs/`, model `STATE.json` และ immutable `ATTEMPTS.jsonl` ตรวจสุขภาพงาน เมื่อมี `BATCH_COMPLETION.json` แสดงว่า batch หยุดตามขอบเขต ต้องอ่าน completed/failed counts ประกอบ รายงานสุดท้ายอัตโนมัติอยู่ที่ `BATCH_REPORT_TH.md`

การปรับชื่อ metric ใช้ `src/readable_report.py` ในขั้นแสดงผล เมื่อ batch จบจะปรับรายงาน `BATCH_REPORT_TH.md` เดิมด้วยชื่อเต็มและคำอธิบาย เก็บสำเนารายงานก่อนปรับชื่อและตรวจ hashes ของ canonical JSON พร้อมยืนยันว่าตัวเลขในตารางตรงเดิม การปรับนี้มี model/scorer calls เท่ากับศูนย์ รหัสใน frozen config และ canonical results ใช้ตรวจย้อนกลับตามเดิม

เมื่อจบ batch มี `gallery/index.html` และ PNG ชุด 16 ภาพเดิมจาก detector manifest แสดง original, GT และเก้าโมเดลบนพิกัดเดียวกัน พร้อม SHA256/provenance ภาพทั้งหมดคงอยู่ใน private outputs ตามนโยบาย dataset

Launcher รอบแรกใช้หลัง preparation, CPU acceptance และ commit ตรงกับ frozen bindings และปฏิเสธ run ที่เคยเปิดไว้

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/launch.py
```

รอบที่กำลังทำใช้ continuation `timer-r2` ผ่าน `src/correct_qanet.py` เพื่อแก้ timer attachment ของ QANet ซึ่ง native API เป็น methods รอบแรกหยุดที่ setup ก่อน model calls ทุกครั้ง CPU correction acceptance ผ่าน 17 cases รวม method-style dispatch ตัว controller เดิมปิดคิวขณะ M2FP worker ที่แยก session ทำงานต่อ จากนั้น continuation รับผล M2FP เดิมแล้วทำ QANet และโมเดลที่เหลือตาม roster โดยเก็บ native inference/scorer เดิมและประวัติ setup failure

ช่วง `QUEUED_WAITING_PARENT_MODEL_COMPLETION` หมายถึง controller ใหม่รอ M2FP เดิมจบ เมื่อผ่านจุดนั้นจะเปิด worker ตามคิว `CURRENT_WORKER.json` และ progress viewer ระบุ PID ที่ทำงานจริง รายละเอียดคำสั่ง launch และ parent/config/source hashes อยู่ใน private `LAUNCH.json` กับ `SCIENTIFIC_FREEZE.json`

แต่ละ ROI เก็บ native selected crop mask, native candidates, geometry และ hashes แต่ละภาพเก็บ original-grid owned native planes, YOLO extents/IDs/boxes/scores, composed native และ projected mask พร้อม references ไป ROI packets เก็บ raw native inputs/outputs เพื่อ rescore protocol revision ถัดไปได้โดยไม่รัน parser ซ้ำ

Packet commit ใช้ temporary directory, fsync และ atomic rename หาก interruption ต้องตรวจ committed/partial/orphan/not-attempted ก่อนเปิด continuation segment ใหม่ ผู้ช่วยต้องใช้ hashes และ frozen bindings ตรวจ reuse งานที่สำเร็จแล้ว ไม่มี automatic restart

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/reconcile.py --model QANet_CIHP_HRNetW48_512x384
```

คำสั่ง reconciliation อ่านผลและเขียน derived record ใหม่ เก็บ partial/orphan files ไว้ การเปิด continuation ยังต้องจัด linked segment หลังตรวจ record ดังกล่าว

### CPU acceptance

```bash
.venv/bin/python Human_Parsing_CCTV600_Benchmark/src/acceptance.py
```

ใช้ independent literal fixtures ตรวจ union, OTHER-before-overlap, unknown/VOID, under-50, ties/ID renaming, missed/duplicate/wrong-owner, denominator-zero, geometry/channel order และ timer boundary รวม packet → scorer → reporter พร้อม corruption/binding/immutability checks อ่าน accepted packet ของ QANet/M2FP/D2FP/RP-R-CNN อย่างละหนึ่งแบบ read-only โดยไม่ใช้ผลเก่าเป็นคะแนน CCTV600

ทดสอบ timer correction บน CPU แยกได้ด้วยคำสั่งนี้

```bash
.venv/bin/python -m unittest discover -s Human_Parsing_CCTV600_Benchmark/src -p correct_qanet.py -v
```

## แหล่งนิยามและข้อจำกัดการใช้โมเดล

นิยาม IoU/Dice/confusion อิง standard semantic segmentation counts การจำกัด valid pixels เป็นวิธีจัดการ partial labels ของการทดลองนี้ ต้องกล่าว scope คู่กับตัวเลขเสมอ Dataset และ controlled ROI เก็บแหล่ง annotation, taxonomy และ source revisions ของตัวเอง

- CIHP อธิบาย semantic parts และ person grouping ใช้ประกอบการแยก global semantic กับ person-associated metrics [Gong et al., ECCV 2018](https://arxiv.org/abs/1808.00157v1)
- Code native adapters และ mappings อ้าง source revisions QANet `61615c4`, M2FP `17a89de`, D2FP `9d41594`, RP-R-CNN `dfcc227`, CPI `d62cae5`, Sapiens2 `7e5bae8` โดย full hashes อยู่ใน frozen run config
- QANet/M2FP/D2FP/RP-R-CNN/CPI/Sapiens มีข้อกำหนด code และ weights ของแต่ละโครงการ Sapiens2 License ใน source checkout มีข้อจำกัด surveillance รวม research/development ที่เกี่ยวข้อง การนำไปใช้ปลายทางต้องมีหลักฐานสิทธิ์จากเจ้าของโมเดลตามเงื่อนไข [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/LICENSE.md)

CPI R50 ใช้ feature-branch scheduling revision 3 พร้อม buffer-reuse/person-chunk dependencies ส่วน ConvNeXt ใช้ person-chunk revision 1 ตาม adapter เดิม หลักฐาน parity เดิมครอบคลุม numeric controls และ diagnostic inputs ที่บันทึกไว้ แยกจาก packet integrity และ saved-mask scorer agreement และยังไม่มี bitwise proof ทั้งประชากร CCTV600

เมื่อ batch หยุด จึงรวมผลจาก canonical records และรายงานระบบที่ครบเท่านั้น Incomplete primary เป็น `null` ผล CCTV600 ชุดนี้มี GT domain ต่างจาก CIHP/CCTV90 จึงต้องแยกตารางตาม protocol ไม่ใช้ตัวเลขข้ามชุดแทนกัน
