# สถานะ Human Parsing CCTV600

**ตรวจข้อมูลก่อนเปิดรอบใหม่ — คิว inference พักอยู่**

Dataset ส่งมอบมี 600 ภาพ, 774 bbox, 2,449 part shapes และ 171 ignore shapes ผลตรวจไฟล์/schema ผ่านครบชุด ผู้จัดทำยืนยันว่ามีคนบางส่วนที่ไม่ได้ annotate และปิดขอบเขตข้อมูลตามชุดนี้แล้ว Source ID0 คงเป็น Unassigned อ่าน [เอกสาร dataset](https://github.com/folklazy/CCTV-Human-Parsing-600-Dataset)

## ผลตรวจ GT domain บน CPU

| รายการ | ผล |
| --- | --- |
| Exact population | 600 unique IDs |
| คนผ่านเกณฑ์ด้านยาว ≥50 px | 769 |
| Hair / Face / Upper / Lower GT parts หลัง shared domain | 544 / 384 / 745 / 668 |
| Hair parts ที่ถูกยกเว้นทั้งหมดจาก conflict domain | 1 |
| Image identity เทียบ native input manifest | ตรงครบ600 |
| GT bbox geometry เปลี่ยน | 2 ภาพ ต้องผูก GT/matching ใหม่ |
| Model calls ในการตรวจข้อมูลครั้งนี้ | 0 |

Membership SHA256 ของคำตอบที่ใช้ตรวจคือ `256a191352a2f32150fdf2d6a8ddc24845780f5885a8d7b2f785008a505a3706` ผล bindings ของรอบเก่ายังคงอ้างคำตอบเดิม การเตรียมรอบใหม่ต้องสร้าง GT bindings และ output แยก

## สิ่งที่ต้องปรับก่อนเปิดรอบใหม่

1. ใช้ manifest จาก private dataset CURRENT.json สร้าง GT masks, shared valid domains, bbox matches และ eligibility ใหม่ Source IDs Hair1/Face2 ต้องแปลงเป็น reference target IDs Hair2/Face1 ให้ตรง scorer
2. ประกาศขอบเขตเป็น annotated-person-part agreement ใช้ Macro4-local, mIoU3 (part-mean), Mean-part IoU และ pooled Precision/Recall/F1-Dice ตาม domain ที่กำหนด Positive GT parts ของคนที่พลาดใช้ empty prediction และคงอยู่ในตัวหาร
3. ให้ reporter แสดง annotated-person recall และ unmatched predictions การนับทุก unmatched เป็น confirmed FP ไม่สอดคล้องกับ partial-person coverage Person AP/Precision/F1 ทั้งภาพและ global semantic metrics ที่ยังไม่มี negative domain รับรองให้เป็น null พร้อมเหตุผล
4. ตรวจ CPU acceptance เรื่องคนที่ไม่ได้ annotate, source0, prediction background, native ID mappings, conflicts และ STRICT_COMPLETE ก่อน freeze config ของรอบใหม่ Source/scorer/config ที่ freeze แล้วเก็บเป็นหลักฐานเดิม
5. ตรวจ reuse compatibility ของ native ROI packets พร้อม crop/model/checkpoint/source hashes แล้วรวบรวมเฉพาะ native outputs ที่ยังขาด การเปลี่ยน GT ใช้เป็นเหตุผลในการสร้าง bindings และคะแนนใหม่ ส่วนการ inference เพิ่มขึ้นกับความตรงกันของ model inputs และ packets

## ผล native ที่เก็บอยู่

หกโมเดล QANet/M2FP/D2FP/RP-R-CNN/CPI R50/CPI ConvNeXt-B มี assembled prediction completion ครบ600 ภาพต่อโมเดล Sapiens2 0.4B มี collector completion600 IDs และ763ROIs โดยต้องผ่าน reconciliation สำหรับรอบใหม่ Sapiens2 0.8B/1B ยังอยู่ในคิวที่พักไว้ คะแนนจาก GT ปัจจุบันยังไม่ได้คำนวณ

## ตำแหน่งตรวจงาน

- Dataset bindings อ้างจาก private dataset CURRENT.json
- ผล preflight เก็บใน private dataset provenance และเชื่อมไว้ใน CURRENT.json
- Native run และ logs อ้างจาก outputs/CURRENT.json ของ benchmark
- บันทึกการเปิดและพักงานอยู่ใน WORK_LOG.md และ immutable run records

[นิยามและ pipeline](README.md) · [Frozen reference protocol เดิม](configs/SHARED_REFERENCE_V1.json)
