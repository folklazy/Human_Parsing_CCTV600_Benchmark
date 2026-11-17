# บันทึกการดำเนินงาน

## 17 พฤศจิกายน 2026

- ผู้ใช้ให้ตรวจละเอียดและเริ่มงาน human parsing ถัดจาก detector CCTV600 ที่ปิดรอบแล้ว
- อ่าน proposal, dataset specification, accepted detector PREPARED/PACKET_AUDIT และ controlled ROI adapter/config เดิม ตรวจ process/GPU/disk ก่อนเปิดงาน
- ตอนตรวจเริ่มต้น Tesla T4 ว่าง 14,909 MiB ไม่มี compute PID พื้นที่ filesystem ว่างประมาณ 151 GiB มีบริการเว็บ resident ที่ยังไม่ใช้ GPU จึงคงบริการไว้
- แยก experiment `Human_Parsing_CCTV600_Benchmark` ไว้ใต้ workspace root รักษา dataset และผลเดิม
- ปิด valid-domain policy เป็น annotated target pixels ที่ไม่มี ambiguity ร่วมกันทุกระบบ เนื่องจาก unassigned pixels ยังไม่มี verified negative semantics เรียกคะแนน AD4 และคง full-frame primary null
- ตรวจภาพ diagnostic ของ Lower-at-head QA เดิม ยืนยันว่าข้อสังเกตมีหลักฐาน และกำหนด common QA VOID สำหรับ source shape เดิมโดยไม่เปลี่ยน annotation
- ใช้ RGB ROI + post-parsing YOLO-mask ownership เดิม เก็บ native outputs ก่อน projection และไม่ใช้ GT ช่วย inference
- เพิ่ม CPU fixture acceptance, atomic packets, sequential controller, saved-mask scoring checkpoints และ timing records พร้อมขอบเขต module API ที่ระบุจริง
- การเตรียมรอบแรกหยุดก่อน model calls เพราะ legacy environment ไม่มี pip เปลี่ยนเฉพาะตัวอ่าน package inventory ให้ใช้ importlib.metadata ของ environment เดิม เก็บ attempt directory ไว้ ไม่มีการติดตั้ง package หรือเปลี่ยน environment
- เตรียมครบ exact 600 IDs และ 763 predicted person ROIs ต่อโมเดล GT หลัง common VOID มี eligible person-parts Hair/Face/Upper/Lower เท่ากับ 544/390/740/665 และ valid pixels 7,373,840
- ตรวจ weights/config/source/environment ทั้งเก้า bindings ใช้สอง model environments เดิม Source/shim bindings 1,612 รายการ CPU acceptance 16 cases ผ่านก่อน launch
- ระหว่างเตรียม บริการเว็บเปิด detector/parser GPU processes เพิ่มขึ้น ว่างประมาณ 13,549 MiB จึงเก็บ resource observations และใช้ admission guards ตามโมเดล โดยคง process ของเว็บไว้
- ผูก 100 timing images และ 16 visual samples จาก detector manifests เดิม Timing มี 124 ROI calls ต่อรอบและ TRAIN warmup 10 calls ต่อรอบ Quality มี 763 ROI calls และ TRAIN warmup หนึ่ง call ต่อโมเดล
