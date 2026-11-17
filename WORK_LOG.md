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
- เปิด parent batch บน tmux ด้วย launch commit `e38d2be` พบว่า timer ของ QANet จับ method เป็น nn.Module ทำให้ setup หยุดก่อน model calls 0 ครั้ง เป็นข้อผิดพลาดของ instrumentation ที่เพิ่มในงานนี้ เก็บ STATE/FAILURE/ATTEMPTS และ native source เดิมไว้
- M2FP ประมวลผล inference จริงหลัง TRAIN warmup และเขียน packets สำเร็จ ตรวจ packet แรกแบบ read-only พร้อม reconstruction/ownership/hash/timer consistency และเปิด RGB/native/owned overlay เพื่อตรวจ alignment หนึ่งภาพ การตรวจครั้งนี้ไม่มีการสรุปคะแนน partial
- เพิ่ม `correct_qanet.py` เป็น correction แยก ใช้ hooks กับ registered child modules ของ QANet รวม CPU acceptance 17 cases ปิดเฉพาะ controller คิวเดิมหลังตรวจ PID และ process-group separation แล้วให้ M2FP worker เดิมทำงานต่อ คิวใหม่รอ parent จบและใช้ผลเดิม จากนั้นเปิด QANet ที่ยังไม่มี forward กับโมเดลที่เหลือ
- Parent config/source/GT/prediction bytes เก็บตามเดิม Continuation ผูก reuse compatibility, protocol/GT/source/config hashes และแต่ละผลระบุ source segment Native model retry/OOM retry ยังคงเป็นศูนย์ ส่วน QANet setup correction เป็น bounded attempt ที่ประกาศแยก
- เปิด detached continuation controller PID 3579861 บน tmux `parsing_cctv600_r2` ด้วย launch commit `12d2ff7` ตรวจ process/session จริงและ parent M2FP มี packets ใหม่เพิ่มต่อเนื่อง
- ตรวจ automatic handover สำเร็จ M2FP ปิด prediction collection ครบ 600 ภาพ/763 ROIs/764 API calls รวม warmup หนึ่งครั้ง Controller เปิด QANet corrected worker PID 3584087 ต่อ และตรวจ native20 packet ใหม่พร้อม module timer บน GPU จริงผ่าน ผลคะแนนคงรอ batch-end scoring

### ปรับชื่อและคำอธิบาย metric

- ผู้ใช้ขอให้ชื่อ metric สื่อความหมายและย้ำว่ามี repo dataset ส่งมอบแล้ว อ่าน README/ANNOTATION_SPEC/labels.json/DATASET.json ซ้ำ ตรวจ remote HEAD ตรง local `d77ad2eec84e3e44b1465abd17031c8f57e1f18f`
- อธิบายคำยืนยันตรวจแก้ครบ 600 ภาพควบคู่กับ field `unassigned_negative_domain=not_certified` ใน schema การตีความ field นี้ใช้กำหนดพื้นที่คะแนน ส่วนข้อสรุปว่ามี annotation ขาดต้องอ้างการตรวจภาพเพิ่มเติม
- ใช้ชื่อแสดงผล “mIoU เฉลี่ย 4 คลาส — เฉพาะพิกเซลที่ GT ระบุคลาส”, “mIoU เฉลี่ย 3 คลาส — ผม เสื้อผ้าบน เสื้อผ้าล่าง” และ “IoU รายคนเฉลี่ย 4 คลาส” พร้อมสูตร หน่วย denominator และ raw-key references
- เพิ่มการอธิบาย null ตามเหตุผล ผลไม่ครบ รอ scoring ตัวหารศูนย์ และพื้นที่ทั้งภาพที่กติกาปัจจุบันยังไม่ประเมิน
- เพิ่ม renderer ชื่อ metric และสถานะภาษาไทยเป็นขั้นแสดงผลหลัง batch พร้อมตรวจ numeric table cells, canonical small-record hashes, backup และ idempotence จาก synthetic fixture ใช้ model/scorer calls ศูนย์
- เปิดขั้นแสดงผลใน tmux window `metric_labels` PID 3625445 ให้รอ BATCH_COMPLETION แล้วปรับรายงานไฟล์เดิม ตรวจ process จริงและ scientific source hashes ของ parent/continuation ตรงเดิม

### เริ่มวิธีประเมินใหม่ตามคำขอผู้ใช้

- อ่าน publication และ working-copy รายงาน CIHP/CCTV พร้อม shared-reference scorer พบชุดชื่อ Mean-part IoU, Face-local, mIoU3 (part-mean), Macro4-local ตาม run shared-c3-face-local-20261108T083533Z และแยกจาก global C4-P ในประวัติ
- ตรวจ GitHub main snapshot 8505cbf4 และ local publication 29f7030 ซึ่งมีรายงานคนละ revision บันทึกแหล่งที่เลือกและ hashes แยกชัดเจน
- ผู้ใช้สั่งเริ่มใหม่ จึงปิดเฉพาะคิว controller R2 และให้ CPI ConvNeXt worker เดิมทำต่อจนจบ โดยเก็บผลและ failure records เดิมไว้
- สร้าง reference-parts R3 ภายใน experiment เดิม ใช้ accepted GT person-parts บน shared valid grid ทั้งภาพสำหรับ Hair/Upper/Lower และ tight GT Face bbox สำหรับ Face-local GT source และ native predictions เก็บตามเดิม
- ปรับ matching เป็น bbox IoU ตาม GT CCTV600 ที่มี ประกาศต่างจาก GT-mask matching ของ shared CIHP/CCTV90 และไม่สร้าง GT silhouette ขึ้นแทน
- ใช้ counts/summarize เดิม ตรวจ literal fixtures รวม Mean-part 0.5 เทียบ pooled 0.9, FP บนพื้นที่นอก GT part, Face-local FP ภายในกรอบและการเว้นนอกกรอบ, missed/empty/VOID, union/conflicts, ID renaming, label contract, null/STRICT_COMPLETE และ packet→scorer→reporter
- ตรวจ empty-output repair ของ RP-R-CNN บน CPU และตรวจ unused-cache-only release ของ CPI โดยคง physical guard เดิม ไม่เปลี่ยน allocator environment หรือ inference settings
- Actual packet integration ใช้สี่โมเดลที่ครบแบบ read-only บนตัวอย่างที่มีครบสี่ target เพื่อทดสอบ contract ไม่มีการใช้คะแนนเลือก mapping/settings
- Prelaunch reconciliation ตรวจผ่าน 3,282 ROIs เหลือ 3,585 ROIs โดย invalid/partial/orphan เป็นศูนย์ เก็บค่ารายโมเดลพร้อม packet/config/model/input bindings
- เปิด R3 detached controller PID 3684804 บน tmux parsing_cctv600_r3 ด้วย commit c409eb2 ตรวจ RP-R-CNN worker PID 3686707 และ packet ใหม่ของ failed ROI เดิมผ่าน Native BoxList มีศูนย์คนจริงจึงเก็บ empty prediction ตาม contract ผลช่วง nonempty ใช้ native API เดิม

## 2026-11-17 — GT revision audit and queue hold

ผู้จัดทำรายงานการแก้ annotation และให้ตรวจ dataset/เอกสารก่อนดำเนิน benchmark ต่อ ตรวจภาพและ JSON ครบ 600 รายการ พบ annotation hash เปลี่ยน 23 ไฟล์ ภาพต้นฉบับและ bbox geometry/owners คงเดิมทุกภาพ Dataset รุ่นตรวจสอบ 1.0.1-rc.1 มี part shapes 2,449 ชิ้น ผ่าน schema และ owner links โดยยังมี overlap QA กำกับ เอกสาร dataset และ private snapshot/manifest เก็บแยก revision

R3 controller หยุดเพื่อรอ GT revision ส่วน native Sapiens2 0.4B ที่ปล่อยให้จบมี collector completion exact 600 IDs และ state 763 ROIs จบ 2026-11-17T13:30:49+07:00 failed/invalid 0 หกโมเดลก่อนหน้ามี assembled completion ครบ600 ภาพ ข้อมูลนี้เป็นสถานะ native outputs การนำ 0.4B เข้ารอบใหม่ต้องผ่าน reconciliation งาน0.8B/1B, scoring และ timing ยังพัก การตรวจ dataset รอบนี้ใช้ CPU และ model calls0

Updated README/STATUS only for benchmark state. Frozen scientific source/config, prior prediction packets and historical run records remain bound to their original revision. The requested post-edit launcher was deferred when the user redirected work to dataset validation and Markdown documentation.

## 2026-11-17 — User adjudication of the three overlap examples

ผู้จัดทำตรวจสามกรณี Hair/Upper ที่แจ้งแล้วและยืนยันรูปทรงตาม source โดยอธิบายว่าเป็นผลจากกล้องมุมกดลง บันทึกสถานะ USER_REVIEWED_ACCEPTED_OVERLAP ในพื้นที่ข้อมูล private และเอกสาร dataset การตรวจ CPU เฉพาะสามกรณีด้วย gt_export เดิมพบ Hair valid support หนึ่งส่วนเป็น0px อีกส่วน41px หลังยกเว้น cross-class overlaps จึงเพิ่มข้อจำกัด support/eligibility ในเอกสาร คะแนนต่อไปต้องเปิดเผยตัวหารตาม export จริง รอบนี้เก็บ source polygons และ membership hash เดิม พร้อมบันทึกคำยืนยันและผลกระทบต่อการวัดแยกกัน

## 2026-11-17 — Partial-person coverage confirmed before parsing restart

ผู้จัดทำยืนยันว่ามีคนบางส่วนที่ยังไม่ได้ annotate และปิดขอบเขตงานตามชุด600 ภาพ Source ID0 จึงคงเป็น Unassigned เอกสาร dataset เพิ่มขอบเขต annotated-person-part agreement และแยก unmatched detections ออกจาก confirmed FP ของ detector คะแนน full-scene ที่ยังไม่มี negative domain รับรองใช้ null

ตรวจ CPU GT-domain preflight บน manifest ที่มี membership SHA256 256a191352a2f32150fdf2d6a8ddc24845780f5885a8d7b2f785008a505a3706 ได้769size-eligible GT people และ Hair/Face/Upper/Lower parts544/384/745/668 พบ Hair1ส่วนไม่มี valid support และ bbox geometry ต่างจาก native-run GT2ภาพ Input image identityตรง600 แต่ GT bindingsเดิมใช้ต่อกับคำตอบใหม่ไม่ได้ Full-scene person scalar counts ของ frozen scorer ต้องมี reporter scope correction ในรอบใหม่ แก้เฉพาะเอกสารและเก็บผลตรวจ private ไม่มี model calls หรือการแก้ frozen scientific source/config

## 2026-11-17 — Agreed test outputs and launch preparation status

บันทึกตารางตัวชี้วัดที่ตกลงกับผู้ใช้ใน README รวม cross-class/binary confusion, TP=0 part counts, เวลา, peak VRAM และ STRICT_COMPLETE ตรวจสคริปต์จริงพบว่ามี binary confusion และ raw peak allocated/reserved logs แล้ว ส่วน cross-class matrix, scoped partial-person reporter, ตัวนับส่วนพลาดทั้งหมด และ peak summary ต้องเตรียมใน source แยกก่อนเปิดคิวใหม่ GT-domain preflight ผ่าน600ภาพแล้ว รอบตรวจสถานะนี้มี model calls0

## 2026-11-17 — User requests all parser predictions fresh

ผู้ใช้เปลี่ยนคำสั่งเป็นเริ่ม parsing ใหม่ทั้งหมด ตั้ง parser_prediction_reuse0 และเก็บรอบก่อนเป็นประวัติ แยก scoped_run/scoped_metrics/scoped_acceptance จาก frozen sources ใช้ GT ส่งมอบล่าสุดกับ frozen detector/RGB input assets เดิม CPU fixtures ผ่านทั้ง literal cross-class/binary counts, Face-local, missed/empty/VOID/duplicate/null handling, source-ID geometry conversion, atomic packets, timer/VRAM reporting, current native-config bindings และ worker failure หลัง packet สุดท้าย Resource limits และ inference settings เดิมคงตาม source bindings เปิดคิว sequential ใน tmux หลัง preparation/readiness ผ่าน ไม่มี model retry อัตโนมัติ

## 2026-11-17T16:34:18+07:00 — Fresh run launch verified

Run cctv600-fresh-20261117T092941639648Z ใน tmux cctv600_fresh_20261117T092941639648Z Controller 3955217 และ QANet worker 3955495 มีผลใหม่จริง 85/600ภาพ 114/763ROIs ณ snapshot failed/invalid0 First native packet readback, integer labels, original-grid geometry, frozen YOLO ownership และ current config binding ผ่าน Source/code freeze อ้าง launch commit b42f435338cc79e0d12f68841aa052e0cb06cef3 Main launch command จบแล้วและ controller มี parent เป็น tmux server ผู้ใช้ดูสถานะต่อได้โดยไม่ต้องให้ Codexเฝ้า Quality scoresยังไม่คำนวณระหว่าง collection
