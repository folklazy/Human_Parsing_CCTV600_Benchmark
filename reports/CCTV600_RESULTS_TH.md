# ผล Human Parsing บน CCTV600

ปิดรอบ 18 พฤศจิกายน 2026 เวลา 11:12:31 น. (Asia/Bangkok, UTC+07:00)

ตรวจรับผลคุณภาพครบ 9 โมเดล โมเดลละ 600 ภาพและ 763 ROI ไม่ขาดหรือซ้ำ จับเวลาแยกครบโมเดลละ 3 รอบ รวม 27 รอบ บน Tesla T4

ประชากร600ภาพเป็น annotation ของบุคคลที่เลือก Source ID0 เป็น Unassigned คะแนนอ้าง GT person-parts ที่มีคำตอบพร้อมขอบเขตอ้างอิง ใช้ predicted YOLO ownership และ bbox matchingที่0.50 Face-local ใช้กรอบ GT Face พอดีใน scoring ผลนี้เปรียบเทียบ checkpoint พร้อม application configuration

GT peopleผ่านเกณฑ์ 769 รายการ Hair/Face/Upper/Lower parts 544/384/745/668

| Model | Hair IoU % | Face-local IoU % | Upper IoU % | Lower IoU % | mIoU3 part-mean % | Macro4-local % | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QANet_CIHP_HRNetW48_512x384 | 49.606 | 61.551 | 62.933 | 55.977 | 56.172 | 57.517 | COMPLETE |
| M2FP_R101_CIHP | 37.568 | 45.256 | 45.421 | 40.018 | 41.002 | 42.066 | COMPLETE |
| D2FP_R101_CIHP | 33.616 | 42.013 | 41.333 | 36.514 | 37.155 | 38.369 | COMPLETE |
| RPR_CIHP | 31.190 | 39.629 | 41.328 | 35.630 | 36.049 | 36.944 | COMPLETE |
| CPI_R50_CIHP | 31.989 | 43.922 | 31.677 | 27.018 | 30.228 | 33.652 | COMPLETE |
| CPI_ConvNeXtB_CIHP | 51.133 | 59.953 | 51.985 | 45.155 | 49.425 | 52.057 | COMPLETE |
| Sapiens2_0.4b | 46.412 | 60.227 | 55.404 | 46.890 | 49.569 | 52.233 | COMPLETE |
| Sapiens2_0.8b | 46.787 | 61.729 | 55.848 | 48.806 | 50.480 | 53.293 | COMPLETE |
| Sapiens2_1b | 54.501 | 66.266 | 61.586 | 54.384 | 56.823 | 59.184 | COMPLETE |

## ข้อสรุปจากคะแนนเฉลี่ยรายส่วน

ตารางหลักใช้ Mean-part IoU: คำนวณ IoU ของ GT แต่ละส่วนก่อน แล้วเฉลี่ยภายในคลาส แต่ละส่วนจึงมีน้ำหนักเท่ากัน ส่วนตาราง pooled รวมพิกเซลก่อนคำนวณและตอบคนละคำถาม

| เป้าหมาย | คะแนนสูงสุด | Mean-part IoU % | สูงกว่าอันดับถัดไป (จุดเปอร์เซ็นต์) |
| --- | --- | --- | --- |
| ผม | Sapiens2_1b | 54.501 | 3.368 |
| ใบหน้าภายในกรอบ GT Face | Sapiens2_1b | 66.266 | 4.537 |
| เสื้อผ้าส่วนบน non-Dress | QANet_CIHP_HRNetW48_512x384 | 62.933 | 1.347 |
| เสื้อผ้าส่วนล่าง non-Dress | QANet_CIHP_HRNetW48_512x384 | 55.977 | 1.594 |

Sapiens2 1B ได้ Macro4-local สูงสุด 59.184% ตามด้วย QANet 57.517% ต่างกัน 1.667 จุดเปอร์เซ็นต์ จึงเป็นตัวเลือกแรกหากให้ทั้งสี่ส่วนมีน้ำหนักเท่ากัน ส่วน QANet เป็นตัวเลือกเมื่อเน้นคะแนนเฉลี่ยรายส่วนของเสื้อผ้าบนและล่าง ข้อสรุปนี้ผูกกับเกณฑ์ Mean-part: เมื่อรวมพิกเซลก่อนหาร Sapiens2 1B มี pooled IoU สูงสุดทั้งสี่คลาส ห้ามสลับสองวิธีรวมคะแนนโดยไม่ระบุ

ผลนี้เป็นการเปรียบเทียบ checkpoint พร้อม application configuration บนส่วนที่มี annotation ในชุดนี้ Face-local ของ Sapiens ใช้ Face_Neck และรายละเอียดใบหน้าเป็นตัวแทน ไม่รับรอง face-only equivalence หรือผลทั้งฉาก และยังไม่ได้ทดสอบนัยสำคัญทางสถิติ

## ความครบถ้วนของผล

| Model | Completed images | Missing images | Duplicate / unexpected | Completed ROIs | Failed / invalid / setup |
| --- | --- | --- | --- | --- | --- |
| QANet_CIHP_HRNetW48_512x384 | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |
| M2FP_R101_CIHP | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |
| D2FP_R101_CIHP | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |
| RPR_CIHP | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |
| CPI_R50_CIHP | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |
| CPI_ConvNeXtB_CIHP | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |
| Sapiens2_0.4b | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |
| Sapiens2_0.8b | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |
| Sapiens2_1b | 600 | 0 | 0 / 0 | 763 | 0 / 0 / 0 |

## นิยามและวิธีรวมคะแนน

IoU = TP/(TP+FP+FN), Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1/Dice = 2TP/(2TP+FP+FN) พิกเซลนอก reference part ของเจ้าของที่ประเมินเป็น non-target ภายใน shared valid domain Mean-part IoU เฉลี่ย IoU รายส่วนภายในคลาส ส่วน pooled metrics รวม counts ก่อนหาร Macro4-local เฉลี่ย Hair/Face-local/Upper/Lower คลาสละ25% mIoU3 part-mean เฉลี่ย Hair/Upper/Lower

Missed GT person ใช้ empty prediction และเก็บ positive parts ในตัวหาร ส่วนที่ TP=0 นับเป็นส่วนพลาดทั้งหมด ส่วนที่ไม่มี valid GT support ยกเว้นจาก denominator ตาม GT bindings STRICT_COMPLETE ต้องครบ600ภาพและ ROI jobs ตาม manifest คะแนนหลักของระบบที่ไม่ครบเป็น null

## Counts, IoU และ F1 รายคลาส

| Model | Target | GT parts | TP px | FP px | FN px | TN px | Precision % | Recall % | F1/Dice % | Pooled IoU % | TP=0 parts | TP=0 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QANet_CIHP_HRNetW48_512x384 | Hair | 544 | 631886 | 105575 | 193470 | 1132098233 | 85.684 | 76.559 | 80.865 | 67.877 | 106 | 19.485 |
| QANet_CIHP_HRNetW48_512x384 | Face_local | 384 | 205318 | 15693 | 59898 | 137597 | 92.899 | 77.415 | 84.454 | 73.091 | 48 | 12.500 |
| QANet_CIHP_HRNetW48_512x384 | Upper | 745 | 3404732 | 461166 | 990057 | 1546880385 | 88.071 | 77.472 | 82.432 | 70.115 | 146 | 19.597 |
| QANet_CIHP_HRNetW48_512x384 | Lower | 668 | 1386698 | 249727 | 506266 | 1389935619 | 84.739 | 73.255 | 78.580 | 64.718 | 135 | 20.210 |
| M2FP_R101_CIHP | Hair | 544 | 526564 | 79956 | 298792 | 1132123852 | 86.817 | 63.798 | 73.549 | 58.164 | 216 | 39.706 |
| M2FP_R101_CIHP | Face_local | 384 | 167066 | 12118 | 98150 | 141172 | 93.237 | 62.992 | 75.187 | 60.240 | 129 | 33.594 |
| M2FP_R101_CIHP | Upper | 745 | 2884813 | 343979 | 1509976 | 1546997572 | 89.347 | 65.642 | 75.681 | 60.877 | 307 | 41.208 |
| M2FP_R101_CIHP | Lower | 668 | 1170986 | 193946 | 721978 | 1389991400 | 85.791 | 61.860 | 71.886 | 56.111 | 288 | 43.114 |
| D2FP_R101_CIHP | Hair | 544 | 503014 | 78180 | 322342 | 1132125628 | 86.548 | 60.945 | 71.525 | 55.672 | 250 | 45.956 |
| D2FP_R101_CIHP | Face_local | 384 | 167649 | 11367 | 97567 | 141923 | 93.650 | 63.212 | 75.478 | 60.614 | 154 | 40.104 |
| D2FP_R101_CIHP | Upper | 745 | 2621930 | 287047 | 1772859 | 1547054504 | 90.132 | 59.660 | 71.797 | 56.002 | 340 | 45.638 |
| D2FP_R101_CIHP | Lower | 668 | 1102170 | 160447 | 790794 | 1390024899 | 87.293 | 58.225 | 69.855 | 53.675 | 321 | 48.054 |
| RPR_CIHP | Hair | 544 | 518123 | 112865 | 307233 | 1132090943 | 82.113 | 62.776 | 71.154 | 55.224 | 230 | 42.279 |
| RPR_CIHP | Face_local | 384 | 168562 | 14770 | 96654 | 138520 | 91.944 | 63.556 | 75.159 | 60.204 | 146 | 38.021 |
| RPR_CIHP | Upper | 745 | 2553543 | 306678 | 1841246 | 1547034873 | 89.278 | 58.104 | 70.394 | 54.314 | 236 | 31.678 |
| RPR_CIHP | Lower | 668 | 1083399 | 211449 | 809565 | 1389973897 | 83.670 | 57.233 | 67.971 | 51.482 | 251 | 37.575 |
| CPI_R50_CIHP | Hair | 544 | 459781 | 125574 | 365575 | 1132078234 | 78.547 | 55.707 | 65.184 | 48.351 | 201 | 36.949 |
| CPI_R50_CIHP | Face_local | 384 | 151844 | 13146 | 113372 | 140144 | 92.032 | 57.253 | 70.591 | 54.549 | 111 | 28.906 |
| CPI_R50_CIHP | Upper | 745 | 2077517 | 224983 | 2317272 | 1547116568 | 90.229 | 47.272 | 62.041 | 44.970 | 274 | 36.779 |
| CPI_R50_CIHP | Lower | 668 | 835605 | 118343 | 1057359 | 1390067003 | 87.594 | 44.143 | 58.703 | 41.545 | 306 | 45.808 |
| CPI_ConvNeXtB_CIHP | Hair | 544 | 652399 | 145147 | 172957 | 1132058661 | 81.801 | 79.045 | 80.399 | 67.223 | 107 | 19.669 |
| CPI_ConvNeXtB_CIHP | Face_local | 384 | 202778 | 14806 | 62438 | 138484 | 93.195 | 76.458 | 84.001 | 72.415 | 60 | 15.625 |
| CPI_ConvNeXtB_CIHP | Upper | 745 | 3046434 | 334374 | 1348355 | 1547007177 | 90.110 | 69.319 | 78.359 | 64.418 | 194 | 26.040 |
| CPI_ConvNeXtB_CIHP | Lower | 668 | 1279508 | 152040 | 613456 | 1390033306 | 89.379 | 67.593 | 76.974 | 62.568 | 223 | 33.383 |
| Sapiens2_0.4b | Hair | 544 | 679898 | 253627 | 145458 | 1131950181 | 72.831 | 82.376 | 77.310 | 63.013 | 98 | 18.015 |
| Sapiens2_0.4b | Face_local | 384 | 221049 | 26093 | 44167 | 127197 | 89.442 | 83.347 | 86.287 | 75.881 | 63 | 16.406 |
| Sapiens2_0.4b | Upper | 745 | 3605006 | 741620 | 789783 | 1546599931 | 82.938 | 82.029 | 82.481 | 70.185 | 141 | 18.926 |
| Sapiens2_0.4b | Lower | 668 | 1469877 | 412390 | 423087 | 1389772956 | 78.091 | 77.649 | 77.870 | 63.759 | 185 | 27.695 |
| Sapiens2_0.8b | Hair | 544 | 645577 | 179240 | 179779 | 1132024568 | 78.269 | 78.218 | 78.244 | 64.262 | 119 | 21.875 |
| Sapiens2_0.8b | Face_local | 384 | 224238 | 28248 | 40978 | 125042 | 88.812 | 84.549 | 86.628 | 76.411 | 57 | 14.844 |
| Sapiens2_0.8b | Upper | 745 | 3682770 | 592824 | 712019 | 1546748727 | 86.135 | 83.799 | 84.951 | 73.838 | 153 | 20.537 |
| Sapiens2_0.8b | Lower | 668 | 1505577 | 374356 | 387387 | 1389810990 | 80.087 | 79.535 | 79.810 | 66.403 | 189 | 28.293 |
| Sapiens2_1b | Hair | 544 | 677888 | 136304 | 147468 | 1132067504 | 83.259 | 82.133 | 82.692 | 70.491 | 89 | 16.360 |
| Sapiens2_1b | Face_local | 384 | 226294 | 27205 | 38922 | 126085 | 89.268 | 85.324 | 87.252 | 77.386 | 53 | 13.802 |
| Sapiens2_1b | Upper | 745 | 3672766 | 578158 | 722023 | 1546763393 | 86.399 | 83.571 | 84.962 | 73.855 | 135 | 18.121 |
| Sapiens2_1b | Lower | 668 | 1579864 | 416421 | 313100 | 1389768925 | 79.140 | 83.460 | 81.243 | 68.411 | 135 | 20.210 |

## Confusion matrix ของ QANet_CIHP_HRNetW48_512x384

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 631886 | 12311 | 1754 | 7 | 179398 |
| Face_local | 8819 | 205318 | 1949 | 0 | 49130 |
| Upper | 10932 | 3851 | 3404732 | 41592 | 933682 |
| Lower | 597 | 0 | 99405 | 1386698 | 406264 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 631886 | 193470 | 105575 | 1132098233 |
| Face_local | 205318 | 59898 | 15693 | 137597 |
| Upper | 3404732 | 990057 | 461166 | 1546880385 |
| Lower | 1386698 | 506266 | 249727 | 1389935619 |

## Confusion matrix ของ M2FP_R101_CIHP

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 526564 | 25255 | 4453 | 0 | 269084 |
| Face_local | 6005 | 167066 | 4587 | 0 | 87558 |
| Upper | 10328 | 13987 | 2884813 | 39941 | 1445720 |
| Lower | 607 | 79 | 62583 | 1170986 | 658709 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 526564 | 298792 | 79956 | 1132123852 |
| Face_local | 167066 | 98150 | 12118 | 141172 |
| Upper | 2884813 | 1509976 | 343979 | 1546997572 |
| Lower | 1170986 | 721978 | 193946 | 1389991400 |

## Confusion matrix ของ D2FP_R101_CIHP

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 503014 | 15632 | 3170 | 17 | 303523 |
| Face_local | 5799 | 167649 | 2596 | 73 | 89099 |
| Upper | 14416 | 20020 | 2621930 | 30437 | 1707986 |
| Lower | 1026 | 577 | 68065 | 1102170 | 721126 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 503014 | 322342 | 78180 | 1132125628 |
| Face_local | 167649 | 97567 | 11367 | 141923 |
| Upper | 2621930 | 1772859 | 287047 | 1547054504 |
| Lower | 1102170 | 790794 | 160447 | 1390024899 |

## Confusion matrix ของ RPR_CIHP

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 518123 | 14293 | 3332 | 20 | 289588 |
| Face_local | 10791 | 168562 | 2405 | 47 | 83411 |
| Upper | 25836 | 21584 | 2553543 | 78640 | 1715186 |
| Lower | 1878 | 364 | 77890 | 1083399 | 729433 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 518123 | 307233 | 112865 | 1132090943 |
| Face_local | 168562 | 96654 | 14770 | 138520 |
| Upper | 2553543 | 1841246 | 306678 | 1547034873 |
| Lower | 1083399 | 809565 | 211449 | 1389973897 |

## Confusion matrix ของ CPI_R50_CIHP

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 459781 | 8434 | 11020 | 1594 | 344527 |
| Face_local | 9455 | 151844 | 2169 | 0 | 101748 |
| Upper | 60134 | 14344 | 2077517 | 43259 | 2199535 |
| Lower | 5764 | 499 | 58615 | 835605 | 992481 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 459781 | 365575 | 125574 | 1132078234 |
| Face_local | 151844 | 113372 | 13146 | 140144 |
| Upper | 2077517 | 2317272 | 224983 | 1547116568 |
| Lower | 835605 | 1057359 | 118343 | 1390067003 |

## Confusion matrix ของ CPI_ConvNeXtB_CIHP

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 652399 | 10840 | 1250 | 0 | 160867 |
| Face_local | 9627 | 202778 | 2517 | 0 | 50294 |
| Upper | 37830 | 30248 | 3046434 | 24995 | 1255282 |
| Lower | 5383 | 8373 | 54056 | 1279508 | 545644 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 652399 | 172957 | 145147 | 1132058661 |
| Face_local | 202778 | 62438 | 14806 | 138484 |
| Upper | 3046434 | 1348355 | 334374 | 1547007177 |
| Lower | 1279508 | 613456 | 152040 | 1390033306 |

## Confusion matrix ของ Sapiens2_0.4b

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 679898 | 6637 | 6006 | 0 | 132815 |
| Face_local | 11733 | 221049 | 3120 | 0 | 29314 |
| Upper | 53854 | 19718 | 3605006 | 21030 | 695181 |
| Lower | 25780 | 4250 | 125765 | 1469877 | 267292 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 679898 | 145458 | 253627 | 1131950181 |
| Face_local | 221049 | 44167 | 26093 | 127197 |
| Upper | 3605006 | 789783 | 741620 | 1546599931 |
| Lower | 1469877 | 423087 | 412390 | 1389772956 |

## Confusion matrix ของ Sapiens2_0.8b

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 645577 | 9597 | 2705 | 231 | 167246 |
| Face_local | 7217 | 224238 | 1263 | 0 | 32498 |
| Upper | 49921 | 22842 | 3682770 | 26876 | 612380 |
| Lower | 9613 | 4695 | 84610 | 1505577 | 288469 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 645577 | 179779 | 179240 | 1132024568 |
| Face_local | 224238 | 40978 | 28248 | 125042 |
| Upper | 3682770 | 712019 | 592824 | 1546748727 |
| Lower | 1505577 | 387387 | 374356 | 1389810990 |

## Confusion matrix ของ Sapiens2_1b

Cross-class matrix ใช้เฉพาะ GT positive pixels ของ person-part แถวเป็น GT คอลัมน์เป็น prediction โดย OTHER รวม native background และคลาสนอกสี่เป้าหมาย

| GT / Prediction | Hair | Face-local | Upper | Lower | OTHER |
| --- | --- | --- | --- | --- | --- |
| Hair | 677888 | 9097 | 3773 | 0 | 134598 |
| Face_local | 9564 | 226294 | 2397 | 1 | 26960 |
| Upper | 10528 | 6337 | 3672766 | 48216 | 656942 |
| Lower | 1763 | 1377 | 57633 | 1579864 | 252327 |

Binary matrix แต่ละคลาสมีรูป [[TP,FN],[FP,TN]] บน domain ของ person-part จึงรวมพิกเซลที่ทายเกินขอบ GT ด้วย หน่วยพิกเซลสะสมแยกตามงานรายส่วน

| Target | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Hair | 677888 | 147468 | 136304 | 1132067504 |
| Face_local | 226294 | 38922 | 27205 | 126085 |
| Upper | 3672766 | 722023 | 578158 | 1546763393 |
| Lower | 1579864 | 313100 | 416421 | 1389768925 |

## การพบคนที่มี GT

Recall อ้างเฉพาะคนที่ annotate ส่วน unmatched predictions ยังรวมคนจริงที่ไม่มี GT ได้ full-scene person Precision/F1/AP เป็น null เพราะยังไม่มี negative domain ที่รับรองครบ

| Model | Bbox IoU | TP | FN | Unmatched | Ignored | Annotated-person recall % |
| --- | --- | --- | --- | --- | --- | --- |
| QANet_CIHP_HRNetW48_512x384 | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| QANet_CIHP_HRNetW48_512x384 | 0.75 | 561 | 208 | 176 | 26 | 72.952 |
| M2FP_R101_CIHP | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| M2FP_R101_CIHP | 0.75 | 561 | 208 | 176 | 26 | 72.952 |
| D2FP_R101_CIHP | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| D2FP_R101_CIHP | 0.75 | 561 | 208 | 176 | 26 | 72.952 |
| RPR_CIHP | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| RPR_CIHP | 0.75 | 561 | 208 | 176 | 26 | 72.952 |
| CPI_R50_CIHP | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| CPI_R50_CIHP | 0.75 | 561 | 208 | 176 | 26 | 72.952 |
| CPI_ConvNeXtB_CIHP | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| CPI_ConvNeXtB_CIHP | 0.75 | 561 | 208 | 176 | 26 | 72.952 |
| Sapiens2_0.4b | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| Sapiens2_0.4b | 0.75 | 561 | 208 | 176 | 26 | 72.952 |
| Sapiens2_0.8b | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| Sapiens2_0.8b | 0.75 | 561 | 208 | 176 | 26 | 72.952 |
| Sapiens2_1b | 0.5 | 624 | 145 | 113 | 26 | 81.144 |
| Sapiens2_1b | 0.75 | 561 | 208 | 176 | 26 | 72.952 |

## เวลาประมวลผลและ Peak VRAM

ตารางแรกด้านล่างเป็นเวลาจาก quality collection คอลัมน์ Clean rounds ระบุความครบของรอบจับเวลาแยก ไม่ได้หมายความว่าค่าเวลาในแถวเป็นค่าเฉลี่ยสามรอบ ดู dedicated timing จริงในหัวข้อ [เทียบกับ CCTV90](#cctv90-comparison)

Adapter intervals ครอบคลุม native classes ทั้งหมด Mean s/image บวกงาน parsing ของ ROI ในแต่ละภาพ ส่วน application-from-cached-inputs รวม source decode, cache/crop reads, ownership และ packet I/O ไม่มี new YOLO inference ใน interval นี้ Peak VRAM เป็น CUDA allocated/reserved ของ process คุณภาพและ clean timing rounds แสดงแยกกันตาม resource checks

| Model | Mean ms/ROI | Mean s/image | Total parsing s | Complete image timers | Peak allocated MiB | Peak reserved MiB | Clean rounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QANet_CIHP_HRNetW48_512x384 | 102.999 | 0.131 | 78.588 | 600/600 | 346.626 | 528.000 | 3/3 |
| M2FP_R101_CIHP | 255.005 | 0.324 | 194.569 | 600/600 | 1616.263 | 3804.000 | 3/3 |
| D2FP_R101_CIHP | 732.038 | 0.931 | 558.545 | 600/600 | 1697.649 | 2962.000 | 3/3 |
| RPR_CIHP | 251.848 | 0.320 | 192.160 | 600/600 | 698.982 | 1276.000 | 3/3 |
| CPI_R50_CIHP | 3168.402 | 4.029 | 2417.491 | 600/600 | 12505.118 | 13508.000 | 3/3 |
| CPI_ConvNeXtB_CIHP | 556.078 | 0.707 | 424.288 | 600/600 | 10365.286 | 13386.000 | 3/3 |
| Sapiens2_0.4b | 1379.845 | 1.755 | 1052.822 | 600/600 | 2158.113 | 2586.000 | 3/3 |
| Sapiens2_0.8b | 2887.896 | 3.672 | 2203.464 | 600/600 | 3798.549 | 4346.000 | 3/3 |
| Sapiens2_1b | 4068.984 | 5.174 | 3104.635 | 600/600 | 6222.808 | 6938.000 | 3/3 |

## ขอบเขตและข้อจำกัด

Face candidate ของ Sapiens รวม Face_Neck และ facial subparts คะแนน Face-local วัดภายใน GT Face box Upper/Lower ใช้ garment unions ที่ประกาศ ส่วน Dress/Apparel เป็น OTHER GT conflicts ใช้ shared VOID และเปิดเผยส่วนที่ถูกยกเว้นใน GT bindings ข้อมูลมีการใช้ helper-assisted annotation และคนที่เลือก annotate จึงต้องอ้างขอบเขตนี้เมื่อสรุปผล

Timing เป็น scoped adapter/application diagnostics ตาม API และสภาพ GPU ที่บันทึก รายงานนี้มี provenance ของ config, source, dataset, checkpoints และ run records กำกับ ประวัติ failures ของรอบนี้คงอยู่ใน immutable attempts และ completion records

Sapiens2 1B quality timing รวม adapter intervals ของ accepted ROI observations จากสอง segments เวลาของเฟรมที่ถูกขัดจังหวะด้าน application/I-O ให้ null ส่วน dedicated timing ใช้ process ใหม่สามรอบ raw interruption และ storage-scanner failure เก็บใน parent records

## แหล่งผลและประวัติการปิดรอบ

Canonical scores ใช้ `METRICS.json` ที่เชื่อมผ่าน `RESULT_INDEX.json` ของ run `cctv600-1b-continuation-20261118T032146603433Z` ภายใต้ protocol SHA256 `b409043aca666cbfeb14700252faa0795a8d920b7f971048855bb864c186c13b` ผลแปดโมเดลอ้าง run `cctv600-fresh-20261117T093921424617Z` ที่ตรวจรับแล้ว ส่วน 1B รวม accepted parent 737 ROI กับ continuation 26 ROI ครบ 763 งาน ไม่มีการรันแปดโมเดลซ้ำเพื่อสรุปตาราง

ผล final failed/invalid เป็นศูนย์หมายถึงผลที่ตรวจรับครบ ไม่ได้หมายความว่าไม่มีประวัติขัดจังหวะ: เก็บทั้ง Ctrl+C ของ attempt แรก และ storage-inspection failure ของ 1B ไว้ การหยุด 1B ครั้งหลังเกิดจาก controller ตรวจพื้นที่ระหว่าง atomic rename ไม่ใช่หลักฐาน OOM คะแนนอ่านจาก canonical metrics; ช่อง primary ที่เป็น null ใน historical collector completion ไม่ใช่แหล่งคะแนนหลัง scoring

ตรวจปิดด้วย membership/ROI completion และ audit ที่มีอยู่ ตรวจ TP+FN=GT support, union=TP+FP+FN และสถานะ clean timing ทั้ง 27 รอบโดยไม่ inference หรือ full rescore ซ้ำ ผลภาพส่วนตัวอยู่ที่ `outputs/cctv600-1b-continuation-20261118T032146603433Z/gallery/index.html` บน server และไม่เผยแพร่ภาพ dataset ใน Git

<a id="cctv90-comparison"></a>

## ทำไมคะแนนและเวลาต่างจาก CCTV90

ตรวจ canonical results, frozen configurations/source bindings, manifests และ detector caches ที่มีอยู่ โดยไม่มี model/YOLO inference หรือ full rescore ใหม่ ใช้ CCTV90 shared-reference ฉบับ 9 พฤศจิกายน 2026 ที่ประเมิน Face-local 35 ส่วน ไม่ใช้ตาราง historical Face-domain 21 ส่วนปะปน

### สิ่งที่ตรงกัน และสิ่งที่เปลี่ยน

| ประเด็น | CCTV90 ล่าสุด | CCTV600 รอบนี้ | ผลต่อการตีความ |
| --- | --- | --- | --- |
| Parser checkpoints / native configs / source commits | 9 checkpoints เดิม | hash bindings ตรงกันครบ 9 โมเดล | ไม่ได้เปลี่ยน weights เป็นรุ่นอื่น |
| Crop adapters / geometry helpers | Controlled YOLO person ROI | source hash bindings ตรงกัน 11 ไฟล์ที่ใช้ร่วมกัน | preprocessing/letterbox/map-back หลักคงเดิม ไม่ใช่ Sapiens full-frame รุ่นแรก |
| Input / ownership | RGB crop ไม่ระบายพื้นหลัง; หลังทำนายใช้ YOLO mask จำกัดเจ้าของ | แนวทางเดียวกัน, context 10%, FP32/batch1/no AMP/TF32/extra TTA | รูปแบบเดียวกันแต่ detector cache และ crop bytes ไม่ได้เหมือนกันข้ามงาน |
| สูตรคะแนน | Mean-part IoU, Face-local, mIoU3 และ Macro4-local | ใช้ฟังก์ชัน counts/summarize ร่วมกัน | สูตรร่วมกันไม่ได้ทำให้คนและขอบ GT เป็นชุดเดียวกัน |
| จับคู่คนกับ GT | mask IoU, maximum cardinality แล้ว maximum sum IoU | bbox IoU, greedy ตาม confidence/geometry | เป็นคนละวิธีจับคู่ แม้ threshold เท่ากับ 0.50 |
| GT person / Hair / Face / Upper / Lower | 127 / 69 / 35 / 120 / 102 | 769 / 544 / 384 / 745 / 668 | เปลี่ยนกลุ่มคนและจำนวนส่วนในตัวหาร |
| ขนาดคน / VOID | GT person masks และ overlap VOID ตาม shared-reference | bbox ด้านยาว ≥50 px; explicit ignore/คนเล็ก/class-owner conflicts เป็น shared VOID | valid domain และ eligibility ต่างกันตาม GT contract |
| Person Precision/F1/AP ทั้งฉาก | มี historical localization counts ตาม GT reference | เป็น null; unmatched detections เป็น diagnostics | ยังมีคนไม่ annotate จึงรับรอง global negative ทั้งฉากไม่ได้ |
| จับเวลา | shared GPU, หลายกลุ่ม timer A/B/C | quality adapter timer พร้อม stage hooks; dedicated timing แยก 3 รอบ | ห้ามตีความเวลาต่างกันเป็นความเร็วโมเดลที่เปลี่ยนโดยตรง |

ตรวจ binding ใน configs ที่เก็บไว้ ไม่ bulk rehash weights หรืออ้าง output equivalence ทั้งประชากรจากชื่อรุ่นเดียวกัน การแก้ orchestration/empty-output/resource guard และ timer มีบันทึกแยกจาก learned model ใน README

### คะแนนเปลี่ยนเท่าไร

ทุกค่าเป็น Macro4-local จากค่าเต็มก่อนปัดเศษ ผลต่างคือ CCTV600 ลบ CCTV90 หน่วยจุดเปอร์เซ็นต์ แต่ละแถวเป็นผลคนละ dataset/GT/matcher จึงไม่ใช่การทดสอบผลของจำนวนภาพอย่างเดียว

| Model | CCTV90 Macro4-local % | CCTV600 Macro4-local % | ผลต่าง (จุดเปอร์เซ็นต์) |
| --- | --- | --- | --- |
| QANet | 60.397 | 57.517 | -2.880 |
| M2FP | 47.569 | 42.066 | -5.503 |
| D2FP | 45.902 | 38.369 | -7.533 |
| RP-R-CNN | 44.520 | 36.944 | -7.576 |
| CPI R50 | 39.755 | 33.652 | -6.103 |
| CPI ConvNeXt-B | 53.438 | 52.057 | -1.382 |
| Sapiens2 0.4B | 59.231 | 52.233 | -6.997 |
| Sapiens2 0.8B | 58.865 | 53.293 | -5.572 |
| Sapiens2 1B | 63.729 | 59.184 | -4.545 |

Sapiens2 1B ยังได้ Macro4-local สูงสุดทั้งสองชุด Hair และ Face-local สูงสุดทั้งสองชุดเช่นกัน Upper เป็น QANet ทั้งสองชุด ส่วน Lower เปลี่ยนจาก Sapiens2 1B เป็น QANet เมื่อใช้ Mean-part IoU ของ CCTV600 การจัดอันดับ pooled IoU เป็นอีก aggregation และต้องอ่านแยก

Hair มี GT support เฉลี่ย 3,544.1 → 1,517.2 พิกเซลต่อส่วน และ Face 1,516.3 → 690.7 พิกเซลต่อส่วน (CCTV90 → CCTV600) เป็นขนาด mask ที่สังเกตได้ ไม่ใช่การวัดระยะจากกล้องโดยตรง ส่วนเล็กอาจได้รับผลจากขอบที่คลาดไม่กี่พิกเซลมากขึ้น แต่ยังไม่ได้ทดลองแยกว่าสาเหตุนี้ลดคะแนนกี่จุด จึงไม่สรุปว่า CCTV600 ยากกว่าทุกด้าน หรือคะแนนลดจากขนาดเพียงอย่างเดียว

### ตรวจ input จริงบนภาพซ้ำ

ภาพ SHA256 ของไฟล์ RGB ตรงกัน 24 ภาพ: 22 ภาพมี ROI และ 2 ภาพไม่มี ROI ทั้งสองรอบ จำนวน detection เท่ากันทุกภาพที่ซ้ำ แต่ bbox/mask arrays ของ 22 ภาพที่มี ROI ไม่ตรงกันทุกไบต์ ชุด crop coordinates พร้อม PNG hashes ตรงกัน 7/22 ภาพที่มี ROI (รวมภาพว่างเป็น 9/24) มี 15 ภาพที่ crop set ต่างกันจริง

จับคู่ bbox ระหว่าง detector caches ด้วย maximum-sum bbox IoU เฉพาะเพื่อวัดความต่างของ input บนภาพซ้ำได้ 33 ROI: median bbox IoU 0.993412 (ต่ำสุด 0.849947), median mask IoU 0.993879 (ต่ำสุด 0.900925), median ของพิกัดที่ขยับมากที่สุดต่อกรอบ 0.3254 px และสูงสุด 2.9672 px หลายกรอบจึงต่างเพียงเล็กน้อย แม้ hash ไม่ตรง ห้ามใช้ hash mismatch อย่างเดียวอ้างว่า detector เสียหรือเป็นสาเหตุหลักของคะแนนที่ลด

ทั้งสอง cache ใช้ YOLO26x-Seg checkpoint hash `92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3` ขั้นสร้าง cache ต่างกัน: รอบ 90 เรียก model.predict ที่ conf 0.25/max_det 200 และไม่ได้ระบุ rect ใน call; รอบ 600 ใช้ SegmentationPredictor โดยตรง ระบุ rect=False/conf floor 0.001/max_det 1000 แล้วเลือก confidence ≥0.25 สำหรับ parser โค้ด Ultralytics ที่ตรวจใน environment ปัจจุบันมี method default rect=True ของ model.predict แต่ config เก่าไม่เก็บ effective rect แยก จึงไม่ใช้ default ปัจจุบันเพียงอย่างเดียวยืนยัน runtime ทุกค่าของรอบเก่า

ความต่าง cache/crops และ ownership ยืนยันได้จาก saved artifacts ส่วนการระบุว่ามาจาก rect, precision ภายใน detector, API หรือข้ออื่นเป็นสัดส่วนเท่าไรยังไม่ได้ทดสอบแยก เงื่อนไขข้ามสองงานจึงไม่ได้เหมือนกันทุกประการ ภายใน CCTV600 ทั้งเก้าระบบยังใช้ detector cache, crop population, matching และ GT domain เดียวกัน

### เวลา: ต่อ ROI กับต่อภาพตอบคนละคำถาม

CCTV90 มี 130 ROI/90 ภาพ = 1.4444 ROI ต่อภาพ; CCTV600 มี 763 ROI/600 ภาพ = 1.2717 ROI ต่อภาพ ภาพไม่มี parser call เพิ่มจาก 8/90 เป็น 112/600 ภาพ หมายถึง YOLO ไม่สร้าง ROI ตาม threshold ไม่ใช่การรับรองว่าไม่มีคนจริง

เมื่อเวลาต่อ ROI ใกล้เดิม เวลาต่อภาพจึงลดได้เพราะงาน ROI เฉลี่ยน้อยลง ทั้งสองชุดหารผลรวมเวลางาน ROI ด้วยจำนวนภาพทั้งชุด รวมภาพศูนย์ ROI จำนวนภาพเพิ่มทำให้เวลารวมเพิ่มด้วย แต่ละ parser call ครอบ native classes ทั้งหมดก่อน map เป็นสี่ targets ไม่ใช่เวลาสี่คลาสอย่างเดียว

| Model | CCTV90 บัญชี ms/ROI | CCTV600 quality ms/ROI | CCTV90 บัญชี s/ภาพ | CCTV600 quality s/ภาพ |
| --- | --- | --- | --- | --- |
| QANet | 76.325 | 102.999 | 0.110 | 0.131 |
| M2FP | 220.187 | 255.005 | 0.318 | 0.324 |
| D2FP | 694.885 | 732.038 | 1.004 | 0.931 |
| RP-R-CNN | 226.784 | 251.848 | 0.328 | 0.320 |
| CPI R50 | 2994.517 | 3168.402 | 4.325 | 4.029 |
| CPI ConvNeXt-B | 490.608 | 556.078 | 0.709 | 0.707 |
| Sapiens2 0.4B | 1384.706 | 1379.845 | 2.000 | 1.755 |
| Sapiens2 0.8B | 2881.421 | 2887.896 | 4.162 | 3.672 |
| Sapiens2 1B | 4096.048 | 4068.984 | 5.917 | 5.174 |

Sapiens2 1B ต่อ ROI ประมาณ 4.096 → 4.069 วินาที ต่าง -0.66% แต่ต่อภาพ 5.917 → 5.174 วินาที ต่าง -12.54% จึงไม่ใช่หลักฐานว่า parser เร็วขึ้น 12.54% ต่อคน เวลารวมที่บันทึกทั้งชุดเป็น 532.486 → 3,104.635 วินาที เพิ่มตาม workload ที่มากขึ้น

ตารางนี้เป็นบัญชีเวลาเดิมเทียบ quality adapter timer ไม่ใช่ clean-speed comparison: CCTV90 มี A=adapter timer, B=continuation event interval, C=original forward interval โดย C ไม่ครอบ preprocessing/map-back แบบ A รอบ 600 เพิ่ม synchronize/hooks ที่ขอบ module เพื่อแยก stage ซึ่งเพิ่ม measurement overhead ขนาด crop/padding/native instance proposals/file I/O/สภาพ GPU ต่างกันด้วย ยังไม่ได้ทดลองแยกสัดส่วนผล จึงไม่หักเวลาเองหรือสรุปว่า QANet ช้าลงเพราะ weights เปลี่ยน

CCTV600 มี dedicated timing แยกจาก quality run: 100 ภาพ/124 ROI ต่อรอบ, warmup 10 calls ต่อ process, ไม่มี external compute PID ตาม guards, ครบ 3 รอบทุกโมเดล ตารางนี้เฉลี่ยเวลารายรอบที่มีจำนวนงานเท่ากัน ช่วงต่ำสุด–สูงสุดเป็นค่าของ mean ms/ROI รายรอบ ไม่ใช่ช่วงความเชื่อมั่นทางสถิติ

| Model | Dedicated mean ms/ROI (3 รอบ) | ช่วง mean ms/ROI รายรอบ | Dedicated mean parsing s/ภาพ |
| --- | --- | --- | --- |
| QANet | 102.939 | 102.102–103.761 | 0.128 |
| M2FP | 244.411 | 243.281–245.435 | 0.303 |
| D2FP | 729.973 | 720.468–737.514 | 0.905 |
| RP-R-CNN | 253.253 | 251.261–256.025 | 0.314 |
| CPI R50 | 3203.272 | 3193.997–3211.148 | 3.972 |
| CPI ConvNeXt-B | 568.085 | 566.255–571.115 | 0.704 |
| Sapiens2 0.4B | 1432.402 | 1431.391–1433.996 | 1.776 |
| Sapiens2 0.8B | 2950.805 | 2948.902–2952.723 | 3.659 |
| Sapiens2 1B | 4212.592 | 4196.541–4233.937 | 5.224 |

Stage hooks วัดขอบเขต adapter/module API ไม่ใช่ native kernel-only latency ที่เหมือนกันทุก architecture และไม่ได้แยกเวลาแต่ละคลาส ไม่มี YOLO inference ใหม่ใน adapter intervals จึงไม่ใช่ end-to-end YOLO+parser+color latency Peak allocated/reserved เป็น CUDA memory ของ process ไม่ใช่ค่าเดียวกับ nvidia-smi หรือขนาด weights

### ขอบเขตข้อสรุปและการตรวจต่อ

สรุปผลภายใน CCTV600 ตาม protocol ที่ประกาศ และใช้ CCTV90 เป็นหลักฐานอีก dataset หนึ่ง ห้ามรวมคะแนนเป็นชุดเดียวหรืออ้างว่าขยายจาก 90 เป็น 600 ภาพภายใต้ทุกเงื่อนไขเดิม การเทียบเชิงสาเหตุต้องควบคุมภาพ/GT/YOLO cache/crops/matcher ให้ตรงกันก่อน

หากต้องตรวจสาเหตุเพิ่ม เริ่ม saved-mask CPU diagnostic บนภาพซ้ำและ GT/cache binding ที่กำหนดตายตัว เพื่อแยกผล matching/GT ก่อน เฉพาะความต่าง input ที่แก้ย้อนหลังจาก hard masks ไม่ได้จึงพิจารณา bounded rerun แยก canonical benchmark ห้ามใช้ GT หรือคะแนนเลือก output ที่ดีกว่า รอบนี้ยังไม่เปิด diagnostic ดังกล่าว และยังไม่มีเหตุผลให้ลบหรือรันใหม่ทั้งหมดจากคะแนนต่างกันอย่างเดียว

แหล่งตรวจ: [CCTV90 ฉบับที่ใช้เทียบ](https://github.com/folklazy/Human_Parsing_Study/blob/29f7030/CCTV_Human_Parsing_Benchmark/reports/CONTROLLED_ROI_FOUR_TARGET_RESULTS_TH.md), [CCTV90 canonical index](https://github.com/folklazy/Human_Parsing_Study/blob/29f7030/Human_Parsing_Benchmark/reports/SHARED_REFERENCE_RESULT_INDEX.json), [shared scorer](https://github.com/folklazy/Human_Parsing_Study/blob/29f7030/Human_Parsing_Benchmark/src/common_view/reference_parts.py), [CCTV600 matcher/GT export](../src/reference_parts600.py), [current scorer](../src/scoped_metrics.py), [current timer](../src/worker.py), [QANet timer correction](../src/correct_qanet.py) การตรวจภาพซ้ำอ่าน cache บน CPU และไม่สร้าง model outputs ใหม่
