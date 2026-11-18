"""Independent numeric fixtures for the publication contract; no dataset/model IO."""
import copy
import json
import tempfile
from pathlib import Path
import unittest

import publication as pub


def fixture():
    # Literal 10-pixel task: TP=2, FP=1, FN=2, TN=5.
    # IoU=2/5, P=2/3, R=1/2, F1=4/7. One part per class.
    index = dict(protocol_sha256='fixture', GT=dict(images=600, people=1,
        parts={t: 1 for t in pub.TARGETS}, support={t: 4 for t in pub.TARGETS}),
        native_mapping={
            'CIHP_GT_AND_NATIVE_PRED': {str(i): 0 for i in range(20)},
            'SAPIENS2_DOME29_PRED': {str(i): 0 for i in range(29)}}, models=[])
    for mid in pub.ROSTER:
        targets = {t: dict(GT_parts=1, TP=2, FP=1, FN=2, TN=5, GT_pixels=4,
            prediction_pixels=3, union=5, mean_part_IoU=.4, pooled_IoU=.4,
            precision=2/3, recall=.5, F1_Dice=4/7, entirely_missed_GT_parts=0,
            binary_confusion_target_other=[[2,2],[1,5]]) for t in pub.TARGETS}
        cross = [[2 if j == i else 2 if j == 4 else 0 for j in range(5)] for i in range(4)]
        time = dict(complete=True, images_with_complete_timer=600, observed_ROIs=763,
            mean_ms_per_ROI=10, mean_s_per_image=7.63/600, total_s=7.63,
            observed_totals_ms=dict(adapter_ms=7630, pre_module_ms=1526,
                module_call_ms=3815, inter_module_ms=763, post_module_ms=1526))
        dedicated = dict(time, images_with_complete_timer=100, observed_ROIs=124,
            mean_s_per_image=.0124, total_s=1.24, clean_complete=True,
            observed_totals_ms=dict(adapter_ms=1240, pre_module_ms=248,
                module_call_ms=620, inter_module_ms=124, post_module_ms=248))
        membership = lambda n: dict(strict_complete=True, expected=n, completed=n,
                                   missing=0, duplicate=0, unexpected=0)
        index['models'].append(dict(model_id=mid, membership=dict(images=membership(600), ROI=membership(763)),
            result=dict(protocol_sha256='fixture', strict_complete=True, completed=600,
                expected=600, missing=0, duplicate=0, targets=targets,
                Macro4_local=.4, primary_mIoU3_part_mean=.4,
                cross_class_confusion=dict(counts=cross),
                localization={'0.5': dict(TP=1, FN=0, GT=1, unmatched_predictions=762,
                    ignored_predictions=0, all_predictions=763, precision=None, F1=None, AP=None)}),
            timing=dict(quality=time, dedicated_rounds={str(i): copy.deepcopy(dedicated) for i in (1,2,3)})))
    return index


class PublicationAcceptance(unittest.TestCase):
    def test_literal_counts_and_units(self):
        pub.validate(fixture())

    def test_FN_decomposition_from_independent_GT_positive_fixture(self):
        m = fixture()['models'][0]
        # Hair row: 2 correct, 1 Face error, 1 OTHER error; FN=2.
        m['result']['cross_class_confusion']['counts'][0] = [2,1,0,0,1]
        t = m['result']['targets']['Hair']
        t.update(GT_parts=2, entirely_missed_GT_parts=1, unmatched_GT_parts=1)
        for other in pub.TARGETS[1:]:
            m['result']['targets'][other].update(unmatched_GT_parts=0)
        d = pub.error_split(m)['Hair']
        self.assertEqual(d, dict(wrong_target_pixels=1, OTHER_pixels=1,
            OTHER_fraction_of_FN=.5, matched_person_zero_overlap_parts=0))

    def test_null_precision_is_not_zero_or_missing_result(self):
        index = fixture()
        r = index['models'][0]['result']
        r['targets']['Hair'].update(TP=0, FP=0, FN=4, GT_pixels=4, prediction_pixels=0,
            union=4, mean_part_IoU=0., pooled_IoU=0., precision=None, recall=0., F1_Dice=0.,
            entirely_missed_GT_parts=1, binary_confusion_target_other=[[0,4],[0,5]])
        r['cross_class_confusion']['counts'][0] = [0,0,0,0,4]
        r.update(Macro4_local=.3, primary_mIoU3_part_mean=4/15)
        pub.validate(index)
        self.assertEqual(pub.pct(None), '—')
        self.assertEqual(pub.pct(0), '0.000')
        self.assertIsNone(json.loads(json.dumps(index))['models'][0]['result']['targets']['Hair']['precision'])

    def test_incomplete_or_mismatched_confusion_is_rejected(self):
        for field in ('membership', 'cross', 'FP'):
            index = fixture()
            m = index['models'][0]
            if field == 'membership':
                m['membership']['images']['completed'] = 599
            elif field == 'cross':
                m['result']['cross_class_confusion']['counts'][0] = [1,0,0,0,3]
            else:
                m['result']['targets']['Hair']['FP'] = 2
            with self.assertRaises(AssertionError):
                pub.validate(index)

    def test_unknown_mapping_and_private_paths_are_rejected(self):
        for private in (False, True):
            index = fixture()
            if private:
                index['leaked'] = '/home/someone/private.png'
            else:
                del index['native_mapping']['SAPIENS2_DOME29_PRED']['28']
            with self.assertRaises(AssertionError):
                pub.validate(index)

    def test_ownership_is_not_added_again_to_adapter(self):
        index = fixture()
        index['models'][0]['timing']['quality']['observed_totals_ms']['ownership_ms'] = 999
        pub.validate(index)  # Still 7.63 adapter seconds, not 8.629.

    def test_duplicate_JSON_rejected_without_writing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'fixture.json'
            p.write_text('{"completed":600,"completed":599}')
            before = p.read_bytes()
            with self.assertRaises(ValueError):
                pub.read(p)
            self.assertEqual(p.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
