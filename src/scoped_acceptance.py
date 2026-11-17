"""Independent CPU fixtures for the fresh annotated-person-part run."""
import json
from pathlib import Path
import tempfile

import numpy as np

from common import atomic, install_shared_paths, load, npz, sha, WORKSPACE
import reference_parts600 as reference
import scoped_metrics as scoped


def matching(found=True, predictions=1):
    return {t: dict(by_GT={'7': 0} if found else {}, TP=int(found), FP=predictions - int(found),
                    FN=int(not found), GT=1, predictions=predictions, all_predictions=predictions,
                    ignored_predictions=0, ambiguous=0, identical_GT_geometry_groups=0)
            for t in ('0.5', '0.75')}


def acceptance():
    install_shared_paths()
    reference.acceptance()
    spec = load(WORKSPACE / 'Human_Parsing_Benchmark/protocol/CIHP_SAPIENS_COMMON_VIEW_REVISION2.json')
    masks = np.zeros((4, 2, 5), bool)
    for index in range(4):
        masks[index, 0, index] = True
    gt = dict(valid=np.ones((2, 5), bool), part_masks=masks,
              part_owner_ids=np.full(4, 7, np.int32), part_target_ids=np.array([2, 1, 3, 4], np.uint8))
    prediction = np.zeros((1, 2, 5), np.uint8)
    prediction[0, 0, :4] = [13, 2, 6, 9]
    row = dict(image_id='fixture', **scoped.score_frame(gt, matching(), prediction, 'CIHP_GT_AND_NATIVE_PRED', spec))
    result = scoped.aggregate([row], ['fixture'])
    assert result['cross_class_confusion']['counts'] == [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]
    assert result['targets']['Hair']['binary_confusion_target_other'] == [[0, 1], [1, 8]]
    assert result['targets']['Face_local']['binary_confusion_target_other'] == [[0, 1], [0, 0]]
    assert result['Macro4_local'] == .25
    assert abs(result['primary_mIoU3_part_mean'] - 1 / 3) < 1e-12
    assert [result['targets'][t]['entirely_missed_GT_parts'] for t in scoped.TARGETS] == [1, 1, 1, 0]
    assert result['targets']['Face_local']['precision'] is None
    assert result['targets']['Face_local']['F1_Dice'] == 0
    assert result['localization']['0.5']['precision'] is None
    assert result['localization']['0.5']['F1'] is None
    assert result['localization']['0.5']['AP'] is None
    assert 'FP' not in result['localization']['0.5']

    partial = dict(image_id='fixture', **scoped.score_frame(gt, matching(predictions=2), prediction,
                                                         'CIHP_GT_AND_NATIVE_PRED', spec))
    diagnostics = scoped.aggregate([partial], ['fixture'])['localization']['0.5']
    assert diagnostics['annotated_person_recall'] == 1 and diagnostics['unmatched_predictions'] == 1
    missed = dict(image_id='fixture', **scoped.score_frame(gt, matching(False, 0), np.empty((0, 2, 5), np.uint8),
                                                        'CIHP_GT_AND_NATIVE_PRED', spec))
    value = scoped.aggregate([missed], ['fixture'])
    assert value['Macro4_local'] == 0
    assert all(value['targets'][t]['entirely_missed_GT_parts'] == 1 for t in scoped.TARGETS)
    assert value['cross_class_confusion']['counts'] == [[0, 0, 0, 0, 1]] * 4
    assert scoped.aggregate([row, row], ['fixture'])['Macro4_local'] is None
    assert scoped.aggregate([], ['fixture'])['Macro4_local'] is None
    void = dict(gt, valid=np.zeros_like(gt['valid']))
    excluded = dict(image_id='fixture', **scoped.score_frame(void, matching(), prediction, 'CIHP_GT_AND_NATIVE_PRED', spec))
    assert not excluded['parts'] and scoped.aggregate([excluded], ['fixture'])['Macro4_local'] is None
    for invalid in (np.full((1, 2, 5), 255, np.uint8), np.zeros((1, 5, 2), np.uint8)):
        try:
            scoped.score_frame(gt, matching(), invalid, 'CIHP_GT_AND_NATIVE_PRED', spec)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid native label/geometry accepted')

    from scoped_run import assemble, gt_frame, native_config
    document = dict(imageWidth=100, imageHeight=80, shapes=[
        dict(label='bbox', group_id=7, points=[[1, 1], [20, 51]]),
        dict(label='bbox', group_id=9, points=[[40, 1], [60, 50.9]]),
        dict(label='ignore', group_id=None, points=[[70, 1], [90, 30]])])
    rebuilt = gt_frame({}, document)
    assert [p['person_id'] for p in rebuilt['people']] == [7]
    assert [p['person_id'] for p in rebuilt['excluded_people']] == [9]
    assert len(rebuilt['ignore']) == 2
    assert scoped.PREDICTION_COLUMNS == ('Hair', 'Face_local', 'Upper', 'Lower', 'OTHER')

    from common_view.packets import write_packet, read_packet
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        packet = write_packet(root / 'packets', 'fixture', dict(person_semantic_masks=prediction),
                              dict(image_id='fixture'), {})
        before = sha(packet / 'prediction.npz')
        _, arrays = read_packet(packet)
        actual = dict(image_id='fixture', **scoped.score_frame(gt, matching(), arrays['person_semantic_masks'],
                                                             'CIHP_GT_AND_NATIVE_PRED', spec))
        assert scoped.aggregate([actual], ['fixture'])['cross_class_confusion'] == result['cross_class_confusion']
        assert sha(packet / 'prediction.npz') == before
        frames = [dict(image_id='fixture', jobs=[dict(job_id='roi')]), dict(image_id='empty', jobs=[])]
        timing = dict(adapter_ms=10, pre_module_ms=2, module_call_ms=6, post_module_ms=2,
                      peak_allocated_MiB=100, peak_reserved_MiB=200)
        values = dict(source_read_ms=1, cached_detector_read_ms=2, composition_projection_ms=3,
                      frame_packet_write_and_validation_ms=4, application_from_cached_inputs_ms=20)
        atomic(root / 'image_records/fixture.json', dict(records=[dict(job_id='roi', timing=timing)],
                                                        parsing_work_ms=10, **values))
        atomic(root / 'image_records/empty.json', dict(records=[], parsing_work_ms=0, **values))
        timer, images = scoped.runtime(root, frames)
        assert timer['mean_ms_per_ROI'] == 10 and timer['mean_s_per_image'] == .005 and timer['total_s'] == .01
        assert timer['peak_allocated_MiB'] == 100 and timer['peak_reserved_MiB'] == 200
        assert images[1]['parsing_work_s'] == 0
        (root / 'image_records/fixture.json').unlink()
        assert scoped.runtime(root, frames)[0]['mean_s_per_image'] is None
        atomic(root / 'CONFIG.json', dict(models={'synthetic': {}}))
        atomic(root / 'MANIFEST.json', dict(frames=frames, timing_ids=['fixture']))
        atomic(root / 'GT_BINDINGS.json', dict(GT_people=1, **{'1': 1, '2': 1, '3': 1, '4': 1}))
        atomic(root / 'RESULT_INDEX.json', dict(synthetic=dict(result, status='COMPLETE')))
        (root / 'models/synthetic').mkdir(parents=True)
        scoped.report(root)
        text = (root / 'BATCH_REPORT_TH.md').read_text()
        assert 'OTHER' in text and 'TP=0 parts' in text and 'Peak allocated MiB' in text and 'unmatched' in text
        assert json.loads((root / 'TIMING_SUMMARY.json').read_text())['synthetic']['quality']['mean_s_per_image'] is None

    # Exercise the real native-config -> packet -> assembly path. A worker
    # failure after the final packet must keep the canonical model incomplete.
    for failed in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'parent'
            mid = 'synthetic'
            native_protocol = dict(execution=dict(precision='FP32', batch_size=1), route=dict(name='fixed'))
            atomic(parent / 'CONFIG.json', dict(protocol=native_protocol))
            config = dict(parent_segment=str(parent), source_hashes={}, models={mid: dict(family='QANet', checkpoint_sha256='weight')},
                          input_membership_sha256='current-GT', GT_bindings_sha256='GT-bindings', GT_packet_bindings={'fixture': 'GT-mask'})
            atomic(root / 'CONFIG.json', config)
            job = dict(job_id='roi', crop_sha256='crop', crop_hw=[2, 5], crop_xyxy=[0, 0, 5, 2])
            frame = dict(image_id='fixture', image_sha256='image', height=2, width=5, jobs=[job], detector_sha256='detector')
            atomic(root / 'MANIFEST.json', dict(frames=[frame], expected_ids=['fixture'], ROI_jobs=1))
            atomic(root / 'LAUNCH.json', dict(launch_commit='fixture'))
            cache = dict(extents=np.ones((1, 2, 5), bool), scores=np.array([.9]),
                         boxes=np.array([[0, 0, 5, 2]]), ids=np.array([1]))
            npz(root / 'detector/fixture.npz', **cache)
            reconciled = dict(pending=['roi'], accepted={}, invalid=[], source=str(root / 'providers' / mid))
            cp = native_config(root, mid, reconciled)
            created = load(cp)
            assert created['input_membership_sha256'] == 'current-GT'
            assert created['GT_packet_bindings'] == {'fixture': 'GT-mask'}
            assert created['protocol']['execution'] == native_protocol['execution']
            assert created['protocol']['route'] == native_protocol['route']
            assert created['protocol']['GT']['role'] == 'GT_NOT_USED_FOR_INFERENCE'
            out = cp.parent / mid
            write_packet(out / 'roi_packets', 'roi', dict(native_crop_mask=np.zeros((2, 5), np.uint8)),
                         dict(model_id=mid, image_id='fixture', job_id='roi', checkpoint_sha256='weight',
                              crop_sha256='crop', config_sha256=sha(cp)), {})
            write_packet(out / 'packets', 'fixture', dict(person_semantic_masks=np.zeros((1, 2, 5), np.uint8),
                         person_extents=cache['extents'], person_scores=cache['scores'], bboxes_xyxy=cache['boxes'], person_ids=cache['ids']),
                         dict(model_id=mid, image_id='fixture', input_sha256='image', checkpoint_sha256='weight'), {})
            atomic(out / 'STATE.json', dict(status='FAILED' if failed else 'PREDICTIONS_COMPLETE',
                                            failed=int(failed), invalid=0, setup_failures=0))
            assembled = assemble(root, mid, reconciled, cp.parent)
            assert assembled['status'] == ('INCOMPLETE' if failed else 'PREDICTIONS_COMPLETE')
            assert assembled['parser_prediction_reuse'] == 0
            assert assembled['image_membership']['strict_complete']
            assert assembled['ROI_membership']['strict_complete']
    print('Scoped CPU acceptance PASS — literal counts, domains, mappings, partial-person nulls, missed parts, packets, timers and reporter', flush=True)
