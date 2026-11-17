"""Literal independent CPU expectations, including real packet/scorer/report path."""
import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common import PROJECT, WORKSPACE, archive, atomic, install_shared_paths, load, sha
from evaluation import aggregate, compose, confusion, derive_gt, metrics, project, score_frame

install_shared_paths()


class Acceptance(unittest.TestCase):
    def setUp(self):
        self.protocol = load(PROJECT / 'configs/PROTOCOL.json')
        self.mapping = self.protocol['mapping']['CIHP20']

    def test_total_mapping_contract(self):
        for name, count in (('CIHP20', 20), ('SAPIENS29', 29)):
            self.assertEqual(len(self.protocol['mapping'][name]), count)
        self.assertEqual(project(np.array([2, 13, 5, 7, 9, 12, 6], np.uint8), self.mapping).tolist(), [1, 2, 3, 3, 4, 4, 0])
        sapiens = self.protocol['mapping']['SAPIENS29']
        self.assertEqual(project(np.array([4, 3, 24, 28, 23, 13, 1], np.uint8), sapiens).tolist(), [1, 2, 2, 2, 3, 4, 0])
        for value in (np.array([255], np.uint8), np.array([20], np.uint8), np.array([-1]), np.array([1.5])):
            with self.assertRaises(ValueError):
                project(value, self.mapping)

    def test_literal_confusion_and_ratios(self):
        gt = np.array([[1, 1, 2, 255], [3, 3, 4, 4]], np.uint8)
        pred = np.array([[1, 2, 2, 3], [3, 0, 4, 4]], np.uint8)
        expected = [[0,0,0,0,0], [0,1,1,0,0], [0,0,1,0,0], [1,0,0,1,0], [0,0,0,0,2]]
        self.assertEqual(confusion(gt, pred).tolist(), expected)
        result = metrics(expected)
        self.assertEqual(result['IoU'][1:], [.5, .5, .5, 1.])
        self.assertEqual(result['AD4_mIoU4'], .625)
        self.assertAlmostEqual(result['C3_global'], 2/3)
        self.assertEqual(result['precision'][1:], [1., .5, 1., 1.])
        self.assertEqual(result['recall'][1:], [.5, 1., .5, 1.])
        self.assertAlmostEqual(result['F1_Dice'][1], 2/3)

    def test_union_and_other_before_overlap(self):
        native = np.array([[[5, 7]], [[6, 0]]], np.uint8)
        composed = compose(native, [.4, .9], [[0,0,2,1], [0,0,2,1]], [1,2], (1,2))
        self.assertEqual(composed.tolist(), [[6,7]])
        self.assertEqual(project(composed, self.mapping).tolist(), [[0,3]])
        merged = project(np.array([[5,7]], np.uint8), self.mapping)
        self.assertEqual(metrics(confusion(np.array([[3,3]], np.uint8), merged))['IoU'][3], 1.)

    def test_composition_ties_use_geometry(self):
        planes = np.array([[[2]], [[13]]], np.uint8)
        first = compose(planes, [.5,.5], [[1,0,2,1], [0,0,1,1]], [1,2], (1,1))
        second = compose(planes[::-1], [.5,.5], [[0,0,1,1], [1,0,2,1]], [20,10], (1,1))
        self.assertEqual(first.tolist(), [[13]])
        np.testing.assert_array_equal(first, second)

    def test_gt_unknown_conflict_and_ignore(self):
        def shape(label, points, owner=1):
            return dict(label=label, points=points, group_id=owner, shape_type='rectangle')
        annotation = dict(imageWidth=64, imageHeight=64, shapes=[
            shape('bbox', [[0,0],[63,63]]), shape('hair', [[1,1],[3,3]]),
            shape('face', [[2,2],[2,2]]), shape('ignore', [[3,3],[3,3]], None)])
        # A degenerate face shape is excluded, its one raster pixel remains VOID.
        gt, binding = derive_gt(annotation)
        self.assertEqual(binding['valid_pixels'], 7)
        self.assertEqual(int(gt['labels'][0,0]), 255)
        self.assertEqual(int(gt['owners'][2,2]), 0)
        self.assertEqual(binding['persons'][0]['support'], [7,0,0,0])
        annotation['shapes'][2]['points'] = [[2,2],[4,4]]
        gt, binding = derive_gt(annotation)
        self.assertEqual(binding['conflict_pixels'], 4)
        self.assertEqual(binding['valid_pixels'], 10)

    def test_under50_whole_box_void(self):
        annotation = dict(imageWidth=64,imageHeight=64,shapes=[
            dict(label='bbox',points=[[0,0],[63,63]],group_id=1,shape_type='rectangle'),
            dict(label='bbox',points=[[2,2],[4,4]],group_id=2,shape_type='rectangle'),
            dict(label='hair',points=[[1,1],[5,5]],group_id=1,shape_type='rectangle')])
        gt, binding = derive_gt(annotation)
        self.assertEqual(binding['valid_pixels'], 16)
        self.assertEqual(len(binding['persons']), 1)

    def test_packet_scorer_reporter_and_immutability(self):
        from common_view.packets import read_packet, write_packet
        from report import report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = dict(labels=np.array([[1,2,3,4]],np.uint8), owners=np.ones((1,4),np.int32))
            owned = np.array([[[2,13,5,9]]],np.uint8)
            cache = dict(extents=np.ones((1,1,4),bool), scores=np.array([.8]),
                         boxes=np.array([[0,0,4,1]]), ids=np.array([9]))
            packet = write_packet(root/'packets', 'fixture', dict(person_semantic_masks=owned),
                                  dict(image_id='fixture', config_sha256='literal-binding'), {})
            before = sha(packet/'prediction.npz')
            metadata, data = read_packet(packet, dict(config_sha256='literal-binding'))
            frame = dict(people=[dict(person_id=1,bbox=[0,0,4,1])], ignore=[])
            row = score_frame(gt,dict(valid_pixels=4),frame,data['person_semantic_masks'],cache,self.mapping)
            result = aggregate([dict(image_id='fixture',**row)], ['fixture'])
            self.assertEqual(result['primary']['AD4_mIoU4'], 1.)
            self.assertEqual(result['person']['0.5']['eligible_person_parts'], [1,1,1,1])
            self.assertEqual(result['person']['0.5']['Q_AD4_GT'], 1.)
            atomic(root/'CONFIG.json',dict(models={'fixture':{}}))
            atomic(root/'RESULT_INDEX.json',{'fixture':dict(model_id='fixture',status='COMPLETE_AD4_DIAGNOSTIC',**result)})
            report(root)
            self.assertIn('100.000', (root/'BATCH_REPORT_TH.md').read_text())
            self.assertEqual(before, sha(packet/'prediction.npz'))
            with self.assertRaises(FileExistsError):
                write_packet(root/'packets','fixture',{}, {}, {})
            with self.assertRaises(ValueError):
                read_packet(packet,dict(config_sha256='wrong-binding'))
            with (packet/'prediction.npz').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaises(ValueError):
                read_packet(packet)

    def test_missed_duplicate_wrong_owner_and_id_renaming(self):
        gt = dict(labels=np.array([[1,1]],np.uint8), owners=np.array([[1,2]],np.int32))
        frame = dict(people=[dict(person_id=1,bbox=[0,0,1,1]),dict(person_id=2,bbox=[1,0,2,1])], ignore=[])
        cache = dict(scores=np.array([.9,.8,.7]), boxes=np.array([[0,0,1,1],[1,0,2,1],[0,0,1,1]]),ids=np.array([1,2,3]))
        # Each parser assigned the neighbouring person's Hair. Duplicate is empty.
        owned = np.array([[[0,2]], [[2,0]], [[0,0]]],np.uint8)
        result = score_frame(gt,dict(valid_pixels=2),frame,owned,cache,self.mapping)
        self.assertEqual(result['person']['0.5']['tp'], 2)
        self.assertEqual(result['person']['0.5']['fp'], 1)
        self.assertEqual(result['person']['0.5']['part_IoU_sum'][0], 0.)
        self.assertEqual(metrics(result['histogram'])['IoU'][1], 1.)
        gt['owners'] = np.array([[88,3]],np.int32)
        frame['people'][0]['person_id'],frame['people'][1]['person_id'] = 88,3
        renamed = score_frame(gt,dict(valid_pixels=2),frame,owned,cache,self.mapping)
        self.assertEqual(result, renamed)
        empty_cache = dict(scores=np.array([]), boxes=np.empty((0,4)), ids=np.array([],int))
        missed = score_frame(gt,dict(valid_pixels=2),frame,np.empty((0,1,2),np.uint8),empty_cache,self.mapping)
        self.assertEqual(missed['person']['0.5']['fn'], 2)
        self.assertEqual(missed['person']['0.5']['part_IoU_sum'], [0.,0.,0.,0.])
        self.assertEqual(missed['person']['0.5']['eligible_person_parts'], [2,0,0,0])

    def test_gt_first_before_ignore(self):
        from cctv600.scoring import fixed
        frame = dict(people=[dict(person_id=1,bbox=[0,0,10,10])],ignore=[[0,0,20,20]])
        result = fixed(frame,[dict(confidence=.9,bbox_xyxy=[0,0,10,10]),dict(confidence=.8,bbox_xyxy=[10,0,20,10])])
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['ignored_predictions']), (1,0,0,1))

    def test_zero_union_unknown_empty_and_strict_null(self):
        result = metrics(np.zeros((5,5),np.int64))
        self.assertIsNone(result['AD4_mIoU4'])
        self.assertEqual(result['IoU'],[None]*5)
        result = aggregate([], ['missing'])
        self.assertIsNone(result['primary'])
        self.assertFalse(result['membership']['strict_complete'])
        self.assertEqual(aggregate([{'image_id':'a'},{'image_id':'a'}],['a'])['primary'],None)

    def test_geometry_rgb_asymmetric(self):
        from controlled_roi_core import place, letterbox_canvas
        from sapiens_input_variants import detector_crop
        rgb = np.zeros((7,11,3),np.uint8)
        rgb[1,8] = [249,17,33]
        extent = np.zeros((7,11),bool)
        extent[1,8] = True
        crop, box = detector_crop(rgb,[7,1,9,3],extent,.1)
        self.assertEqual(crop[1-box[1],8-box[0]].tolist(),[249,17,33])
        native = np.zeros(crop.shape[:2],np.uint8)
        native[1-box[1],8-box[0]] = 13
        raw, owned = place(native,box,extent,20)
        self.assertEqual(int(owned[1,8]),13)
        self.assertEqual(int(np.count_nonzero(owned)),1)
        canvas, geometry = letterbox_canvas(rgb,(13,17),np.array([100.,110.,120.]))
        self.assertEqual(canvas.shape,(13,17,3))
        self.assertEqual(geometry['original_hw'],[7,11])
        self.assertEqual(geometry['input_hw'],[13,17])

    def test_four_accepted_packet_contracts_read_only(self):
        from common_view.packets import read_packet
        parent = WORKSPACE / 'CCTV_Human_Parsing_Benchmark/outputs/cctv-controlled-roi-20261101T115653Z'
        for mid in ('QANet_CIHP_HRNetW48_512x384', 'M2FP_R101_CIHP', 'D2FP_R101_CIHP', 'RPR_CIHP'):
            packets = sorted((parent/mid/'packets').glob('*/packet.json'))
            self.assertTrue(packets, mid)
            path = packets[0].parent
            before = sha(path/'packet.json'), sha(path/'prediction.npz')
            metadata, data = read_packet(path)
            planes, extents = data['person_semantic_masks'], data['person_extents']
            self.assertEqual(planes.shape, extents.shape)
            self.assertEqual(planes.shape[-2:], tuple(metadata['original_wh'][::-1]))
            self.assertFalse(np.any(planes[~extents]))
            projected = project(planes, self.mapping)
            self.assertEqual(projected.shape, planes.shape)
            self.assertEqual(before, (sha(path/'packet.json'), sha(path/'prediction.npz')))

    def test_native_method_timer_preserves_outputs(self):
        import torch
        from types import SimpleNamespace
        from worker import AdapterTimer
        module = torch.nn.Module()
        module.first = torch.nn.Linear(2,2,bias=False)
        module.second = torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():
            module.first.weight.copy_(torch.eye(2))
            module.second.weight.copy_(2*torch.eye(2))
        fake_cuda = SimpleNamespace(synchronize=lambda:None, reset_peak_memory_stats=lambda:None,
                                    max_memory_allocated=lambda:0, max_memory_reserved=lambda:0)
        timer = AdapterTimer(SimpleNamespace(family='RP-R-CNN',model=module), SimpleNamespace(cuda=fake_cuda))
        output, timing = timer.measure(lambda:module.second(module.first(torch.tensor([[3.,5.]]))))
        self.assertEqual(output.tolist(),[[6.,10.]])
        self.assertEqual(timing['module_calls'],2)
        total = sum(timing[k] for k in ('pre_module_ms','module_call_ms','post_module_ms','inter_module_ms'))
        self.assertAlmostEqual(timing['adapter_ms'],total)

    def test_interrupted_atomic_packet_keeps_orphan(self):
        from unittest.mock import patch
        from common_view.packets import write_packet
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('common_view.packets.np.savez_compressed',side_effect=RuntimeError('synthetic interruption')):
                with self.assertRaises(RuntimeError):
                    write_packet(root,'fixture',{}, {}, {})
            self.assertFalse((root/'fixture').exists())
            self.assertTrue((root/'.partial-fixture').exists())
            with self.assertRaises(FileExistsError):
                write_packet(root,'fixture',{}, {}, {})

    def test_qanet_rgb_normalization_and_geometry(self):
        from types import SimpleNamespace
        from controlled_roi_models import prepare_qanet_rgb,restore_qanet_probabilities
        import torch
        cfg = SimpleNamespace(TEST=SimpleNamespace(SCALE=[5,3]),PIXEL_MEANS=[.1,.2,.3],PIXEL_STDS=[1,2,4])
        rgb = np.zeros((3,5,3),np.uint8)
        rgb[0,0] = [255,0,0]
        tensor,geometry = prepare_qanet_rgb(rgb,cfg)
        np.testing.assert_allclose(tensor[0,:,0,0].numpy(),[.9,-.1,-.075],atol=1e-7)
        probs = torch.zeros((1,20,3,5),dtype=torch.float32)
        probs[:,0] = 1.
        probs[0,0,0,4],probs[0,13,0,4] = 0.,1.
        native = restore_qanet_probabilities(probs,geometry)
        self.assertEqual(native.shape,(3,5))
        self.assertEqual(int(native[0,4]),13)
        self.assertEqual(int(native[2,0]),0)

    def test_private_gallery_manifest_and_alignment(self):
        from common_view.packets import write_packet
        from common import npz
        from gallery import gallery
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixels = np.array([[[255,0,0],[0,255,0],[0,0,255],[230,90,10]]],np.uint8)
            Image.fromarray(pixels).save(root/'input.png')
            npz(root/'GT/fixture.npz',labels=np.array([[1,2,3,4]],np.uint8))
            atomic(root/'CONFIG.json',dict(models={'fixture':{}},protocol_sha256='synthetic'))
            atomic(root/'MANIFEST.json',dict(visual_ids=['fixture'],frames=[dict(image_id='fixture',
                image_path=str(root/'input.png'),image_sha256=sha(root/'input.png'),gt_npz_sha256=sha(root/'GT/fixture.npz'))]))
            write_packet(root/'fixture/packets','fixture',dict(common_mask=np.array([[1,2,3,4]],np.uint8)),{}, {})
            gallery(root)
            manifest = load(root/'gallery/MANIFEST.json')
            self.assertEqual(len(manifest['panels']),1)
            self.assertEqual(list(manifest['panels'][0]['models']),['fixture'])
            self.assertEqual(Image.open(root/'gallery/fixture.png').size,(1200,295))
            self.assertEqual(manifest['panels'][0]['png_sha256'],sha(root/'gallery/fixture.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.config and result.wasSuccessful():
        from common import verify_sources
        config = load(args.config)
        verify_sources(config)
        atomic(Path(config['root'])/'CPU_ACCEPTANCE.json',dict(status='PASS',test_cases=result.testsRun,
               failures=0,config_sha256=sha(args.config),protocol_sha256=config['protocol_sha256'],
               expected_values='independent literal tiny fixtures; real immutable packet -> scorer -> reporter'))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
