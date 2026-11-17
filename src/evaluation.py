"""Annotation-supported semantic comparison. Unassigned GT stays unknown."""
import numpy as np
from PIL import Image, ImageDraw

LABELS = {'hair': 1, 'face': 2, 'upper': 3, 'lower': 4}


def raster(shape, wh):
    canvas = Image.new('1', wh)
    points = [tuple(p) for p in shape['points']]
    if shape['shape_type'] == 'rectangle':
        xs, ys = zip(*points)
        ImageDraw.Draw(canvas).rectangle([min(xs), min(ys), max(xs), max(ys)], fill=1)
    else:
        ImageDraw.Draw(canvas).polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)


def degenerate(shape):
    points = shape['points']
    if shape['shape_type'] == 'rectangle':
        xs, ys = zip(*points)
        return max(xs) <= min(xs) or max(ys) <= min(ys)
    area2 = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1]))
    return abs(area2) < 1e-9


def derive_gt(annotation, excluded_indices=()):
    wh = (annotation['imageWidth'], annotation['imageHeight'])
    labels = np.zeros(wh[::-1], np.uint8)
    owners = np.zeros(wh[::-1], np.int32)
    void = np.zeros(wh[::-1], bool)
    conflicts = np.zeros(wh[::-1], bool)
    eligible = set()
    for shape in annotation['shapes']:
        if shape['label'] == 'ignore':
            void |= raster(shape, wh)
        elif shape['label'] == 'bbox':
            xs, ys = zip(*shape['points'])
            if max(max(xs) - min(xs), max(ys) - min(ys)) >= 50:
                eligible.add(shape['group_id'])
            else:
                void |= raster(shape, wh)
    for index, shape in enumerate(annotation['shapes']):
        if shape['label'] not in LABELS:
            continue
        region = raster(shape, wh)
        if index in excluded_indices or degenerate(shape) or shape['group_id'] not in eligible:
            void |= region
            continue
        value, owner = LABELS[shape['label']], shape['group_id']
        conflicts |= region & (labels > 0) & ((labels != value) | (owners != owner))
        free = region & (labels == 0)
        labels[free], owners[free] = value, owner
    void |= conflicts
    valid = (labels > 0) & ~void
    labels[~valid], owners[~valid] = 255, 0
    persons = []
    for pid in sorted(eligible):
        supports = [int(((owners == pid) & (labels == c)).sum()) for c in range(1, 5)]
        persons.append(dict(person_id=pid, support=supports))
    return dict(labels=labels, owners=owners), dict(
        valid_pixels=int(valid.sum()), conflict_pixels=int(conflicts.sum()),
        explicit_or_QA_void_pixels=int(void.sum()), unknown_pixels=int((~valid).sum()),
        support=[int((labels == c).sum()) for c in range(5)], persons=persons)


def project(native, mapping):
    native = np.asarray(native)
    if native.dtype.kind not in 'ui' or (native.size and (native.min() < 0 or native.max() >= len(mapping))):
        raise ValueError('Unknown/255 native prediction label or noninteger dtype')
    return np.asarray(mapping, np.uint8)[native]


def compose(planes, scores, boxes, ids, shape):
    if planes.shape != (len(scores), *shape) or len(boxes) != len(scores) or len(ids) != len(scores):
        raise ValueError('Prediction geometry/identity mismatch')
    result = np.zeros(shape, np.uint8)
    # Highest confidence wins. Smaller canonical geometry/ID wins equal scores.
    canonical = sorted(range(len(scores)), key=lambda j: (*boxes[j], int(ids[j])))
    rank = {j: i for i, j in enumerate(canonical)}
    for j in sorted(range(len(scores)), key=lambda j: (float(scores[j]), -rank[j])):
        selected = planes[j] > 0
        result[selected] = planes[j][selected]
    return result


def confusion(gt, prediction):
    if gt.shape != prediction.shape or np.any((gt != 255) & (gt > 4)) or np.any(prediction > 4):
        raise ValueError('Mapped grid/label contract')
    valid = gt != 255
    return np.bincount(5 * gt[valid].astype(np.int64) + prediction[valid], minlength=25).reshape(5, 5)


def metrics(histogram):
    h = np.asarray(histogram, np.int64)
    if h.shape != (5, 5) or np.any(h < 0):
        raise ValueError('Nonnegative 5x5 confusion required')
    tp, gs, ps = np.diag(h), h.sum(1), h.sum(0)
    union = gs + ps - tp
    ratio = lambda a, b: float(a / b) if b else None
    iou = [ratio(tp[c], union[c]) for c in range(5)]
    return dict(confusion=h.tolist(), GT_support=gs.tolist(), prediction_support=ps.tolist(),
                union=union.tolist(), IoU=iou,
                precision=[ratio(tp[c], ps[c]) for c in range(5)],
                recall=[ratio(tp[c], gs[c]) for c in range(5)],
                F1_Dice=[ratio(2 * tp[c], gs[c] + ps[c]) for c in range(5)],
                AD4_mIoU4=float(np.mean(iou[1:])) if all(gs[1:] > 0) else None,
                C3_global=float(np.mean([iou[c] for c in (1, 3, 4)])) if all(gs[[1, 3, 4]] > 0) else None,
                full_frame_mIoU4=None)


def score_frame(gt, gt_binding, frame, owned, cache, mapping):
    from cctv600.scoring import fixed
    scores, boxes, ids = cache['scores'], cache['boxes'], cache['ids']
    projected = project(owned, mapping)
    global_mask = project(compose(owned, scores, boxes, ids, gt['labels'].shape), mapping)
    hist = confusion(gt['labels'], global_mask)
    predictions = [dict(confidence=float(s), bbox_xyxy=b.tolist()) for s, b in zip(scores, boxes)]
    order = sorted(range(len(predictions)), key=lambda i: (-predictions[i]['confidence'], *predictions[i]['bbox_xyxy']))
    person = {}
    for threshold in (.5, .75):
        detection = fixed(frame, predictions, threshold=threshold)
        by_gt = {m['gt_index']: order[m['prediction_index']] for m in detection['matches']}
        total = [0.] * 4
        count = [0] * 4
        for index, item in enumerate(frame['people']):
            pid = item['person_id']
            region = gt['owners'] == pid
            valid = gt['labels'] != 255
            predicted = projected[by_gt[index]] if index in by_gt else np.zeros(region.shape, np.uint8)
            for c in range(1, 5):
                truth = region & (gt['labels'] == c)
                if not truth.any():
                    continue
                selected = (predicted == c) & valid
                union = np.count_nonzero(truth | selected)
                total[c - 1] += float(np.count_nonzero(truth & selected) / union)
                count[c - 1] += 1
        person[str(threshold)] = dict(tp=detection['tp'], fp=detection['fp'], fn=detection['fn'],
                                     ignored_predictions=detection['ignored_predictions'],
                                     part_IoU_sum=total, eligible_person_parts=count)
    return dict(histogram=hist.tolist(), person=person, valid_pixels=gt_binding['valid_pixels'])


def aggregate(rows, expected_ids):
    from common import membership
    checked = membership(expected_ids, [r['image_id'] for r in rows])
    if not checked['strict_complete']:
        return dict(membership=checked, primary=None)
    histogram = sum((np.array(r['histogram'], np.int64) for r in rows), np.zeros((5, 5), np.int64))
    person = {}
    for t in ('0.5', '0.75'):
        total = np.sum([r['person'][t]['part_IoU_sum'] for r in rows], axis=0)
        counts = np.sum([r['person'][t]['eligible_person_parts'] for r in rows], axis=0)
        values = [float(a / b) if b else None for a, b in zip(total, counts)]
        person[t] = dict(part_mean_IoU=values, eligible_person_parts=counts.tolist(),
                         Q_AD4_GT=float(np.mean(values)) if all(counts > 0) else None,
                         **{k: sum(r['person'][t][k] for r in rows) for k in ('tp', 'fp', 'fn', 'ignored_predictions')})
        p = person[t]
        p['precision'] = p['tp'] / (p['tp'] + p['fp']) if p['tp'] + p['fp'] else None
        p['recall'] = p['tp'] / (p['tp'] + p['fn']) if p['tp'] + p['fn'] else None
        p['F1'] = 2*p['tp'] / (2*p['tp'] + p['fp'] + p['fn']) if 2*p['tp'] + p['fp'] + p['fn'] else None
    return dict(membership=checked, primary=metrics(histogram), person=person,
                score_scope='ANNOTATION_SUPPORTED_DOMAIN_ONLY', full_frame_primary=None)
