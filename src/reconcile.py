"""Explicit read-only packet reconciliation. Never starts or repairs inference."""
import argparse
from pathlib import Path

from common import PROJECT, atomic, install_shared_paths, load, membership, now, sha


def reconcile(root, mid):
    install_shared_paths()
    from common_view.packets import read_packet
    config = root/'CONFIG.json'
    manifest = load(root/'MANIFEST.json')
    expected_jobs = [job['job_id'] for f in manifest['frames'] for job in f['jobs']]
    accepted = []
    invalid = []
    for job_id in expected_jobs:
        path = root/mid/'roi_packets'/job_id
        if not path.exists():
            continue
        try:
            metadata, arrays = read_packet(path,dict(config_sha256=sha(config),job_id=job_id,model_id=mid))
            accepted.append(dict(job_id=job_id,path=str(path),packet_sha256=metadata['packet_sha256'],
                                 metadata_sha256=sha(path/'packet.json')))
        except Exception as error:
            invalid.append(dict(job_id=job_id,path=str(path),reason=str(error)))
    completed_frames = []
    for iid in manifest['expected_ids']:
        path = root/mid/'packets'/iid
        if not path.exists():
            continue
        try:
            read_packet(path,dict(config_sha256=sha(config),image_id=iid,model_id=mid))
            completed_frames.append(iid)
        except Exception as error:
            invalid.append(dict(image_id=iid,path=str(path),reason=str(error)))
    known = set(expected_jobs)
    orphan = [str(p) for p in (root/mid/'roi_packets').glob('*') if p.is_dir() and not p.name.startswith('.') and p.name not in known]
    partial = [str(p) for p in (root/mid).glob('**/.partial-*')]
    checked = membership(expected_jobs,[a['job_id'] for a in accepted])
    result = dict(model_id=mid,at_utc=now(),ROI_membership=checked,
                  image_membership=membership(manifest['expected_ids'],completed_frames),
                  accepted_ROI_references=accepted,invalid=invalid,orphan=orphan,partial=partial,
                  primary=None,source_files_changed=0,model_calls=0,
                  continuation='requires a new linked segment and scientific-binding reuse review; no automatic retry')
    path = root/'reconciliation'/(mid+'-'+now().replace(':','').replace('.','')+'.json')
    atomic(path,result)
    print('accepted_ROIs',len(accepted),'pending_ROIs',checked['missing'],'completed_images',len(completed_frames),
          'invalid',len(invalid),'partial',len(partial),'orphan',len(orphan),'record',path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--root',type=Path)
    args = parser.parse_args()
    reconcile(args.root or Path(load(PROJECT/'outputs/CURRENT.json')['root']),args.model)
