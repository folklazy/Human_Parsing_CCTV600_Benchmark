"""Private fixed-sample visual comparison, using saved outputs only."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from common import archive, atomic, load, sha

PALETTE = np.array([[0,0,0],[128,64,192],[255,166,77],[0,158,115],[0,114,178]],np.uint8)


def panel(rgb, mask, title):
    rendered = rgb.copy()
    if mask is not None:
        valid = (mask >= 1) & (mask <= 4)
        rendered[valid] = np.rint(.55*rendered[valid] + .45*PALETTE[mask[valid]]).astype(np.uint8)
    canvas = Image.new('RGB',(400,260),'white')
    image = ImageOps.contain(Image.fromarray(rendered),(400,225))
    canvas.paste(image,((400-image.width)//2,30+(225-image.height)//2))
    ImageDraw.Draw(canvas).text((8,8),title,fill='black')
    return canvas


def gallery(root):
    from common import install_shared_paths
    install_shared_paths()
    from common_view.packets import read_packet
    config = load(root/'CONFIG.json')
    manifest = load(root/'MANIFEST.json')
    frames = {f['image_id']:f for f in manifest['frames']}
    output = root/'gallery'
    output.mkdir(exist_ok=True)
    entries = []
    for iid in manifest['visual_ids']:
        frame = frames[iid]
        rgb = np.asarray(Image.open(frame['image_path']).convert('RGB'))
        gt = archive(root/'GT'/(iid+'.npz'))
        panels = [panel(rgb,None,'Original RGB'),panel(rgb,gt['labels'],'GT annotated targets; unknown transparent')]
        sources = {}
        for mid in config['models']:
            path = root/mid/'packets'/iid
            if path.exists():
                metadata, arrays = read_packet(path)
                panels.append(panel(rgb,arrays['common_mask'],mid))
                sources[mid] = dict(packet_sha256=metadata['packet_sha256'],metadata_sha256=sha(path/'packet.json'))
            else:
                panels.append(panel(rgb,None,mid+' [missing]'))
                sources[mid] = None
        canvas = Image.new('RGB',(1200,260*math.ceil(len(panels)/3)+35),'#e8eef3')
        for number, value in enumerate(panels):
            canvas.paste(value,((number%3)*400,(number//3)*260))
        legend = ImageDraw.Draw(canvas)
        for index, name in enumerate(('Hair','Face proxy','Upper','Lower'),1):
            x,y = (index-1)*220+10,canvas.height-25
            legend.rectangle([x,y,x+16,y+16],fill=tuple(PALETTE[index]))
            legend.text((x+23,y),name,fill='black')
        filename = iid+'.png'
        canvas.save(output/filename)
        entries.append(dict(image_id=iid,file=filename,png_sha256=sha(output/filename),
                            input_sha256=frame['image_sha256'],GT_sha256=frame['gt_npz_sha256'],models=sources))
    atomic(output/'MANIFEST.json',dict(selection='same 16 fixed images from accepted detector manifest, before parser scores',
           protocol_sha256=config['protocol_sha256'],panels=entries,public_upload=False))
    links = ['<!doctype html><meta charset="utf-8"><title>CCTV600 private parsing gallery</title>',
             '<h1>CCTV600 fixed-sample parsing gallery</h1>',
             '<p>GT displays annotated targets. Unknown is transparent. Predictions can extend into unscored unknown.</p>']
    for entry in entries:
        links.append(f'<h2>{entry["image_id"]}</h2><a href="{entry["file"]}"><img style="max-width:100%" src="{entry["file"]}"></a>')
    (output/'index.html').write_text('\n'.join(links))
