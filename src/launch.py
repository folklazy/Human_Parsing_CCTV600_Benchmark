"""Launch the accepted frozen batch exactly once in a detached tmux session."""
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from common import PROJECT, WORKSPACE, atomic, load, now, sha, verify_sources


def launch():
    current = load(PROJECT/'outputs/CURRENT.json')
    root = Path(current['root'])
    config_path = root/'CONFIG.json'
    config = load(config_path)
    acceptance = load(root/'CPU_ACCEPTANCE.json')
    if acceptance['status'] != 'PASS' or acceptance['config_sha256'] != sha(config_path):
        raise ValueError('CPU acceptance does not bind this config')
    verify_sources(config)
    if sha(PROJECT/'configs/PROTOCOL.json') != config['protocol_sha256']:
        raise ValueError('Protocol binding mismatch')
    if (root/'LAUNCH.json').exists() or (root/'BATCH_STATE.json').exists():
        raise FileExistsError('Existing attempt must be reconciled before any continuation')
    session = 'parsing_cctv600'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode == 0:
        raise RuntimeError('Existing tmux session; inspect it before launching')
    resources = config['protocol']['resources']
    available = shutil.disk_usage(root).free / 2**30
    if available < resources['minimum_free_disk_GiB'] + resources['max_run_GiB']:
        raise RuntimeError('Insufficient free disk for declared run cap and reserved headroom')
    commit = subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(PROJECT),'status','--porcelain'],text=True).strip():
        raise ValueError('Commit this experiment source before launch')
    command = [str(WORKSPACE/'.venv/bin/python'),'-u',str(PROJECT/'src/batch.py'),'--config',str(config_path)]
    record = dict(run_id=config['run_id'],created_utc=now(),launch_commit=commit,command=command,
                  tmux_session=session,config_sha256=sha(config_path),free_disk_GiB=available,
                  controller_log=str(root/'controller.log'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','0'),
                  authorization='User requested detailed verification and execution of next CCTV600 parsing benchmark',
                  scope='Nine existing checkpoints; reuse YOLO600; zero training/downloads; AD4 partial-domain scores')
    atomic(root/'SCIENTIFIC_FREEZE.json',dict(at_utc=now(),config_sha256=sha(config_path),
           protocol_sha256=config['protocol_sha256'],CPU_acceptance_sha256=sha(root/'CPU_ACCEPTANCE.json'),
           GT_bindings_sha256=config['GT_bindings_sha256'],launch_commit=commit,
           freeze_before_any_CCTV600_parser_calls=True))
    atomic(root/'LAUNCH.json',record)
    shell_command = 'exec ' + shlex.join(command) + ' > ' + shlex.quote(str(root/'controller.log')) + ' 2>&1'
    result = subprocess.check_output(['tmux','new-session','-d','-s',session,'-c',str(WORKSPACE),
              '-P','-F','#{session_id} #{pane_pid}',shell_command],text=True).strip()
    subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
    record['tmux_session_and_initial_pane_pid'] = result
    atomic(root/'LAUNCH.json',record)
    print(record['run_id'],result,'log',record['controller_log'])


if __name__ == '__main__':
    launch()
