"""Pinned SDXL ControlNet load components; one explicitly scaled component per load."""
import hashlib
import json
import re
from pathlib import Path
from .contracts import WorkerError, digest, minor, strict, approved_snapshot_link, safetensors_header

FILES={'config.json','diffusion_pytorch_model.safetensors'}


def validate(entries,catalog,version,verify=True):
    if not isinstance(entries,list) or len(entries)>1:
        raise WorkerError(400,'invalid_request','controlnets')
    if entries and minor(version)<4: raise WorkerError(400,'unsupported','controlnets')
    result=[]
    for entry in entries:
        strict(entry,('controlnet','scale'))
        record=entry['controlnet'];strict(record,('schema_version','controlnet_id','artifact_handle','family','files','source','digest'))
        if not isinstance(record['controlnet_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',record['controlnet_id']):raise WorkerError(400,'invalid_request','controlnet_id')
        if record['schema_version']!=1 or record['family']!='sdxl-base':raise WorkerError(400,'unsupported','controlnet')
        if digest({k:v for k,v in record.items() if k!='digest'})!=record['digest']:raise WorkerError(400,'artifact_changed','controlnet')
        strict(record['source'],(),('repository','revision','license'))
        if any(not isinstance(v,str) for v in record['source'].values()):raise WorkerError(400,'invalid_request','source')
        if not isinstance(record['files'],dict) or set(record['files'])!=FILES:raise WorkerError(400,'unsupported','controlnet.files')
        handle=record['artifact_handle']
        if not isinstance(handle,str) or handle not in catalog:raise WorkerError(400,'missing_component','artifact_handle')
        root=Path(catalog[handle]).resolve()
        if not root.is_dir():raise WorkerError(400,'missing_component','controlnet')
        actual={p.name:p for p in root.iterdir()}
        if set(actual)!=FILES:raise WorkerError(400,'artifact_changed','controlnet.files')
        for name,path in actual.items():
            if not path.is_file() or path.is_symlink() and not approved_snapshot_link(path,root):raise WorkerError(400,'unsupported','controlnet.files')
            if not isinstance(record['files'][name],str) or not re.fullmatch(r'[a-f0-9]{64}',record['files'][name]):raise WorkerError(400,'invalid_request','controlnet.files')
            if verify:
                with path.open('rb') as stream: actual_hash=hashlib.file_digest(stream,'sha256').hexdigest()
                if actual_hash!=record['files'][name]:raise WorkerError(400,'artifact_changed','controlnet')
        try:config=json.loads((root/'config.json').read_text())
        except (OSError,ValueError):raise WorkerError(400,'invalid_request','controlnet.config') from None
        if config.get('_class_name')!='ControlNetModel' or config.get('addition_embed_type')!='text_time':raise WorkerError(400,'unsupported','controlnet.config')
        safetensors_header(root/'diffusion_pytorch_model.safetensors')
        scale=entry['scale']
        if type(scale) not in (int,float) or not 0<=scale<=2:raise WorkerError(400,'invalid_request','controlnet.scale')
        result.append((record,root,scale))
    return result


def attach(backend,resolved):
    import torch
    from diffusers import ControlNetModel
    backend.controlnet=None;backend.controlnet_scale=None
    if not resolved:return
    record,root,scale=resolved[0]
    backend.controlnet=ControlNetModel.from_pretrained(str(root),local_files_only=True,use_safetensors=True,
        torch_dtype=getattr(torch,backend.load_options['precision'])).to(backend.run_device)
    backend.controlnet_scale=float(scale)
