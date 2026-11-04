"""Local reference DiT pipelines; no custom code, downloads or scheduler substitution."""
from pathlib import Path

FAMILIES = ('flux-schnell', 'sd35')
PAIR = {'sampler': 'flow-euler', 'scheduler': 'flow-match'}


def load(backend, root, options):
    import torch
    from diffusers import FluxPipeline, StableDiffusion3Pipeline
    from .backend import offline_environment, uniform_precision
    offline_environment()
    backend.apply_budget()
    cls = FluxPipeline if backend.family == 'flux-schnell' else StableDiffusion3Pipeline
    pipe = cls.from_pretrained(str(root), torch_dtype=getattr(torch, options['precision']),
                               local_files_only=True, use_safetensors=True)
    uniform_precision(pipe, ('text_encoder', 'text_encoder_2', 'text_encoder_3'), options['precision'])
    if backend.family == 'flux-schnell' and pipe.transformer.config.guidance_embeds:
        raise ValueError('flux-schnell requires a guidance-free transformer')
    pipe.to(backend.run_device)
    pipe.set_progress_bar_config(disable=True)
    backend.pipe = pipe
    backend.scheduler_config = dict(pipe.scheduler.config)


def generate(backend, request, directory, emit, cancelled, image_path=None):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from .backend import Cancelled
    from .contracts import output_ids, WorkerError
    params, inputs = request['parameters'], request['inputs']
    if {k: params[k] for k in PAIR} != PAIR:
        raise WorkerError(400, 'unsupported', 'scheduler')
    if backend.family == 'flux-schnell' and (params['guidance'] != 0 or inputs.get('negative_prompt')):
        raise WorkerError(400, 'unsupported', 'guidance')
    pipe = backend.pipe
    if request['task'] in ('image-to-image','inpaint'):
        from diffusers import FluxImg2ImgPipeline, FluxInpaintPipeline, StableDiffusion3Img2ImgPipeline, StableDiffusion3InpaintPipeline
        classes={'flux-schnell':(FluxImg2ImgPipeline,FluxInpaintPipeline),'sd35':(StableDiffusion3Img2ImgPipeline,StableDiffusion3InpaintPipeline)}
        pipe=classes[backend.family][request['task']=='inpaint'].from_pipe(pipe)
        pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(backend.scheduler_config)
    backend.begin(cancelled)
    try:
        def step(_pipe, index, timestep, values):
            if cancelled.is_set(): raise Cancelled()
            emit('progress', {'stage':'sampling','completed_steps':index+1,'total_steps':params['steps'],'scope':'stage'})
            return values
        emit('stage', {'stage':'encoding'})
        seeds = backend.seeds(params)
        generators = [torch.Generator(device=backend.run_device).manual_seed(seed) for seed in seeds]
        args = dict(prompt=inputs['prompt'], width=params['width'], height=params['height'],
                    num_inference_steps=params['steps'], guidance_scale=params['guidance'],
                    generator=generators[0] if len(seeds)==1 else generators,
                    num_images_per_prompt=len(seeds), callback_on_step_end=step, output_type='pil',
                    max_sequence_length=256)
        if request['task'] in ('image-to-image','inpaint'):
            from PIL import Image
            source=image_path['image'] if isinstance(image_path,dict) else image_path
            with Image.open(source) as image:args['image']=image.convert('RGB')
            args['strength']=params['strength']
            if request['task']=='inpaint':
                with Image.open(image_path['mask']) as mask:args['mask_image']=mask.convert('L')
        if backend.family == 'sd35': args['negative_prompt'] = inputs.get('negative_prompt')
        emit('stage', {'stage':'sampling'})
        with torch.inference_mode(): images = pipe(**args).images
        if cancelled.is_set(): raise Cancelled()
        emit('stage', {'stage':'saving'})
        records=[]
        for output_id, image in zip(output_ids(request['task'],params),images):
            if cancelled.is_set(): raise Cancelled()
            path=Path(directory)/(output_id+'.png'); image.save(path,format='PNG')
            records.append(dict(output_id=output_id,path=path,mime_type='image/png',width=image.width,height=image.height))
        return records, {'effective_steps':params['steps'], **backend.measurements()}
    finally: backend.finish()
