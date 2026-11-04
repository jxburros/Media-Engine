"""Resident component weights used consistently for planning and load estimates."""
import math
from .contracts import PRECISION_BYTES, safetensors_parameters


def inventory(controlnets, conditioning):
    records = [(record,root) for record,root,_scale in controlnets]
    records.extend((entry['component'],root) for entry,root in conditioning)
    parameters = 0
    replacement = False
    for record,root in records:
        replacement |= record.get('kind') == 'vae'
        for name in record['files']:
            if name.endswith('.safetensors'):
                parameters += sum(safetensors_parameters(root/name).values())
    return {'parameters':parameters, 'replaces_vae':replacement}


def include(base, stats, precision):
    """Base estimates already count the original VAE: remove it when replaced."""
    weights_mb = math.ceil(stats['parameters'] * PRECISION_BYTES[precision] / 1024**2)
    removed_mb = base.get('roles',{}).get('vae',{}).get('loaded_mb',0) if stats['replaces_vae'] else 0
    resident_mb = base['value_mb'] - removed_mb + weights_mb
    return {**base, 'value_mb':resident_mb, 'load_peak_mb':max(base['value_mb'],resident_mb),
        'additional_components_mb':weights_mb, 'replaced_base_vae_mb':removed_mb,
        'uncertainty':str(base.get('uncertainty','')) +
            '; includes resident component weights at resolved precision; load_peak_mb also covers the base load before VAE replacement; excludes CPU replacement staging, activations and image-encoder working memory'}
