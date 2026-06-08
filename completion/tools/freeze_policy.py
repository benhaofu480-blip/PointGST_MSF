"""Trainable-parameter policies for partial fine-tuning experiments."""


def _is_msf_adapter(name):
    return 'gft_adapter' in name


def _is_rebuild_head(name):
    """Fine reconstruction head on AdaPoinTr_PGST (exclude base_model coarse stack)."""
    if name.startswith('decode_head.'):
        return True
    if name.startswith('reduce_map.'):
        return True
    if name.startswith('increase_dim.') and not name.startswith('base_model.'):
        return True
    return False


def _is_decoder_block(name, block_idx):
    return f'decoder.blocks.blocks.{block_idx}.' in name


def param_trainable_for_part(name, part, opti_config):
    if part == 'gft_single_decoder':
        block_idx = int(getattr(opti_config, 'trainable_decoder_block', 7))
        return (
            _is_msf_adapter(name)
            or _is_rebuild_head(name)
            or _is_decoder_block(name, block_idx)
        )
    if part == 'gft_msf_head_only':
        return _is_msf_adapter(name) or _is_rebuild_head(name)
    return None
