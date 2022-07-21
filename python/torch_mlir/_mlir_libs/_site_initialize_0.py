from . import _torchMlir

def context_init_hook(context):
    _torchMlir.register_required_dialects(context)
