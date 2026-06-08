import os, importlib, sys, time
os.chdir('/home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion')
mods = [
    'tools',
    'utils',
    'utils.parser',
    'models',
    'models.PGST',
    'datasets',
    'models.gft',
    'tools.runner',
]
for m in mods:
    print('start', m, flush=True)
    importlib.import_module(m)
    print('done', m, flush=True)
print('all done', flush=True)
