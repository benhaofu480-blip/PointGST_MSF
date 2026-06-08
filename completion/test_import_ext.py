print('start', flush=True)
import importlib
mods = ['extensions', 'extensions.chamfer_dist']
for m in mods:
    print('import', m, flush=True)
    importlib.import_module(m)
    print('done', m, flush=True)
print('ok', flush=True)
