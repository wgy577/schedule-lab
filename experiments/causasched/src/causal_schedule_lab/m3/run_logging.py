"""Concise console; full parent-process messages retained separately."""
import builtins
_file=None
_verbose=False

def configure(folder, verbose=False):
    global _file,_verbose
    if _file is not None:_file.close()
    _file=open(folder/'detail.log','a',buffering=1)
    _verbose=verbose

def log(*args, **kwargs):
    text=kwargs.get('sep',' ').join(map(str,args))
    if _file is not None:
        _file.write(text+kwargs.get('end','\n'))
    hidden=(text.startswith(('[collect]','[continue]','[choice-perturb]','[perturb]',
        '[reward-mix]','[E2E]','[episode-clock]','[replay parity]','[parity CPU verified]','[timing]'))
        or (text.startswith('[parity]') and 'decisions=' in text)
        or (text.startswith('[update]') and 'decisions=' in text and 'OOM' not in text)
        or (text.startswith('[episode]') and 'update=' in text))
    if _verbose or not hidden:builtins.print(*args,**kwargs)
