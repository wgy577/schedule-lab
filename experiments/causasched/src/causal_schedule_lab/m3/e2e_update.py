"""One normalized optimizer step per epoch, memory-bounded real minibatches."""
import gc
import copy
import json
import math
import os
import time
import torch
from .run_logging import log as print
from .e2e_batch import replay_batch, loss_batch


def verify_feature_roundoff(gpu, recorded, reference):
    """CPU replay must reproduce behavior before scaled GPU tolerance applies."""
    gpu=gpu.detach().float().cpu(); recorded=recorded.detach().float().cpu()
    reference=reference.detach().float().cpu()
    if gpu.shape!=recorded.shape or reference.shape!=recorded.shape:
        raise RuntimeError('Feature shape mismatch: optimizer not stepped')
    if not all(bool(torch.isfinite(t).all()) for t in (gpu,recorded,reference)):
        raise RuntimeError('Nonfinite replay features: optimizer not stepped')
    cpu_error=float((reference-recorded).abs().max())
    scaled=float(((gpu-reference).abs()/(1e-4+1e-4*reference.abs())).max())
    if cpu_error>.005 or scaled>1:
        raise RuntimeError(f'Replay mismatch after CPU verification: cpu_abs={cpu_error:.6g} '
                           f'gpu_scaled={scaled:.6g}; optimizer not stepped')
    return cpu_error,scaled


def chunks(records, size):
    # Similar graph sizes reduce padding. A node/appearance budget guards large cases.
    ordered=sorted(records,key=lambda s:(len(s['e2e']['input']['batch'].node_numeric),s['M']))
    rows=[]; nodes=cells=0
    for rec in ordered:
        b=rec['e2e']['input']['batch']; n=len(b.node_numeric); c=n*b.symptom_block_count
        if rows and (len(rows)>=size or nodes+n>int(os.getenv('MAX_BATCH_NODES','24000'))
                     or cells+c>int(os.getenv('MAX_APPEARANCE_CELLS','400000'))):
            yield rows; rows=[]; nodes=cells=0
        rows.append(rec); nodes+=n; cells+=c
    if rows: yield rows


def update(r,groups,opt,named,device,epochs,check,decision_batch=32):
    start=time.perf_counter(); records=[s for g in groups for tr in g for s in tr['steps']]
    count=len(records)
    informative2=sum(any(s['inf2'] for s in tr['steps']) for g in groups for tr in g)
    informative3=sum(any(s['inf3'] for s in tr['steps']) for g in groups for tr in g)
    for g in groups:
        for tr in g:
            n2=sum(s['inf2'] for s in tr['steps']); n3=sum(s['inf3'] for s in tr['steps'])
            for s in tr['steps']:
                s['_w2']=float(bool(s['inf2']))/max(n2*informative2,1)
                s['_w3']=float(bool(s['inf3']))/max(n3*informative3,1)
    parity=dict(m2_logp=0.,m3_logp=0.,features=0.)
    size=decision_batch
    cpu_reference=None
    feature_rechecks=0
    if check:
        while True:
            try:
                done=0
                with torch.no_grad():
                    for rows in chunks(records,size):
                        out=replay_batch(r,rows,device)
                        old2=out['lp2'].new_tensor([sum(d['logp'] for d in s['m2_rec']['draws']) for s in rows])
                        old3=out['lp3'].new_tensor([s['logp_old'] for s in rows])
                        if not all(bool(torch.isfinite(t).all()) for t in (out['lp2'],out['lp3'],old2,old3)):
                            raise RuntimeError('Nonfinite replay log probabilities: optimizer not stepped')
                        err2=float((out['lp2']-old2).abs().max()); err3=float((out['lp3']-old3).abs().max())
                        errf=float(torch.stack([(f-s['F_pool'].to(device)).abs().max() for f,s in zip(out['features'],rows)]).max())
                        for f,s in zip(out['features'],rows):
                            if not bool(torch.isfinite(f).all()) or not bool(torch.isfinite(s['F_pool']).all()):
                                raise RuntimeError('Nonfinite replay features: optimizer not stepped')
                            if float((f-s['F_pool'].to(device)).abs().max())<=.005:
                                continue
                            from .end_to_end import recompute, modules
                            if cpu_reference is None:
                                # Clone only active models, not the replay/memory bank.
                                cpu_reference=copy.deepcopy({k:r[k] for k in
                                    ('model_b5','single_head','direct_head','scorer','policy')})
                                for m in modules(cpu_reference).values():
                                    m.to('cpu'); m.eval(); m.requires_grad_(False)
                            c2,c3,_,aux=recompute(cpu_reference,s,'cpu')
                            old_c2=sum(d['logp'] for d in s['m2_rec']['draws'])
                            if not math.isfinite(float(c2)) or not math.isfinite(float(c3)) or max(
                                abs(float(c2)-old_c2),abs(float(c3)-s['logp_old']))>.005:
                                raise RuntimeError('CPU behavior logp mismatch: optimizer not stepped')
                            try:
                                ce,se=verify_feature_roundoff(f,s['F_pool'],aux['F'])
                            except RuntimeError as exc:
                                # Preserve the entire offending batch: packing errors
                                # may disappear when only one decision is reconstructed.
                                exc.parity_payload=dict(records=rows,reference_models=cpu_reference,
                                    gpu_features=[v.detach().cpu() for v in out['features']],
                                    failed_index=next(i for i,row in enumerate(rows) if row is s),
                                    cpu_features=aux['F'].detach().cpu(),error=str(exc),
                                    torch_version=str(torch.__version__),device=str(device),
                                    matmul_precision=torch.get_float32_matmul_precision())
                                raise
                            feature_rechecks+=1
                            print(f'[parity CPU verified] feature_abs={float((f.cpu()-s["F_pool"].cpu()).abs().max()):.6g} '
                                  f'cpu_abs={ce:.6g} gpu_scaled={se:.6g}',flush=True)
                        parity={k:max(parity[k],v) for k,v in zip(parity,(err2,err3,errf))}
                        done+=len(rows)
                        print(f'[parity] decisions={done}/{count} batch={len(rows)} seconds={time.perf_counter()-start:.1f}',flush=True)
                        del out
                break
            except torch.OutOfMemoryError:
                if size<=1: raise
                size=max(1,size//2); gc.collect(); torch.cuda.empty_cache()
                print(f'[parity] OOM: retry batch={size}',flush=True)
        print('[replay parity] '+json.dumps(parity),flush=True)
        if max(parity['m2_logp'],parity['m3_logp'])>.005:
            raise RuntimeError('Batched behavior replay mismatch: optimizer not stepped')
        del cpu_reference
    parity_seconds=time.perf_counter()-start
    initial={n:p.detach().cpu().clone() for n,p in named}
    grads={n:0. for n,p in named}; losses=[]; timings=[]
    for epoch in range(epochs):
        if not(informative2 or informative3):
            losses.append(0.); print('[update] no informative trajectories; no optimizer step',flush=True); break
        while True:
            opt.zero_grad(set_to_none=True); epoch_start=time.perf_counter(); done=0; total=0.
            if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            print(f'[update] epoch={epoch+1}/{epochs} START decisions={count} decision_batch={size}',flush=True)
            try:
                for rows in chunks(records,size):
                    t=time.perf_counter()
                    out=replay_batch(r,rows,device)
                    if str(device).startswith('cuda'): torch.cuda.synchronize()
                    forward=time.perf_counter()-t
                    loss=loss_batch(out,rows)
                    loss.backward()
                    value=float(loss.detach())
                    if not math.isfinite(value): raise RuntimeError('Nonfinite loss; optimizer not stepped')
                    total+=value; done+=len(rows)
                    elapsed=time.perf_counter()-epoch_start
                    peak=torch.cuda.max_memory_allocated()/1e9 if str(device).startswith('cuda') else 0.
                    timing=dict(batch=len(rows),forward=forward,total=time.perf_counter()-t)
                    timings.append(timing)
                    print(f'[update] epoch={epoch+1}/{epochs} decisions={done}/{count} batch={len(rows)} '
                          f'forward_s={forward:.2f} batch_s={timing["total"]:.2f} seconds={elapsed:.1f} '
                          f'peak_allocated_GB={peak:.2f}',flush=True)
                    del out,loss
                break
            except torch.OutOfMemoryError:
                # Partial gradients are discarded; restart the SAME epoch before
                # any optimizer step. No samples are dropped and weights stay fixed.
                if size<=1: raise
                out=None; loss=None; opt.zero_grad(set_to_none=True)
                size=max(1,size//2); gc.collect(); torch.cuda.empty_cache()
                print(f'[update] OOM: discarded partial gradients; restarting epoch batch={size}',flush=True)
        norms=torch.stack([p.grad.detach().norm() if p.grad is not None else p.new_zeros(()) for n,p in named]).cpu().tolist()
        for (n,p),norm in zip(named,norms): grads[n]=max(grads[n],norm)
        torch.nn.utils.clip_grad_norm_([p for n,p in named],1.,error_if_nonfinite=True)
        opt.step(); losses.append(total)
        print(f'[update] epoch={epoch+1}/{epochs} DONE seconds={time.perf_counter()-epoch_start:.1f}',flush=True)
    audit={n:dict(grad_norm=grads[n],max_weight_change=float((p.detach().cpu()-initial[n]).abs().max())) for n,p in named}
    return dict(loss=losses,decisions=count,parity=parity if check else None,parity_checked=check,
        parity_seconds=parity_seconds,backward_and_audit_seconds=time.perf_counter()-start-parity_seconds,
        parameters=audit,decision_batch_effective=size,batch_timings=timings,
        pair_candidates=sum(int(s['pair_mask'].sum()) for s in records),
        pair_actions=sum(bool(s['selected_is_pair']) for s in records),
        nonzero_advantages=sum(bool(s['adv2'] or s['adv3']) for s in records))
