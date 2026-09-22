"""Exact encoding-only path; disconnected graphs share GAT/Gantt kernels.

Unused legacy proposal decoders and SVD diagnostics are intentionally not run.
No active encoding parameter is frozen or replaced.
"""
from types import SimpleNamespace
import torch
from torch.nn.utils.rnn import pad_sequence
from ..sg_sct_model_v1 import _attention_pool, _scatter_mean


def pack(batches, device):
    fields = ('node_numeric','node_type','edge_type','edge_role','edge_features',
              'symptom_node_mask','candidate_node_mask','gantt_numeric',
              'gantt_machine_index','gantt_job_index','gantt_time_index',
              'appearance_type','appearance_features')
    out = {k: torch.cat([getattr(b,k) for b in batches]).to(device) for k in fields}
    nodes, blocks, edges, ng, gg, go, bn, nb, masks = [], [], [], [], [], [], [], [], []
    no = bo = 0
    for i,b in enumerate(batches):
        n = len(b.node_numeric); count = b.symptom_block_count
        nodes.append(n); blocks.append(count)
        edges.append(b.edge_index+no)
        ng.append(torch.full((n,),i,dtype=torch.long))
        gg.append(torch.full((len(b.gantt_numeric),),i,dtype=torch.long))
        go.append(torch.where(b.gantt_operation_node>=0,b.gantt_operation_node+no,b.gantt_operation_node))
        bn.append(b.symptom_block_node_index+b.symptom_block_node_index.new_tensor([[bo],[no]]))
        nb.append(torch.where(b.node_symptom_block>=0,b.node_symptom_block+bo,b.node_symptom_block))
        masks.append(b.reverse_causal_mask if b.reverse_causal_mask is not None else
                     ((b.edge_type==7)|(b.edge_type==8)))
        no += n; bo += count
    for k,items,axis in [('edge_index',edges,1),('node_graph',ng,0),('gantt_graph',gg,0),
                         ('gantt_operation_node',go,0),('symptom_block_node_index',bn,1),
                         ('node_symptom_block',nb,0),('reverse_causal_mask',masks,0)]:
        out[k]=torch.cat(items,dim=axis).to(device)
    out['symptom_block_count']=bo
    return SimpleNamespace(**out),nodes,blocks


def encode_many(model, batches, device):
    batch, sizes, block_sizes = pack(batches, device)
    count=len(batches)
    nodes0=model.node_projection(batch.node_numeric,batch.node_type)
    context=nodes0; cm=(batch.edge_role==0)|(batch.edge_role==1)
    for layer in model.context_layers:
        context,_,_=layer(context,batch.edge_index,batch.edge_type,batch.edge_features,cm)
    causal=nodes0; hard=batch.edge_role==2; soft=batch.edge_role==3
    for layer in model.causal_layers:
        causal,_,_=layer(causal,batch.edge_index,batch.edge_type,batch.edge_features,
                         hard|soft,hard_edge_mask=hard,soft_edge_mask=soft)
    ge=model.gantt_encoder
    hidden=ge.numeric(batch.gantt_numeric)
    if ge.use_identity_embeddings:
        if ge.use_machine_identity_embedding:
            hidden=hidden+ge.machine(batch.gantt_machine_index)
        hidden=hidden+ge.job(batch.gantt_job_index)+ge.time(batch.gantt_time_index)
    lengths=[len(b.gantt_numeric) for b in batches]
    if any(n==0 for n in lengths):
        raise ValueError('E2E packed encoder requires nonempty scheduling sequences')
    padded=pad_sequence(list(hidden.split(lengths)),batch_first=True)
    valid=torch.arange(padded.shape[1],device=device)[None]<torch.tensor(lengths,device=device)[:,None]
    encoded=ge.encoder(padded,src_key_padding_mask=~valid)
    gantt=ge.norm(torch.cat([encoded[i,:n] for i,n in enumerate(lengths)]))
    fused,_=model._align_graph_and_gantt(.5*(context+causal),gantt,batch,count)
    state=model.state_query(model._masked_graph_mean(
        fused,batch.node_graph,batch.symptom_node_mask.bool(),count))
    bi,ni=batch.symptom_block_node_index
    bc=batch.symptom_block_count
    if bc:
        # Preserve the legacy pooling's local indexing exactly. Its implementation
        # indexes weights by block IDs, so concatenating block IDs changes the
        # computation. Keep this small reduction per graph; GAT/Transformer stay batched.
        pools=[]; no=0
        for raw,n,b in zip(batches,sizes,block_sizes):
            ids=raw.symptom_block_node_index.to(device)
            pools.append(_attention_pool(fused[no+ids[1]],ids[0],b,model.block_attention))
            no+=n
        h=torch.cat(pools)
        qa=model.appearance_query_mlp(torch.cat([h,model.appearance_type_embedding(batch.appearance_type),
                                                model.appearance_feature_encoder(batch.appearance_features)],-1))
        qa=model._refine_block_state(qa,batch)
        nb=batch.node_symptom_block
        query=torch.where((nb>=0)[:,None],qa[nb.clamp_min(0)],state[batch.node_graph])
    else:
        qa=fused.new_empty((0,fused.shape[-1])); query=state[batch.node_graph]
    hc=fused
    for i,layer in enumerate(model.reverse_layers):
        delta,_,_=layer(hc,batch.edge_index.flip(0),batch.edge_type,batch.edge_features,
                       batch.reverse_causal_mask)
        gate=torch.sigmoid(model.reverse_gate[i](hc))
        hc=gate*delta+(1-gate)*hc+model.reverse_query(query)
    hi=fused; im=model._intervention_mask(batch)
    # The existing path bypasses the layer if a graph has no intervention edges.
    has_inter=[bool(model._intervention_mask(b).any()) for b in batches]
    active=torch.repeat_interleave(torch.tensor(has_inter,device=device),torch.tensor(sizes,device=device))
    for i,layer in enumerate(model.reverse_layers):
        delta,_,_=layer(hi,batch.edge_index,batch.edge_type,batch.edge_features,im)
        delta=torch.where(active[:,None],delta,hi)
        gate=torch.sigmoid(model.reverse_gate[i](hi))
        hi=gate*delta+(1-gate)*hi+model.reverse_query(query)
    result=[]; no=bo=0
    for n,b in zip(sizes,block_sizes):
        c=hc[no:no+n]; inter=hi[no:no+n]; q=qa[bo:bo+b]
        cb=c[None].expand(b,-1,-1); ib=inter[None].expand(b,-1,-1)
        gate=torch.sigmoid(model.dual_fusion(torch.cat([cb,ib,q[:,None].expand(-1,n,-1)],-1)))
        result.append((gate*cb+(1-gate)*ib,c,q,b))
        no+=n; bo+=b
    return result
