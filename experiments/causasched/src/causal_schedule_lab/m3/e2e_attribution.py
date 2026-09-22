"""Same B5 propagation, all appearance blocks evaluated together."""
import torch

def attribution(model,batch,c_prior,encoded):
    from ..b5_attribution import B5AttributionOutput, _RESOURCE_SEQUENCE_TYPE
    ha,hc,qa,bc=encoded; n=len(hc); dev=hc.device
    role=batch.edge_role==2; ei=batch.edge_index[:,role]; src,dst=ei
    ef=batch.edge_features[role]; et=batch.edge_type[role]; e=len(src)
    if not model._b5_use_physics: ef=torch.zeros_like(ef)
    re=model.b5_transmission_relation((et==_RESOURCE_SEQUENCE_TYPE).long())
    if not e or not bc:
        return B5AttributionOutput(per_block_candidate_score=hc.new_zeros((bc,n)),
            per_block_edge_transmission=hc.new_zeros((bc,e)),causal_edge_index=ei,
            used_appearance=model._b5_use_appearance,used_physics=model._b5_use_physics)
    app=(model.b5_appearance_proj(batch.appearance_features) if model._b5_use_appearance
         and batch.appearance_features is not None and batch.appearance_features.numel()
         else hc.new_zeros((bc,model._b5_appearance_ctx_dim)))
    rep=ha if model._b5_use_appearance else hc[None].expand(bc,-1,-1)
    base=torch.cat([rep[:,src],rep[:,dst],ef[None].expand(bc,-1,-1),re[None].expand(bc,-1,-1)],-1)
    if model._b5_use_adapter:
        logit=model.b5_base_head(base).squeeze(-1)
        if model._b5_use_appearance:
            key=model.b5_edge_key(torch.cat([ef,re],-1))
            modulation=model.b5_mod_head(torch.cat([app,model.b5_query_mod(qa)],-1))
            logit=logit+(key[None]*modulation[:,None]).sum(-1)
    else:
        logit=model.edge_transmission_head(torch.cat([base,app[:,None].expand(-1,e,-1)],-1)).squeeze(-1)
    transmission=torch.sigmoid(logit)
    bi,ni=batch.symptom_block_node_index
    mass=hc.new_zeros((bc,n)); mass[bi,ni]=1.
    c=c_prior.to(dev)[None]
    for _ in range(model._b5_hops):
        msg=transmission*mass[:,dst]*model._b5_gamma
        mass=mass+c*hc.new_zeros((bc,n)).index_add(1,src,msg)
    if batch.candidate_node_mask is not None: mass=mass*batch.candidate_node_mask[None].to(mass.dtype)
    return B5AttributionOutput(per_block_candidate_score=mass,
        per_block_edge_transmission=transmission,causal_edge_index=ei,
        used_appearance=model._b5_use_appearance,used_physics=model._b5_use_physics)
