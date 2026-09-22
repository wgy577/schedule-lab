"""Unified learned delay-responsibility propagation for M2.

The output is a policy distribution over intervention nodes, not an identified
causal posterior. Appearance-specific seeds, conditional edge weights and depth
preference are trained end to end from the same global net-makespan return.
"""
from __future__ import annotations

import torch
from torch import nn


TRACE_MAX_HOPS = 4


def build_trace_graph(ast, ops, max_hops=None):
    """Retain bounded ancestors while preserving each appearance's identity."""
    if max_hops is None:
        max_hops = TRACE_MAX_HOPS
    appearances = []
    for block_id, raw_seeds in ast.get("trace_appearance_seeds", ()):
        seeds = tuple(sorted({node for node in raw_seeds
                              if node in ast["node_index"]}))
        if seeds:
            appearances.append((str(block_id), seeds))
    seed_union = {node for _, seeds in appearances for node in seeds}
    incoming = {}
    for cause, effect in ast.get("trace_causal_edges", ()):
        incoming.setdefault(effect, set()).add(cause)
    seen, frontier = set(seed_union), set(seed_union)
    for _ in range(max_hops):
        frontier = {u for v in frontier for u in incoming.get(v, ())
                    if u in ast["node_index"]} - seen
        seen.update(frontier)
    nodes = sorted(seen)
    index = {node: i for i, node in enumerate(nodes)}
    edges = sorted({(index[effect], index[cause]) for effect in nodes
                    for cause in incoming.get(effect, ()) if cause in index})
    outgoing = {source for source, _ in edges}
    edges.extend((i, i) for i in range(len(nodes)) if i not in outgoing)
    seed_nodes, seed_groups = [], []
    for group, (_block_id, seeds) in enumerate(appearances):
        for node in seeds:
            seed_nodes.append(index[node])
            seed_groups.append(group)
    h = ast["h_c"].detach().float()
    return {
        "node_indices": tuple(ast["node_index"][node] for node in nodes),
        "nodes": h[[ast["node_index"][node] for node in nodes]],
        "edges": torch.tensor(edges, dtype=torch.long).reshape(-1, 2),
        "seeds": torch.tensor([node in seed_union for node in nodes],
                              dtype=torch.bool),
        "seed_nodes": torch.tensor(seed_nodes, dtype=torch.long),
        "seed_groups": torch.tensor(seed_groups, dtype=torch.long),
        "appearance_ids": tuple(block_id for block_id, _ in appearances),
        "n_appearances": len(appearances),
        "roots": torch.tensor([index.get(f"operation:{op}", -1) for op in ops]),
    }


def propagate(q, edges, weights, hops):
    """Apply total-probability transitions without enumerating reverse paths."""
    levels = [q]
    src, dst = edges.unbind(1)
    for layer in range(hops):
        weight = weights if weights.dim() == 1 else weights[layer]
        q = torch.zeros_like(q).index_add(0, dst, q[src] * weight)
        levels.append(q)
    return torch.stack(levels)


def _segment_softmax(logits, owner, segments):
    maxima = logits.new_full((segments,), -torch.inf).scatter_reduce(
        0, owner, logits, reduce="amax", include_self=True)
    exp = (logits - maxima[owner]).exp()
    denom = logits.new_zeros(segments).index_add(0, owner, exp)
    return exp / denom[owner].clamp_min(1e-12)


def _seed_layout(graph, device):
    """Return seed entries and appearance owners, including K compatibility."""
    seed_nodes = graph.get("seed_nodes")
    seed_groups = graph.get("seed_groups")
    if seed_nodes is not None and len(seed_nodes):
        groups = seed_groups.to(device).long()
        count = int(graph.get("n_appearances", int(groups.max()) + 1))
        return seed_nodes.to(device).long(), groups, count
    seeds = graph.get("seeds")
    if seeds is not None:
        nodes = seeds.to(device).bool().nonzero(as_tuple=False).reshape(-1)
        if len(nodes):
            return nodes, torch.arange(len(nodes), device=device), len(nodes)
    return (torch.zeros(1, device=device, dtype=torch.long),
            torch.zeros(1, device=device, dtype=torch.long), 1)


class ProbabilityTrace(nn.Module):
    """Unified intervention-node distribution learned from net makespan."""

    def __init__(self, latent_dim, state_dim, hops=4, hop_dim=8):
        super().__init__()
        self.hops = int(hops)
        self.hop_dim = int(hop_dim)
        self.hop_embedding = nn.Embedding(self.hops, self.hop_dim)
        self.edge = nn.Sequential(
            nn.Linear(3 * latent_dim + state_dim + self.hop_dim, 32),
            nn.Tanh(), nn.Linear(32, 1))
        self.seed = nn.Linear(latent_dim, 1)
        self.appearance_potential = nn.Sequential(
            nn.Linear(latent_dim + state_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.depth = nn.Linear(latent_dim + state_dim, self.hops + 1)
        # T2-M starts causal propagation as a weak prior (~0.10) and lets GRPO
        # raise it only when real future-best makespan credit supports it.
        # Starting near one made an untrained trace dominate frozen B5 evidence.
        self.strength = nn.Parameter(torch.tensor(-2.2))

    def _transition_weights(self, x, edges, context, states, owner):
        src, dst = edges.unbind(1)
        rows = []
        for layer in range(self.hops):
            hop = self.hop_embedding.weight[layer].expand(len(src), -1)
            edge_input = torch.cat((x[src], x[dst], context[owner[src]],
                                    states[owner[src]], hop), 1)
            logits = self.edge(edge_input).squeeze(-1)
            rows.append(_segment_softmax(logits, src, len(x)))
        return torch.stack(rows)

    def distribution(self, graph, state):
        device = self.strength.device
        x = graph["nodes"].to(device).float()
        roots = graph["roots"].to(device).long()
        if len(x) == 0:
            return x.new_full((len(roots),), 1.0 / max(len(roots), 1)), {}
        edges = graph["edges"].to(device).long()
        state = state.detach().to(device).float().reshape(1, -1)
        seed_nodes, seed_groups, n_app = _seed_layout(graph, device)
        app_sum = x.new_zeros((n_app, x.shape[1])).index_add(
            0, seed_groups, x[seed_nodes])
        app_count = x.new_zeros(n_app).index_add(
            0, seed_groups, torch.ones(len(seed_nodes), device=device))
        app_ctx = app_sum / app_count[:, None].clamp_min(1)
        app_logits = self.appearance_potential(torch.cat(
            (app_ctx, state.expand(n_app, -1)), 1)).squeeze(-1)
        app_weight = torch.softmax(app_logits, 0)
        seed_weight = _segment_softmax(
            self.seed(x[seed_nodes]).squeeze(-1), seed_groups, n_app)
        q0 = x.new_zeros(len(x)).index_add(
            0, seed_nodes, app_weight[seed_groups] * seed_weight)
        context = (app_weight[:, None] * app_ctx).sum(0, keepdim=True)
        owner = torch.zeros(len(x), device=device, dtype=torch.long)
        transitions = self._transition_weights(x, edges, context, state, owner)
        levels = propagate(q0, edges, transitions, self.hops)
        depth = torch.softmax(self.depth(torch.cat((context[0], state[0]))), 0)
        mass = (depth[:, None] * levels).sum(0)
        valid = roots >= 0
        p = mass[roots.clamp_min(0)] * valid.to(mass.dtype) + 1e-6
        p = p / p.sum().clamp_min(1e-12)
        entropy = -(app_weight * app_weight.clamp_min(1e-12).log()).sum()
        return p, {"levels": levels, "transition": transitions, "depth": depth,
                   "appearance_weight": app_weight,
                   "appearance_entropy": entropy,
                   "n_appearances": n_app}

    def forward(self, graph, state):
        p, _ = self.distribution(graph, state)
        logp = p.clamp_min(1e-12).log()
        return torch.sigmoid(self.strength) * (logp - logp.mean())

    def batch_bias(self, queries, states, width, *, end_to_end=False):
        """Vectorized disjoint-graph evaluation of the unified distribution."""
        device = self.strength.device
        xs, edges_all, roots_all, owners, root_owners = [], [], [], [], []
        seed_nodes_all, seed_apps_all, app_owners = [], [], []
        counts, node_offset, app_offset = [], 0, 0
        for graph_index, (graph, ids) in enumerate(queries):
            if graph is None:
                raise ValueError("missing trace graph in GPU batch")
            x = graph["nodes"].to(device).float()
            if len(x) == 0:
                x = states.new_zeros((1, self.seed.in_features))
                edges = torch.zeros((1, 2), device=device, dtype=torch.long)
                seed_nodes = torch.zeros(1, device=device, dtype=torch.long)
                seed_groups = torch.zeros(1, device=device, dtype=torch.long)
                n_app = 1
            else:
                edges = graph["edges"].to(device).long()
                seed_nodes, seed_groups, n_app = _seed_layout(graph, device)
            roots = graph["roots"].to(device).long()
            xs.append(x if end_to_end else x.detach())
            edges_all.append(edges + node_offset)
            roots_all.append(torch.where(roots >= 0, roots + node_offset, roots))
            owners.append(torch.full((len(x),), graph_index, device=device,
                                     dtype=torch.long))
            root_owners.append(torch.full((len(roots),), graph_index, device=device,
                                          dtype=torch.long))
            seed_nodes_all.append(seed_nodes + node_offset)
            seed_apps_all.append(seed_groups + app_offset)
            app_owners.append(torch.full((n_app,), graph_index, device=device,
                                         dtype=torch.long))
            counts.append(len(roots))
            node_offset += len(x)
            app_offset += n_app
        x, edges = torch.cat(xs), torch.cat(edges_all)
        owner, root_owner = torch.cat(owners), torch.cat(root_owners)
        roots = torch.cat(roots_all)
        seed_nodes, seed_apps = torch.cat(seed_nodes_all), torch.cat(seed_apps_all)
        app_owner = torch.cat(app_owners)
        batch_size, n_app = len(queries), len(app_owner)
        app_sum = x.new_zeros((n_app, x.shape[1])).index_add(
            0, seed_apps, x[seed_nodes])
        app_count = x.new_zeros(n_app).index_add(
            0, seed_apps, torch.ones(len(seed_nodes), device=device))
        app_ctx = app_sum / app_count[:, None].clamp_min(1)
        app_logits = self.appearance_potential(torch.cat(
            (app_ctx, states.detach()[app_owner]), 1)).squeeze(-1)
        app_weight = _segment_softmax(app_logits, app_owner, batch_size)
        seed_weight = _segment_softmax(
            self.seed(x[seed_nodes]).squeeze(-1), seed_apps, n_app)
        q0 = x.new_zeros(len(x)).index_add(
            0, seed_nodes, app_weight[seed_apps] * seed_weight)
        context = x.new_zeros((batch_size, x.shape[1])).index_add(
            0, app_owner, app_weight[:, None] * app_ctx)
        transitions = self._transition_weights(
            x, edges, context, states.detach(), owner)
        levels = propagate(q0, edges, transitions, self.hops)
        depth = self.depth(torch.cat((context, states.detach()), 1)).softmax(1)
        mass = (levels * depth[owner].T).sum(0)
        p = mass[roots.clamp_min(0)] * (roots >= 0).to(mass.dtype) + 1e-6
        p = p / p.new_zeros(batch_size).index_add(
            0, root_owner, p)[root_owner].clamp_min(1e-12)
        logp = p.log()
        means = logp.new_zeros(batch_size).index_add(0, root_owner, logp)
        means = means / logp.new_tensor(counts).clamp_min(1)
        bias = torch.sigmoid(self.strength) * (logp - means[root_owner])
        output, offset = [], 0
        for count, (_graph, ids) in zip(counts, queries):
            selected = bias[offset:offset + count][ids]
            output.append(torch.nn.functional.pad(selected,
                                                   (0, width - len(ids))))
            offset += count
        return torch.stack(output)
