"""A fixed-ID training-search ruler; retiring a graph never erases its gain."""
from __future__ import annotations


class FixedCohortMetrics:
    def __init__(self, graphs, saved=None):
        initial = {g.iid: [int(g.s0_schedule.makespan), int(g.s0_schedule.makespan)]
                   for g in graphs}
        if len(initial) != len(graphs):
            raise ValueError("fixed cohort requires unique instance IDs")
        self.rows = ({k: list(v) for k, v in saved.items()} if saved is not None
                     else initial)
        if set(self.rows) != set(initial):
            raise ValueError("fixed cohort changed across resume")
        if any(v[0] != initial[k][0] or not 0 <= v[1] <= v[0]
               for k, v in self.rows.items()):
            raise ValueError("fixed cohort baseline/best mismatch")

    def update(self, arena):
        for ag in arena:
            if ag.iid in self.rows:
                row = self.rows[ag.iid]
                if row[0] != int(ag.s0_schedule.makespan):
                    raise ValueError("same instance ID has a different S0")
                row[1] = min(row[1], int(ag.best.ms))

    def state_dict(self):
        return {k: list(v) for k, v in self.rows.items()}

    def metrics(self):
        gains = [s0-best for s0,best in self.rows.values()]
        n = len(gains)
        return {
            "sum_best_makespan_reduction_all_graphs": float(sum(gains)),
            "mean_best_makespan_reduction_per_graph": sum(gains)/max(n,1),
            "improved_graph_rate": sum(g>0 for g in gains)/max(n,1),
            "mean_best_makespan_reduction_pct": sum(
                100*(s0-best)/max(s0,1) for s0,best in self.rows.values())/max(n,1)}

    def publish(self, writer, step, arena):
        if writer is None:
            return
        for name,value in self.metrics().items():
            writer.add_scalar("cumulative/"+name,value,step)
        writer.add_scalar("fixed_cohort/n_instances",len(self.rows),step)
        writer.add_scalar("fixed_cohort/n_still_active",
                          sum(ag.iid in self.rows for ag in arena),step)
        writer.flush()
