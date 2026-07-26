# Round 6 Incremental Additions

Additive dynamic and distributed scheduling knowledge for the existing multi-view candidate graph.

This package intentionally does not create separate candidate stores by composite problem label. Merge canonical additions by `metric_id`/`diagnostic_id`, append view relations, and deduplicate retrieval results by candidate ID.
