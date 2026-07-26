# Unresolved Questions — Dynamic and Distributed Additions

1. Which factory capacity normalizer should be used for heterogeneous factories: nominal machine time, calendar-adjusted capacity, or lower-bound load?
2. Does the project permit whole-job factory assignment only, or operation-level transfers between factories?
3. Are transfer-ready, departure, arrival and downstream-release timestamps available as distinct events?
4. Does event detection use simulation time, MES event time or wall-clock ingestion time?
5. Is the rescheduling trigger timestamp separate from solver start and revised-schedule publication?
6. How is the affected-operation closure constructed across machine, precedence, transport, blocking and auxiliary-resource edges?
7. Should new-job response delay be weighted by urgency or due-date class in any project view?
8. Under heterogeneous factories, when is factory workload imbalance meaningful rather than a misleading equality target?
