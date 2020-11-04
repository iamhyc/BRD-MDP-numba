# MDP with Partial Information via Broadcast Information Sharing (numba version)

> No details provided now.

### Lesson to Learn

1) Always `assert` when you are not sure;

2) Range Check Assertion: $(a,b), [a,b], [a,b), (a,b]$;

3) When initialize array: `argmin` with `np.full`, `argmax` with `np.zeros`.

### Todo

- [x] try `fastmath` option in `@njit` (no performance difference)
- [x] commit a `main_one_shot` function, with reduced and formatted output
    - ~~move all related traces under same folder~~
    - ~~recording somehow per-stage~~
    - ~~complete `main_one_shot` function~~
- [x] add static policy replacement function
- [x] test policy replacement (with 50 submission)
    - ~~running on vps and get 50 results~~
- [ ] semi-analytical average cost calculation
    - ~~one-step/n-step policy improvement for any stage~~ (n < STAGE_EVAL)
    - Possible: enhance one evaluation with multi-step policy improvement's return?
- [ ] finish two simple analysis
    - reinforcement learning
    - optimized baseline policy (aware of start)
- [ ] touch a `plot-traces2.py` with new plot function
    - calculate the record data and display