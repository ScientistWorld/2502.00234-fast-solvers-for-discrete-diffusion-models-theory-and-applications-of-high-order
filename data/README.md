# data/

No data files are needed.  Problem instances, exact reverse-process
intensities and ground truth are generated analytically by the evaluators
(`eval/train/` for the public slice, `eval/test/` for the hidden slice) at
evaluation time; solvers reach an instance only through `problem.rates()`.
