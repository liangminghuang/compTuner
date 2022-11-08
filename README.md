# Compiler Auto-tuning through Multiple Phase Learning

In this repository, we provide our code and the data.

## Environment

- Python == 3.7.3
- Numpy == 1.14.3
- sklearn == 0.19.1

## Dataset

The folder `benchmarks` contains all the programs.

The folder `cbench` contains the programs of cBench.

The folder `polybench` contains the programs of PolyBench.

## Algorithm

The folder `algorithm` contains all the code we used.

 `boca.py` is the BOCA algorithm.

 `ga.py` is the GA algorithm.

 `rio.py` is the Random Iteration Optimization algorithm.

 `tpe.py` is the TPE algorithm.

 `CompTuner.py` is our proposed  **CompTuner** algorithm.

We add a README file in the `algorithm` folder to help you understand and run the programs.

## Result

The folder `results` contains all the results of (speedup and time consumption for five techniques).

We add a README file in the `result` folder to help you understand the source data and result data for our expriments.

## Run

- In order to tune `gcc`'s optimization for program `benchmarks/cbench/automotive_bitcount`, execute the following command:

```
python3 runCompTuner.py --bin-path (your gcc location) --driver (your gcc driver) --linker (your gcc linker) --src-dir 'benchmarks/cbench/automotive_bitcount' --execute-params 20
```

- In order to tune `llvm`'s optimization sequence for program `benchmarks/polybench/3mm`, execute the followoing:

```
python3 runCompTuner.py --bin-path (your llvm location) --driver (your llvm driver) --linker (your llvm linker) --src-dir (3mm location) --libs '-I (resource for 3mm)
```

## Note

Different versions of compilers use different compilation commands, so please pay attention to modify the relevant statements.