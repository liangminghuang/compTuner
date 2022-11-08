The folder `Result for overview effectives`  contains the results for our Table **Results of Compiler Auto-tuning Techniques** 

-  `results.txt` presents five techniques' results for 20 programs (10 from PolyBench and 10 from cBench) on GCC and LLVM. For each program, we first presents the five techniques' speedup of 3 experiments and time consumption for each speedup.

-  `Results of Compiler Auto-tuning Techniques.pdf` presents the Table. For our technique **CompTuner** we presents the **median** result (speedup - time consumption)  of each program on gcc or llvm. For other techniques, if it can reach the similar speedup as  **CompTuner**  in fixed time on a program, we present its time consumption, otherwise, we present as ❌.

The folder `Result for ablation study`  contains the results for our **Ablation Study**

-  `results_for_speedup.txt` presents $CompTuner's, CompTuner_{high}'s, CompTuner_{impact}'s$ speedup performance on P1, P2, P3, C1, C2, C3 of three experiments. We use the mean value of three experiments as the result of Table 5 in our paper.

-  `results_for_prediction_err.txt` presents $CompTuner's, CompTuner_{high}'s$ prediction error on P1, P2, P3, C1, C2, C3. We use the mean value of the 50 test samples' prediction error as the result of Figure 2 and Figure 3 in our paper.

