We implement our technique and compared techniques in five python file.

`CompTuner.py` implements the ComTuer technique by **compTuner** class, and the *run* function is the main runtime function. We implements the function `build_RF_by_BOCA` for our ablation study 1 and the  function `search_by_impactful` for our ablation study 2. We have comment this part of the code, and you can experiment by browsing through the comments

`boca.py` implements the BOCA technique , and the *run* function is the main runtime function. 

`rio.py` implements the RIO technique , and the *run* function is the main runtime function. 

`ga.py` implements the RIO technique , and the *GA_main* function is the main runtime function. 

For the above four techniques, you need to use the entry file for method calls. We implement the entry file for CompTuner in the main directory `runCompTuner`, and the run process is to **declare the parameters - call the compTuner class - execute the *run* method**. For other techniques, you can use the same process for calls, just adjust the parameters and classes. 

`tpe.py` implements the RIO technique , and its main runtime function directly contains the ' \__main__'.  You can directly run it for expriment.

