# Quora Insincere Questions Kaggle Competiton

## To submit

Run the script `generate_script.sh`. This will generate a file `run.py`. Copy and paste this file into a script kernel 
to submit it to the competition. This relies on the `stickytape` python library.

Note: I am not able to run the generated script except by "committing" it. Highlighting the 
code and clicking the interactive run button results in a `Module Not Found` error.

## To automatically submit random search hyperparameter configurations as kernels

Configure the search space by manually editing `run_kernel_hp_search.py` to define
hyperparameter options/ranges for the search. This involves defining a list of lists 
of model configurations, `config_list`. Each element of this list is a list of models 
specified as a class and associated hyperparameters. The models in the inner lists will be 
trained in a single kernel and ensembled (i.e. they are a single submission.). For each 
inner list of models, we will do a random search through non-model hyperparameters.
Set the `search_size` to define how many 
random configurations to try for each ensemble. The total number of kernels launched 
will be `len(config_list) * search_size`.

Configure `kernel-metadata.json` to point to the kernel you would like to push each 
configuration to. You must have your local environment configured to support this push. 
See the kaggle api documentation for instructions.

When finished, run `run_kernel_hp_search.py` as a script to launch the kernels as 
commit versions of the kernel specified in `kernel-metadata.json`.

Kaggle's API doesn't support downloading output from past versions of kernels, so 
the results must be manually inspected through the 