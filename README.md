# Quora Insincere Questions Kaggle Competiton

## To submit

Run the script `generate_script.sh`. This will generate a file `run.py`. Copy and paste this file into a script kernel 
to submit it to the competition. This relies on the `stickytape` python library.

Note: I am not able to run the generated script except by "committing" it. Highlighting the 
code and clicking the interactive run button results in a `Module Not Found` error.

## To automatically submit random search hyperparameter configurations as kernels

Configure the search space by manually editing `run_kernel_hp_search.py` to define
hyperparameter options/ranges for the search. Set the `search_size` to define how many 
random configurations to try. 

Configure `kernel-metadata.json` to point to the kernel you would like to push each 
configuration to. You must have your local environment configured to support this push. 
See the kaggle api documentation for instructions.

When finished, run `run_kernel_hp_search.py` as a script to launch the kernels as 
commit versions of the kernel specified in `kernel-metadata.json`.

For now, results must be manually collected and analysed using the kaggle website.