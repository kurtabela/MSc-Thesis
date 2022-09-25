# Model 1: Fairseq Baseline Implementation Model

To train this model, one needs to run the preprocessing script and the training & evaluation script. 
```console
foo@bar:~$ ./install.sh
foo@bar:~$ ./preprocessing.sh mt en  data/ results/ 
foo@bar:~$ ./run_baseline_system.sh mt en data/ results/ 10
```

Note, that these can be combined into one bash script, however they were kept separate as the preprocessing does not require any GPUs. This scenario is helpful if one is running this in a [SLURM](https://slurm.schedmd.com/documentation.html) cluster or similar, to not hog GPUs for the (potentially) long preprocessing time. 

The parameters passed are the following, in order:

1. Target Language
2. Source Language
3. Data Directory
4. Save Directory
5. Number of Epochs

These files can handle both Maltese and Icelandic translations, one simply needs to pass `ic` or `mt` for Icelandic or Maltese respectively.