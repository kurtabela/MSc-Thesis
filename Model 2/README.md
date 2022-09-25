# Model 2 - Tensorflow Transformer Baseline

To run this model for the **Maltese - English** language pair, the following scripts need to be run first, to train the subwords tokenizer, to train the model and to evaluate the model.

```console
foo@bar:~$ ./install.sh
foo@bar:~$ ipython subwords_tokenizer.py
foo@bar:~$ ipython transformer.py
foo@bar:~$ ./evaluate.sh
```

Similarly, to run this model for the **Icelandic - English** language pair, the following scripts need to be run:

```console
foo@bar:~$ ./install.sh
foo@bar:~$ ipython subwords_tokenizer-ic.py
foo@bar:~$ ipython transformer-ic.py
foo@bar:~$ ./evaluate-ic.sh
```


``evaluate.sh`` and ``evaluate-ic.sh`` all assume that the code is being run in a [SLURM](https://slurm.schedmd.com/documentation.html) environment. These bash scripts can be configured otherwise, without affecting the actual python files. 

To change the direction of the MT model (such as from Maltese -> English to English -> Maltese), one can change the constant variables at the top of the python files.

