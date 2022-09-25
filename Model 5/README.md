# Model 5 - Multi-Sense Words Improvement Model

To run this model for the **Maltese - English** language pair, the following scripts need to be run first, to train the subwords tokenizer, train the baseline model (to learn to keep ``<MASK>`` in the translation), to train the model and finally to evaluate the model.

```console
foo@bar:~$ ./install.sh
foo@bar:~$ ipython subwords_tokenizer.py
foo@bar:~$ ipython train_baseline.py
foo@bar:~$ ipython create_updated_data.py
foo@bar:~$ ipython create_updated_data_step3.py
foo@bar:~$ ipython transformer.py
foo@bar:~$ ./evaluate.sh
```

Similarly, to run this model for the **Icelandic - English** language pair, the following scripts need to be run:

```console
foo@bar:~$ ./install.sh
foo@bar:~$ ipython subwords_tokenizer-ic.py
foo@bar:~$ ipython train_baseline-ic.py
foo@bar:~$ ipython create_updated_data_en_ic.py
foo@bar:~$ ipython create_updated_data_step3_ic.py
foo@bar:~$ ipython transformer-ic.py
foo@bar:~$ ./evaluate-ic.sh
```


``evaluate.sh`` and ``evaluate-ic.sh`` all assume that the code is being run in a [SLURM](https://slurm.schedmd.com/documentation.html) environment. These bash scripts can be configured otherwise, without affecting the actual python files. 

To change the direction of the MT model (such as from Maltese -> English to English -> Maltese), one can change the constant variables at the top of the python files.

