# CNN_sentence.pytorch
A PyTorch implementation of CNNs for sentence classification.  
- The original source is located at [yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence).
- Preprocessed data files located in `resource/data` were borrowed from  [harvardnlp/sent-conv-torch](https://github.com/harvardnlp/sent-conv-torch).

## Dependencies

- Download word2vec binary file, `GoogleNews-vectors-negative300.bin`, from https://code.google.com/archive/p/word2vec/
  and paste it under the `./resource` folder.
- pytorch:1.2, cuda10.0, and cudnn7 were used. Please refer `docker/dockerfile` for more details.


## Train and eval
1. Turn visdom on
    ```cmd
    CNN_sentence.pytorch\scripts>visdom
    ```
2. Run script files
    ```cmd
    CNN_sentence.pytorch\scripts>MR.bat
    CNN_sentence.pytorch\scripts>SST.bat
    CNN_sentence.pytorch\scripts>Subj.bat
    CNN_sentence.pytorch\scripts>TREC.bat
    CNN_sentence.pytorch\scripts>CR.bat
    CNN_sentence.pytorch\scripts>MPQA.bat
    ```
