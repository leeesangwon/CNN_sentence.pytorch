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

## Results

### Quantitative results
|  Data   |     | CNN-rand | CNN-staic | CNN-nonstatic | CNN-multichannel |
| :-----: |-----| :------: | :-------: | :-----------: | :--------------: |
|**MR**   |Paper|     76.1 |      81.0 |          81.5 |             81.1 |
|     |Reproduce|     72.9 |      77.9 |          79.2 |             78.6 |
|**Subj** |Paper|     89.6 |      93.0 |          93.4 |             93.2 |
|     |Reproduce|     88.1 |      92.1 |          92.6 |             92.5 |
|**CR**   |Paper|     79.8 |      84.7 |          84.3 |             85.0 |
|     |Reproduce|     77.1 |      82.3 |          82.0 |             82.0 |
|**MPQA** |Paper|     83.4 |      89.6 |          89.5 |             89.4 |
|     |Reproduce|     81.8 |      86.8 |          86.8 |             87.1 |
|**SST-1**|Paper|     45.0 |      45.5 |          48.0 |             47.4 |
|     |Reproduce|     36.2 |      40.9 |          43.0 |             39.7 |
|**SST-2**|Paper|     82.7 |      86.8 |          87.2 |             88.1 |
|     |Reproduce|     79.0 |      81.9 |          82.0 |             81.7 |
|**TREC** |Paper|     91.2 |      92.8 |          93.6 |             92.2 |
|     |Reproduce|     88.6 |      89.6 |          91.0 |             90.8 |

* Performance degradation was observed.
* **TODO** Finding the reason of the degradation and Fixing it. 
  
### Top 4 neighboring words
|        | Static     |            | Non-static |            |
|:------:|:------:    |:---------: |:----------:|:---------: |
|        | Paper      | Reproduce  | Paper      | Reproduce  |
|        |            |            |            |            |
|**bad** | *good*     | *good*     | *terrible* | *horrible* |
|        | *terrible* | *terrible* | *horrible* | *lousy*    |
|        | *horrible* | *horrible* | *lousy*    | *terrible* |
|        | *lousy*    | *lousy*    | *stupid*   | *dreadful* |
|        |            |            |            |            |
|**good**| *great*    | *great*    | *nice*     | *great*    |
|        | *bad*      | *bad*      | *decent*   | *terrific* |
|        | *terrific* | *terrific* | *solid*    | *nice*     |
|        | *decent*   | *decent*   | *terrific* | *decent*   |
|        |            |            |            |            |
|**n't** | *os*       | *os*       | *not*      | *os*       |
|        | *ca*       | *ca*       | *never*    | *never*    |
|        | *ireland*  | *ireland*  | *nothing*  | *ireland*  |
|        | *wo*       | *wo*       | *neither*  | *lisa*     |
|        |            |            |            |            |
| **!**  | *2,500*    | *fussing*  | *2,500*    | *fussing*  |
|        | *entire*   | *moaning*  | *lush*     | *talking*  |
|        | *jez*      | *weeping*  | *beautiful*| *ticking*  |
|        | *changer*  | *deepa*    | *terrific* | *tartly*   |
|        |            |            |            |            |
| **,**  | *decasia*  |*numbingly*    | *but*   | *but*      |
|        | *abysmally*|*exhaustingly* | *dragon*| *the*      |
|        | *demise*   |*maddeningly*  | *a*     | *a*        |
|        | *valiant*  |*disquietingly*| *and*   | *thrillers*|