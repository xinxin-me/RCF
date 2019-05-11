# RCF 
This is the implementation of paper:


>Xin Xin, Xiangnan He, Yongfeng Zhang, Yongdong Zhang and Joemon Jose (2019). [Relational Collaborative Filtering: Modeling Multiple Item Relations for Recommendation](https://arxiv.org/abs/1904.12796).

Please note that this code may be slow, but it' not the problem of the algorithm. At this moment, the code spends much time to generate training batch.

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{RCF,
  author    = {Xin Xin and
               Xiangnan He and
               Yongfeng Zhang and
               Yongdong Zhang and
               Joemon Jose},
  title     = {Relational Collaborative Filtering: Modeling Multiple Item Relations for Recommendation},
  booktitle = {{SIGIR}},
  year      = {2019}
}
```
## Environemnt
Tensorfow with python 2.7


## Dataset
We provide two processed datasets: ML100K and KKBOX
* `train.txt`
  * Train file.
  * Each line is a user with one of her/his interaced items: (`userID` and `itemID`).
  
* `test.txt`
  * Test file (positive instances).
  * Same format with train.txt
  
* `test_negative.txt`
  * Test file (for KKBOX).
  * For KKBOX, the ranking is performed between 1 postive instance vs 999 negative instances
  
* `auxiliary-mapping.txt`
  * For ML100K, itemID|genreIDs|directorIDs|actorsIDs|.
  * For KKBOX, itemID|genreIDs|singerIDs|composerIDs|lyricistIDs
