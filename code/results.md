## Run 01
* HMModel
* train dl: OAAT HMDataset
* val dl: OAAT HMDataset
* Took ~30 min
Epoch 1 Loss: 4.414722630018788 : 100%|█| 86480/86480 [04:45<00:00, 302.99it
Epoch 1
Train Loss: 4.414722626092124
Validation Loss: 3.4684507963117723

Epoch 2 Loss: 1.7030972961049524 : 100%|█| 86480/86480 [04:26<00:00, 324.06i
Epoch 2
Train Loss: 1.703097292005175
Validation Loss: 3.164008537234519

* Run 01b
* MF
* train dl: OAAT HMDataset
* val dl: OAAT HMDataset
11:50 Start
12:40 End
val loss ~3.43

01c
1:21 Start
1:23 End epoch 1 val loss 3.468
epoch 2 val loss 3.16
1:29 epoch 3 val loss 3.10
1:33 epoch 4 val loss 3.09

## Run 02
15-17 secs per epoch
laptop
* MF
* train dl: User USDataset
* val dl: OAAT

70 runs

100 additional
12:54 Epcoh 130 val loss 3.60
1:00 Epoch 150 val loss 3.56
1:06 epoch 170 val loss 3.54
stop

50 additional
lr = .001
1:07 start
1:12 epoch 20 val loss 3.538
1:20 epoch 50 val loss 3.535


## Run 03
sparse MF
User USdataset
OAAT val set
satrt 1:03
epoch 70 at 1:07 ~3.85
additonal 70
Epoch 70 validation Loss: 3.632
additional OAAT Training
1:16 start
epoch 1 val loss 3.58
1:18 epoch 2 val loss 3.56

## Run 04
MF not sparse, Tensor train and val
1:34 start
1:36 epoch 1 val loss 3.4578
epoch 2 val loss 3.158
1:39 epoch 3 val loss 3.10