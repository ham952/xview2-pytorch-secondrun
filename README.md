# xview2-pytorch-secondrun
Training for localization and classification for xview2 through Mask R-CNN

### Overview of Experiment
 1. Use of (only) socal-fire dataset imagery Localization task (only)
 2. Use of train dataset for training
 3. Use of hold dataset for validation 
 4. Use of test dataset for testing (Offline)

### Major Experiments
 - E1 : Training with learning rate **0.02** and its variations
 - E2 : Training with learning rate **0.01** and its variations
 - E3 : Use of AdamW optimizer 
 - E4 : Training with No Augmentations
 - E5 : Freezing model layers during transfer training
 
### Ablation Studies
 - ToDo : Writeup

### Experiments Settings

 **Base Settings**

 - Training Images : 408
 - Validation Images : 133
 - Test Images : 307
 - Batch Size : 2
 - Learning rate reduction factor : 0.1
 - Backbone : Resnet-50-fpn
 
 **E1 Settings**
 
  - Optimizer : SGD
  - Epochs : 250 (51k iterations)
  - Learning rate : 0.02
  - Learning rate schedular : 200 (epoch)
  - Augmentations : Horizontal Flip
 
  **E2 Settings**
 
  - Optimizer : SGD
  - Epochs : 250 (51k iterations)
  - Learning rate : 0.01
  - Learning rate schedular : 200 (epoch)
  - Augmentations : Horizontal Flip
  
  **E3 Settings**
 
  - Optimizer : AdamW
  - Epochs : 110
  - Learning rate : 0.02
  - Learning rate schedular : [30,70] (epoch)
  - Augmentations : Horizontal Flip
 
  **E4 Settings**
 
  - Optimizer : SGD
  - Epochs : 250 (51k iterations)
  - Learning rate : 0.02
  - Learning rate schedular : 200 (epoch)
  - Augmentations : Nil
  
   **E5 Settings**
 
  - Optimizer : SGD
  - Epochs : 250 (51k iterations)
  - Learning rate : 0.02
  - Learning rate schedular : 200 (epoch)
  - Augmentations : Horizontal Flip
  - Freezing layer : 2nd Layer (only) of Resnet-50

### Conclusion
 
 - Model overfitting persists under all experiments
 - Accuracy better then **first experiment**
 - Best peformance by **E1**
 - Highest Overfitting by **E4** i-e with no augmentations

### Way forward 

 - Use of multi-scale augmentations
 - Use of data grouping (epoch wise) for training
 - Early stopping
 - Use of Combination of L1 and L2 regulariztion
 - Data engineering (logical addition of dataset from inter/intra disaster sets) 

### Results

 - ToDo Writeup
