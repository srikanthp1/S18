# S18

## PART 1

### Model 
* model is saved in model.py

### dataset, dataloader etc
* rest of the functionalities are saved in main.py

### results

* dice loss performed very well, which converged faster and able to reached lowest loss
* maxpool, strided convolution i didnt see any remarkable difference
* same holds for upsampling and transposeconv 

## PART 2 

### Architecture 

* right before nn.linear for mean values we are concatnating nn.linear(one-hot) results to encoder embeddings

### wrong label encoding images 
* wrong label encoding images are given label as their next ones. so that it is easier to visualize. 
* not to most extent but compared to correct label image generators, wrong labeled ones did give some smudges and also a 5% of it is going in label direction 

* for cifar10 most of the image recieved are smudgy and loss keeps going into negative so end at close to zero.
* so tried with mse loss, but in a way gaussian likelihood is performing better  