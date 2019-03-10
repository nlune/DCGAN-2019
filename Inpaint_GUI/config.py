# model parameters
BATCH_SIZE=64
z_dim = 100 # random noise vector dimension

beta1=0.5 # β1 Adam training param
lmda = 0.003 # λ ratio for prior loss

# backprop parameters
niters = 100
momentum = 0.9
learning_rate = 0.01 # learning rate

# image
image_size = 64
channels = 3
weighted_mask = False
window_size = 7 # window size of neighbours for weighted mask 
mask_ratio = 0.25 # must be <0.5
