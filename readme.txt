main.py:
Prints the means and stds of the MSE test errors of a model
args.model_type againts a target function of type args.target_type 
with varying number of layers/embedding dimension (specified by args.experiment)
args.model_type = hyper or emb
args.target_type = dot or poly or net
arg.experiment = layers / emb_dim


plots.py:
Plots the results in error bars 
means_hyper_layers = mean errors of varying the number of layers for hypernets
stds_hyper_layers = stds of the errors of varying the number of layers for hypernets
    
means_emb_layers = mean errors of varying the number of layers for embedding method
stds_emb_layers = stds of the errors of varying the number of layers for embedding method
        
means_hyper_dim, stds_hyper_dim, means_emb_dim, stds_emb_dim = same for varying the embedding dimension

For example, running main.py with parameters: 
args.model_type = hyper
args.target_type = net
args.experiment = layers
Prints the data for 'means_hyper_layers' and for 'stds_hyper_layers'.

By running main.py with parameters: 
args.model_type = emb
args.target_type = net
args.experiment = layers
Prints the data for 'means_emb_layers' and for 'stds_emb_layers'.


models.py:
Classes of functions.
DotProd = target functions of the form <h(I),x>
where h is a three layers mlp neural network 
with softmax layer of top of it and ELU activations.
PolynomialNet = target functions of the form h(I*x)
where h is a three layers ml neural network with ELU
activations.
Net = target functions of the form h(x,I)
where h is a three layers mlp neural network with ELU
activations.
Embedding = embedding method. 
Hypernet = hypernet model.

identifiability:
A directory that contains the relevant code for the
testing Assumption 1.

realdata:
code for rotations prediction and colorization.
