import torch
import torch.nn.functional as F


words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


def build_dataset(words, context):
    xs= []
    ys = []
    for word in words:

        new_word =  word + '.'

        new_added = ['.'] * (context)

        contextt = "".join(new_added)
        new_word = contextt + new_word

        for i in range(len(new_word)):

            slided_word = new_word[i: (i +context)%len(new_word)]
            if not (i+context) == len(new_word):
                xs.append([stoi[x] for x in list(slided_word)])
                ys.append(stoi[new_word[(i+context)%len(new_word)]])
            else:
                break

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys   

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

# META_VARIABLES
H_weights_length = 200
Context_length = 3
Feature_dimensions = 10


Xtr, Ytr = build_dataset(words[:8], Context_length)
# Xdev, Ydev = build_dataset(words[n1:n2],2)
# Xte, Yte = build_dataset(words[n2:],2)

# Parameters
feature_vector_C = torch.rand((27, Feature_dimensions))
H_weights = torch.rand((Context_length*Feature_dimensions, H_weights_length))
H_bias = torch.rand((H_weights_length))
G_weights = torch.rand((H_weights_length, 27))
G_bias = torch.rand(27)

parameters = [feature_vector_C, H_weights, H_bias, G_weights, G_bias]
# print(sum(p.nelement() for p in parameters), 'total parameters')
# ========================================================================
# Forward_Pass
for p in parameters:
    p.requires_grad = True

for i in range(20000):
    # minibatch contruct
    minibatch_ints = torch.randint(0, Xtr.shape[0], (32,))
    minibatch_construct = Xtr[minibatch_ints]
# forward pass
    feature_activation_layer = feature_vector_C[minibatch_construct] 
    merged_feature_vector = feature_activation_layer.view((minibatch_construct.shape[0], -1)) 
    first_activation = torch.tanh(merged_feature_vector @ H_weights + H_bias)
    second_activation = first_activation @ G_weights + G_bias
    
# loss
    normalized_probabilities = F.log_softmax(second_activation, dim=1)
    negative_loss_likelihood = F.nll_loss(normalized_probabilities, Ytr[minibatch_ints])
# backward pass
    for p in parameters:
        p.grad = None

    negative_loss_likelihood.backward()

#gradient descent
    for p in parameters:
        p.data += -0.0001*p.grad
    

    print(negative_loss_likelihood)
    

