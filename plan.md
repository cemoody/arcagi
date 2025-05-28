1. Try a simple fixed-weights UNET model for a single example
    - try with dropout
2. Try a factorized UNET model for many examples
3. Try an S4/S5 model for a single example
4. Try an S4/S5 model factorized for many examples

From NCA-like models, and from interesting models with synap synchronization we have the interesting
idea of small, local cell models, that attend to their neighborhood, to their own history, and
to the current timestep of cells.

So inputs to a cell are:
    - It's history (shape: T timesteps)
    - The current state of cells in the neighborhood (9 x D)
    - Should also take into account the outputs of all other cells at current timestamp (30x30)

So maybe three functions per cell:
    - A history attention, that listens to the cell output state over time, return embedding shape D
    - A cell neighborhood MLP, that listens to local cell neighborhood, returns D
    - A global connection attention that also return D

Write a pytorch module that takes as input 30x30x11 one-hot images, embeds them to get 
30x30xD. It then outputs a 30x30xD embedding. It should have a single local MLP which transforms a D-dimensional embedding into another D-dim embedding, which wehn applied in parallel gets you an 30x30xD output. That is ultimately transformed back into a 30x30x11 image, which then has a softmax loss to compare the target image, which is also 30x30x11 one-hot.


Could also just slap it all together as 
    - Token inputs  
        - 30x30 input embeddings that connect cell to all other cells
            - Compute with pos embeddings, eg embed_x(i) + embed_y(j) + mlp(cell_output)
        - T history tokens that connect cell to previous states
            - computed like history_embed(t) +  mlp(history[1..T])
        - 1 passthru token that informs network of current cell location
            - computed like embed_x(i)  + embed_y(j) + embed('passthru')
    - residual: last_cell_output + attention(rms_norm(tokens)) -> cell output
    - when computing attention, tokens are 30x30=900 grid cell attentions, and then computed *again* for each cell, 
        there is a way to save on recomputing the attention costs -- but need to correctly use with position embeddings 
        see example code below
    - after each forward pass for each layer, append a 2D list of cell states to a concatenated history matrix (B, T, 30, 30, D)
        - To get a single cell history, index into (B, 1:T, i, j, D)
    - use grouped query attention (only for inference)
    - use rms norm, use prenorm (https://www.stephendiehl.com/posts/post_transformers/)
    - use swiglu 
    - add noise at each layer / tick / step so as to make the network self-correcting
    - todo: use rotary embeds to preserve relative distance between two embeds (how do we do this for 2D?)

add tests
    - ensure 1D relative distances are indeed relatively clsoe to each other
    - ensure 2D relative distances are indeed relatively clsoe to each other
    - ensure shapes and sizes are right

Implementaiton: https://chatgpt.com/share/6837110f-ea18-800c-8e23-ac723f31e9a8