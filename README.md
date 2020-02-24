# ECES641

# How to use GLoVe embedding

## Tools

The GLoVe embedding software includes 4 tools:

    1. vocab_count
    1. cooccur
    1. shuffle
    1. glove

Each of these has a unique feature.

### vocab_count

This one counts the number of times a word appears in a text. I don't think this is applicable to our use cases.

### cooccur

Generates a co-occurrence matrix based on an input text file. See vocab.txt in the GLoVe directory for what that's supposed to look like.

### shuffle

Shuffle shuffles the data and does some preprocessing. This is abstracted out from our understanding. We just need to feed it a co-occurrence matrix.

### glove

Glove performs the actual glove embedding. Returns a vector per word. This will be useful for machine learning stuff.

