# JAX Entropy Coders

## About

This project is a proof-of-concept for executing entropy coding in JAX. There
are two coders included: 1) a simple range/arithmetic coder, and 2) an ANS coder.
The arithmetic coder is based on that of a blog post:

https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html

The ANS coder is based on Crayjax:

https://github.com/j-towns/crayjax

Both implementations include some minor modifications that allow functional
probabilities and context-adaptive coding. In principle, you could embed a
neural network in these probability functions for entropy coding.

## Installation

Run `pip install -r requirements.txt` to get JAX. To verify your installation,
run `pytest test_entropy_coders.py`. For example usage, please examine the tests.
