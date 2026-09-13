"""Dependency-free numerical primitives for action-head integration tests.

These operate on user-supplied velocity/logit functions. They neither load model
weights nor stand in for the SmolVLA and LLaDA-V training implementations.
"""
import math
from .contracts import finite


def flow_sample(vector_field, noise, steps):
    """Euler integrate dx/dt=v(x,t), t in [0,1], configurable denoising count."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    x = [finite(v, "noise") for v in noise]
    if not x:
        raise ValueError("empty action vector")
    for i in range(steps):
        velocity = list(vector_field(tuple(x), i/steps))
        if len(velocity) != len(x):
            raise ValueError("vector field dimension mismatch")
        x = [finite(a + finite(v, "velocity")/steps, "integrated action") for a,v in zip(x, velocity)]
    return x


def flow_matching_loss(vector_field, noise, target, t):
    """Conditional straight-path FM squared-error objective."""
    if not 0 <= finite(t, "t") <= 1 or not noise or len(noise) != len(target):
        raise ValueError("invalid flow matching batch")
    noise = [finite(v, "noise") for v in noise]
    target = [finite(v, "target") for v in target]
    x = [(1-t)*n+t*a for n,a in zip(noise,target)]
    prediction = list(vector_field(tuple(x),t))
    if len(prediction) != len(x):
        raise ValueError("vector field dimension mismatch")
    return sum((finite(p,"prediction")-(a-n))**2 for p,a,n in zip(prediction,target,noise))/len(x)


def masked_decode(predict, slots, steps, mask_id=-1):
    """Confidence-ordered iterative unmasking for a fixed action-token block.

    predict(tokens) returns one (token_id, confidence) pair per slot. Resolved
    slots are frozen. The actual LLaDA-V remasking decoder lives in robotics/.
    """
    if type(slots) is not int or slots < 1 or type(steps) is not int or steps < 1:
        raise ValueError("slots and steps must be positive integers")
    tokens = [mask_id]*slots
    for iteration in range(min(slots, steps)):
        candidates = list(predict(tuple(tokens)))
        if len(candidates) != slots:
            raise ValueError("prediction dimension mismatch")
        remaining = [i for i,t in enumerate(tokens) if t == mask_id]
        for i in remaining:
            token, confidence = candidates[i]
            if type(token) is not int or token == mask_id or not 0 <= finite(confidence,"confidence") <= 1:
                raise ValueError("invalid token or confidence")
        count = math.ceil(len(remaining)/(min(slots, steps)-iteration))
        ordered = sorted(remaining, key=lambda i: (-candidates[i][1], i))
        for i in ordered[:count]:
            tokens[i] = candidates[i][0]
    return tokens
