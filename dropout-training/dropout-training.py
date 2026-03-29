import numpy as np
def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    # drop_pattern = None
    drop_pattern = dropout_pattern(p,x,rng)
    
    scale = 1.0 / (1.0 - p)
    drop_pattern = drop_pattern * scale
    val = x * drop_pattern
    # val = np.array(x)*drop_pattern
    # print (val)
    # print(drop_pattern)
    return  val , drop_pattern

def dropout_pattern(p,x,rng):
    
    rand = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    dropout_pattern = (rand >= p).astype(int)
    return dropout_pattern