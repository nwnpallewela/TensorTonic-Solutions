import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param_d = np.array(param)
    grad_d = np.array(grad)
    m_d = np.array(m)
    v_d = np.array(v)
    # Write code here
    mt = beta1 * m_d + (1-beta1) * grad_d
    vt = beta2 * v_d + (1-beta2) * (grad_d ** 2)
    # vt = np.sqrt(v_d + eps)
    m_new = mt / (1-beta1**t)
    v_new = vt / (1-beta2**t)
    param_new = param_d - lr * (m_new / (np.sqrt(v_new) + eps))

    # m_new = beta1 * m + (1 - beta1) * grad

    # Step 2: Update biased second raw moment estimate
    # v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Step 3: Bias correction
    # m_hat = m_new / (1 - beta1 ** t)
    # v_hat = v_new / (1 - beta2 ** t)

    # Step 4: Parameter update
    # theta_new = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)

    # return theta_new, m_new, v_new
    return (param_new, mt, vt)