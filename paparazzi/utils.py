# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as tt


__all__ = ["Adam"]


def Adam(cost, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    """https://gist.github.com/Newmu/acb738767acb4788bac3

    """
    updates = []
    grads = tt.grad(cost, params)
    i = theano.shared(np.array(0.,dtype=theano.config.floatX))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (tt.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tt.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tt.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates