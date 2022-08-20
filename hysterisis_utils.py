import numpy as np
from matplotlib import pyplot as plt


def create_temperatures(T_nodes, dT=0.1):
    if T_nodes[0] > T_nodes[1]:
        T = np.arange(T_nodes[0], T_nodes[1], -dT)
    else:
        T = np.arange(T_nodes[0], T_nodes[1], dT)
    for k in range(1, len(T_nodes)-1):
        if T_nodes[k] > T_nodes[k+1]:
            T = np.append(T, np.arange(T_nodes[k], T_nodes[k+1], -dT))
        else:
            T = np.append(T, np.arange(T_nodes[k], T_nodes[k+1], dT))
    return T


def hys_tan(T, args, x_0=0, h_0=0):

    H = args['H']
    ctl = args['ctl']
    ctr = args['ctr']
    a = args['a']
    s = args['s']
    cs = args['cs']
    Tl = args['Tl']
    Tr = args['Tr']

    maj_l = lambda t: H/2*np.tanh((t-ctl*a)) + s*(T-(ctl+ctr)/2) + H/2 + cs
    maj_r = lambda t: H/2*np.tanh((t-ctr*a)) + s*(T-(ctl+ctr)/2) + H/2 + cs
    min_l = lambda t, h: h/2*np.tanh((t-ctl*a)) + s*(T-(ctl+ctr)/2) + H - h/2 + cs
    min_r = lambda t, h: h/2*np.tanh((t-ctr*a)) + s*(T-(ctl+ctr)/2) + H - h/2 + cs
    h_l = lambda t, h_prev: (h_prev * (np.tanh((t-ctl)*a)+1) - 2*H) / (np.tanh((t-ctr)*a)-1)
    h_r = lambda t, h_prev: (h_prev * (np.tanh((t-ctr)*a)-1) + 2*H) / (np.tanh((t-ctl)*a)+1)

    if T[1] > T[0]:
        dir = 'rise'
    else:
        dir = 'lower'

    x = np.zeros(T.shape)

    if T[0] <= Tl or T[0] >= Tr:
        if dir == 'rise':
            x[0] = maj_r(T[0])
        if dir == 'lower':
            x[0] = maj_l(T[0])
        h_pre = H
    else:
        x[0] = x_0
        h_pre = h_0

    for k in range(1, T.size):
        if T[k] > T[k-1]:
            dir = 'rise'
        else:
            dir = 'lower'



    return 0


if __name__ == '__main__':
    args = dict()
    args['H'] = 0.031
    args['ctl'] = 46
    args['ctr'] = 65
    args['a'] = 0.147
    args['s'] = 1.25*10**(-5)
    args['cs'] = 0.001
    args['Tl'] = 35
    args['Tr'] = 75
    m_0 = 0

    T_nodes = [0, 100, 20, 80, 40, 60, 50]
    T = create_temperatures(T_nodes)

    pass
