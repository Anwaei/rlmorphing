import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


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


def hys_tan(T_nodes, args, x_0=0, h_0=0):

    H = args['H']
    ctl = args['ctl']
    ctr = args['ctr']
    a = args['a']
    s = args['s']
    cs = args['cs']
    Tl = args['Tl']
    Tr = args['Tr']

    maj_l = lambda t: H/2*np.tanh((t-ctl)*a) + s*(t-(ctl+ctr)/2) + H/2 + cs
    maj_r = lambda t: H/2*np.tanh((t-ctr)*a) + s*(t-(ctl+ctr)/2) + H/2 + cs
    min_l = lambda t, h: h/2*np.tanh((t-ctl)*a) + s*(t-(ctl+ctr)/2) + h/2 + cs
    min_r = lambda t, h: h/2*np.tanh((t-ctr)*a) + s*(t-(ctl+ctr)/2) + H - h/2 + cs
    h_l = lambda t, h_prev: (h_prev * (np.tanh((t-ctr)*a)-1) + 2*H) / (np.tanh((t-ctl)*a)+1)
    h_r = lambda t, h_prev: (h_prev * (np.tanh((t-ctl)*a)+1) - 2*H) / (np.tanh((t-ctr)*a)-1)


    T = create_temperatures(T_nodes)

    if T[1] > T[0]:
        Tdir = 'rise'
    else:
        Tdir = 'lower'

    x = np.zeros(T.shape)
    h = np.zeros(T.shape)

    if T[0] <= Tl or T[0] >= Tr:
        if Tdir == 'rise':
            x[0] = maj_r(T[0])
        if Tdir == 'lower':
            x[0] = maj_l(T[0])
        h[0] = H
        which_loop = 'major'
    else:
        x[0] = x_0
        h[0] = h_0
        which_loop = 'minor'

    dir_prev = Tdir
    for k in range(1, T.size):
        if T[k] > T[k-1]:
            Tdir = 'rise'
        else:
            Tdir = 'lower'

        if Tdir == dir_prev:
            h[k] = h[k-1]
            if Tdir == 'rise':
                    x[k] = min_r(T[k], h[k])
            if Tdir == 'lower':
                    x[k] = min_l(T[k], h[k])
        if Tdir != dir_prev:
            if Tdir == 'rise':
                h[k] = h_r(T[k-1], h[k-1])
                x[k] = min_r(T[k], h[k])
            if Tdir == 'lower':
                h[k] = h_l(T[k-1], h[k-1])
                x[k] = min_l(T[k], h[k])

        # if Tdir == dir_prev:
        #     h[k] = h[k-1]
        #     if Tdir == 'rise':
        #         if which_loop == 'major':
        #             x[k] = maj_r(T[k])
        #         if which_loop == 'minor':
        #             x[k] = min_r(T[k], h[k])
        #     if Tdir == 'lower':
        #         if which_loop == 'major':
        #             x[k] = maj_l(T[k])
        #         if which_loop == 'minor':
        #             x[k] = min_l(T[k], h[k])
        # if Tdir != dir_prev:
        #     if T[k] <= Tl or T[k] >= Tr:
        #         h[k] = H
        #         which_loop = 'major'
        #         if Tdir == 'rise':
        #             x[k] = maj_r(T[k])
        #         if Tdir == 'lower':
        #             x[k] = maj_l(T[k])
        #     else:
        #         which_loop = 'minor'
        #         if Tdir == 'rise':
        #             h[k] = h_r(T[k-1], h[k-1])
        #             x[k] = min_r(T[k], h[k])
        #         if Tdir == 'lower':
        #             h[k] = h_l(T[k-1], h[k-1])
        #             x[k] = min_l(T[k], h[k])

        dir_prev = Tdir

        # x[k] = maj_r(T[k])

    return T, x, h


def hys_tan2(v, args, T0=0, h_0=0, gamma_0=0):
    """
    Hysteresis loop in form of 2-order diff equation. Include voltage as input.
    :param v:
    :return: T, h, gamma
    """
    H = args['H']
    ctl = args['ctl']
    ctr = args['ctr']
    a = args['a']
    s = args['s']
    cs = args['cs']

    mw = args['mw']
    Aw = args['Aw']
    cw = args['cw']
    Rw = args['Rw']
    Tf = args['Tf']
    hw = args['hw']

    dt = args['dt']

    maj_l = lambda t: H/2*np.tanh((t-ctl)*a) + s*(t-(ctl+ctr)/2) + H/2 + cs
    maj_r = lambda t: H/2*np.tanh((t-ctr)*a) + s*(t-(ctl+ctr)/2) + H/2 + cs
    f_l = lambda t, h: h/2*np.tanh((t-ctl)*a) + s*(t-(ctl+ctr)/2) + h/2 + cs
    f_r = lambda t, h: h/2*np.tanh((t-ctr)*a) + s*(t-(ctl+ctr)/2) + H - h/2 + cs
    g_l = lambda t, h_prev: (h_prev * (np.tanh((t-ctr)*a)-1) + 2*H) / (np.tanh((t-ctl)*a)+1)
    g_r = lambda t, h_prev: (h_prev * (np.tanh((t-ctl)*a)+1) - 2*H) / (np.tanh((t-ctr)*a)-1)

    sgn = lambda x: 1 if x >= 0 else 0
    K1 = 1/(mw*cw*Rw)
    K2 = -hw*Aw/(mw*cw)
    sigma = lambda T, v: K1*v**2 + K2*(T-Tf)
    func_T = lambda T_p, v_p: T_p + dt * sigma(T_p, v_p)
    func_h = lambda T, T_p, T_pp, h_p, v_p, v_pp: (1-sgn(sigma(T_p, v_p)*sigma(T_pp, v_pp))) * \
                                                  (sgn(sigma(T_p, v_p))*g_r(T, h_p) + (1-sgn(sigma(T_p, v_p)))*g_l(T, h_p)) \
                                                  + sgn(sigma(T_p, v_p)*sigma(T_pp, v_pp)) * h_p
    func_gamma = lambda T, T_p, h, v_p: sgn(sigma(T_p, v_p))*f_r(T, h) + (1-sgn(sigma(T_p, v_p)))*f_l(T, h)

    K = v.size

    gamma = np.zeros(shape=K)
    T = np.zeros(shape=K)
    h = np.zeros(shape=K)

    T[0] = Tf
    h[0] = H
    gamma[0] = maj_r(T[0])
    T[1] = func_T(T[0], v[0])
    h[1] = H
    gamma[1] = func_gamma(T[1], T[0], h[1], v[0])

    for k in range(2, K):
        T[k] = func_T(T[k-1], v[k-1])
        h[k] = func_h(T[k], T[k-1], T[k-2], h[k-1], v[k-1], v[k-2])
        hc = h[k]
        gamma[k] = func_gamma(T[k], T[k-1], h[k], v[k-1])

    return T, h, gamma


def hys_tan1(v, args, T0=0, h_0=0, gamma_0=0):
    """
    Hysteresis loop in form of 2-order diff equation. Include voltage as input.
    :param v:
    :return: T, h, gamma
    """
    H = args['H']
    ctl = args['ctl']
    ctr = args['ctr']
    a = args['a']
    s = args['s']
    cs = args['cs']

    mw = args['mw']
    Aw = args['Aw']
    cw = args['cw']
    Rw = args['Rw']
    Tf = args['Tf']
    hw = args['hw']
    h2 = args['h2']

    dt = args['dt']

    maj_l = lambda t: H/2*np.tanh((t-ctl)*a) + s*(t-(ctl+ctr)/2) + H/2 + cs
    maj_r = lambda t: H/2*np.tanh((t-ctr)*a) + s*(t-(ctl+ctr)/2) + H/2 + cs
    f_l = lambda t, h: h/2*np.tanh((t-ctl)*a) + s*(t-(ctl+ctr)/2) + h/2 + cs
    f_r = lambda t, h: h/2*np.tanh((t-ctr)*a) + s*(t-(ctl+ctr)/2) + H - h/2 + cs
    g_l = lambda t, h_prev: (h_prev * (np.tanh((t-ctr)*a)-1) + 2*H) / (np.tanh((t-ctl)*a)+1)
    g_r = lambda t, h_prev: (h_prev * (np.tanh((t-ctl)*a)+1) - 2*H) / (np.tanh((t-ctr)*a)-1)

    sgn = lambda x: 1 if x >= 0 else 0
    K1 = 1/(mw*cw*Rw)
    K2 = -hw*Aw/(mw*cw)
    sigma = lambda T, v: K1*v**2 + (K2 + K2/hw*h2*T**2)*(T-Tf)
    func_T = lambda T_p, v_p: T_p + dt * sigma(T_p, v_p)
    func_h = lambda T, T_p, h_p, v, v_p: (1-sgn(sigma(T, v)*sigma(T_p, v_p))) * \
                                      (sgn(sigma(T, v))*g_r(T, h_p) + (1-sgn(sigma(T, v)))*g_l(T, h_p)) + \
                                      sgn(sigma(T, v)*sigma(T_p, v_p)) * h_p
    func_gamma = lambda T, h, v: sgn(sigma(T, v))*f_r(T, h) + (1-sgn(sigma(T, v)))*f_l(T, h)

    K = v.size

    gamma = np.zeros(shape=K)
    T = np.zeros(shape=K)
    h = np.zeros(shape=K)

    T[0] = Tf
    h[0] = H
    gamma[0] = maj_r(T[0])

    for k in range(1, K):
        T[k] = func_T(T[k-1], v[k-1])
        h[k] = func_h(T[k], T[k-1], h[k-1], v[k], v[k-1])
        gamma[k] = func_gamma(T[k], h[k], v[k])

    return T, h, gamma


def gene_voltage_sin(time_steps):
    w = 0.6
    a = 9
    phi = 0
    h = a
    vol = a*np.sin(w*time_steps*np.pi-phi) + h
    return vol


def plot_hys(T, x):
    plt.figure()
    plt.plot(T, x)
    plt.xlabel('Temperature')
    plt.ylabel('Length Factor')
    plt.show()
    return


def plot_hys_tan2(timesteps, T, h, gamma, v):

    gs = gridspec.GridSpec(4, 2)
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[2, 0])
    ax3 = plt.subplot(gs[3, 0])
    ax4 = plt.subplot(gs[:, 1])

    ax0.plot(time_steps, v)
    ax0.set_xlabel('t')
    ax0.set_ylabel('v')
    ax1.plot(time_steps, T)
    ax1.set_xlabel('t')
    ax1.set_ylabel('T')
    ax2.plot(time_steps, h)
    ax2.set_xlabel('t')
    ax2.set_ylabel('h')
    ax3.plot(time_steps, gamma)
    ax3.set_xlabel('t')
    ax3.set_ylabel('gamma')

    ax4.plot(T, gamma)
    ax4.set_xlabel('T')
    ax4.set_ylabel('gamma')

    # fig1, ax = plt.subplots(4, 1)
    # ax[0].plot(time_steps, v)
    # ax[0].set_xlabel('t')
    # ax[0].set_ylabel('v')
    # ax[1].plot(time_steps, T)
    # ax[1].set_xlabel('t')
    # ax[1].set_ylabel('T')
    # ax[2].plot(time_steps, h)
    # ax[2].set_xlabel('t')
    # ax[2].set_ylabel('h')
    # ax[3].plot(time_steps, gamma)
    # ax[3].set_xlabel('t')
    # ax[3].set_ylabel('gamma')

    # fig2 = plt.figure()
    # plt.plot(T, gamma)
    # plt.xlabel('T')
    # plt.ylabel('gamma')

    plt.show()
    # fig1.show()

    return


def args_setting():
    args = dict()
    args['H'] = 0.995
    args['ctl'] = 46
    args['ctr'] = 65
    args['a'] = 0.147
    args['s'] = 1.25 * 10**(-5)
    # args['s'] = 0
    args['cs'] = 0.001
    args['Tl'] = 35
    args['Tr'] = 75

    args['mw'] = 1.14 * 10**(-4)
    args['Aw'] = 4.72 * 10**(-4)
    args['cw'] = 837.4
    args['Rw'] = 50.8
    args['Tf'] = 20
    args['hw'] = 120
    args['h2'] = 0.001

    args['time_final'] = 10
    args['dt'] = 0.01

    return args


if __name__ == '__main__':
    args = args_setting()

    # T_nodes = [0, 100, 30, 80, 35, 65, 40, 70, 45, 75, 15, 80, 25, 85, 25, 95]
    # T, x, _  = hys_tan(T_nodes, args)
    # plot_hys(T, x)

    time_steps = np.arange(start=0, stop=args['time_final'], step=args['dt'])
    vol = gene_voltage_sin(time_steps)
    T, h, gamma = hys_tan2(vol, args)
    plot_hys_tan2(time_steps, T, h, gamma, vol)

    pass
