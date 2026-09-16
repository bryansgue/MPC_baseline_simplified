import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import shutil
import subprocess


def _latex_available():
    """Use LaTeX text rendering only if latex and the cm-super fonts (type1ec.sty) exist."""
    if shutil.which('latex') is None:
        return False
    try:
        return subprocess.run(['kpsewhich', 'type1ec.sty'], capture_output=True).returncode == 0
    except Exception:
        return False


plt.rc('text', usetex=_latex_available())
def fancy_plots_2():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    return fig, ax1, ax2


def fancy_plot():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(111)
    return fig, ax1

def plot_pose(x, xref, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    def yaw_from_quat(q):
        # q = [qw, qx, qy, qz] (4 x n)
        return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                          1 - 2 * (q[2] ** 2 + q[3] ** 2))

    n = x.shape[1]
    if x.shape[0] >= 11:      # quaternion model: x = [p(3), q(4), v(3), w]
        yaw = yaw_from_quat(x[3:7, :])
        yaw_d = yaw_from_quat(xref[3:7, :n])
    else:                     # Euler model: x = [p(3), psi, v(3), w]
        yaw = x[3, :]
        yaw_d = xref[3, :n]
    signals = [x[0, :], x[1, :], x[2, :], yaw]
    refs = [xref[0, :n], xref[1, :n], xref[2, :n], yaw_d]

    for i in range(4):
        ax.plot(t[0:n], signals[i],
                color=colors[i], lw=2, ls="-", label=labels[i])

        ax.plot(t[0:n], refs[i],
                color=colors[i], lw=2, ls="--", label=labels[i] + r'$d$')

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_error(error, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(3):
        ax.plot(t[0:error.shape[1]], error[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_time(ts, delta_t, t):
    fig, ax = fancy_plot()
    ax.set_xlim((t[0], t[-1]))
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(1):
        ax.plot(t[0:ts.shape[1]], ts[i, :],
                color='#BB5651', lw=2, ls="--", label=labels[i])
        
        ax.plot(t[0:ts.shape[1]], delta_t[i, 0:ts.shape[1]],
                color='#69BB51', lw=2, ls="-", label=labels[i] + r'$d$')

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig


def plot_cbf_distance(dist, d_safe, r_obs, t):
    fig, ax = fancy_plot()
    n = dist.shape[1]
    for i in range(dist.shape[0]):
        ax.plot(t[0:n], dist[i, :], lw=2, label=r"$\|r_" + str(i + 1) + r"\|$")
    ax.axhline(d_safe, color="k", ls="--", lw=1.5, label=r"$d_s$")
    ax.axhline(r_obs, color="r", ls=":", lw=1.5, label=r"$r_{obs}$")
    ax.set_ylabel(r"$[m]$")
    ax.set_xlabel(r"$[t]$")
    ax.legend(loc="best", frameon=True)
    ax.grid(color="#949494", linestyle="-.", linewidth=0.5)
    return fig


def plot_cbf_control(u_nmpc, u_safe, t):
    n = u_nmpc.shape[1]
    labels = [r"$u_l$", r"$u_m$", r"$u_n$", r"$w$"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    for i in range(4):
        axes[i].plot(t[0:n], u_nmpc[i, :], color="#BB5651", lw=1.5, ls="--", label=labels[i] + " NMPC")
        axes[i].plot(t[0:n], u_safe[i, :], color="#5189BB", lw=1.5, label=labels[i] + " CBF")
        axes[i].legend(loc="best", frameon=True)
        axes[i].grid(color="#949494", linestyle="-.", linewidth=0.5)
    axes[3].set_xlabel(r"$[t]$")
    fig.tight_layout()
    return fig


def plot_cbf_xy(x, xref, obstacles, r_obs, d_safe):
    n = x.shape[1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xref[0, 0:n], xref[1, 0:n], color="#69BB51", ls="--", lw=2, label="ref")
    ax.plot(x[0, :], x[1, :], color="#BB5651", lw=2, label="drone")
    for i in range(obstacles.shape[0]):
        ax.add_patch(plt.Circle(obstacles[i, 0:2], r_obs, color="k", alpha=0.6))
        ax.add_patch(plt.Circle(obstacles[i, 0:2], d_safe, color="k", fill=False, ls="--"))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x [m]$")
    ax.set_ylabel(r"$y [m]$")
    ax.legend(loc="best", frameon=True)
    ax.grid(color="#949494", linestyle="-.", linewidth=0.5)
    return fig


def plot_cbf_views(x, xref, obstacles, r_obs, d_safe):
    # Vistas laterales (x-z, y-z) y 3D
    n = x.shape[1]
    fig = plt.figure(figsize=(14, 5))
    idx = [[0, 2], [1, 2]]
    lab = [[r"$x [m]$", r"$z [m]$"], [r"$y [m]$", r"$z [m]$"]]
    for k in range(2):
        i = idx[k][0]
        j = idx[k][1]
        ax = fig.add_subplot(1, 3, k + 1)
        ax.plot(xref[i, 0:n], xref[j, 0:n], color="#69BB51", ls="--", lw=2, label="ref")
        ax.plot(x[i, :], x[j, :], color="#BB5651", lw=2, label="drone")
        for m in range(obstacles.shape[0]):
            ax.add_patch(plt.Circle((obstacles[m, i], obstacles[m, j]), r_obs, color="k", alpha=0.6))
            ax.add_patch(plt.Circle((obstacles[m, i], obstacles[m, j]), d_safe, color="k", fill=False, ls="--"))
        ax.set_aspect("equal")
        ax.set_xlabel(lab[k][0])
        ax.set_ylabel(lab[k][1])
        ax.grid(color="#949494", linestyle="-.", linewidth=0.5)
        if k == 0:
            ax.legend(loc="best", frameon=True)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.plot(xref[0, 0:n], xref[1, 0:n], xref[2, 0:n], color="#69BB51", ls="--", lw=1.5)
    ax.plot(x[0, :], x[1, :], x[2, :], color="#BB5651", lw=1.5)
    uu, vv = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    for m in range(obstacles.shape[0]):
        p_o = obstacles[m, :]
        ax.plot_surface(p_o[0] + d_safe * np.cos(uu) * np.sin(vv), p_o[1] + d_safe * np.sin(uu) * np.sin(vv),
                        p_o[2] + d_safe * np.cos(vv), color="k", alpha=0.15, linewidth=0)
        ax.plot_surface(p_o[0] + r_obs * np.cos(uu) * np.sin(vv), p_o[1] + r_obs * np.sin(uu) * np.sin(vv),
                        p_o[2] + r_obs * np.cos(vv), color="k", alpha=0.7, linewidth=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((8, 8, 4))
    fig.tight_layout()
    return fig


def plot_camera(r_log, vis_log, known_log, t, sense_range, fov_deg):
    # Lo que la camara entrega al filtro: rango, bearing (azimut, elevacion), visible/conocido
    n_obs = r_log.shape[0]
    n = r_log.shape[2]
    colors = ["#BB5651", "#69BB51", "#5189BB"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    for i in range(n_obs):
        r = r_log[i, :, :]
        rango = np.linalg.norm(r, axis=0)
        az = np.rad2deg(np.arctan2(r[1, :], r[0, :]))
        el = np.rad2deg(np.arctan2(r[2, :], np.hypot(r[0, :], r[1, :])))
        known = known_log[i, :]
        c = colors[i % 3]
        axes[0].plot(t[0:n], np.where(known, rango, np.nan), color=c, lw=1.8, label=r"$\|r_" + str(i + 1) + r"\|$")
        axes[1].plot(t[0:n], np.where(known, az, np.nan), color=c, lw=1.8, label="az" + str(i + 1))
        axes[1].plot(t[0:n], np.where(known, el, np.nan), color=c, lw=1.2, ls=":", label="el" + str(i + 1))
        axes[2].plot(t[0:n], vis_log[i, :] * 1.0 + 2.2 * i, color=c, lw=1.8, label="obs " + str(i + 1) + " visible")
        axes[2].plot(t[0:n], known_log[i, :] * 1.0 + 2.2 * i, color=c, lw=1.0, ls="--", label="obs " + str(i + 1) + " known")
    axes[0].axhline(sense_range, color="k", ls="--", lw=1, label="range")
    axes[0].set_ylabel("range [m]")
    axes[1].axhline(fov_deg / 2, color="k", ls="--", lw=1, label="FOV/2")
    axes[1].axhline(-fov_deg / 2, color="k", ls="--", lw=1)
    axes[1].set_ylabel("bearing [deg]")
    axes[2].set_ylabel("visible / known")
    axes[2].set_yticks([])
    axes[2].set_xlabel(r"$[t]$")
    for i in range(3):
        axes[i].grid(color="#949494", linestyle="-.", linewidth=0.5)
        axes[i].legend(loc="upper right", frameon=True, fontsize=7, ncol=3)
    fig.tight_layout()
    return fig
