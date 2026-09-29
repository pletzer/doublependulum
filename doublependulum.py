"""Double pendulum solver.

Point mass m1 at the joint (end of arm l1), point mass m2 at the tip (end of
arm l2), massless rigid arms. Angles theta1, theta2 are measured from the
downward vertical. State vector: y = (theta1, theta2, omega1, omega2).

Usage:
    python doublependulum.py --theta1 120 --theta2 -10 --tmax 30
    python doublependulum.py --animate
    python doublependulum.py --no-show --movie pendulum.mp4
"""

import argparse

import numpy as np
from scipy.integrate import solve_ivp


def rhs(t, y, l1, l2, m1, m2, g):
    """Time derivative of the state (theta1, theta2, omega1, omega2)."""
    th1, th2, w1, w2 = y
    d = th1 - th2
    sd, cd = np.sin(d), np.cos(d)
    # 2*m1 + m2 - m2*cos(2d) == 2*(m1 + m2*sin(d)**2), singular if m1 == 0 and d == 0
    den = 2.0 * m1 + m2 - m2 * np.cos(2.0 * d)
    a1 = (-g * (2.0 * m1 + m2) * np.sin(th1)
          - m2 * g * np.sin(th1 - 2.0 * th2)
          - 2.0 * sd * m2 * (w2**2 * l2 + w1**2 * l1 * cd)) / (l1 * den)
    a2 = (2.0 * sd * (w1**2 * l1 * (m1 + m2)
                      + g * (m1 + m2) * np.cos(th1)
                      + w2**2 * l2 * m2 * cd)) / (l2 * den)
    return [w1, w2, a1, a2]


def energy(y, l1, l2, m1, m2, g):
    """Total energy T + V; y may be a single state or an array of shape (4, n)."""
    th1, th2, w1, w2 = y
    T = (0.5 * (m1 + m2) * l1**2 * w1**2
         + 0.5 * m2 * l2**2 * w2**2
         + m2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2))
    V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return T + V


def positions(th1, th2, l1, l2):
    """Cartesian positions of the joint (x1, y1) and the tip (x2, y2)."""
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th2)
    y2 = y1 - l2 * np.cos(th2)
    return x1, y1, x2, y2


def solve(y0, tmax, l1=1.0, l2=1.0, m1=1.0, m2=1.0, g=9.81,
          nt=2001, rtol=1e-10, atol=1e-10, method="DOP853"):
    """Integrate from t=0 to tmax; returns the scipy OdeResult sampled at nt times."""
    if m1 <= 0.0:
        raise ValueError("m1 must be > 0: with a massless joint the equations are singular")
    t_eval = np.linspace(0.0, tmax, nt)
    sol = solve_ivp(rhs, (0.0, tmax), y0, method=method, t_eval=t_eval,
                    args=(l1, l2, m1, m2, g), rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l1", type=float, default=1.0, help="length of arm 1 [m]")
    p.add_argument("--l2", type=float, default=1.0, help="length of arm 2 [m]")
    p.add_argument("--m1", type=float, default=1.0, help="mass at the joint [kg]")
    p.add_argument("--m2", type=float, default=1.0, help="mass at the tip [kg]")
    p.add_argument("--g", type=float, default=9.81, help="gravity [m/s^2]")
    p.add_argument("--theta1", type=float, default=120.0, help="initial theta1 [deg]")
    p.add_argument("--theta2", type=float, default=-10.0, help="initial theta2 [deg]")
    p.add_argument("--omega1", type=float, default=0.0, help="initial omega1 [deg/s]")
    p.add_argument("--omega2", type=float, default=0.0, help="initial omega2 [deg/s]")
    p.add_argument("--tmax", type=float, default=20.0, help="final time [s]")
    p.add_argument("--nt", type=int, default=2001, help="number of output times")
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--method", default="DOP853", help="solve_ivp method")
    p.add_argument("--animate", action="store_true", help="show an animation")
    p.add_argument("--movie", default="", help="save the animation (.mp4 or .gif)")
    p.add_argument("--fps", type=float, default=30.0, help="animation frame rate")
    p.add_argument("--trail", type=float, default=2.0, help="length of the tip trail [s]")
    p.add_argument("--save", default="", help="save the summary plot to this file")
    p.add_argument("--no-show", action="store_true", help="do not open plot windows")
    args = p.parse_args()

    params = dict(l1=args.l1, l2=args.l2, m1=args.m1, m2=args.m2, g=args.g)
    y0 = np.deg2rad([args.theta1, args.theta2, args.omega1, args.omega2])
    sol = solve(y0, args.tmax, nt=args.nt, rtol=args.rtol, atol=args.atol,
                method=args.method, **params)

    E = energy(sol.y, **params)
    rel_drift = np.abs(E - E[0]) / max(abs(E[0]), 1e-300)
    print(f"method={args.method} rtol={args.rtol} atol={args.atol} nfev={sol.nfev}")
    print(f"E(0) = {E[0]:.12g} J, max |E - E(0)|/|E(0)| = {rel_drift.max():.3e}")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th1, th2 = sol.y[0], sol.y[1]
    x1, y1, x2, y2 = positions(th1, th2, args.l1, args.l2)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(sol.t, np.rad2deg(th1), label=r"$\theta_1$")
    axs[0].plot(sol.t, np.rad2deg(th2), label=r"$\theta_2$")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("angle [deg]")
    axs[0].legend()
    axs[0].set_title("Angles (unwrapped)")

    axs[1].plot(x2, y2, lw=0.6)
    axs[1].plot(0, 0, "ko")
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_title("Tip trajectory")

    axs[2].semilogy(sol.t[1:], np.maximum(rel_drift[1:], 1e-17))
    axs[2].set_xlabel("t [s]")
    axs[2].set_ylabel(r"$|E-E_0|/|E_0|$")
    axs[2].set_title("Relative energy drift")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")

    if args.animate or args.movie:
        from matplotlib.animation import FuncAnimation
        L = args.l1 + args.l2
        fig2, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.1 * L, 1.1 * L)
        ax.set_ylim(-1.1 * L, 1.1 * L)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        trail, = ax.plot([], [], lw=1.0, color="tab:red", alpha=0.7)
        rods, = ax.plot([], [], "-", lw=2.5, color="k")
        masses = ax.scatter([0, 0], [0, 0], s=[80 * args.m1 / max(args.m1, args.m2),
                                              80 * args.m2 / max(args.m1, args.m2)],
                            color=["tab:blue", "tab:red"], zorder=3)
        ax.plot(0, 0, "ks", ms=5)
        label = ax.text(0.02, 0.95, "", transform=ax.transAxes)
        dt = sol.t[1] - sol.t[0]
        # subsample output times so playback runs in real time at the requested fps
        step = max(1, int(round(1.0 / (dt * args.fps))))
        frames = range(0, len(sol.t), step)
        ntrail = int(round(args.trail / dt))

        def update(i):
            rods.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
            masses.set_offsets([[x1[i], y1[i]], [x2[i], y2[i]]])
            j = max(0, i - ntrail)
            trail.set_data(x2[j:i + 1], y2[j:i + 1])
            label.set_text(f"t = {sol.t[i]:.2f} s")
            return trail, rods, masses, label

        anim = FuncAnimation(fig2, update, frames=frames,
                             interval=1000 * dt * step, blit=True)
        if args.movie:
            fps = 1.0 / (dt * step)
            writer = "pillow" if args.movie.endswith(".gif") else "ffmpeg"
            anim.save(args.movie, writer=writer, fps=fps, dpi=100)
            print(f"saved {args.movie} ({len(frames)} frames at {fps:.1f} fps)")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
