from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
import casadi as ca
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math

from fancy_plots import plot_pose, plot_error, plot_time
from Functions_SimpleModel import f_system_simple_model_quat, log_cuaternion_casadi
from Functions_SimpleModel import f_d, euler_to_quaternion, quaternion_error


def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    ocp = AcadosOcp()

    model, f_system = f_system_simple_model_quat()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    ocp.solver_options.N_horizon = N_horizon

    Q_mat = ca.MX.zeros(3, 3)
    Q_mat[0, 0] = 1.1
    Q_mat[1, 1] = 1.1
    Q_mat[2, 2] = 1.1

    K_mat = ca.MX.zeros(3, 3)
    K_mat[0, 0] = 1.1
    K_mat[1, 1] = 1.1
    K_mat[2, 2] = 1.1

    R_mat = ca.MX.zeros(4, 4)
    R_mat[0, 0] = 1
    R_mat[1, 1] = 1
    R_mat[2, 2] = 1
    R_mat[3, 3] = 1

    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = model.p[0:3] - model.x[0:3]
    quat_error = quaternion_error(model.x[3:7], model.p[3:7])
    log_q = log_cuaternion_casadi(quat_error)

    ocp.model.cost_expr_ext_cost = (
        error_pose.T @ Q_mat @ error_pose
        + model.u.T @ R_mat @ model.u
        + log_q.T @ K_mat @ log_q
    )
    ocp.model.cost_expr_ext_cost_e = (
        error_pose.T @ Q_mat @ error_pose
        + log_q.T @ K_mat @ log_q
    )

    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.tf = t_horizon

    return ocp


def get_initial_state():
    """Return a default initial state [x, y, z, qw, qx, qy, qz, ul, um, un, w]."""
    return np.array([1.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def send_control(u):
    print(f"u: {np.round(u, 3)}")


def main():
    t_final = 60
    frec = 30
    t_s = 1 / frec
    N_horizont = 50
    t_prediction = N_horizont / frec

    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    t = np.arange(0, t_final + t_s, t_s)

    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s * np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    x = np.zeros((11, t.shape[0] + 1 - N_prediction), dtype=np.double)
    x[:, 0] = get_initial_state()

    value = 6
    xd  = lambda t: 4 * np.sin(value * 0.04 * t) + 3
    yd  = lambda t: 4 * np.sin(value * 0.08 * t)
    zd  = lambda t: 2 * np.sin(value * 0.08 * t) + 6
    xdp = lambda t: 4 * value * 0.04 * np.cos(value * 0.04 * t)
    ydp = lambda t: 4 * value * 0.08 * np.cos(value * 0.08 * t)
    zdp = lambda t: 2 * value * 0.08 * np.cos(value * 0.08 * t)

    hxd  = xd(t)
    hyd  = yd(t)
    hzd  = zd(t)
    hxdp = xdp(t)
    hydp = ydp(t)

    psid  = np.arctan2(hydp, hxdp)
    quatd = np.zeros((4, t.shape[0]), dtype=np.double)
    for i in range(t.shape[0]):
        quatd[:, i] = euler_to_quaternion(0, 0, psid[i])

    xref = np.zeros((15, t.shape[0]), dtype=np.double)
    xref[0, :] = hxd
    xref[1, :] = hyd
    xref[2, :] = hzd
    xref[3, :] = quatd[0, :]
    xref[4, :] = quatd[1, :]
    xref[5, :] = quatd[2, :]
    xref[6, :] = quatd[3, :]
    # xref[7:11] = 0 (already zero)

    u_control = np.zeros((4, t.shape[0] - N_prediction), dtype=np.double)

    zp_ref_max = 3
    phi_max    = 3
    theta_max  = 3
    psi_max    = 2

    model, f = f_system_simple_model_quat()

    ocp = create_ocp_solver_description(
        x[:, 0], N_prediction, t_prediction,
        zp_ref_max, -zp_ref_max,
        phi_max, -phi_max,
        theta_max, -theta_max,
        psi_max, -psi_max,
    )

    solver_json = 'acados_ocp_' + model.name + '.json'
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))

    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    Error = np.zeros((3, t.shape[0] - N_prediction), dtype=np.double)

    for k in range(0, t.shape[0] - N_prediction):
        tic = time.time()

        print(xref[3:7, k])
        Error[:, k] = xref[0:3, k] - x[0:3, k]

        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        for j in range(N_prediction):
            acados_ocp_solver.set(j, "p", xref[:, k + j])
        acados_ocp_solver.set(N_prediction, "p", xref[:, k + N_prediction])

        status = acados_ocp_solver.solve()

        for i in range(N_prediction):
            simX[:, i] = acados_ocp_solver.get(i, "x")
            simU[:, i] = acados_ocp_solver.get(i, "u")
        simX[:, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        u_control[:, k] = acados_ocp_solver.get(0, "u")

        toc_solver = time.time() - tic

        send_control(u_control[:, k])

        x[:, k + 1] = f_d(x[:, k], u_control[:, k], t_s, f)

        delta_t[:, k] = toc_solver

        elapsed = time.time() - tic
        sleep_time = t_s - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    send_control([0, 0, 0, 0])

    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")
    fig2 = plot_error(Error, t)
    fig2.savefig("2_error_pose.png")
    fig3 = plot_time(t_sample, delta_t, t)
    fig3.savefig("3_Time.png")

    print(f'Mean iteration time: {1000 * np.mean(delta_t):.1f}ms -- {1 / np.mean(delta_t):.0f}Hz')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted")
    else:
        print("Complete Execution")
