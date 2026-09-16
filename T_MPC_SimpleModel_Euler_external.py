# NMPC (acados) standalone para el modelo simplificado con yaw en Euler.
#   x = [nx, ny, nz, psi, ul, um, un, w]      u = [ul_ref, um_ref, un_ref, w_ref]
# Portado de T_MPC_SimpleModel_External.py (ROS1): sin rospy, API acados nueva,
# yaw de referencia activo con error envuelto atan2(sin e, cos e).
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca
import numpy as np
import time
import os

from fancy_plots import plot_pose, plot_error, plot_time
from Functions_SimpleModel import f_system_simple_model, f_d


def create_ocp_solver_description(x0, N_horizon, t_horizon, u_max) -> AcadosOcp:
    ocp = AcadosOcp()

    model, f_system = f_system_simple_model()
    model.name = 'Drone_ode_euler'                 # no pisar el build del modelo cuaternion
    ocp.model = model
    ocp.code_export_directory = 'c_generated_code_euler'
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    ocp.solver_options.N_horizon = N_horizon

    Q_mat = np.diag([1, 1, 1])                     # posicion
    K_psi = 1                                      # yaw
    R_mat = 0.2 * np.diag([1, 1, 1, 1])

    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = model.p[0:3] - model.x[0:3]
    e_psi = model.p[3] - model.x[3]
    e_psi = ca.atan2(ca.sin(e_psi), ca.cos(e_psi))   # envuelto a (-pi, pi]

    ocp.model.cost_expr_ext_cost = error_pose.T @ Q_mat @ error_pose + K_psi * e_psi**2 + model.u.T @ R_mat @ model.u
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose + K_psi * e_psi**2

    ocp.constraints.lbu = np.array([-u_max, -u_max, -u_max])
    ocp.constraints.ubu = np.array([u_max, u_max, u_max])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = t_horizon

    return ocp


def get_initial_state():
    # [x, y, z, psi, ul, um, un, w]
    return np.array([1.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def main():
    t_final = 60
    frec = 30
    t_s = 1 / frec
    N_horizont = frec                # horizonte 1 s (como en el script ROS1)
    t_prediction = N_horizont / frec

    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]
    t = np.arange(0, t_final + t_s, t_s)

    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s * np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    x = np.zeros((8, t.shape[0] + 1 - N_prediction), dtype=np.double)
    x[:, 0] = get_initial_state()

    # TAREA DESEADA
    value = 6
    xd = lambda t: 4 * np.sin(value * 0.04 * t) + 3
    yd = lambda t: 4 * np.sin(value * 0.08 * t)
    zd = lambda t: 2 * np.sin(value * 0.08 * t) + 6
    xdp = lambda t: 4 * value * 0.04 * np.cos(value * 0.04 * t)
    ydp = lambda t: 4 * value * 0.08 * np.cos(value * 0.08 * t)

    hxd = xd(t)
    hyd = yd(t)
    hzd = zd(t)
    hxdp = xdp(t)
    hydp = ydp(t)
    psid = np.arctan2(hydp, hxdp)

    ref = np.zeros((12, t.shape[0]), dtype=np.double)
    ref[0, :] = hxd
    ref[1, :] = hyd
    ref[2, :] = hzd
    ref[3, :] = psid

    u_control = np.zeros((4, t.shape[0] - N_prediction), dtype=np.double)
    u_max = 3

    model, f = f_system_simple_model()
    ocp = create_ocp_solver_description(x[:, 0], N_prediction, t_prediction, u_max)

    solver_json = 'acados_ocp_' + ocp.model.name + '.json'
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)

    nu = ocp.model.u.size()[0]
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    Error = np.zeros((3, t.shape[0] - N_prediction), dtype=np.double)

    for k in range(0, t.shape[0] - N_prediction):
        tic = time.time()
        Error[:, k] = ref[0:3, k] - x[0:3, k]

        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])
        for j in range(N_prediction):
            acados_ocp_solver.set(j, "p", ref[:, k + j])
        acados_ocp_solver.set(N_prediction, "p", ref[:, k + N_prediction])

        status = acados_ocp_solver.solve()
        if status != 0:
            print("acados status", status, "en k =", k)
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        x[:, k + 1] = f_d(x[:, k], u_control[:, k], t_s, f)

        delta_t[:, k] = time.time() - tic
        sleep_time = t_s - delta_t[0, k]
        if sleep_time > 0:
            time.sleep(sleep_time)

    pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "baseline_euler")
    os.makedirs(pwd, exist_ok=True)
    fig1 = plot_pose(x, ref, t)
    fig1.savefig(os.path.join(pwd, "1_pose.png"), dpi=150)
    fig2 = plot_error(Error, t)
    fig2.savefig(os.path.join(pwd, "2_error_pose.png"), dpi=150)
    fig3 = plot_time(t_sample, delta_t, t)
    fig3.savefig(os.path.join(pwd, "3_Time.png"), dpi=150)
    np.savez(os.path.join(pwd, "run_data.npz"), x=x, ref=ref, u=u_control, t=t, delta_t=delta_t)

    print("Figuras guardadas en", pwd)
    print(f'Mean iteration time: {1000 * np.mean(delta_t):.1f}ms -- {1 / np.mean(delta_t):.0f}Hz')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted")
    else:
        print("Complete Execution")
