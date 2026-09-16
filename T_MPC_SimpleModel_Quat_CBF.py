# NMPC (modelo cuaternion) + filtro de seguridad CBF orden 2.
# Cadena por paso:  x_k -> NMPC (acados) -> u_nmpc -> QP-CBF -> u_safe -> planta
#
# El obstaculo lo ve una camara de profundidad: lo que recibe el filtro es el vector
# relativo r_b en marco cuerpo (pixel + depth -> punto 3D). La barrera vive en marco
# cuerpo, asi que el filtro no necesita la orientacion del dron.
#   h  = |r_b|^2 - d_s^2
#   h'' + (a1 + a2) h' + a1 a2 h >= 0      (ver Functions_CBF.py)
from acados_template import AcadosOcpSolver
import numpy as np
import time
import os

from fancy_plots import plot_pose, plot_error, plot_time
from fancy_plots import plot_cbf_distance, plot_cbf_control, plot_cbf_xy, plot_cbf_views, plot_camera
from Functions_SimpleModel import f_system_simple_model_quat, f_d, euler_to_quaternion
from Functions_CBF import rot_quat, camera_depth_sim, obstacle_memory, create_cbf_qp, cbf_filter
from T_MPC_SimpleModel_Quat_external import create_ocp_solver_description, get_initial_state


def main():
    # Opcion: "CBF" filtra la accion del NMPC, "NMPC" la manda directa
    opcion = "CBF"

    # Tiempos
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

    # Estado inicial
    x = np.zeros((11, t.shape[0] + 1 - N_prediction), dtype=np.double)
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
    quatd = np.zeros((4, t.shape[0]), dtype=np.double)
    for i in range(t.shape[0]):
        quatd[:, i] = euler_to_quaternion(0, 0, psid[i])

    xref = np.zeros((15, t.shape[0]), dtype=np.double)
    xref[0, :] = hxd
    xref[1, :] = hyd
    xref[2, :] = hzd
    xref[3:7, :] = quatd

    # OBSTACULOS (sobre la trayectoria deseada, para que el NMPC solo choque)
    r_obs = 0.4                      # radio fisico
    margin = 0.6                     # holgura
    d_safe = r_obs + margin          # distancia de barrera
    t_obs = [9.0, 24.0, 40.0]
    obstacles = np.zeros((3, 3))
    for i in range(3):
        obstacles[i, :] = [xd(t_obs[i]), yd(t_obs[i]), zd(t_obs[i])]
    n_obs = obstacles.shape[0]

    # CAMARA DE PROFUNDIDAD (simulada)
    sense_range = 6.0                # alcance [m]
    fov_deg = 87.0                   # FOV horizontal (RealSense D435)
    t_forget = 2.0                   # memoria fuera del FOV [s]
    r_mem = np.zeros((n_obs, 3))
    age = np.inf * np.ones(n_obs)

    # FILTRO CBF
    alpha1 = 1.5
    alpha2 = 1.5
    rho = 1e4                        # penalizacion del slack
    u_max = np.array([3.5, 3.5, 3.5, 2.0])
    u_min = -u_max
    qp, H = create_cbf_qp(n_obs, rho)

    # Limites (mismos que el baseline, no se usan en el costo)
    zp_ref_max = 3
    phi_max = 3
    theta_max = 3
    psi_max = 2

    # NMPC
    model, f = f_system_simple_model_quat()
    ocp = create_ocp_solver_description(x[:, 0], N_prediction, t_prediction,
                                        zp_ref_max, -zp_ref_max, phi_max, -phi_max,
                                        theta_max, -theta_max, psi_max, -psi_max)
    solver_json = 'acados_ocp_' + model.name + '.json'
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)

    nu = ocp.model.u.size()[0]
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Vectores para guardar
    n_steps = t.shape[0] - N_prediction
    u_nmpc = np.zeros((4, n_steps), dtype=np.double)
    u_safe = np.zeros((4, n_steps), dtype=np.double)
    dist = np.zeros((n_obs, n_steps), dtype=np.double)
    r_log = np.zeros((n_obs, 3, n_steps), dtype=np.double)
    vis_log = np.zeros((n_obs, n_steps), dtype=bool)
    known_log = np.zeros((n_obs, n_steps), dtype=bool)
    cbf_active = np.zeros(n_steps, dtype=bool)
    Error = np.zeros((3, n_steps), dtype=np.double)

    for k in range(0, n_steps):
        tic = time.time()
        Error[:, k] = xref[0:3, k] - x[0:3, k]

        # ---- NMPC ----
        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])
        for j in range(N_prediction):
            acados_ocp_solver.set(j, "p", xref[:, k + j])
        acados_ocp_solver.set(N_prediction, "p", xref[:, k + N_prediction])

        status = acados_ocp_solver.solve()
        if status != 0:
            print("acados status", status, "en k =", k)
        u_nmpc[:, k] = acados_ocp_solver.get(0, "u")

        # ---- Camara de profundidad + memoria ----
        p = x[0:3, k]
        q = x[3:7, k]
        v = x[7:10, k]
        w = x[10, k]
        R = rot_quat(q)                                   # solo el simulador usa la orientacion
        r_meas, visible = camera_depth_sim(p, R, obstacles, sense_range, fov_deg)
        r_mem, age, known = obstacle_memory(r_mem, age, r_meas, visible, v, w, t_s, t_forget)
        r_log[:, :, k] = r_mem
        vis_log[:, k] = visible
        known_log[:, k] = known

        # ---- Filtro CBF ----
        if opcion == "CBF":
            u_safe[:, k], cbf_active[k] = cbf_filter(qp, H, u_nmpc[:, k], r_mem, known, v, w,
                                                     d_safe, alpha1, alpha2, u_min, u_max)
        elif opcion == "NMPC":
            u_safe[:, k] = u_nmpc[:, k]
        else:
            print("Opcion no valida")

        # distancia real (solo para evaluar)
        for i in range(n_obs):
            dist[i, k] = np.linalg.norm(obstacles[i, :] - x[0:3, k])

        # ---- Planta ----
        x[:, k + 1] = f_d(x[:, k], u_safe[:, k], t_s, f)

        delta_t[:, k] = time.time() - tic
        sleep_time = t_s - delta_t[0, k]
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ---- Resultados ----
    if opcion == "CBF":
        carpeta = "cbf"
    else:
        carpeta = "nocbf"
    pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", carpeta)
    os.makedirs(pwd, exist_ok=True)

    fig1 = plot_pose(x, xref, t)
    fig1.savefig(os.path.join(pwd, "1_pose.png"), dpi=150)
    fig2 = plot_error(Error, t)
    fig2.savefig(os.path.join(pwd, "2_error_pose.png"), dpi=150)
    fig3 = plot_time(t_sample, delta_t, t)
    fig3.savefig(os.path.join(pwd, "3_Time.png"), dpi=150)
    fig4 = plot_cbf_distance(dist, d_safe, r_obs, t)
    fig4.savefig(os.path.join(pwd, "4_distance.png"), dpi=150)
    fig5 = plot_cbf_control(u_nmpc, u_safe, t)
    fig5.savefig(os.path.join(pwd, "5_control.png"), dpi=150)
    fig6 = plot_cbf_xy(x, xref, obstacles, r_obs, d_safe)
    fig6.savefig(os.path.join(pwd, "6_xy.png"), dpi=150)
    fig7 = plot_camera(r_log, vis_log, known_log, t, sense_range, fov_deg)
    fig7.savefig(os.path.join(pwd, "7_camera.png"), dpi=150)
    fig8 = plot_cbf_views(x, xref, obstacles, r_obs, d_safe)
    fig8.savefig(os.path.join(pwd, "8_side3d.jpg"), dpi=150)

    np.savez(os.path.join(pwd, "run_data.npz"), x=x, xref=xref, u_nmpc=u_nmpc, u_safe=u_safe,
             dist=dist, obstacles=obstacles, d_safe=d_safe, r_obs=r_obs, t=t, delta_t=delta_t)

    print("Figuras guardadas en", pwd)
    print(f'Mean iteration time: {1000 * np.mean(delta_t):.1f}ms -- {1 / np.mean(delta_t):.0f}Hz')
    print("Pasos con CBF activo:", cbf_active.sum(), "/", n_steps)
    for i in range(n_obs):
        print(f"obstaculo {i + 1}: distancia minima {dist[i, :].min():.3f} m  (d_safe {d_safe}, r_obs {r_obs})")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted")
    else:
        print("Complete Execution")
