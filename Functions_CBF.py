# Funciones para el filtro de seguridad CBF (orden 2) y la camara de profundidad simulada.
# Todo en marco cuerpo: el filtro solo necesita r_b (vector relativo al obstaculo),
# v = [ul, um, un] y w. No necesita la orientacion del dron.
import casadi as ca
import numpy as np

# Parametros del modelo simplificado (mismo chi que Functions_SimpleModel, a = b = 0)
chi = [0.6756, 1.0000, 0.6344, 1.0000, 0.4080, 1.0000, 1.0000, 1.0000, 0.2953,
       0.5941, -0.8109, 1.0000, 0.3984, 0.7040, 1.0000, 0.9365, 1.0000, 1.0000, 0.9752]


def M_matrix():
    # M = diag(chi0, chi2, chi4, chi8)
    M = np.diag([chi[0], chi[2], chi[4], chi[8]])
    return M


def C_matrix(w):
    C = np.zeros((4, 4))
    C[0, 0] = chi[9]
    C[0, 1] = w * chi[10]
    C[1, 0] = w * chi[12]
    C[1, 1] = chi[13]
    C[2, 2] = chi[15]
    C[3, 3] = chi[18]
    return C


def rot_quat(q):
    # Matriz de rotacion cuerpo -> inercial desde q = [qw, qx, qy, qz] (Rodrigues)
    q = np.array(q) / np.linalg.norm(q)
    q_hat = np.array([[0, -q[3], q[2]],
                      [q[3], 0, -q[1]],
                      [-q[2], q[1], 0]])
    R = np.eye(3) + 2 * q_hat @ q_hat + 2 * q[0] * q_hat
    return R


def rot_euler(psi):
    # Matriz de rotacion cuerpo -> inercial, solo yaw
    R = np.array([[np.cos(psi), -np.sin(psi), 0],
                  [np.sin(psi), np.cos(psi), 0],
                  [0, 0, 1]])
    return R


def camera_depth_sim(p, R, obstacles, sense_range, fov_deg):
    # Simula la camara de profundidad: devuelve el vector relativo de cada obstaculo
    # en marco cuerpo (lo que sale de pixel + depth) y si es visible.
    # p: posicion del dron (3,), R: rotacion cuerpo->inercial (3x3)
    n_obs = obstacles.shape[0]
    r_b = np.zeros((n_obs, 3))
    visible = np.zeros(n_obs, dtype=bool)
    half_fov = np.deg2rad(fov_deg) / 2
    for i in range(n_obs):
        r = R.T @ (obstacles[i, :] - p)          # inercial -> cuerpo
        dist = np.linalg.norm(r)
        bearing = np.arctan2(r[1], r[0])         # camara mira hacia +x cuerpo
        if dist <= sense_range and abs(bearing) <= half_fov:
            visible[i] = True
            r_b[i, :] = r
    return r_b, visible


def obstacle_memory(r_mem, age, r_meas, visible, v, w, ts, t_forget):
    # Guarda la ultima medida y la propaga con odometria cuando el obstaculo sale del FOV:
    #   r_b <- r_b - (v + w x r_b) * ts
    # Se olvida despues de t_forget segundos sin medida.
    omega = np.array([0, 0, w])
    for i in range(r_mem.shape[0]):
        if visible[i]:
            r_mem[i, :] = r_meas[i, :]
            age[i] = 0
        elif age[i] < t_forget:
            r_mem[i, :] = r_mem[i, :] - (v + np.cross(omega, r_mem[i, :])) * ts
            age[i] = age[i] + ts
    known = age < t_forget
    return r_mem, age, known


def cbf_terms(r_b, v, w, d_safe):
    # Barrera en marco cuerpo:
    #   h  = |r_b|^2 - d_s^2
    #   h' = -2 r_b' v
    #   h''= 2|v|^2 + 2 (w x r_b)' v - 2 r_b' v_dot,   v_dot = M_l^-1 u_l - [M^-1 C nu]_l
    # Devuelve h, h_dot y (a0, b) tales que h'' = a0 + b' u
    M = M_matrix()
    C = C_matrix(w)
    nu = np.array([v[0], v[1], v[2], w])
    omega = np.array([0, 0, w])

    h = r_b @ r_b - d_safe ** 2
    h_dot = -2 * r_b @ v
    v_dot_free = -(np.linalg.inv(M) @ C @ nu)[0:3]
    a0 = 2 * v @ v + 2 * np.cross(omega, r_b) @ v - 2 * r_b @ v_dot_free
    b = np.zeros(4)
    b[0:3] = -2 * r_b / np.diag(M)[0:3]
    return h, h_dot, a0, b


def create_cbf_qp(n_obs, rho):
    # QP:  min 1/2 |u - u_nmpc|^2 + rho/2 |delta|^2
    #      s.t. b_i' u + delta_i >= c_i ,  u_min <= u <= u_max
    # Variables z = [u (4), delta (n_obs)]
    nz = 4 + n_obs
    H = np.eye(nz)
    H[4:, 4:] = rho * np.eye(n_obs)
    H = ca.DM(H)
    A_sp = ca.DM.ones(n_obs, nz).sparsity()
    qp = ca.conic("cbf_qp", "qpoases", {"h": H.sparsity(), "a": A_sp}, {"printLevel": "none"})
    return qp, H


def cbf_filter(qp, H, u_nmpc, r_list, known, v, w, d_safe, alpha1, alpha2, u_min, u_max):
    # Filtro HOCBF orden 2:  h'' + (a1 + a2) h' + a1 a2 h >= 0  (lineal en u)
    n_obs = r_list.shape[0]
    nz = 4 + n_obs

    if not np.any(known):
        u = np.clip(u_nmpc, u_min, u_max)
        return u, False

    A = np.zeros((n_obs, nz))
    lba = -np.inf * np.ones(n_obs)
    for i in range(n_obs):
        A[i, 4 + i] = 1                       # slack
        if known[i]:
            h, h_dot, a0, b = cbf_terms(r_list[i, :], v, w, d_safe)
            A[i, 0:4] = b
            lba[i] = -a0 - (alpha1 + alpha2) * h_dot - alpha1 * alpha2 * h

    g = np.concatenate([-u_nmpc, np.zeros(n_obs)])
    lbx = np.concatenate([u_min, np.zeros(n_obs)])
    ubx = np.concatenate([u_max, np.inf * np.ones(n_obs)])
    uba = np.inf * np.ones(n_obs)

    sol = qp(h=H, g=g, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
    z = np.array(sol["x"]).reshape(-1)
    u = z[0:4]
    return u, True
