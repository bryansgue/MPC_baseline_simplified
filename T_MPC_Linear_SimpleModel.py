from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import horzcat
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from casadi import mtimes
from fancy_plots import fancy_plot
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
from geometry_msgs.msg import TwistStamped
import math
import os


# Global variables Odometry Drone
x_real = 0.0
y_real = 0.0
z_real = 5
vx_real = 0.0
vy_real = 0.0
vz_real = 0.0
qw_real = 1.0
qx_real = 0
qy_real = 0.0
qz_real = 0.0
wx_real = 0.0
wy_real = 0.0
wz_real = 0.0

# CARGA FUNCIONES DEL PROGRAMA
from fancy_plots import plot_pose, plot_error, plot_time
#from Functions_SimpleModel import f_system_simple_model
##from Functions_SimpleModel import f_d, odometry_call_back, get_odometry_simple, send_velocity_control, pub_odometry_sim

def FLUtoENU(u, quaternion):
    # Cuaterniones (qw, qx, qy, qz)
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    R_aux = R.from_quat([qx, qy, qz, qw])
    # Obtiene la matriz de rotación
    Rot = R_aux.as_matrix()
    v = Rot@u 

    return v

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]

def calc_M(chi, a, b):
    

    M = MX.zeros(4, 4)
    M[0,0] = chi[0]
    M[0,1] = 0
    M[0,2] = 0
    M[0,3] = b * chi[1]
    M[1,0] = 0
    M[1,1] = chi[2]
    M[1,2] = 0
    M[1,3] = a* chi[3]
    M[2,0] = 0
    M[2,1] = 0
    M[2,2] = chi[4]
    M[2,3] = 0
    M[3,0] = b*chi[5]
    M[3,1] = a* chi[6]
    M[3,2] = 0
    M[3,3] = chi[7]*(a**2+b**2) + chi[8]
    
    return M

def calc_C(chi, a, b, w):
    
    C = MX.zeros(4, 4)
    C[0,0] = chi[9]
    C[0,1] = w*chi[10]
    C[0,2] = 0
    C[0,3] = a * w * chi[11]
    C[1,0] = w*chi[12]
    C[1,1] = chi[13]
    C[1,2] = 0
    C[1,3] = b * w * chi[14]
    C[2,0] = 0
    C[2,1] = 0
    C[2,2] = chi[15]
    C[2,3] = 0
    C[3,0] = a *w* chi[16]
    C[3,1] = b * w * chi[17]
    C[3,2] = 0
    C[3,3] = chi[18]

    return C

def calc_G():
    G = MX.zeros(4, 1)
    G[0, 0] = 0
    G[1, 0] = 0
    G[2, 0] = 0
    G[3, 0] = 0

    return G


def calc_J(x, a, b):
    
    psi = x[3]

    RotZ = MX.zeros(4, 4)
    RotZ[0, 0] = cos(psi)
    RotZ[0, 1] = -sin(psi)
    RotZ[0, 2] = 0
    RotZ[0, 3] = -(a*sin(psi) + b*cos(psi))
    RotZ[1, 0] = sin(psi)
    RotZ[1, 1] = cos(psi)
    RotZ[1, 2] = 0 
    RotZ[1, 3] = (a*cos(psi) - b*sin(psi))
    RotZ[2, 0] = 0
    RotZ[2, 1] = 0
    RotZ[2, 2] = 1
    RotZ[2, 3] = 0
    RotZ[3, 0] = 0
    RotZ[3, 1] = 0
    RotZ[3, 2] = 0
    RotZ[3, 3] = 1

    J = RotZ

    return J

def pub_odometry_sim(state_vector, odom_sim_pub, odom_sim_msg):

    quaternion = euler_to_quaternion(0, 0, state_vector[3])
    u = [state_vector[4],state_vector[5],state_vector[6]]
    
    v = FLUtoENU(u, quaternion)

    odom_sim_msg.header.frame_id = "odo"
    odom_sim_msg.header.stamp = rospy.Time.now()
    odom_sim_msg.pose.pose.position.x = state_vector[0]
    odom_sim_msg.pose.pose.position.y = state_vector[1]
    odom_sim_msg.pose.pose.position.z = state_vector[2]
    odom_sim_msg.pose.pose.orientation.x = quaternion[1]
    odom_sim_msg.pose.pose.orientation.y = quaternion[2]
    odom_sim_msg.pose.pose.orientation.z = quaternion[3]
    odom_sim_msg.pose.pose.orientation.w = quaternion[0]
    odom_sim_msg.twist.twist.linear.x = v[0]
    odom_sim_msg.twist.twist.linear.y = v[1]
    odom_sim_msg.twist.twist.linear.z = v[2]
    odom_sim_msg.twist.twist.angular.x = 0
    odom_sim_msg.twist.twist.angular.y = 0
    odom_sim_msg.twist.twist.angular.z = state_vector[7]

    # Publish the message
    odom_sim_pub.publish(odom_sim_msg)

def send_velocity_control(u, vel_pub, vel_msg):
    # velocity message

    vel_msg.header.frame_id = "base_link"
    vel_msg.header.stamp = rospy.Time.now()
    vel_msg.twist.linear.x = u[0]
    vel_msg.twist.linear.y = u[1]
    vel_msg.twist.linear.z = u[2]
    vel_msg.twist.angular.x = 0
    vel_msg.twist.angular.y = 0
    vel_msg.twist.angular.z = u[3]

    # Publish control values
    vel_pub.publish(vel_msg)

def get_odometry_simple():
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real
    
    quaternion = [qx_real, qy_real, qz_real, qw_real ] # cuaternión debe estar en la convención "xyzw",
    r_quat = R.from_quat(quaternion)
    euler =  r_quat.as_euler('zyx', degrees = False)
    psi = euler[0]

    J = np.zeros((3, 3))
    J[0, 0] = np.cos(psi)
    J[0, 1] = -np.sin(psi)
    J[1, 0] = np.sin(psi)
    J[1, 1] = np.cos(psi)
    J[2, 2] = 1

    J_inv = np.linalg.inv(J)
    v = np.dot(J_inv, [vx_real, vy_real, vz_real])
 
    ul_real = v[0]
    um_real = v[1]
    un_real = v[2]

    x_state = [x_real,y_real,z_real,psi,ul_real,um_real,un_real, wz_real]

    return x_state

def odometry_call_back(odom_msg):
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real
    # Read desired linear velocities from node
    x_real = odom_msg.pose.pose.position.x 
    y_real = odom_msg.pose.pose.position.y
    z_real = odom_msg.pose.pose.position.z
    vx_real = odom_msg.twist.twist.linear.x
    vy_real = odom_msg.twist.twist.linear.y
    vz_real = odom_msg.twist.twist.linear.z

    qx_real = odom_msg.pose.pose.orientation.x
    qy_real = odom_msg.pose.pose.orientation.y
    qz_real = odom_msg.pose.pose.orientation.z
    qw_real = odom_msg.pose.pose.orientation.w

    wx_real = odom_msg.twist.twist.angular.x
    wy_real = odom_msg.twist.twist.angular.y
    wz_real = odom_msg.twist.twist.angular.z
    return None

def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)

    
    num = x.size()[0]  # Obtener el número de componentes en x automáticamente
    aux_x = np.array(x[:,0]).reshape((num,))
    return aux_x

def f_system_simple_model():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system

    chi = [0.6756,    1.0000,    0.6344,    1.0000,    0.4080,    1.0000,    1.0000,    1.0000,    0.2953,    0.5941,   -0.8109,    1.0000,    0.3984,    0.7040,    1.0000,    0.9365,    1.0000, 1.0000,    0.9752]# Position
    
    # set up states & controls
    # Position
    nx = MX.sym('nx') 
    ny = MX.sym('ny')
    nz = MX.sym('nz')
    psi = MX.sym('psi')
    ul = MX.sym('ul')
    um = MX.sym('um')
    un = MX.sym('un')
    w = MX.sym('w')

    # General vector of the states
    x = vertcat(nx, ny, nz, psi, ul, um, un, w)

    # Action variables
    ul_ref = MX.sym('ul_ref')
    um_ref = MX.sym('um_ref')
    un_ref = MX.sym('un_ref')
    w_ref = MX.sym('w_ref')

    # General Vector Action variables
    u = vertcat(ul_ref,um_ref,un_ref,w_ref)

    # Variables to explicit function
    nx_p = MX.sym('nx_p')
    ny_p = MX.sym('ny_p')
    nz_p = MX.sym('nz_p')
    psi_p = MX.sym('psi_p')
    ul_p = MX.sym('ul_p')
    um_p = MX.sym('um_p')
    un_p = MX.sym('un_p')
    w_p = MX.sym('w_p')

    # general vector X dot for implicit function
    xdot = vertcat(nx_p,ny_p,nz_p,psi_p,ul_p,um_p,un_p,w_p)

    # Ref system as a external value
    nx_d = MX.sym('nx_d')
    ny_d = MX.sym('ny_d')
    nz_d = MX.sym('nz_d')
    psi_d = MX.sym('psi_d')
    ul_d= MX.sym('ul_d')
    um_d= MX.sym('um_d')
    un_d = MX.sym('un_d')
    w_d = MX.sym('w_d')

    ul_ref_d= MX.sym('ul_ref_d')
    um_ref_d= MX.sym('um_ref_d')
    un_ref_d = MX.sym('un_ref_d')
    w_ref_d = MX.sym('w_ref_d')
    
    p = vertcat(nx_d, ny_d, nz_d, psi_d, ul_d, um_d, un_d, w_d, ul_ref_d, um_ref_d, un_ref_d, w_ref_d)

    # Rotational Matrix
    a = 0
    b = 0
    J = calc_J(x, a, b)
    M = calc_M(chi,a,b)
    C = calc_C(chi,a,b, w)
    G = calc_G()

    # Crear matriz A
    A_top = horzcat(MX.zeros(4, 4), J)
    A_bottom = horzcat(MX.zeros(4, 4), -mtimes(inv(M), C))
    A = vertcat(A_top, A_bottom)

    # Crear matriz B
    B_top = MX.zeros(4, 4)
    B_bottom = inv(M)
    B = vertcat(B_top, B_bottom)

    # Crear vector aux
    aux = vertcat(MX.zeros(4, 1), -mtimes(inv(M), G))

    f_expl = MX.zeros(8, 1)
    f_expl = A @ x + B @ u + aux

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p

    return model, f_system

def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system = f_system_simple_model()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    
    ny = nx + nu
    ny_e = nx

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost
    
    Q_mat = np.zeros((8, 8))
    Q_mat[0, 0] = 1.1
    Q_mat[1, 1] = 1.1
    Q_mat[2, 2] = 1.1
    Q_mat[3, 3] = 1

    R_mat = np.zeros((4, 4))
    R_mat[0, 0] = 1.3*(1/2)
    R_mat[1, 1] = 1.3*(1/2)
    R_mat[2, 2] = 1.3*(1/2)
    R_mat[3, 3] = 1.3*(1/2)

    ocp.parameter_values = np.zeros(ny)
    
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([-2, -2, -2])
    ocp.constraints.ubu = np.array([2, 2, 2])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    #ocp.solver_options.tol = 1e-4

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp


import os

def manage_ocp_solver(model, ocp):
    """
    Maneja la creación o uso del solver OCP.

    Args:
        model: El modelo utilizado para definir el solver.
        ocp: La descripción del OCP a gestionar.

    Returns:
        acados_ocp_solver: La instancia del solver creado o cargado.
    """
    # Definir el nombre del archivo JSON
    solver_json = 'acados_ocp_' + model.name + '.json'

    # Comprobar si el archivo JSON del solver ya existe
    if os.path.exists(solver_json):
        # Preguntar al usuario qué desea hacer
        respuesta = input(f"El solver {solver_json} ya existe. ¿Deseas usar el existente (U) o generarlo nuevamente (G)? [U/G]: ").strip().upper()
        
        if respuesta == 'U':
            print(f"Usando el solver existente: {solver_json}")
            # Crear el solver directamente desde el archivo existente
            return AcadosOcpSolver.create_cython_solver(solver_json)
        elif respuesta == 'G':
            print(f"Regenerando y reconstruyendo el solver: {solver_json}")
            # Generar y construir el solver
            AcadosOcpSolver.generate(ocp, json_file=solver_json)
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
            return AcadosOcpSolver.create_cython_solver(solver_json)
        else:
            raise ValueError("Opción no válida. No se realizó ninguna acción.")
    else:
        print(f"El solver {solver_json} no existe. Generando y construyendo...")
        # Generar y construir el solver
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        return AcadosOcpSolver.create_cython_solver(solver_json)


def main(vel_pub, vel_msg, odom_sim_pub, odom_sim_msg):
    # Initial Values System
    # Simulation Time
    t_final = 30
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = 30
    t_prediction = N_horizont/frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)


    # Vector Initial conditions
    x = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)
    x_sim = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)

    # Read Real data
    x[:, 0] = get_odometry_simple()

    #TAREA DESEADA
    num = 4
    xd = lambda t: 4 * np.sin(5*0.04*t) + 3
    yd = lambda t: 4 * np.sin(5*0.08*t)
    zd = lambda t: 2.5 * np.sin (0.2* t) + 5  
    xdp = lambda t: 4 * 5 * 0.04 * np.cos(5*0.04*t)
    ydp = lambda t: 4 * 5 * 0.08 * np.cos(5*0.08*t)
    zdp = lambda t: 2.5 * 0.2 * np.cos(0.2 * t)

    hxd = xd(t)
    hyd = yd(t)
    hzd = zd(t)
    hxdp = xdp(t)
    hydp = ydp(t)
    hzdp = zdp(t)

    psid = np.arctan2(hydp, hxdp)
    psidp = np.gradient(psid, t_s)

    # Reference Signal of the system
    xref = np.zeros((12, t.shape[0]), dtype = np.double)
    xref[0,:] = hxd 
    xref[1,:] = hyd
    xref[2,:] = hzd  
    xref[3,:] = psid 
    xref[4,:] = 0
    xref[5,:] = 0 
    xref[6,:] = 0 
    xref[7,:] = 0 
    # Initial Control values
    u_control = np.zeros((4, t.shape[0]-N_prediction), dtype = np.double)
    #u_control = np.zeros((4, t.shape[0]), dtype = np.double)

    # Limits Control values
    zp_ref_max = 3
    phi_max = 3
    theta_max = 3
    psi_max = 2

    zp_ref_min = -zp_ref_max
    phi_min = -phi_max
    theta_min = -theta_max
    psi_min = -psi_max

    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate

    #INICIALIZA LECTURA DE ODOMETRIA
    for k in range(0, 10):
        # Read Real data
        x[:, 0] = get_odometry_simple()
        # Loop_rate.sleep()
        rate.sleep() 
        print("Init System")

    # Create Optimal problem
    model, f = f_system_simple_model()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, zp_ref_max, zp_ref_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min)
    
         
    acados_ocp_solver = manage_ocp_solver(model, ocp)


    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Errors of the system
    Error = np.zeros((3, t.shape[0]-N_prediction), dtype = np.double)

    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()

        Error[:,k] = xref[0:3, k] - x[0:3, k]

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # update yref
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "yref", yref)
        
        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "yref", yref_N[0:8])

        # Get Computational Time
        status = acados_ocp_solver.solve()

        toc_solver = time.time()- tic

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")
        #u_control[:, k] = [0.0,0.0,0,0.0]
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)

        # System Evolution
        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_simple()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            pub_odometry_sim(x[:, k+1], odom_sim_pub, odom_sim_msg)
        else:
            print("Opción no válida")
        

        delta_t[:, k] = toc_solver


        print(1/toc_solver)
        
        #print("v_real:", " ".join("{:.2f}".format(value) for value in np.round(x[0:12, k], decimals=2)))
        
        rate.sleep() 
        toc = time.time() - tic 
        #print(1/toc)
        
    
    send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)

    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")
    fig2 = plot_error(Error, t)
    fig2.savefig("2_error_pose.png")
    fig3 = plot_time(t_sample, delta_t , t)
    fig3.savefig("3_Time.png")
  

    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        # SUCRIBER
        velocity_subscriber = rospy.Subscriber("/dji_sdk/odometry", Odometry, odometry_call_back)
        
        # PUBLISHER
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher("/m100/velocityControl", TwistStamped, queue_size=10)

        odometry_sim_msg = Odometry()
        odom_sim_pub = rospy.Publisher('/dji_sdk/odometry', Odometry, queue_size=10)

        main(velocity_publisher, velocity_message, odom_sim_pub, odometry_sim_msg)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("\nError System")
        send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        pass
