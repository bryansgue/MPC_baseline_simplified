# MPC_UAV_ACADOS

Control predictivo por modelo (MPC) para UAV cuadrotor usando [acados](https://github.com/acados/acados) y CasADi. El controlador opera sobre un modelo simplificado con representación de orientación en cuaterniones y se ejecuta de forma **standalone** (sin dependencias de ROS).

---

## Archivos principales

| Archivo | Descripción |
|---|---|
| `T_MPC_SimpleModel_Quat_external.py` | Punto de entrada. Configura y ejecuta el MPC en lazo cerrado. |
| `Functions_SimpleModel.py` | Modelo simbólico CasADi, integradores, utilidades de cuaterniones. |
| `fancy_plots.py` | Funciones de visualización. |

---

## Cambios recientes

### Eliminación de dependencias ROS1
El archivo `T_MPC_SimpleModel_Quat_external.py` y `Functions_SimpleModel.py` fueron migrados para ejecutarse **sin ROS**:

- Removidos: `rospy`, `nav_msgs`, `geometry_msgs`, `std_msgs`
- `rospy.Rate` → `time.sleep` con control de timing manual
- `rospy.Subscriber / Publisher / init_node` → eliminados
- `send_velocity_control(u, pub, msg)` → `send_control(u)` que imprime el control por consola
- `pub_odometry_sim_quat(...)` → stub vacío (estado se mantiene internamente)
- `publish_matrix(...)` → stub vacío
- `get_odometry_simple_quat()` → `get_initial_state()` con estado inicial fijo
- `odometry_call_back` → eliminado

### Actualización de API acados
- `ocp.dims.N` → `ocp.solver_options.N_horizon` (nueva API)
- `ocp.p = model.p` → eliminado (causaba error de serialización MX al volcar JSON)
- Las expresiones de costo ahora referencian `model.p` directamente

### Actualización de CasADi
- Reemplazados múltiples `from casadi import X` → `import casadi as ca`
- Uso estilo moderno: `ca.MX.zeros(...)`, `ca.MX.sym(...)`, etc.

---

## Modelo del sistema en forma matricial

El modelo está implementado en `f_system_simple_model_quat()` dentro de `Functions_SimpleModel.py`.

### Vector de estados (n=11)

```
x = [nx, ny, nz, qw, qx, qy, qz, ul, um, un, w]ᵀ
```

| Variable | Descripción |
|---|---|
| `nx, ny, nz` | Posición en marco inercial |
| `qw, qx, qy, qz` | Orientación como cuaternión unitario |
| `ul, um, un` | Velocidades lineales en marco cuerpo (surge, sway, heave) |
| `w` | Velocidad angular de guiñada (yaw rate) |

### Vector de control (m=4)

```
u = [ul_ref, um_ref, un_ref, w_ref]ᵀ
```

Representan las velocidades de referencia de entrada al modelo dinámico.

### Dinámica explícita

El modelo completo tiene la forma **lineal por partes** (afín en x y u):

```
ẋ = A(x) · x + B · u
```

La matriz A depende del estado (a través de la rotación J(q) y la matriz de Coriolis C(w)), por lo que el sistema es **no lineal** aunque está escrito en estructura matricial por bloques. La dinámica se divide en tres subsistemas:

---

#### 1. Cinemática de posición

```
[ṅx]         [ul]
[ṅy] = J(q)·[um]
[ṅz]         [un]
```

Donde J(q) ∈ ℝ³ˣ³ es la matriz de rotación obtenida del cuaternión mediante la fórmula de Rodrigues (función `QuatToRot`):

```
J(q) = I₃ + 2·q̂² + 2·q₀·q̂
```

Con q̂ la matriz antisimétrica de la parte vectorial del cuaternión normalizado:

```
q̂ = [  0  , -qz,  qy ]
    [  qz ,   0 , -qx ]
    [ -qy ,  qx ,   0 ]
```

En la estructura matricial A esto corresponde al bloque:

```
A₁ = [ 0₃ₓ₇ | J(q) | 0₃ₓ₁ ]   ∈ ℝ³ˣ¹¹
```

---

#### 2. Cinemática del cuaternión

La evolución del cuaternión bajo velocidad angular [p, q, r] = [0, 0, w] es:

```
q̇ = ½ · S(ω) · q
```

Con la matriz de multiplicación cuaterniónica S(ω):

```
S(ω) = [ 0,  -p,  -q,  -r ]     Con p=0, q=0, r=w:
        [ p,   0,   r,  -q ]
        [ q,  -r,   0,   p ]     S(w) = [  0,  0,  0, -w ]
        [ r,   q,  -p,   0 ]            [  0,  0,  w,  0 ]
                                         [  0, -w,  0,  0 ]
                                         [  w,  0,  0,  0 ]
```

En la estructura matricial A este bloque es:

```
A₂ = [ 0₄ₓ₃ | ½·S(w) | 0₄ₓ₄ ]   ∈ ℝ⁴ˣ¹¹
```

---

#### 3. Dinámica de velocidades (modelo de segundo orden)

```
[u̇l]                  [ul]
[u̇m] = -M⁻¹·C(w) ·  [um]  +  M⁻¹ · u
[u̇n]                  [un]
[ẇ ]                  [ w]
```

Donde:
- **M** ∈ ℝ⁴ˣ⁴ es la matriz de masa/inercia (constante, función de parámetros identificados `chi`):

```
M = [ χ₀,   0,   0,  0   ]
    [  0,  χ₂,   0,  0   ]
    [  0,   0,  χ₄,  0   ]
    [  0,   0,   0,  χ₈  ]
```

- **C(w)** ∈ ℝ⁴ˣ⁴ es la matriz de Coriolis/amortiguamiento (depende del yaw rate w):

```
C(w) = [ χ₉,    w·χ₁₀,  0,     0     ]
        [ w·χ₁₂, χ₁₃,   0,     0     ]
        [  0,     0,    χ₁₅,   0     ]
        [  0,     0,     0,    χ₁₈   ]
```

En la estructura matricial A este bloque es:

```
A₃ = [ 0₄ₓ₇ | -M⁻¹·C(w) ]   ∈ ℝ⁴ˣ¹¹
```

---

#### Estructura completa

```
     [ A₁ ]   [ 0₃ₓ₇ |  J(q)    | 0₃ₓ₁      ]
A =  [ A₂ ] = [ 0₄ₓ₃ |  ½·S(w)  | 0₄ₓ₄      ]   ∈ ℝ¹¹ˣ¹¹
     [ A₃ ]   [ 0₄ₓ₇ | -M⁻¹·C(w)            ]


     [ 0₇ₓ₄ ]
B =  [       ]   ∈ ℝ¹¹ˣ⁴
     [ M⁻¹  ]
```

La ecuación de estado queda:

```
ẋ = A(x) · x + B · u
```

El integrador numérico utilizado es **Runge-Kutta de orden 4 (RK4)** implementado en `f_d()`.

---

## Función de costo del MPC

El problema de control óptimo resuelve en cada instante:

```
min  Σ_{k=0}^{N-1} [ eₚᵀ Q eₚ + uᵀ R u + log(qₑ)ᵀ K log(qₑ) ]
 u                + eₚ_N ᵀ Q eₚ_N + log(qₑ_N)ᵀ K log(qₑ_N)
```

Donde:
- **eₚ = p_d - p** ∈ ℝ³ : error de posición
- **qₑ = q⁻¹ ⊗ q_d** : error de orientación en cuaternión (producto cuaterniónico)
- **log(qₑ)** ∈ ℝ³ : mapa logarítmico del cuaternión de error (distancia geodésica en SO(3))
- **Q = diag(1.1, 1.1, 1.1)** : peso posición
- **K = diag(1.1, 1.1, 1.1)** : peso orientación
- **R = diag(1, 1, 1, 1)** : peso control

### Mapa logarítmico del cuaternión

```
log(q) = 2 · arctan2(‖qᵥ‖, q₀) · qᵥ / ‖qᵥ‖
```

Implementado en `log_cuaternion_casadi()`. Si q₀ < 0 se aplica q → -q antes para garantizar la rama principal.

---

## Parámetros del solver

| Parámetro | Valor |
|---|---|
| Horizonte N | 51 nodos |
| Tiempo de predicción | 51/30 ≈ 1.7 s |
| Frecuencia de control | 30 Hz |
| Integrador | ERK (Runge-Kutta explícito) |
| Solver NLP | SQP_RTI |
| Solver QP | FULL_CONDENSING_HPIPM |
| Tolerancia | 1e-3 |

---

## Ejecución

```bash
python3 T_MPC_SimpleModel_Quat_external.py
```

No requiere ROS ni ningún middleware. Los resultados se guardan en:
- `1_pose.png` — trayectoria vs referencia
- `2_error_pose.png` — error de posición
- `3_Time.png` — tiempo de cómputo por iteración
# MPC_baseline_simplified
# MPC_baseline_simplified
