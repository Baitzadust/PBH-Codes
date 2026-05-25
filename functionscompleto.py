# File: functions.py
## Module with functions
### Version BETA
# --- INSERTA ESTO AL INICIO DE functions.py ---
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController
import jax.lax as lax
import numpy as np 
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from numpy import diff
import sys
from scipy import special
from PBHBeta import constants
from PBHBeta import constraints


    
def put_M_array(Mass_min, Mass_max):
    """
    Generate an array of primordial black hole (PBH) masses in grams based on specified limits.

    Parameters:
        Mass_min (float): The minimum PBH mass value in grams.
        Mass_max (float): The maximum PBH mass value in grams.

    Returns:
        np.ndarray: An array of calculated PBH masses.
    """
    i = 0
    M = 0
    delta_M = 0.0123
    M_tot_try = []
    num_values = 20

    mass_array = np.geomspace(Mass_min, 10**(i*delta_M) , num_values)

    while M < constraints.data_mass[0]:
        M = 10**(i*delta_M)
        M_tot_try.append(M)
        i = i+1

    M_tot_try = np.concatenate((mass_array, M_tot_try, constraints.data_mass))
    M_tot_try = np.unique(M_tot_try)
    M = M_tot_try[-1]

    A = M
    j = 0

    while M < Mass_max:
        j = j+1
        M = A*10**(j*delta_M)
        M_tot_try = np.append(M_tot_try,[M])

    constraints.M_tot = np.array(M_tot_try)

    return constraints.M_tot

@eqx.filter_jit
@jax.vmap  
def precalcular_acreccion_lote(Mi_val_g):
    """
    Simulación vectorizada rigurosa. 
    Toma gramos, opera en Unidades de Planck (Hawking exacto) y devuelve gramos.
    """
    # 1. ESCALAS FÍSICAS (PBHBeta)
    M_pl_GeV = 1.22089e19
    M_pl_g = 2.17645e-5
    H_end_GeV = 4.44e13 
    
    # 2. TRANSFORMACIÓN A UNIDADES DE PLANCK (M_pl = 1)
    M_i_pl = Mi_val_g / M_pl_g
    H_end_pl = H_end_GeV / M_pl_GeV

    # Parámetros del campo escalar adaptados al sistema natural
    n = 1000.0
    mu_pl = n * H_end_pl 
    
    # Corrección física: Se incluye el 8*pi de la ecuación de Friedmann
    rho_end_inf_pl = (3.0 * H_end_pl**2.0) / (8.0 * jnp.pi)
    
    # Densidad de Reheating convertida a unidades de Planck
    rho_end_reh_pl = 1e-8 / M_pl_GeV**4 
    
    phi_ini_pl = jnp.sqrt(2.0 * rho_end_inf_pl / (mu_pl**2.0 + 9.0 * H_end_pl**2.0 / 4.0))

    # Cálculo de E-folds
    M_end_pl = 1.0 / H_end_pl
    N_ini = (2.0 / 3.0) * jnp.log(M_i_pl / M_end_pl)
    N_fin = (1.0 / 3.0) * jnp.log(rho_end_inf_pl / rho_end_reh_pl) 
    
    # Constante de evaporación de Hawking (Exacta en G=c=h=1)
    alpha_evap = 1.0 / (15.0 * 256.0 * jnp.pi)

    # 3. FUNCIONES DEL CAMPO ESCALAR OSCILANTE
    def Hubble(N):
        return H_end_pl * jnp.exp(-3.0 * N / 2.0)

    def rho_inf_field(N):
        phi0 = phi_ini_pl * jnp.exp(-3.0*N/2.0) * jnp.sin(2.0*mu_pl*jnp.exp(3.0*N/2.0)/(3.0*H_end_pl)) 
        Pi0 = (-3.0*phi0/2.0) + (phi_ini_pl*mu_pl*jnp.cos(2.0*mu_pl*jnp.exp(3.0*N/2.0)/(3.0*H_end_pl))/H_end_pl)
        return ((Hubble(N)*Pi0)**2.0)/2.0 + ((mu_pl**2.0)*phi0**2.0)/2.0

    def C4(N, M, rho_inf_val):
        w = 0.0 
        a = 0.5 * M
        r_plus = M + jnp.sqrt(M**2.0 - a**2.0)
        u_c = M / (2.0 * r_plus)
        rho_h = 3.0 * M / (4.0 * jnp.pi * r_plus**3.0)
        return u_c * (rho_h / rho_inf_val)**(1.0 / (1.0 + w))

    def vector_field_field(N, M, args):
        # M_safe evita divisiones por 0 matemáticas
        M_safe = jnp.maximum(M, 1.1)
        
        rho_val = rho_inf_field(N) 
        C_4_val = C4(N, M_safe, rho_val)
        a = 0.5 * M_safe
        r_plus = M_safe + jnp.sqrt(M_safe**2.0 - a**2.0)
        
        # --- ACRECIÓN ---
        # Eficiencia calibrada. (Cambia este valor entre 1e-18 y 1e-22 para ajustar las gráficas)
        f_acc_base = 1e-38
        supresion = M_safe * mu_pl
        f_acc= f_acc_base * supresion
        
        tasa_acrecion = f_acc * 4.0 * jnp.pi * C_4_val * r_plus**2.0 * (rho_val / Hubble(N))
        
        # --- EVAPORACIÓN ---
        tasa_evap_real = - (alpha_evap / (M_safe**2.0 * Hubble(N)))
        
        # Regularización numérica para evitar colapsos al tender a 0
        tasa_evap_suave = jnp.maximum(tasa_evap_real, -10.0 * M_safe)
        
        tasa_total = tasa_acrecion + tasa_evap_suave
        
        apagador_suave = 0.5 * (1.0 + jnp.tanh((M - 1.5) * 10.0))
        return tasa_total * apagador_suave
    
    # 4. CONFIGURACIÓN DEL SOLVER
    term = ODETerm(vector_field_field)
    solver = Tsit5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5) 
    saveat = SaveAt(t1=True) 

    def integrar(_):
        sol = diffeqsolve(term, solver, t0=N_ini, t1=N_fin, dt0=None, 
                          y0=M_i_pl, stepsize_controller=stepsize_controller, 
                          saveat=saveat, max_steps=100000)
        return sol.ys[0]
        
    M_final_pl = lax.cond(N_ini < N_fin, integrar, lambda _: M_i_pl, operand=None)
    
    # 5. RETORNO AL SISTEMA DE GRAMOS (Para PBHBeta)
    M_final_g = M_final_pl * M_pl_g
    
    return M_final_g, M_final_g / Mi_val_g

def diff_rad_rel(ln_rho,initial,M,beta0):

    """In the scenario where PBHs evaporate before reaching the energy scale of interest (as is the case, for example,
    before reaching the energy scale of BBN), we calculate the PBH abundance by assuming the existence of remnants with
    a mass equal to the Planck mass. Instead of simultaneously solving Eqs.(10) and (11) with the constraint Eq. (8), we
    focus on solving Eq.(10) with the constraint $\Omega_{PBH} = (m_{Pl}/M_{PBH})\beta(M_{PBH})$."""

    # Extract initial scale factor b and calculate Om_0
    b = initial[0]
    Om_0 = beta0 * b * (constants.M_pl_g / M)

    # Calculate the derivative of the scale factor b
    dy = -(Om_0 - 1.) * b / (Om_0 - 4.)

    return dy


def diff_rad(ln_rho,initial,M,beta0):
    """This function corresponds to Eqs.(10) and (11) with the constraint Eq.(8) in our reference paper. It is employed
    to calculate the abundance of PBHs in a radiation-dominated universe as a function of total energy density."""

    # Initialize dy array
    dy = np.zeros(initial.shape)

    # Extract initial values of scale factor b and time
    b = initial[0]
    time = initial[1]

    # Calculate Delta_t and Om_0
    Delta_t = constants.t_pl * (M / constants.M_pl_g) ** 3
    Om_0 = beta0 * b * (1. - time / Delta_t) ** (1. / 3)

    # Calculate the derivative of the scale factor b and the time derivative of the density of radiation
    dy[0] = -(Om_0 - 1.) * b / (Om_0 - 4.)
    dy[1] = 3 ** (1. / 2) * constants.M_pl / ((Om_0 - 4.) * np.exp(ln_rho) ** (1. / 2))

    return dy


def end_evol(ln_rho,initial,M,beta0):
    """This function is used to determine whether a PBH reaches the Planck mass (thus becoming a Planck relic) or not.
    By solving the system of equations (10) and (11) with the constraint (8) from our reference article, this function
    is used as a stopping condition for the evolution of the system. In the event that the evolution is halted before
    reaching the desired energy scale (such as the scale of BBN), the evolution of PBHs is carried out considering them
    as Planck mass relics."""
    # Calculate Delta_t and Mass_end
    Delta_t = constants.t_pl * (M / constants.M_pl_g) ** 3
    Mass_end = M * (1. - diff_rad(ln_rho,initial,M,beta0)[1] / Delta_t) ** (1. / 3)

    # Return the difference between the final mass of a system and the Planck mass
    return Mass_end - constants.M_pl_g



def k_end_over_k(Mpbh, omega):
    """
    Calculates the ratio of k_end/k for a given PBH mass and radiation energy density parameter.

    Parameters:
        - Mpbh (float): The mass of the PBH, in grams.
        - omega (float): The energy density parameter for radiation.

    Returns:
        - ratio (float): The ratio of k_end/k for the given PBH mass and radiation energy density parameter.
    """
    if omega==1/3:
        res = (Mpbh/(7.1*10**-2*constants.gam_rad*(1.8*10**15/constants.H_end)))**(1/2)
    else:
        z = (1+3*omega)/(3*(1 + omega))
        ratio = (Mpbh*constants.H_end/(3*constants.gam_rad*(constants.M_pl**2.)))**z
        res = np.array(ratio)
    return res


def rho_f(Mpbh, omega):
    """
    Calculates the final density of black holes after evaporation.

    Parameters:
        - Mpbh (float): The initial mass of a black hole, in grams.
        - omega (float): The ratio of the energy density of dark matter to the critical density of the universe.

    Returns:
        - rho (float): The final density of black holes, in grams per cubic centimeter.
    """
    if omega==1/3:
        k_end_over_k_rad = (Mpbh/(7.1*10**-2*constants.gam_rad*(1.8*10**15/constants.H_end)))**(1/2)
        rho_f = constants.rho_end_inf/(k_end_over_k_rad)**4
    else:
        z = (1+3*omega)/(3*(1 + omega))
        ratio = (Mpbh*constants.H_end/(3*constants.gam_rad*(constants.M_pl**2.)))**z
        res = np.array(ratio)
        i = (6*(1 + omega))/(1+(3*omega))
        rho_f = constants.rho_end_inf/(res)**i
    return rho_f



ln_den_end = np.log(constants.rho_end)


def Betas_DM(M_tot, omega, M_f_tot=None, mu_tot=None):
    M_n, betas_prim, M_relic, betas_relic_prim = [], [], [], []
    betas_tot, Omegas_tot, Omegas, Omegas_relic_pbbn = [], [], [], []
    Omegas_relic, M_dm, M_dm_rel_pbbn, M_dm_rel = [], [], [], []

    M_pl_g = 2.17645e-5
    t_pl_s = 5.39 * 10 ** -44
    s_to_evm1 = (1. / 6.5823) * 10 ** 25
    t_pl = t_pl_s * s_to_evm1
    gam_rad = (1. / 3) ** (3. / 2)

    rho_form_rad = rho_f(M_tot, omega)
    rho_end = (1e-2) ** 4
    ln_den_end = np.log(rho_end)

    # --- CICLO 1: LECTURA DE ACRECIÓN Y CÁLCULO DE BETA ---
    for i in range(len(M_tot)):
        M_i = M_tot[i]
        
        # Leemos el lote precalculado
        if M_f_tot is not None and mu_tot is not None:
            M_f, mu = M_f_tot[i], mu_tot[i]
        else:
            M_f, mu = M_i, 1.0 
            
        if M_f <= 1.0: 
            beta = constants.ev1
        elif M_f > 4.1 * 10 ** 14:
            M_n.append(M_i) 
            beta_std = 1.86 * 10 ** -18 * (M_f / (10 ** 15)) ** (1 / 2)
            beta = beta_std / mu
            betas_prim.append(beta)
        elif M_f < 10 ** 11 * constants.M_pl_g: 
            M_relic.append(M_i)
            beta_std = 2 * 10 ** -28 * (M_f / constants.M_pl_g) ** (3 / 2)
            beta = beta_std / mu
            betas_relic_prim.append(beta)
        else:
            beta = constants.ev1 / mu 
            
        betas_tot.append(beta / gam_rad ** (1 / 2))

    betas_prim = np.array(betas_prim)
    betas_relic_prim = np.array(betas_relic_prim)
    betas_tot = np.array(betas_tot)
    constraints.betas_DM_tot = betas_tot

    M_n = np.array(M_n)
    M_relic = np.array(M_relic)
    betas = betas_prim / constants.gam_rad ** (1 / 2)
    betas_relic = betas_relic_prim / constants.gam_rad ** (1 / 2)
    
    # --- CICLO 2: EVOLUCIÓN EN RADIACIÓN ---
    for i in range(len(constraints.betas_DM_tot)):
        if M_f_tot is not None: M_f = M_f_tot[i]
        else: M_f = M_tot[i]

        if constraints.betas_DM_tot[i] == constants.ev1 / gam_rad ** (1 / 2):
            Omegas_tot.append(constants.ev2)
            continue
            
        ln_den_f = np.log(rho_form_rad[i])
        if ln_den_f <= ln_den_end:
            Omegas_tot.append(constants.ev2)
            continue
            
        ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
        sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), events=end_evol, t_eval=ln_den,
                            args=(M_f, betas_tot[i]), method="DOP853")
        if sol_try.t[-1] > ln_den_end:
            sol_try = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end), np.array([1.]), t_eval=ln_den,
                                args=(M_f, betas_tot[i]), method="DOP853")
            y = betas_tot[i] * sol_try.y[0][-1] * (M_pl_g / M_f)
            if M_f < 10 ** 11 * M_pl_g:
                Omegas_relic_pbbn.append(y)
                M_dm_rel_pbbn.append(M_tot[i])
        else:
            Delta_t = t_pl * (M_f / M_pl_g) ** 3
            y = betas_tot[i] * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t) ** (1. / 3)
            if M_f > 4.1 * 10 ** 14:
                Omegas.append(y)
                M_dm.append(M_tot[i])
            elif M_f < 10 ** 11 * M_pl_g:
                Omegas_relic.append(y)
                M_dm_rel.append(M_tot[i])
                
        Omegas_tot.append(y)

    Omegas_tot = np.array(Omegas_tot)
    constraints.Omega_DM_tot = Omegas_tot
    return M_n, betas, M_relic, betas_relic, Omegas_tot


    
def Betas_BBN(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_bbn, M_bbn, M_bbn_bbn = [], [], []
    Omegas_bbn, Omegas_bbn_tot, Omegas_bbn_pbbn, M_bbn_pbbn = [], [], [], []

    rho_form_rad = rho_f(M_tot, omega)
    
    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_f_tot is not None and mu_tot is not None:
            M_f, mu = M_f_tot[i], mu_tot[i]
        else:
            M_f, mu = M_i, 1.0

        if M_f <= 1.0: 
            constraints.betas_BBN_tot.append(constants.ev1)
            Omegas_bbn_tot.append(constants.ev2)
            continue
            
        ln_den_f = np.log(rho_form_rad[i])

        if constraints.data_mass[0] <= M_f < constraints.data_mass[76]:
            M_bbn.append(M_i) 
            abundancia_limite = np.interp(M_f, constraints.data_mass, constraints.data_abundances)
            beta = (abundancia_limite / constants.gam_rad**(1/2)) / mu
            betas_bbn.append(beta)
        
            if ln_den_f > ln_den_end:
                ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), 
                                    events=end_evol, t_eval=ln_den, args=(M_f, beta), method="DOP853")
                if sol_try.t[-1] > ln_den_end:
                    sol_try = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end), np.array([1.]), 
                                        t_eval=ln_den, args=(M_f, beta), method="DOP853")
                    y = beta * sol_try.y[0][-1] * (constants.M_pl_g / M_f)
                    Omegas_bbn_pbbn.append(y)
                    M_bbn_pbbn.append(M_i)
                else:
                    Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                    y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                    Omegas_bbn.append(y)
                    M_bbn_bbn.append(M_i)
                    
        elif constraints.data_mass[76] <= M_f < 2.5 * 10**13:
            M_bbn.append(M_i)
            abundancia_limite = np.interp(M_f, constraints.data_mass, constraints.data_abundances)
            beta = (abundancia_limite / constants.gam_rad**(1/2)) / mu
            betas_bbn.append(beta)
        
            if ln_den_f > ln_den_end:
                ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), 
                                    events=end_evol, t_eval=ln_den, args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_bbn.append(y)
                M_bbn_bbn.append(M_i)
        else:
            beta = constants.ev1
            y = constants.ev2
            
        constraints.betas_BBN_tot.append(beta)
        Omegas_bbn_tot.append(y)

    constraints.Omega_BBN_tot = np.array(Omegas_bbn_tot)
    return np.array(M_bbn), np.array(betas_bbn), np.array(Omegas_bbn_tot)

def Betas_SD(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_sd, M_sd, M_sd_bbn, Omegas_sd = [], [], [], []
    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_f_tot is not None and mu_tot is not None:
            M_f, mu = M_f_tot[i], mu_tot[i]
        else:
            M_f, mu = M_i, 1.0

        if M_f > 1.0 and M_f > 10**11 and M_f < 10**13:
            M_sd.append(M_i) 
            beta = (10**(-21)/constants.gam_rad**(1/2)) / mu 
            betas_sd.append(beta)
        
            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end:
                ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), events=end_evol, t_eval=ln_den, args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_sd.append(y)
                M_sd_bbn.append(M_i)
        else:
            beta = constants.ev1
            y = constants.ev2
            
        constraints.betas_SD_tot.append(beta)
        constraints.Omega_SD_tot.append(y)
    
    return np.array(M_sd), np.array(betas_sd), np.array(Omegas_sd)

def Betas_CMB_AN(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_an, M_an, M_an_bbn, Omegas_an = [], [], [], []
    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_f_tot is not None and mu_tot is not None:
            M_f, mu = M_f_tot[i], mu_tot[i]
        else:
            M_f, mu = M_i, 1.0

        if M_f > 1.0 and M_f > 2.5*10**13 and M_f < 2.4*10**14:
            M_an.append(M_i)
            beta = (3*10**(-30)*(M_f/10**13)**3.1/constants.gam_rad**(1/2)) / mu
            betas_an.append(beta)
        
            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end:
                ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), events=end_evol, t_eval=ln_den, args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_an.append(y)
                M_an_bbn.append(M_i)
        else:
            beta = constants.ev1
            y = constants.ev2
            
        constraints.betas_CMB_AN_tot.append(beta)
        constraints.Omega_CMB_AN_tot.append(y)
    
    return np.array(M_an), np.array(betas_an), np.array(Omegas_an)


def Betas_GRB(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_grb1, M_grb1, betas_grb2, M_grb2 = [], [], [], []
    M_grb1_bbn, Omegas_grb1, M_grb2_bbn, Omegas_grb2 = [], [], [], []
    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_f_tot is not None and mu_tot is not None:
            M_f, mu = M_f_tot[i], mu_tot[i]
        else:
            M_f, mu = M_i, 1.0

        if M_f > 1.0 and M_f > 3*10**13 and M_f < 4.1*10**14:
            M_grb1.append(M_i)
            beta = (5*10**(-28)*(M_f/(4.1*10**14))**-3.3/constants.gam_rad**(1/2)) / mu
            betas_grb1.append(beta)
        
            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end:
                ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), events=end_evol, t_eval=ln_den, args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_grb1.append(y)
                M_grb1_bbn.append(M_i)
        
        elif M_f > 1.0 and M_f >= 4.1*10**14 and M_f < 7*10**16:
            M_grb2.append(M_i)
            beta = (5*10**(-26)*(M_f/(4.1*10**14))**3.9/constants.gam_rad**(1/2)) / mu
            betas_grb2.append(beta)
        
            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end:
                ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), events=end_evol, t_eval=ln_den, args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_grb2.append(y)
                M_grb2_bbn.append(M_i)
        else:
            beta = constants.ev1
            y = constants.ev2
            
        constraints.betas_GRB_tot.append(beta)
        constraints.Omega_GRB_tot.append(y)
    
    return np.array(M_grb1), np.array(M_grb2), np.array(betas_grb1), np.array(betas_grb2), np.array(Omegas_grb1), np.array(Omegas_grb2)

def Betas_Reio(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_reio, M_reio, M_reio_bbn, Omegas_reio = [], [], [], []
    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_f_tot is not None and mu_tot is not None:
            M_f, mu = M_f_tot[i], mu_tot[i]
        else:
            M_f, mu = M_i, 1.0

        if M_f > 1.0 and M_f > 10**15 and M_f < 10**17:
            M_reio.append(M_i)
            beta = (2.4*10**(-26)*(M_f/(4.1*10**14))**4.3/constants.gam_rad**(1/2)) / mu
            betas_reio.append(beta)
        
            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end:
                ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), events=end_evol, t_eval=ln_den, args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_reio.append(y)
                M_reio_bbn.append(M_i)
        else:
            beta = constants.ev1
            y = constants.ev2
            
        constraints.betas_Reio_tot.append(beta)
        constraints.Omega_Reio_tot.append(y)
    
    return np.array(M_reio), np.array(betas_reio), np.array(Omegas_reio)

def Betas_LSP(M_tot, w, M_f_tot=None, mu_tot=None):
    ev1, ev2 = 1e-5, 1e-2
    M_pl, gam_rad = 1.22089*10**19, (1./3)**(3./2)
    t_pl_s, s_to_evm1 = 5.39 * 10**-44, (1./6.5823) * 10**25
    t_pl = t_pl_s * s_to_evm1
    
    betas_lsp, betas_lsp_tot = [], []
    M_lsp, M_lsp_bbn, Omegas_lsp = [], [], []
    M_lsp_pbbn, Omegas_lsp_pbbn, Omegas_lsp_tot = [], [], []

    rho_form_rad = rho_f(M_tot, w)
    rho_end = (1e-2)**4
    ln_den_end = np.log(rho_end)

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_f_tot is not None and mu_tot is not None:
            M_f, mu = M_f_tot[i], mu_tot[i]
        else:
            M_f, mu = M_i, 1.0

        if M_f > 1.0 and M_f < 10**11:
            M_lsp.append(M_i)
            beta = (10**(-18) * (M_f / (10**11))**(-1/2) / gam_rad**(1/2)) / mu
            betas_lsp.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end:
                ln_den = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]), events=end_evol, t_eval=ln_den, args=(M_f, beta), method="DOP853")
                
                if sol_try.t[-1] > ln_den_end:
                    sol_try = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end), np.array([1.]), t_eval=ln_den, args=(M_f, beta), method="DOP853")
                    y = beta * sol_try.y[0][-1] * (constants.M_pl_g / M_f)
                    Omegas_lsp_pbbn.append(y)
                    M_lsp_pbbn.append(M_i)
                else:
                    Delta_t = t_pl * (M_f / constants.M_pl_g)**3
                    y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                    Omegas_lsp.append(y)
                    M_lsp_bbn.append(M_i)
        else:
            beta = ev1
            y = ev2
            
        constraints.betas_LSP_tot.append(beta)
        constraints.Omega_LSP_tot.append(y)
        betas_lsp_tot.append(beta)
        Omegas_lsp_tot.append(y)

    return np.array(M_lsp), np.array(betas_lsp), np.array(Omegas_lsp_tot)

def get_Betas_full(M_tot):
    """
    This function calculates composite constraint values derived from various PBH constraints.
        Parameters:
                - M_tot (array-like): Array of masses in grams.

        Returns:
                - betas_full (numpy.ndarray): Represent the most robust constraints across diverse scenarios for each specific mass value. This output is saved in the module called constraints into variable named betas_full.
    """

    DM_tot = np.array(constraints.betas_DM_tot)
    BBN_tot = np.array(constraints.betas_BBN_tot)
    SD_tot = np.array(constraints.betas_SD_tot)
    CMB_tot = np.array(constraints.betas_CMB_AN_tot)
    GRB_tot = np.array(constraints.betas_GRB_tot)
    Reio_tot = np.array(constraints.betas_Reio_tot)
    LSP_tot = np.array(constraints.betas_LSP_tot)

    constraints.betas_full = M_tot * 0

    for i in range(len(M_tot)):
        # Collect only the non-empty arrays
        values = []
        if DM_tot.size: values.append(DM_tot[i])
        if BBN_tot.size: values.append(BBN_tot[i])
        if SD_tot.size: values.append(SD_tot[i])
        if CMB_tot.size: values.append(CMB_tot[i])
        if GRB_tot.size: values.append(GRB_tot[i])
        if Reio_tot.size: values.append(Reio_tot[i])
        if LSP_tot.size: values.append(LSP_tot[i])

        if values:  # Ensure there are values to calculate the minimum
            constraints.betas_full[i] = min(values)

    return constraints.betas_full


def get_Omegas_full(M_tot):
    """
    This function calculates composite constraint values derived from various PBH constraints.
        Parameters:
                - M_tot (array-like): Array of masses in grams.

        Returns:
                - Omegas_full (numpy.ndarray): Represent the most robust constraints across diverse scenarios for each specific mass value. This output is saved in the module called constraints into variable named Omegas_full.
    """

    DM_tot = np.array(constraints.Omega_DM_tot)
    BBN_tot = np.array(constraints.Omega_BBN_tot)
    SD_tot = np.array(constraints.Omega_SD_tot)
    CMB_tot = np.array(constraints.Omega_CMB_AN_tot)
    GRB_tot = np.array(constraints.Omega_GRB_tot)
    Reio_tot = np.array(constraints.Omega_Reio_tot)
    LSP_tot = np.array(constraints.Omega_LSP_tot)

    constraints.Omegas_full = M_tot * 0

    for i in range(len(M_tot)):
        # Collect only the non-empty arrays
        values = []
        if DM_tot.size: values.append(DM_tot[i])
        if BBN_tot.size: values.append(BBN_tot[i])
        if SD_tot.size: values.append(SD_tot[i])
        if CMB_tot.size: values.append(CMB_tot[i])
        if GRB_tot.size: values.append(GRB_tot[i])
        if Reio_tot.size: values.append(Reio_tot[i])
        if LSP_tot.size: values.append(LSP_tot[i])

        if values:  # Ensure there are values to calculate the minimum
            constraints.Omegas_full[i] = min(values)

    return constraints.Omegas_full



def inverse_error(betas, delta_c):
    aux = []
    for i in range(len(betas)):
        aux.append(delta_c/(np.sqrt(2)*special.erfcinv(betas[i])))
    return aux


def a_endre(rho_r0, rho_end_re):
    return (rho_r0 / rho_end_re) ** (1. / 4)


def k_rad(M):
    a_end_inf_rad = (constants.rho_r0 / constants.rho_end_inf) ** (1. / 4)
    k_end = a_end_inf_rad * constants.H_end
    k_end_over_k_rad = (M/(7.1*10**-2*constants.gam_rad*(1.8*10**15/constants.H_end)))**(1/2)
    k = (k_end/k_end_over_k_rad)*constants.GeV*constants.metter_m1
    k = np.array(k)
    return k