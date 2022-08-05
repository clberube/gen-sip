# @Author: charles
# @Date:   2021-09-08 13:09:44
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2022-08-05 15:08:16


import numpy as np


epsilon_0 = 8.854e-12

mixture = {
    'log_a_i': np.log10(200e-6),
    'log_phi_i': np.log10(0.1),
    'log_D_i': np.log10(1e-6),
    'log_sigma_i': np.log10(100),
    'log_epsilon_i': np.log10(10*epsilon_0),
    'log_D_h': np.log10(1e-9),
    'log_sigma_h': np.log10(0.1),
    'log_epsilon_h': np.log10(80*epsilon_0),
}


def forward_spherical(w, log_a_i, log_phi_i,
                      log_D_i, log_sigma_i, log_epsilon_i,
                      log_D_h, log_sigma_h, log_epsilon_h):

    a_i = 10**log_a_i
    phi_i = 10**log_phi_i
    D_h = 10**log_D_h
    D_i = 10**log_D_i
    sigma_h = 10**log_sigma_h
    sigma_i = 10**log_sigma_i
    epsilon_h = 10**log_epsilon_h
    epsilon_i = 10**log_epsilon_i

    n = 2

    K_h = sigma_h + 1j*w*epsilon_h
    K_i = sigma_i + 1j*w*epsilon_i

    gamma_h = np.sqrt(1j*w/D_h + sigma_h/(epsilon_h*D_h))
    gamma_i = np.sqrt(1j*w/D_i + sigma_i/(epsilon_i*D_i))

    ag_i = a_i*gamma_i
    ag_h = a_i*gamma_h

    F_i_over_H_i = a_i*(ag_i - np.tanh(ag_i)) / \
        (2*ag_i - ag_i**2 * np.tanh(ag_i) - 2*np.tanh(ag_i))
    E_h_over_G_h = a_i*(ag_h + 1) / (ag_h**2 + 2*ag_h + 2)

    numerator = 3*1j*w
    term1 = (2 * sigma_h * E_h_over_G_h) / (a_i * epsilon_h)
    term2 = (2 * K_h * sigma_i * F_i_over_H_i) / (a_i * K_i * epsilon_i)
    term3 = 1j*w*(2 * K_h / K_i + 1)
    denominator = 2 * (term1 - term2 + term3)
    f_w = -0.5 + numerator/denominator
    K_eff = K_h * (phi_i*f_w*n + 1) / (1 - phi_i*f_w)

    return K_eff
