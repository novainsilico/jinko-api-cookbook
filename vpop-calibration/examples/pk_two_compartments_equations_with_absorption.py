def pk_two_compartments_model(t, y, k_a, k12, k21, k_el):
    # y[0] is A_absorption, y[1] is A_central, y[2] is A_peripheral
    A_absorption, A_central, A_peripheral = y[0], y[1], y[2]
    dA_absorption_dt = -k_a * A_absorption
    dA_central_dt = (
        k_a * A_absorption + k21 * A_peripheral - k12 * A_central - k_el * A_central
    )
    dA_peripheral_dt = k12 * A_central - k21 * A_peripheral

    ydot = [dA_absorption_dt, dA_central_dt, dA_peripheral_dt]
    return ydot
