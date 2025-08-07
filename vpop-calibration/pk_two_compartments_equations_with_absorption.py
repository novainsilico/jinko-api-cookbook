def pk_two_compartments_model(t, y, k_a, k12, k21, k_el):
    # differential equation
    ydot = [  # y[0] is A0, y[1] is A1, y[2] is A2
        -2 * k_a * y[0],
        k_a * y[0] + k21 * y[2] - k12 * y[1] - k_el * y[1],
        k_a * y[0] + k12 * y[1] - k21 * y[2],
    ]
    return ydot
