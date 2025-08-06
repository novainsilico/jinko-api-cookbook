def pk_two_compartments_model(t, y, k12, k21, k_el):
    # differential equation
    ydot = [  # y[0] is A1, y[1] is A2
        k21 * y[1] - k12 * y[0] - k_el * y[0],
        k12 * y[0] - k21 * y[1],
    ]
    return ydot
