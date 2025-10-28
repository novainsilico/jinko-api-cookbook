from .ode_model import OdeModel


def equations(t, y, k_12, k_21, k_el):
    # differential equation
    ydot = [  # y[0] is A1, y[1] is A2
        k_21 * y[1] - k_12 * y[0] - k_el * y[0],
        k_12 * y[0] - k_21 * y[1],
    ]
    return ydot


variable_names = ["A0", "A1"]
parameter_names = ["k_12", "k_21", "k_el"]

pk_two_compartments_model = OdeModel(equations, variable_names, parameter_names)
