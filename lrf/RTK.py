import numpy as np
from scipy.interpolate import splrep, BSpline


def unit_response_triangle(T, K, duration=200):
    response = np.zeros(duration)
    recession_time = K * T
    total_duration = int(np.ceil(T + recession_time))

    if total_duration > duration:
        total_duration = duration

    peak = 1 / (0.5 * T * (1 + K))  

    for t in range(total_duration):
        if t < T:
            response[t] = peak * (t / T)
        else:
            response[t] = peak * ((T + recession_time - t) / recession_time)

    return response


def combined_unit_response(params, duration=200):
    urf = np.zeros(duration)
    for i in range(3):
        R, T, K = params[i]
        triangle = unit_response_triangle(T, K, duration)
        urf += R * triangle
    return urf

def init_c_from_splrep(tau, urf, n_points, T0, k=3):
    tck = splrep(tau, urf, k=k, s=0)
    spline_func = lambda x: BSpline(*tck)(x)
    control_point_locs = np.linspace(0, T0, n_points)
    return spline_func(control_point_locs)

