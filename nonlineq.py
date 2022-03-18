"""
Generic module to solve nonlinear equations with the Newton's method
"""
import warnings

import numpy as np

DEFAULT_NEWTON_SOLVER_PRM = {
    'absolute_tolerance': 1e-8,
    'relative_tolerance': 1e-10,
    'maximum_iterations': 50
}

# from typing import TypeVar, Callable
# A = TypeVar('A')
# SUBPROB_TYPE = Callable[[A,], Tuple[Callable[[], A], Callable[[A,], A]]]
def newton_solve(x_0, linear_subproblem, norm=None, step_size=1.0, params=None):
    """
    Solve a non-linear problem with the Newton-Raphson method

    Parameters
    ----------
    x_0 : A
        Initial guess
    linear_subproblem : fn(A) -> (fn() -> A, fn(A) -> A)
        Callable returning a residual function and linear solver about a state.
    norm : fn(A) -> float
        Callable returning a norm for vectors of type `A`
    step_size : float
        Step size control for Newton-Raphson
    params : dict
        Dictionary of parameters for newton solver
        {'absolute_tolerance', 'relative_tolerance', 'maximum_iterations'}

    Returns
    -------
    xn : A
    info : dict
        Dictionary summarizing run info
    """
    _params = DEFAULT_NEWTON_SOLVER_PRM.copy()
    params = params if params is not None else {}
    _params.update(params)
    params = _params

    abs_tol = params['absolute_tolerance']
    rel_tol = params['relative_tolerance']
    max_iter = params['maximum_iterations']

    norm = generic_norm if norm is None else norm

    exit_status = -1
    abs_errs, rel_errs = [], []
    n = 0
    x_n = x_0
    while True:
        # compute the residual/residual norm and error measures
        assem_res, solve = linear_subproblem(x_n)
        res_n = assem_res()
 
        abs_err = norm(res_n)
        abs_errs.append(abs_err)
        rel_err = 0.0 if abs_errs[0] == 0 else abs_err/abs_errs[0]
        rel_errs.append(rel_err)

        # check for convergence of error measures/exit conditions
        if rel_errs[-1] <= rel_tol or abs_errs[-1] <= abs_tol:
            exit_status = 0
            exit_message = "solver converged"
        elif np.isnan(rel_errs[-1]) or np.isnan(abs_errs[-1]):
            exit_status = 1
            exit_message = "solver failed due to nan"
        elif n > max_iter:
            exit_status = 2
            exit_message = "solver reached maximum number of iterations"

        if exit_status == -1:
            dx_n = step_size*solve(res_n)
            x_n = x_n - dx_n
            n += 1
        else:
            break
            
    if exit_status == 2:
        warnings.warn(
            "Newton solve failed to converge before maximum"
            " iteration count reached.", UserWarning)

    info = {'status': exit_status,
            'message': exit_message,
            'abs_errs': np.array(abs_errs),
            'rel_errs': np.array(rel_errs)}
    return x_n, info

def generic_norm(x):
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x)
    else:
        return x.norm()
    
