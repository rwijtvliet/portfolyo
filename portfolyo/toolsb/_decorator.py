"""Module with decorators."""

import functools
import inspect
from typing import Callable

# def convert_if_type(additional_type: type, conversion_fn: Callable):
#     """Create a decorator factory.
#
#     The decorators created by the factory check one or more parameters of the wrapped
#     function. If the function is called with a certain argument types, a conversion
#     function is applied. This ensures the wrapped function accepts more diverse argument
#     types, without having to carry out `isinstance` checks from inside the function.
#
#     Parameters
#     ----------
#     additional_type
#         The type that we want (one or more) parameters to also accept for their argument
#         value.
#     conversion_fn
#         A one-argument function that is called on the argument when it is of this type,
#         and which returns an instance of the type expected by the class.
#     """
#
#     def conversiondecorator_factory(*params) -> Callable:
#         f"""Create a conversion decorator which checks the types of certain parameters of
#         wrapped function and calls the conversion function on them if they are not of the
#         correct type.
#
#         Parameters
#         ----------
#         *params
#             The names of the parameters of which the type must be checked - and converted,
#             if their type is {additional_type}.
#         """
#         # Guard clause.
#         if not len(params):
#             raise ValueError(
#                 "Provide the name(s) of the parameters that must be turned into DateOffset objects.."
#             )
#
#         def conversiondecorator(fn: Callable):
#             sig = inspect.signature(fn)
#
#             # Guard clause.
#             not_found = [param for param in params if param not in sig.parameters]
#             if len(not_found):
#                 raise ValueError(
#                     f"The following parameters are not in the function's signature: {', '.join(not_found)}."
#                 )
#
#             # Change / extend type hints.
#             for param in params:
#                 if param in fn.__annotations__:
#                     fn.__annotations__[param] |= additional_type
#                 else:
#                     fn.__annotations__[param] = additional_type
#
#             @functools.wraps(fn)
#             def wrapped(*args, **kwargs):
#                 bound_args = sig.bind(*args, **kwargs)
#                 bound_args.apply_defaults()
#
#                 # Convert the argument.
#                 for param in params:
#                     if isinstance(arg := bound_args.arguments[param], additional_type):
#                         bound_args.arguments[param] = conversion_fn(arg)
#
#                 # Execute function as normal.
#                 return fn(*bound_args.args, **bound_args.kwargs)
#
#             return wrapped
#
#         return conversiondecorator
#
#     return conversiondecorator_factory
#


def apply_validation(validate: Callable, default_param: str | None = None):
    """Create a decorator factory.

    The decorators created by the factory check one or more parameters of the wrapped
    function. The check function is applied to the specified parameters and the
    wrapped function is called with the output of the check function instead of with the
    original parameter value. The check function shall raise an Exception to alert the
    caller.

    To avoid checks from being unnecessarily repeated, an input_output_map is kept in-
    side the function.

    Parameters
    ----------
    validate
        A one-argument function that returns a single value.
    default_param, optional
        Default parameter to check (i.e., if no other parameters are specified in the
        decorator factory).
    """

    # To store checked inputs and their corresponding outputs, to avoid repeated checking of identical input.
    input_output_map = {}

    def decorator_factory(*params) -> Callable:
        """Create a check decorator which performs checks certain parameters of wrapped
        function.

        Parameters
        ----------
        *params
            The names of the parameters which the check-function must be called on.
        """
        # Guard clause.
        if not len(params):
            if not default_param:
                raise ValueError(
                    "Provide the name(s) of the parameters that must be checked."
                )
            params = [default_param]

        def decorator(fn: Callable):
            sig = inspect.signature(fn)

            # Guard clause.
            not_found = [param for param in params if param not in sig.parameters]
            if len(not_found):
                raise ValueError(
                    f"The following parameters are not in the function's signature: {', '.join(not_found)}."
                )

            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Check the argument.
                for param in params:
                    arg = bound_args.arguments[param]

                    # Hash for lookup.
                    try:
                        id = hash(arg)
                    except TypeError:
                        id = hash(repr(arg))

                    # Get value after checkfunction.
                    if id in input_output_map:
                        arg = input_output_map[id]
                    else:
                        arg = validate(arg)  # this may take long
                        input_output_map[id] = arg

                    # Ensure we use this in the wrapped function.
                    bound_args.arguments[param] = arg

                # Execute function as normal.
                return fn(*bound_args.args, **bound_args.kwargs)

            return wrapped

        return decorator

    return decorator_factory
