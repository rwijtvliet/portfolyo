"""Module with decorators."""

import functools
import inspect
from typing import Any, Callable, Hashable

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


def cache_hashable_arguments(fn):
    """Like lru_cache, but skips cache if argument is not hashable."""
    cached_fn = functools.lru_cache()(fn)

    @functools.wraps(fn)
    def wrapper(arg):
        if isinstance(arg, Hashable):
            return cached_fn(arg)
        return fn(arg)

    return wrapper


def create_coercedecorator(
    *,
    conversion: Callable[[Any], Any] | None,
    validation: Callable[[Any], None] | None,
    default_param: str | None = None,
):
    """Create decorator factory.

    Decorators created by factory perform (a) conversion and (b) validation one or more
    parameters of wrapped function.
    * The conversion function is applied to specified parameters and wrapped function is
      called with output of conversion function instead of with original parameter value.
    * The validation function shall raise an Exception to alert caller if necessary.

    Parameters
    ----------
    conversion
        One-argument function that returns a single value. None to do no conversion.
    validation
        One-argument function without return value that raises Exception if input not
        valid. None to do no validation.
    default_param, optional (default: no default parameters)
        Default parameter to check (i.e., if no other parameters are specified in
        decorator factory).
    """
    # Guard clause.
    if conversion is validation is None:
        raise ValueError("Specify at least ``conversion`` or ``validation``.")

    def decorator_factory(*params, validate: bool = True) -> Callable:
        """Create a coerce decorator which performs checks on certain parameters of wrapped
        function.

        Parameters
        ----------
        *params
            Names of parameters which coerce-function must be called on.
        validate, optional (default: True)
            False to skip parameter validation for wrapped function.
        """
        # Guard clause.
        if not len(params):
            if not default_param:
                raise ValueError("Provide name(s) of parameter(s) that must be checked.")
            params = [default_param]

        # Create one-stop function.
        def convert_and_validate(arg):
            if conversion:
                arg = conversion(arg)
            if validation and validate:
                validation(arg)  # may raise error
            return arg

        def decorator(fn: Callable):
            sig = inspect.signature(fn)

            # Guard clause.
            not_found = [param for param in params if param not in sig.parameters]
            if len(not_found):
                raise ValueError(
                    f"Following parameters are not in function's signature: {', '.join(not_found)}."
                )

            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Check the argument.
                for param in params:
                    bound_args.arguments[param] = convert_and_validate(bound_args.arguments[param])

                # Execute function as normal.
                return fn(*bound_args.args, **bound_args.kwargs)

            return wrapped

        return decorator

    return decorator_factory
