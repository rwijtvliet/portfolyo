"""Module with decorators."""

import functools
from typing import Any, Callable, Hashable


def cache_hashable_arguments(fn):
    """Like lru_cache, but skips cache if argument is not hashable."""
    cached_fn = functools.lru_cache()(fn)

    @functools.wraps(fn)
    def wrapper(arg):
        if isinstance(arg, Hashable):
            return cached_fn(arg)
        return fn(arg)

    return wrapper


# def coerce_fn(convert: Callable[[Any], Any], validate: Callable[[Any], None]) -> Callable:
#     @functools.wraps(convert)
#     def coerce(*args, **kwargs):
#         arg = convert(*args, **kwargs)
#         validate(arg)
#         return arg
#
#     return coerce


# def create_coerciondecorator(
#     convert: Callable[[Any], Any] | None,
#     validate: Callable[[Any], None] | None,
#     *,
#     default_param: str | None = None,
# ):
#     """Create decorator factory.
#
#     Decorators created by factory perform (a) conversion and (b) validation one or more parameters
#     of wrapped function.
#     * The conversion function is applied to specified parameters and wrapped function is called with
#       output of conversion function instead of with original parameter value.
#     * The validation function shall raise an Exception to alert caller if necessary.
#
#     Parameters
#     ----------
#     convert
#         One-argument function that returns a single value. None to do no conversion.
#     validate
#         One-argument function without return value that raises Exception if input not valid. None
#         to do no validation.
#     default_param, optional (default: no default parameters)
#         Default parameter to check (i.e., if no other parameters are specified in decorator factory).
#     """
#     # Guard clause.
#     if convert is validate is None:
#         raise ValueError("Specify at least ``convert`` or ``validate``.")
#
#     def decorator_factory(*params, validation: bool = True) -> Callable:
#         """Create a coerce decorator which performs checks on certain parameters of wrapped
#         function.
#
#         Parameters
#         ----------
#         *params
#             Names of parameters which coerce-function must be called on.
#         validation, optional
#             False to skip parameter validation for wrapped function.
#         """
#         # Guard clause.
#         if not len(params):
#             if not default_param:
#                 raise ValueError("Provide name(s) of parameter(s) that must be checked.")
#             params = [default_param]
#
#         # Create one-stop function.
#         def convert_and_validate(arg):
#             if convert:
#                 arg = convert(arg)
#             if validation and validate:
#                 validate(arg)  # may raise error
#             return arg
#
#         def decorator(fn: Callable):
#             sig = inspect.signature(fn)
#
#             # Guard clause.
#             not_found = [param for param in params if param not in sig.parameters]
#             if len(not_found):
#                 raise ValueError(
#                     f"Following parameters are not in function's signature: {', '.join(not_found)}."
#                 )
#
#             @functools.wraps(fn)
#             def wrapped(*args, **kwargs):
#                 bound_args = sig.bind(*args, **kwargs)
#                 bound_args.apply_defaults()
#
#                 # Check the argument.
#                 for param in params:
#                     bound_args.arguments[param] = convert_and_validate(bound_args.arguments[param])
#
#                 # Execute function as normal.
#                 return fn(*bound_args.args, **bound_args.kwargs)
#
#             return wrapped
#
#         return decorator
#
#     return decorator_factory
