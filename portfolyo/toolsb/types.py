"""Help the type checker."""

from typing import TypeVar

import pandas as pd

Frequencylike = str | pd.offsets.BaseOffset
Series_or_DataFrame = TypeVar("Series_or_DataFrame", pd.Series, pd.DataFrame)


# def conversiondecorator_factory_creator(wanted_type: type, conversion_fn: Callable):
#     """Create a decorator factory.
#
#     The decorators created by the factory check one or more parameters of the wrapped
#     function. If the function is called with the incorrect argument types, a conversion
#     function is applied. This ensures the wrapped function accepts more diverse argument
#     types, without having to carry out `isinstance` checks from inside the function.
#
#     Parameters
#     ----------
#     wanted_type
#         The class that the parameter should be an instance of.
#     conversion_fn
#         A one-argument function that is called on the argument when it is of an incorrect
#         type, and which returns an instance of the wanted type.
#     """
#
#     def conversiondecorator_factory(*params) -> Callable:
#         """Create a conversion decorator which checks the types of certain parameters of
#         wrapped function and calls the conversion function on them if they are not of the
#         correct type.
#
#         Parameters
#         ----------
#         *params
#             The names of the parameters of which the type must be checked (and possibly
#             converted).
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
#                     f"The following parameters are not part of the function's definition: {', '.join(not_found)}."
#                 )
#
#             # Change type hints.
#             for param in params:
#                 fn.__annotations__[param] = Frequencylike
#
#             @functools.wraps(fn)
#             def wrapped(*args, **kwargs):
#                 bound_args = sig.bind(*args, **kwargs)
#                 bound_args.apply_defaults()
#
#                 # Turn into DateOffset if it is a string.
#                 for paramname in params:
#                     if isinstance(arg := bound_args.arguments[paramname], str):
#                         bound_args.arguments[paramname] = to_offset(arg)
#
#                 # Execute function as normal.
#                 return fn(*bound_args.args, **bound_args.kwargs)
#
#             return wrapped
#
#         return conversiondecorator
#
#     return conversiondecorator_factory
