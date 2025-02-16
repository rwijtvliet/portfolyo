"""Help the type checker."""

from typing import TypeVar, Callable

import functools
import pandas as pd
import inspect

PintSeries = pd.Series  # aLias to make it more clear, what is being returned
Frequencylike = str | pd.DateOffset
Series_or_DataFrame = TypeVar("Series_or_DataFrame", pd.Series, pd.DataFrame)


def convert_if_type(additional_type: type, conversion_fn: Callable):
    """Create a decorator factory.

    The decorators created by the factory check one or more parameters of the wrapped
    function. If the function is called with a certain argument types, a conversion
    function is applied. This ensures the wrapped function accepts more diverse argument
    types, without having to carry out `isinstance` checks from inside the function.

    Parameters
    ----------
    additional_type
        The type that we want (one or more) parameters to also accept for their argument
        value.
    conversion_fn
        A one-argument function that is called on the argument when it is of this type,
        and which returns an instance of the type expected by the class.
    """

    def conversiondecorator_factory(*params) -> Callable:
        f"""Create a conversion decorator which checks the types of certain parameters of
        wrapped function and calls the conversion function on them if they are not of the
        correct type.

        Parameters
        ----------
        *params
            The names of the parameters of which the type must be checked - and converted,
            if their type is {additional_type}.
        """
        # Guard clause.
        if not len(params):
            raise ValueError(
                "Provide the name(s) of the parameters that must be turned into DateOffset objects.."
            )

        def conversiondecorator(fn: Callable):
            sig = inspect.signature(fn)

            # Guard clause.
            not_found = [param for param in params if param not in sig.parameters]
            if len(not_found):
                raise ValueError(
                    f"The following parameters are not in the function's signature: {', '.join(not_found)}."
                )

            # Change / extend type hints.
            for param in params:
                if param in fn.__annotations__:
                    fn.__annotations__[param] |= additional_type
                else:
                    fn.__annotations__[param] = additional_type

            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Convert the argument.
                for param in params:
                    if isinstance(arg := bound_args.arguments[param], additional_type):
                        bound_args.arguments[param] = conversion_fn(arg)

                # Execute function as normal.
                return fn(*bound_args.args, **bound_args.kwargs)

            return wrapped

        return conversiondecorator

    return conversiondecorator_factory
