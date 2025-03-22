class Pipe:
    """A magittr like pipe functionality


    Examples:

    >>> from pymer4.tidystats._pipes import lm, tidy, augment
    >>> from pymer4.tidystats import load_dataset
    >>> df = load_dataset('sleep')
    >>> df >> lm(formula='Reaction ~ Days') >> tidy()

    >>> df >> lm('Reaction ~ Days') >> augment() >> select('fitted')
    """

    def __init__(self, value):
        self.value = value

    def __rshift__(self, other):
        other = self.pipe_op(other)
        result = other(self.value)
        # Return the actual value if this is the end of the chain
        return result if not isinstance(other, Pipe) else Pipe(result)

    def __rrshift__(self, other):
        result = self.value(other)
        # Return the actual value if this is the end of the chain
        return result if not isinstance(self.value, Pipe) else Pipe(result)

    def __call__(self, value):
        return self.value(value)

    @staticmethod
    def pipe_op(value):
        return Pipe(value) if callable(value) else value


def make_pipeable(func):
    """Usage decorator for making functions pipe-friendly"""

    def wrapper(*args, **kwargs):
        if not args:
            return Pipe(lambda x: func(x, **kwargs))
        return func(*args, **kwargs)

    return wrapper


def pipeable(expect=None):
    """Decorator that makes a function pipe-friendly and automatically curries if first arg isn't of expected_type.

    Args:
        expect (type / tuple / callable): Type or tuple of types that the first argument should be, or a function that checks the argument
    """

    def decorator(func):
        def wrapper(first_arg, *args, **kwargs):
            if expect is None:
                return func(first_arg, *args, **kwargs)

            # Check if first argument is of expected type
            is_valid = (
                isinstance(first_arg, expect)
                if isinstance(expect, type)
                else expect(first_arg)
            )

            if not is_valid:
                # If first arg isn't of expected type, curry the function
                all_args = (first_arg,) + args
                return Pipe(lambda data: func(data, *all_args, **kwargs))
            # Otherwise call function normally
            return func(first_arg, *args, **kwargs)

        return make_pipeable(wrapper)

    return decorator
