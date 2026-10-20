import functools
import sys
from collections.abc import Callable
from typing import Annotated, Any
from pydantic import Field, ValidationError, validate_call

PositiveInt = Annotated[int, Field(ge=1)]


def cli_safe(func: Callable[..., Any]) -> Callable[..., Any]:
    """Validate arguments by their type hints; exit cleanly on bad input."""
    validated = validate_call(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return validated(*args, **kwargs)
        except ValidationError as e:
            for err in e.errors():
                print(f"Error: argument '{err['input']}': {err['msg']}")
            sys.exit(2)

    return wrapper
