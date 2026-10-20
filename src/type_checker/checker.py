import sys
import functools
import fire
from pydantic import validate_call, ValidationError, Field
from typing import Annotated

PositiveInt = Annotated[int, Field(ge=1)]
NoneNegativeInt = Annotated[int, Field(ge=0)]


def cli_safe(func):
    validated = validate_call(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return validated(*args, **kwargs)
        except ValidationError as e:
            for err in e.errors():
                print(f"Error: argument '{err["input"]}': {err['msg']}")
            sys.exit(2)

    return wrapper


@cli_safe
def repeat(text: str, times: PositiveInt = 1):
    return text * times


if __name__ == "__main__":
    fire.Fire(repeat)
