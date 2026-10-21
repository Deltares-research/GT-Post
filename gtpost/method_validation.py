from functools import wraps


def run_method(
    skip_if_present: list[str] = [],
    required: list[str] = [],
    computed: list[str] = [],
):
    """
    Decorator to manage method execution based on dataset variables. It allows to run any
    method at any time as it solves its dependencies automatically.

    Parameters
    ----------
    skip_if_present : list[str], optional
        List of dataset variables that, if present, will cause decorated method to be skipped.
    required : list[str], optional
        List of dataset variables that must be present for the method to run.
    computed : list[str], optional
        List of method names that must have ran before running the decorated method.

    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # First check if we can skip
            skip = all(variable in self.dataset for variable in skip_if_present)
            if skip and len(skip_if_present) > 0:
                print(
                    f"Skipping {method.__name__}: Results already exists in the dataset."
                )
                return

            # Then check that all required variables are present in the dataset
            for variable in required:
                if variable not in self.dataset:
                    raise ValueError(
                        f"Cannot run {method.__name__}: required dataset variable "
                        f"'{variable}' is missing."
                    )

            # Finally, check that all computed dependencies are satisfied
            for dep in computed:
                getattr(self, dep, None)()

            return method(self, *args, **kwargs)

        return wrapper

    return decorator
