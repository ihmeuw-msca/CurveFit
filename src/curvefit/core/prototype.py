from copy import deepcopy


class Prototype:
    """
    {begin_markdown Prototype}

    {spell_markdown deepcopy}

    # `curvefit.core.prototype.Prototype`
    ## Base class for an object that can be cloned

    Some objects in the `curvefit` package need to be treated as prototypes and cloned.
    If an object needs this ability, it will be a subclass of `Prototype` and inherit the `clone` method.

    ## Methods

    ### `clone`
    Returns a deepcopy of itself.

    {end_markdown Prototype}
    """

    def clone(self):
        return deepcopy(self)

