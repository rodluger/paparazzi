"""
Theano throws an error when trying to convert NaNs to float
because by definition, NaN != NaN, so the ``filter`` method
as implemented fails. This script does a quick monkey patch.

"""

import theano
import numpy as np


def filter(self, data, strict=False, allow_downcast=None):
    py_type = self.dtype_specs()[0]
    if strict and not isinstance(data, py_type):
        raise TypeError(
            "%s expected a %s, got %s of type %s"
            % (self, py_type, data, type(data)),
            data,
        )
    try:
        converted_data = py_type(data)
        if (
            allow_downcast
            or (
                allow_downcast is None
                and type(data) is float
                and self.dtype == theano.config.floatX
            )
            or data == converted_data
            or np.all(np.isnan(data))  # PATCHED LINE
        ):
            return py_type(data)
        else:
            raise TypeError(
                "Value cannot accurately be converted to dtype"
                " (%s) and allow_downcast is not True" % self.dtype
            )
    except Exception as e:
        raise TypeError(
            "Could not convert %s (value=%s) to %s"
            % (type(data), data, self.dtype),
            e,
        )


theano.scalar.basic.Scalar.filter = filter