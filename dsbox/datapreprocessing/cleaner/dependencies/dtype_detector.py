import typing
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple

import d3m.metadata.base as mbase
import pandas as pd
from common_primitives import utils
from d3m import container
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from dsbox.datapreprocessing.cleaner import config
import logging
import traceback


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def detector(inputs):
    lookup = {"float": ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'),
              "int": ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute'),
              "Categorical": ('https://metadata.datadrivendiscovery.org/types/CategoricalData',
                              'https://metadata.datadrivendiscovery.org/types/Attribute'),
              "Ordinal": ('https://metadata.datadrivendiscovery.org/types/OrdinalData',
                          'https://metadata.datadrivendiscovery.org/types/Attribute')
              }

    _logger = logging.getLogger(__name__)

    for col in range(inputs.shape[1]):
        temp = inputs.iloc[:, col]
        old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, col)))
        dtype = pd.DataFrame(temp.dropna().str.isnumeric().value_counts())
        ## if there is already a data type, see if that is equal to what we identified, else update
        ## corner case : Integer type, could be a categorical Arrtribute
        # detetct integers and update metadata

        if True in dtype.index:
            try:
                if dtype.loc[True][0] == temp.dropna().shape[0]:
                    if old_metadata["semantic_types"] == lookup["int"] or old_metadata["semantic_types"] == lookup[
                        "Categorical"] or old_metadata["semantic_types"] == lookup["Ordinal"]:
                        old_metadata["structural_type"] = type(10)

                    else:
                        if 'http://schema.org/Integer' not in old_metadata["semantic_types"]:
                            old_metadata["semantic_types"] = lookup["int"]
                        old_metadata["structural_type"] = type(10)
            except KeyError:
                try:
                    if dtype.loc[True].tolist()[0] == temp.dropna().shape[0]:
                        if old_metadata["semantic_types"] == lookup["int"] or old_metadata["semantic_types"] == lookup[
                            "Categorical"] or old_metadata["semantic_types"] == lookup["Ordinal"]:
                            old_metadata["structural_type"] = type(10)

                        else:
                            if 'http://schema.org/Integer' not in old_metadata["semantic_types"]:
                                old_metadata["semantic_types"] = lookup["int"]
                            old_metadata["structural_type"] = type(10)
                except Exception as e:
                    _logger.error(traceback.print_exc(e))
                    pass
            except Exception as e:
                _logger.error(traceback.print_exc(e))
                pass

        # detetct Float and update metadata
        else:
            dtype = pd.DataFrame(temp.dropna().apply(isfloat).value_counts())
            if True in dtype.index:
                try:
                    if dtype.loc[True][0] == temp.dropna().shape[0]:
                        if old_metadata["semantic_types"] == lookup["float"] or old_metadata["semantic_types"] == \
                                lookup[
                                    "Categorical"] or old_metadata["semantic_types"] == lookup["Ordinal"]:
                            old_metadata["structural_type"] = type(10.0)
                        else:
                            if 'http://schema.org/Float' not in old_metadata["semantic_types"]:
                                old_metadata["semantic_types"] = lookup["float"]
                            old_metadata["structural_type"] = type(10.0)
                except KeyError:
                    try:
                        if dtype.loc[True].tolist()[0] == temp.dropna().shape[0]:
                            if old_metadata["semantic_types"] == lookup["float"] or old_metadata["semantic_types"] == \
                                    lookup[
                                        "Categorical"] or old_metadata["semantic_types"] == lookup["Ordinal"]:
                                old_metadata["structural_type"] = type(10.0)
                            else:
                                if 'http://schema.org/Float' not in old_metadata["semantic_types"]:
                                    old_metadata["semantic_types"] = lookup["float"]
                                old_metadata["structural_type"] = type(10.0)
                    except Exception as e:
                        _logger.error(traceback.print_exc(e))
                        pass
                except Exception as e:
                    _logger.error(traceback.print_exc(e))
                    pass

        # _logger.info(
        #     "Integer and float detector. 'column_index': '%(column_index)d', 'old_metadata': '%(old_metadata)s', 'new_metadata': '%(new_metadata)s'",
        #     {
        #         'column_index': col,
        #         'old_metadata': dict(inputs.metadata.query((mbase.ALL_ELEMENTS, col))),
        #         'new_metadata': old_metadata,
        #     },
        # )

        inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, col), old_metadata)

    return inputs
