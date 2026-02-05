from baybe.targets import NumericalTarget
from baybe import Campaign
from baybe.searchspace import SearchSpace, SubspaceContinuous
from baybe.objectives import SingleTargetObjective, DesirabilityObjective

import pandas as pd
import numpy as np  
from pathlib import Path
import json


def create_searchspace(low_bounds_list: list, upper_bounds_list: list,names:list) -> SearchSpace:
    parameters = {}
    for i in range(len(low_bounds_list)):
        key = names[i]
        value = list([low_bounds_list[i], upper_bounds_list[i]])
        parameters[key] = value
    subspace = SubspaceContinuous.from_dataframe(pd.DataFrame(parameters))
    searchspace = SearchSpace(continuous=subspace)
    return searchspace

def init_campaign(param_low_bounds_list, param_upper_bounds_list,names) -> Campaign:
    if(len(param_low_bounds_list) != len(param_upper_bounds_list)):
        raise ValueError("Length of lower and upper bounds lists must be equal.")
    
    searchspace = create_searchspace(param_low_bounds_list, param_upper_bounds_list,names)
    target_1 = NumericalTarget(name="y1", mode="MIN", bounds=(0, 1000))
    objective = SingleTargetObjective(target_1)
    campaign = Campaign(searchspace, objective)
    return campaign

