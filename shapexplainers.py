import InitializeCampaign as ic
import shap
from baybe.insights import SHAPInsight
from shap.explainers import *

def insight_permutation(campaign):
    insight = SHAPInsight.from_campaign(
        campaign,
        explainer_cls='PermutationExplainer',
        use_comp_rep=True,
    )
    return insight

def insight_exact(campaign):
    insight = SHAPInsight.from_campaign(
        campaign,
        explainer_cls='ExactExplainer',
        use_comp_rep=True,
    )
    return insight

def insight_partition(campaign):
    insight= SHAPInsight.from_campaign(
        campaign,
        explainer_cls='PartitionExplainer',
        use_comp_rep=True,
    )
    return insight
