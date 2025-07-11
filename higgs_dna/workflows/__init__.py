from higgs_dna.workflows.dystudies import (
    DYStudiesProcessor,
    TagAndProbeProcessor,
)
from higgs_dna.workflows.taggers import taggers
from higgs_dna.workflows.HHbbgg import HHbbggProcessor
from higgs_dna.workflows.particleLevel import ParticleLevelProcessor
from higgs_dna.workflows.top import TopProcessor
from higgs_dna.workflows.Zmmy import ZmmyProcessor, ZmmyHist, ZmmyZptHist
from higgs_dna.workflows.hpc_processor import HplusCharmProcessor
from higgs_dna.workflows.zee_processor import ZeeProcessor
from higgs_dna.workflows.lowmass import lowmassProcessor
from higgs_dna.workflows.lowmassVH import lowmassVHProcessor
from higgs_dna.workflows.btagging import BTaggingEfficienciesProcessor
from higgs_dna.workflows.stxs import STXSProcessor
from higgs_dna.workflows.diphoton_training import DiphoTrainingProcessor

workflows = {}

workflows["base"] = DYStudiesProcessor
workflows["tagandprobe"] = TagAndProbeProcessor
workflows["HHbbgg"] = HHbbggProcessor
workflows["particleLevel"] = ParticleLevelProcessor
workflows["top"] = TopProcessor
workflows["zmmy"] = ZmmyProcessor
workflows["zmmyHist"] = ZmmyHist
workflows["zmmyZptHist"] = ZmmyZptHist
workflows["hpc"] = HplusCharmProcessor
workflows["zee"] = ZeeProcessor
workflows["lowmass"] = lowmassProcessor
workflows["lowmassVH"] = lowmassVHProcessor
workflows["BTagging"] = BTaggingEfficienciesProcessor
workflows["stxs"] = STXSProcessor
workflows["diphotonID"] = DiphoTrainingProcessor

__all__ = ["workflows", "taggers", "DYStudiesProcessor"]
