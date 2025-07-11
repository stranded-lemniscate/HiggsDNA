from functools import partial
import logging
from .jet_systematics_json import jerc_jet

logger = logging.getLogger(__name__)


def add_jme_corr_syst(corrections_dict, systematics_dict, logger):
    corrections_dict.update(
        {
            # jerc for MC
            "jec_jet": partial(jerc_jet, pt=None, apply_jec=True),
            "jec_jet_syst": partial(jerc_jet, pt=None, apply_jec=True, jec_syst=True),
            "jec_jet_regrouped_syst": partial(
                jerc_jet, pt=None, apply_jec=True, jec_syst=True, split_jec_syst=True
            ),
            "jerc_jet": partial(jerc_jet, pt=None, apply_jec=True, apply_jer=True),
            "jerc_jet_syst": partial(
                jerc_jet,
                pt=None,
                apply_jec=True,
                jec_syst=True,
                apply_jer=True,
                jer_syst=True,
            ),
            "jerc_jet_regrouped_syst": partial(
                jerc_jet,
                pt=None,
                apply_jec=True,
                jec_syst=True,
                split_jec_syst=True,
                apply_jer=True,
                jer_syst=True,
            ),
            # jec for data: Usually corrections on Data innecessary
            # Use jec corrections with Era to re-do jec for data
            "jec_RunA": partial(jerc_jet, pt=None, era="RunA", level="L1L2L3Res"),
            "jec_RunB": partial(jerc_jet, pt=None, era="RunB", level="L1L2L3Res"),
            "jec_RunC": partial(jerc_jet, pt=None, era="RunC", level="L1L2L3Res"),
            "jec_RunD": partial(jerc_jet, pt=None, era="RunD", level="L1L2L3Res"),
            "jec_RunE": partial(jerc_jet, pt=None, era="RunE", level="L1L2L3Res"),
            "jec_RunF": partial(jerc_jet, pt=None, era="RunF", level="L1L2L3Res"),
            "jec_RunG": partial(jerc_jet, pt=None, era="RunG", level="L1L2L3Res"),
            "jec_RunH": partial(jerc_jet, pt=None, era="RunH", level="L1L2L3Res"),
            # For 2023 era C, different version of the datasets have different JECs
            # Details: https://gitlab.cern.ch/cms-analysis/general/HiggsDNA/-/issues/220#note_9180675
            "jec_RunCv123": partial(jerc_jet, pt=None, era="RunCv123", level="L1L2L3Res"),
            "jec_RunCv4": partial(jerc_jet, pt=None, era="RunCv4", level="L1L2L3Res"),
        }
    )
    logger.info(
        f"""_summary_

    Available correction keys:
    {corrections_dict.keys()}
    Available systematic keys:
    {systematics_dict.keys()}
    """
    )
    return corrections_dict, systematics_dict
