import numpy as np
import awkward as ak
import correctionlib
import os
import sys
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


# first dummy, keeping it at this point as reference for even simpler implementations
def photon_pt_scale_dummy(pt, **kwargs):
    return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * pt[:, None]


# Not nice but working: if the functions are called in the base processor by Photon.add_systematic(... "what"="pt"...), the pt is passed to the function as first argument.
# I need the full events here, so I pass in addition the events. Seems to only work if it is explicitly a function of pt, but I might be missing something. Open for better solutions.
def Scale_Trad(pt, events, year="2022postEE", is_correction=True, restriction=None):
    """
    Applies the photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, events.Photon.pt)[0])
    gain = ak.flatten(events.Photon.seedGain)
    eta = ak.flatten(events.Photon.ScEta)
    r9 = ak.flatten(events.Photon.r9)
    _pt = ak.flatten(events.Photon.pt)

    if year == "2016preVFP":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/EGM_ScaleUnc_2016preVFP.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]
    elif year == "2016postVFP":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/EGM_ScaleUnc_2016postVFP.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]
    elif year == "2017":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/EGM_ScaleUnc_2017.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]
    elif year == "2018":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/EGM_ScaleUnc_2018.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]
    elif year == "2022preEE":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/SS_Rereco2022BCD.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoBCD_ScaleJSON"]
    elif year == "2022postEE":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/SS_RerecoE_PromptFG_2022.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoE+PromptFG_ScaleJSON"]
    else:
        logger.error("There are only scale corrections for the year strings [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\"]! \n Exiting. \n")
        sys.exit(1)

    if is_correction:
        # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
        if hasattr(events, "GenPart"):
            raise ValueError("Scale corrections should only be applied to data!")

        if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            # the correction is already applied for Run 2
            logger.info("the scale correction for Run 2  MC is already applied in nAOD, nothing to be done")
        else:
            correction = evaluator.evaluate("total_correction", gain, run, eta, r9, _pt)
            pt_corr = _pt * correction

            corrected_photons = deepcopy(events.Photon)
            pt_corr = ak.unflatten(pt_corr, counts)
            corrected_photons["pt"] = pt_corr

            events.Photon = corrected_photons

        return events

    else:
        if not hasattr(events, "GenPart"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            # the uncertainty is applied in reverse because the correction is meant for data as I understand fro EGM instructions here: https://cms-talk.web.cern.ch/t/pnoton-energy-corrections-in-nanoaod-v11/34327/2
            uncertainty_up = evaluator.evaluate(year, "scaledown", eta, gain)
            uncertainty_down = evaluator.evaluate(year, "scaleup", eta, gain)

            corr_up_variation = uncertainty_up
            corr_down_variation = uncertainty_down

        else:
            correction = evaluator.evaluate("total_correction", gain, run, eta, r9, _pt)
            uncertainty = evaluator.evaluate("total_uncertainty", gain, run, eta, r9, _pt)

            if restriction is not None:
                if restriction == "EB":
                    uncMask = ak.to_numpy(ak.flatten(events.Photon.isScEtaEB))

                elif restriction == "EE":
                    uncMask = ak.to_numpy(ak.flatten(events.Photon.isScEtaEE))
                    if year == "2022preEE":
                        rescaleFactor = 1.5
                        logger.info(f"Increasing EB scale uncertainty by factor {rescaleFactor}.")
                        uncertainty *= rescaleFactor
                    elif year == "2022postEE":
                        rescaleFactor = 2.
                        logger.info(f"Increasing EE scale uncertainty by factor {rescaleFactor}.")
                        uncertainty *= rescaleFactor

                uncertainty = np.where(
                    uncMask, uncertainty, np.zeros_like(uncertainty)
                )

            # divide by correction since it is already applied before
            corr_up_variation = (correction + uncertainty) / correction
            corr_down_variation = (correction - uncertainty) / correction

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * _pt[:, None]


def Smearing_Trad(pt, events, year="2022postEE", is_correction=True):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)

    eta = ak.flatten(events.Photon.ScEta)
    r9 = ak.flatten(events.Photon.r9)
    _pt = ak.flatten(events.Photon.pt)

    # we need reproducible random numbers since in the systematics call, the previous correction needs to be cancelled out
    rng = np.random.default_rng(seed=125)

    if year == "2022preEE":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/SS_Rereco2022BCD.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoBCD_SmearingJSON"]
    elif year == "2022postEE":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/SS_RerecoE_PromptFG_2022.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoE+PromptFG_SmearingJSON"]
    elif year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        logger.info("the systematic variations are taken directly from the dedicated nAOD branches Photon.dEsigmaUp and Photon.dEsigmaDown")
    else:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\"] \n Exiting. \n")
        sys.exit(1)

    if is_correction:

        if year in ["2016", "2016preVFP", "2016postVFP", "2017", "2018"]:
            logger.info("the smearing correction for Run 2 MC is already applied in nAOD")
        else:
            # In theory, the energy should be smeared and not the pT, see: https://mattermost.web.cern.ch/cmseg/channels/egm-ss/6mmucnn8rjdgt8x9k5zaxbzqyh
            # However, there is a linear proportionality between pT and E: E = pT * cosh(eta)
            # Because of that, applying the correction to pT and E is equivalent (since eta does not change)
            # Energy is provided as a LorentzVector mixin, so we choose to correct pT
            # Also holds true for the scale part
            rho = evaluator.evaluate("rho", eta, r9)
            smearing = rng.normal(loc=1., scale=rho)
            pt_corr = _pt * smearing
            corrected_photons = deepcopy(events.Photon)
            pt_corr = ak.unflatten(pt_corr, counts)
            rho_corr = ak.unflatten(rho, counts)

            # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
            try:
                events.GenIsolatedPhoton  # this operation is here because if there is no "events.GenIsolatedPhoton" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
                corrected_photons["pt"] = pt_corr
            except:
                pass

            corrected_photons["rho_smear"] = rho_corr

            events.Photon = corrected_photons  # why does this work? Why do we not need events['Photon'] to assign?

        return events

    else:

        if year in ["2016", "2016preVFP", "2016postVFP", "2017", "2018"]:
            # the correction is already applied for Run 2
            dEsigmaUp = ak.flatten(events.Photon.dEsigmaUp)
            dEsigmaDown = ak.flatten(events.Photon.dEsigmaDown)
            logger.info(f"{dEsigmaUp}, {events.Photon.dEsigmaUp}")

            # the correction is given as additive factor for the Energy (Et_smear_up = Et + abs(dEsigmaUp)) so we have to convert it before applying it to the Pt
            # for EGM instruction on how to calculate uncertainties I link to this CMSTalk post https://cms-talk.web.cern.ch/t/pnoton-energy-corrections-in-nanoaod-v11/34327/2
            smearing_up = (_pt + abs(dEsigmaUp) / np.cosh(eta))
            smearing_down = (_pt - abs(dEsigmaDown) / np.cosh(eta))

            # divide by correction since it is already applied before we also divide for the Pt because it is later multipled when passing it to coffea, to be compatible with Run 3 calculation from json.
            # I convert it to numpy because ak.Arrays don't have the .reshape method needed further on.
            corr_up_variation = (smearing_up / _pt).to_numpy()
            corr_down_variation = (smearing_down / _pt).to_numpy()

        else:
            rho = evaluator.evaluate("rho", eta, r9)
            # produce the same numbers as in correction step
            smearing = rng.normal(loc=1., scale=rho)

            err_rho = evaluator.evaluate("err_rho", eta, r9)
            rho_up = rho + err_rho
            rho_down = rho - err_rho
            smearing_up = rng.normal(loc=1., scale=rho_up)
            smearing_down = rng.normal(loc=1., scale=rho_down)

            # divide by correction since it is already applied before
            corr_up_variation = smearing_up / smearing
            corr_down_variation = smearing_down / smearing

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * _pt[:, None]


def Scale_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G", restriction=None):
    """
    Applies the IJazZ photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py.
    The IJazZ corrections are independent and detached from the Egamma corrections.
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, events.Photon.pt)[0])
    gain = ak.flatten(events.Photon.seedGain)
    eta = ak.flatten(events.Photon.ScEta)
    AbsScEta = abs(eta)
    r9 = ak.flatten(events.Photon.r9)
    # scale uncertainties are applied on the smeared pt but computed from the raw pt
    pt_raw = ak.flatten(events.Photon.pt_raw)
    _pt = ak.flatten(events.Photon.pt)

    if gaussians == "1G":
        gaussian_postfix = ""
    elif gaussians == "2G":
        gaussian_postfix = "2G"
    else:
        logger.error("The selected number of gaussians is not implemented yet! Valid options are [\"1G\", \"2G\"] \n Exiting. \n")
        sys.exit(1)

    valid_years_paths = {
        "2022preEE": "EGMScalesSmearing_Pho_2022preEE",
        "2022postEE": "EGMScalesSmearing_Pho_2022postEE",
        "2023preBPix": "EGMScalesSmearing_Pho_2023preBPIX",
        "2023postBPix": "EGMScalesSmearing_Pho_2023postBPIX"
    }

    ending = ".v1.json"

    if year not in valid_years_paths and year not in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"] \n Exiting. \n")
        sys.exit(1)

    if year in valid_years_paths:
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + gaussian_postfix + ending)
        try:
            cset = correctionlib.CorrectionSet.from_file(path_json)
        except:
            logger.error(f"WARNING: the JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
            sys.exit(1)
        # Convention of Fabrice and Paul: Capitalise IX (for some reason)
        if "BPix" in year:
            year = year.replace("BPix", "BPIX")
        scale_evaluator = cset.compound[f"EGMScale_Compound_Pho_{year}{gaussian_postfix}"]
        smear_and_syst_evaluator = cset[f"EGMSmearAndSyst_PhoPTsplit_{year}{gaussian_postfix}"]
    else:
        logger.info("the systematic variations are taken directly from the dedicated nAOD branches Photon.dEsigmaUp and Photon.dEsigmaDown")

    if is_correction:
        # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
        if hasattr(events, "GenPart"):
            raise ValueError("Scale corrections should only be applied to data!")

        correction = scale_evaluator.evaluate("scale", run, eta, r9, AbsScEta, pt_raw, gain)
        pt_corr = pt_raw * correction
        corrected_photons = deepcopy(events.Photon)
        pt_corr = ak.unflatten(pt_corr, counts)
        corrected_photons["pt"] = pt_corr

        events.Photon = corrected_photons

        return events

    else:
        # Note the conventions in the JSON, both `scale_up`/`scale_down` and `escale` are available.
        # scale_up = 1 + escale
        if not hasattr(events, "GenPart"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        corr_up_variation = smear_and_syst_evaluator.evaluate('scale_up', pt_raw, r9, AbsScEta)
        corr_down_variation = smear_and_syst_evaluator.evaluate('scale_down', pt_raw, r9, AbsScEta)

        if restriction == "EB":
            corr_up_variation[ak.to_numpy(ak.flatten(events.Photon.isScEtaEE))] = 1.
            corr_up_variation[ak.to_numpy(ak.flatten(events.Photon.isScEtaEE))] = 1.
        elif restriction == "EE":
            corr_up_variation[ak.to_numpy(ak.flatten(events.Photon.isScEtaEB))] = 1.
            corr_up_variation[ak.to_numpy(ak.flatten(events.Photon.isScEtaEB))] = 1.
        else:
            logger.error("The restriction is not implemented yet! Valid options are [\"EB\", \"EE\"] \n Exiting. \n")
            sys.exit(1)

        # Coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        # scale uncertainties are applied on the smeared pt
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * _pt[:, None]


def double_smearing(std_normal, std_flat, mu, sigma1, sigma2, frac):
    # compute the two smearing scales from the gaussian draws
    scales = np.array([1 + sigma1 * std_normal, mu * (1 + sigma2 * std_normal)])
    # select the gaussian based on the relative fraction and the flat draw
    binom = (std_flat > frac).astype(int)
    return scales[binom, np.arange(len(mu))]


def Smearing_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G"):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)

    eta = ak.flatten(events.Photon.ScEta)
    AbsScEta = abs(eta)
    r9 = ak.flatten(events.Photon.r9)
    pt_raw = ak.flatten(events.Photon.pt_raw)
    # Need some broadcasting to make the event numbers match
    event_number = ak.flatten(ak.broadcast_arrays(events.event, events.Photon.pt)[0])

    if gaussians == "1G":
        gaussian_postfix = ""
    elif gaussians == "2G":
        gaussian_postfix = "2G"
    else:
        logger.error("The selected number of gaussians is not implemented yet! Valid options are [\"1G\", \"2G\"] \n Exiting. \n")
        sys.exit(1)

    valid_years_paths = {
        "2022preEE": "EGMScalesSmearing_Pho_2022preEE",
        "2022postEE": "EGMScalesSmearing_Pho_2022postEE",
        "2023preBPix": "EGMScalesSmearing_Pho_2023preBPIX",
        "2023postBPix": "EGMScalesSmearing_Pho_2023postBPIX"
    }

    ending = ".v1.json"

    if year not in valid_years_paths and year not in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"] \n Exiting. \n")
        sys.exit(1)

    if year in valid_years_paths:
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + gaussian_postfix + ending)
        try:
            cset = correctionlib.CorrectionSet.from_file(path_json)
        except:
            logger.error(f"Tthe JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
            sys.exit(1)
        # Convention of Fabrice and Paul: Capitalise IX (for some reason)
        if "BPix" in year:
            year_ = year.replace("BPix", "BPIX")
        else:
            year_ = year
        smear_and_syst_evaluator = cset[f"EGMSmearAndSyst_PhoPTsplit_{year_}{gaussian_postfix}"]
        random_generator = cset['EGMRandomGenerator']
    else:
        logger.info("the systematic variations are taken directly from the dedicated nAOD branches Photon.dEsigmaUp and Photon.dEsigmaDown")

    # In theory, the energy should be smeared and not the pT, see: https://mattermost.web.cern.ch/cmseg/channels/egm-ss/6mmucnn8rjdgt8x9k5zaxbzqyh
    # However, there is a linear proportionality between pT and E: E = pT * cosh(eta)
    # Because of that, applying the correction to pT and E is equivalent (since eta does not change)
    # Energy is provided as a LorentzVector mixin, so we choose to correct pT
    # Also holds true for the scale part

    # Calculate upfront since it is needed for both correction and uncertainty
    smearing = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, AbsScEta)
    random_numbers = random_generator.evaluate('stdnormal', pt_raw, r9, AbsScEta, event_number)

    if gaussians == "1G":
        correction = (1 + smearing * random_numbers)
    # Else can only be "2G" due to the checks above
    # Have to use else here to satisfy that correction is always defined in all possible branches of the code
    else:
        correction = double_smearing(
            random_numbers,
            random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
            smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
            smearing,
            smear_and_syst_evaluator.evaluate('reso2', pt_raw, r9, AbsScEta),
            smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta)
        )

    if is_correction:
        pt_corr = pt_raw * correction
        corrected_photons = deepcopy(events.Photon)
        pt_corr = ak.unflatten(pt_corr, counts)
        # For the 2G case, also take the rho_corr from the 1G case as advised by Fabrice
        # Otherwise, the sigma_m/m will be lower on average, new CDFs will be needed etc. not worth the hassle
        if gaussians == "2G":
            path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + ending)
            try:
                cset = correctionlib.CorrectionSet.from_file(path_json)
            except:
                logger.error(f"The JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
                sys.exit(1)
            if "BPix" in year:
                year = year.replace("BPix", "BPIX")
            smear_and_syst_evaluator_for_rho_corr = cset[f"EGMSmearAndSyst_PhoPTsplit_{year}"]
            rho_corr = ak.unflatten(smear_and_syst_evaluator_for_rho_corr.evaluate('smear', pt_raw, r9, AbsScEta), counts)
        else:
            rho_corr = ak.unflatten(smearing, counts)

        # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
        try:
            events.GenIsolatedPhoton  # this operation is here because if there is no "events.GenIsolatedPhoton" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
            corrected_photons["pt"] = pt_corr
        except:
            pass

        corrected_photons["rho_smear"] = rho_corr
        events.Photon = corrected_photons

        return events

    else:
        # Note the conventions in the JSON, both `smear_up`/`smear_down` and `esmear` are available.
        # smear_up = smear + esmear
        if gaussians == "1G":
            corr_up_variation = 1 + smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, AbsScEta) * random_numbers
            corr_down_variation = 1 + smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, AbsScEta) * random_numbers

        else:
            corr_up_variation = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('reso2', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta)
            )

            corr_down_variation = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('reso2', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta)
            )

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        # smearing uncertainties are applied on the raw pt because the smearing is redone from scratch
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * pt_raw[:, None]


def energyErrShift(energyErr, events, year="2022postEE", is_correction=True):
    # See also https://indico.cern.ch/event/1131803/contributions/4758593/attachments/2398621/4111806/Hgg_Differentials_Approval_080322.pdf#page=47
    # 2% with flows justified by https://indico.cern.ch/event/1495536/#20-study-of-the-sigma_mm-mismo
    if is_correction:
        return events
    else:
        _energyErr = ak.flatten(events.Photon.energyErr)
        uncertainty_up = np.ones(len(_energyErr)) * 1.02
        uncertainty_dn = np.ones(len(_energyErr)) * 0.98
        return (
            np.concatenate(
                (uncertainty_up.reshape(-1, 1), uncertainty_dn.reshape(-1, 1)), axis=1
            )
            * _energyErr[:, None]
        )


# Not nice but working: if the functions are called in the base processor by Photon.add_systematic(... "what"="pt"...), the pt is passed to the function as first argument.
# I need the full events here, so I pass in addition the events. Seems to only work if it is explicitly a function of pt, but I might be missing something. Open for better solutions.
def FNUF(pt, events, year="2017", is_correction=True):
    """
    ---This is an implementation of the FNUF uncertainty copied from flashgg,
    --- Preliminary JSON (run2 I don't know if this needs to be changed) file created with correctionlib starting from flashgg: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    Applies the photon pt and energy scale corrections and corresponding uncertainties.
    To be checked by experts
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)
    eta = ak.flatten(events.Photon.ScEta)
    r9 = ak.flatten(events.Photon.r9)
    _energy = ak.flatten(events.Photon.energy)
    _pt = ak.flatten(events.Photon.pt)

    # era/year defined as parameter of the function
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
    if year not in avail_years:
        logger.error(f"Only FNUF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        sys.exit(1)
    elif "2022" or "2023" in year:
        logger.warning(f"""You selected the year_string {year}, which is a 2022 era.
                        FNUF was not re-derived for Run 3 yet, but we fall back to the Run 2 2018 values.
                        These values only constitute up/down variations, no correction is applied.
                        The values are the averaged corrections from Run 2, turned into a systematic and inflated by 25%.
                        Please make sure that this is what you want. You have been warned.""")
        # The values have been provided by Badder for HIG-23-014 and Fabrice suggested to increase uncertainty a bit.
        year = "2022"
    elif "2016" in year:
        year = "2016"

    jsonpog_file = os.path.join(os.path.dirname(__file__), f"JSONs/FNUF/{year}/FNUF_{year}.json")
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)["FNUF"]

    if is_correction:
        correction = evaluator.evaluate("nominal", eta, r9)
        corr_energy = _energy * correction
        corr_pt = _pt * correction

        corrected_photons = deepcopy(events.Photon)
        corr_energy = ak.unflatten(corr_energy, counts)
        corr_pt = ak.unflatten(corr_pt, counts)
        corrected_photons["energy"] = corr_energy
        corrected_photons["pt"] = corr_pt
        events.Photon = corrected_photons

        return events

    else:
        correction = evaluator.evaluate("nominal", eta, r9)
        # When creating the JSON I already included added the variation to the returned value
        uncertainty_up = evaluator.evaluate("up", eta, r9) / correction
        uncertainty_dn = evaluator.evaluate("down", eta, r9) / correction
        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return (
            np.concatenate(
                (uncertainty_up.reshape(-1, 1), uncertainty_dn.reshape(-1, 1)), axis=1
            )
            * _pt[:, None]
        )


# Same old same old, just reiterated: if the functions are called in the base processor by Photon.add_systematic(... "what"="pt"...), the pt is passed to the function as first argument.
# Open for better solutions.
def ShowerShape(pt, events, year="2017", is_correction=True):
    """
    ---This is an implementation of the ShowerShape uncertainty copied from flashgg,
    --- Preliminary JSON (run2 I don't know if this needs to be changed) file created with correctionlib starting from flashgg: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    Applies the photon pt and energy scale corrections and corresponding uncertainties (only on the pt because it is what is used in selection).
    To be checked by experts
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)
    eta = ak.flatten(events.Photon.ScEta)
    r9 = ak.flatten(events.Photon.r9)
    _energy = ak.flatten(events.Photon.energy)
    _pt = ak.flatten(events.Photon.pt)

    # era/year defined as parameter of the function
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018"]
    if year not in avail_years:
        logger.error(f"Only ShowerShape corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        sys.exit(1)
    elif "2016" in year:
        year = "2016"

    jsonpog_file = os.path.join(os.path.dirname(__file__), f"JSONs/ShowerShape/{year}/ShowerShape_{year}.json")
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)["ShowerShape"]

    if is_correction:
        correction = evaluator.evaluate("nominal", eta, r9)
        corr_energy = _energy * correction
        corr_pt = _pt * correction

        corrected_photons = deepcopy(events.Photon)
        corr_energy = ak.unflatten(corr_energy, counts)
        corr_pt = ak.unflatten(corr_pt, counts)
        corrected_photons["energy"] = corr_energy
        corrected_photons["pt"] = corr_pt
        events.Photon = corrected_photons

        return events

    else:
        correction = evaluator.evaluate("nominal", eta, r9)
        # When creating the JSON I already included added the variation to the returned value
        uncertainty_up = evaluator.evaluate("up", eta, r9) / correction
        uncertainty_dn = evaluator.evaluate("down", eta, r9) / correction
        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return (
            np.concatenate(
                (uncertainty_up.reshape(-1, 1), uncertainty_dn.reshape(-1, 1)), axis=1
            )
            * _pt[:, None]
        )


def Material(pt, events, year="2017", is_correction=True):
    """
    ---This is an implementation of the Material uncertainty copied from flashgg,
    --- JSON file for run2 created with correctionlib starting from flashgg: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    Applies the photon pt and energy scale corrections and corresponding uncertainties.
    To be checked by experts
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)
    eta = ak.flatten(abs(events.Photon.ScEta))
    r9 = ak.flatten(events.Photon.r9)
    _energy = ak.flatten(events.Photon.energy)
    _pt = ak.flatten(events.Photon.pt)

    # era/year defined as parameter of the function, only 2017 is implemented up to now
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
    if year not in avail_years:
        logger.error(f"Only eVetoSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        sys.exit(1)
    elif "2016" in year:
        year = "2016"
    # use Run 2 files also for Run 3, preliminary
    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]:
        logger.warning(f"""You selected the year_string {year}, which is a Run 3 era.
                  Material was not rederived for Run 3 yet, but we fall back to the Run 2 2018 values.
                  Please make sure that this is what you want. You have been warned.""")
        year = "2018"

    jsonpog_file = os.path.join(os.path.dirname(__file__), f"JSONs/Material/{year}/Material_{year}.json")
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)["Material"]

    if is_correction:
        correction = evaluator.evaluate("nominal", eta, r9)
        corr_energy = _energy * correction
        corr_pt = _pt * correction

        corrected_photons = deepcopy(events.Photon)
        corr_energy = ak.unflatten(corr_energy, counts)
        corr_pt = ak.unflatten(corr_pt, counts)
        corrected_photons["energy"] = corr_energy
        corrected_photons["pt"] = corr_pt
        events.Photon = corrected_photons

        return events

    else:
        correction = evaluator.evaluate("nominal", eta, r9)
        # When creating the JSON I already included added the variation to the returned value
        uncertainty_up = evaluator.evaluate("up", eta, r9) / correction
        uncertainty_dn = evaluator.evaluate("down", eta, r9) / correction
        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return (
            np.concatenate(
                (uncertainty_up.reshape(-1, 1), uncertainty_dn.reshape(-1, 1)), axis=1
            )
            * _pt[:, None]
        )
