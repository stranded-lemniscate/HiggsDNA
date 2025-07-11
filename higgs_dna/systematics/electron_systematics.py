import numpy as np
import awkward as ak
import correctionlib
import os
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def Electron_Scale(pt, events, year="2022postEE", is_correction=True, restriction=None):
    """
    Applies the photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py
    """

    # for later unflattening:
    counts = ak.num(events.Electron.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, events.Electron.pt)[0])
    gain = ak.flatten(events.Electron.seedGain)
    eta = ak.flatten(events.Electron.ScEta)
    r9 = ak.flatten(events.Electron.r9)
    ecalEnergy_ = ak.flatten(events.Electron.ecalEnergy)
    pt_ = ak.flatten(events.Electron.pt)

    # Adding option to use electron scale corrections
    if year == "2022preEE":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/SS_Electron_Rereco2022BCD.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoBCD_ScaleJSON"]
    elif year == "2022postEE":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/SS_Electron_RerecoE_PromptFG_2022.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoE+PromptFG_ScaleJSON"]
    else:
        logger.info("WARNING: there are only electrons scale corrections for the year strings [\"2022preEE\", \"2022postEE\"]! \n Exiting. \n")
        exit()

    if is_correction:
        # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
        if hasattr(events, "GenPart"):
            raise ValueError("Scale corrections should only be applied to data!")

        if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            # the correction is already applied for Run 2
            logger.info("the scale correction for Run 2  MC is already applied in nAOD, nothing to be done")
        else:
            correction = evaluator.evaluate("total_correction", gain, run, eta, r9, pt_)
            ecal_energy_corr = ecalEnergy_ * correction
            pt_corr = pt_ * correction

            corrected_electrons = deepcopy(events.Electron)
            ecal_energy_corr = ak.unflatten(ecal_energy_corr, counts)
            pt_corr = ak.unflatten(pt_corr, counts)

            corrected_electrons["ele_ecalEnergy"] = ecal_energy_corr
            corrected_electrons["pt"] = pt_corr

            events.Electron = corrected_electrons

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
            correction = evaluator.evaluate("total_correction", gain, run, eta, r9, pt_)
            uncertainty = evaluator.evaluate("total_uncertainty", gain, run, eta, r9, pt_)

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
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * pt_[:, None]


def Electron_Smearing(pt, events, year="2022postEE", is_correction=True):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """

    # for later unflattening:
    counts = ak.num(events.Electron.pt)

    eta = ak.flatten(events.Electron.ScEta)
    r9 = ak.flatten(events.Electron.r9)
    ecalEnergy_ = ak.flatten(events.Electron.ecalEnergy)
    pt_ = ak.flatten(events.Electron.pt)

    # we need reproducible random numbers since in the systematics call, the previous correction needs to be cancelled out
    rng = np.random.default_rng(seed=125)

    # Adding option to use electron scale corrections
    if year == "2022preEE":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/SS_Electron_Rereco2022BCD.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoBCD_SmearingJSON"]
    elif year == "2022postEE":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/SS_Electron_RerecoE_PromptFG_2022.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoE+PromptFG_SmearingJSON"]
    else:
        logger.info("WARNING: there are only electrons scale corrections for the year strings [\"2022preEE\", \"2022postEE\"]! \n Exiting. \n")
        exit()

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

            ecalEnergy_corr = ecalEnergy_ * smearing
            pt_corr = pt_ * smearing

            corrected_electrons = deepcopy(events.Electron)

            ecalEnergy_corr = ak.unflatten(ecalEnergy_corr, counts)
            pt_corr = ak.unflatten(pt_corr, counts)
            rho_corr = ak.unflatten(rho, counts)

            # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
            try:
                events.GenIsolatedPhoton  # this operation is here because if there is no "events.GenIsolatedPhoton" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
                corrected_electrons["ecalEnergy"] = ecalEnergy_corr
                corrected_electrons["pt"] = pt_corr
            except:
                pass

            corrected_electrons["rho_smear"] = rho_corr

            events.Electron = corrected_electrons

        return events

    else:

        if year in ["2016", "2016preVFP", "2016postVFP", "2017", "2018"]:

            logger.info('The electron smearing systematic is not implemented for run2! - Exiting ...')
            exit()

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
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * pt_[:, None]
