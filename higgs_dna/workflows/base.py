from higgs_dna.tools.chained_quantile import ChainedQuantileRegression
from higgs_dna.tools.diphoton_mva import calculate_retrained_diphoton_mva as calculate_diphoton_mva
from higgs_dna.tools.xgb_loader import load_bdt
from higgs_dna.tools.photonid_mva import calculate_photonid_mva, load_photonid_mva
from higgs_dna.tools.photonid_mva import calculate_photonid_mva_run3, load_photonid_mva_run3
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from higgs_dna.tools.sigma_m_tools import compute_sigma_m
from higgs_dna.selections.photon_selections import photon_preselection
from higgs_dna.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, jetvetomap, getBTagMVACut
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from higgs_dna.utils.misc_utils import choose_jet
from higgs_dna.tools.flow_corrections import apply_flow_corrections_to_photons

from higgs_dna.tools.mass_decorrelator import decorrelate_mass_resolution

# from higgs_dna.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from higgs_dna.metaconditions import photon_id_mva_weights
from higgs_dna.metaconditions import diphoton as diphoton_mva_dir
from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections
from higgs_dna.systematics import apply_systematic_variations_object_level

import functools
import operator
import os
import warnings
from typing import Any, Dict, List, Optional
import awkward as ak
import numpy
import sys
import vector
from coffea import processor
from coffea.analysis_tools import Weights
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class HggBaseProcessor(processor.ProcessorABC):  # type: ignore
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Optional[Dict[str, List[str]]],
        corrections: Optional[Dict[str, List[str]]],
        apply_trigger: bool,
        output_location: Optional[str],
        taggers: Optional[List[Any]],
        nano_version: int,
        bTagEffFileName: Optional[str],
        trigger_group: str,
        analysis: str,
        applyCQR: bool,
        skipJetVetoMap: bool,
        year: Optional[Dict[str, List[str]]],
        fiducialCuts: str,
        doDeco: bool,
        Smear_sigma_m: bool,
        doFlow_corrections: bool,
        output_format: str,
    ) -> None:
        self.meta = metaconditions
        self.systematics = systematics if systematics is not None else {}
        self.corrections = corrections if corrections is not None else {}
        self.apply_trigger = apply_trigger
        self.output_location = output_location
        self.nano_version = nano_version
        self.bTagEffFileName = bTagEffFileName
        self.trigger_group = trigger_group
        self.analysis = analysis
        self.applyCQR = applyCQR
        self.skipJetVetoMap = skipJetVetoMap
        self.year = year if year is not None else {}
        self.fiducialCuts = fiducialCuts
        self.doDeco = doDeco
        self.Smear_sigma_m = Smear_sigma_m
        self.doFlow_corrections = doFlow_corrections
        self.output_format = output_format
        self.name_convention = "Legacy"

        # muon selection cuts
        self.muon_pt_threshold = 10
        self.muon_max_eta = 2.4
        self.mu_id_wp = "medium"
        self.mu_iso_wp = "tight"
        self.muon_photon_min_dr = 0.2
        self.global_muon = True
        self.muon_max_dxy = None
        self.muon_max_dz = None

        # electron selection cuts
        self.electron_pt_threshold = 15
        self.electron_max_eta = 2.5
        self.electron_photon_min_dr = 0.2
        self.el_id_wp = "loose"  # this includes isolation
        self.electron_max_dxy = None
        self.electron_max_dz = None

        # jet selection cuts
        self.jet_jetId = "tightLepVeto"  # can be "tightLepVeto" or "tight": https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
        self.jet_dipho_min_dr = 0.4
        self.jet_pho_min_dr = 0.4
        self.jet_ele_min_dr = 0.4
        self.jet_muo_min_dr = 0.4
        self.jet_pt_threshold = 20
        self.jet_max_eta = 4.7
        self.bjet_mva = "particleNet"  # Possible choices: particleNet, deepJet, robustParticleTransformer
        self.bjet_wp = "T"  # Possible choices: L, M, T, XT, XXT

        self.clean_jet_dipho = False
        self.clean_jet_pho = True
        self.clean_jet_ele = True
        self.clean_jet_muo = True

        # diphoton preselection cuts
        self.min_pt_photon = 25.0
        self.min_pt_lead_photon = 35.0
        self.min_mvaid = -0.9
        self.max_sc_eta = 2.5
        self.gap_barrel_eta = 1.4442
        self.gap_endcap_eta = 1.566
        self.max_hovere = 0.08
        self.min_full5x5_r9 = 0.8
        self.max_chad_iso = 20.0
        self.max_chad_rel_iso = 0.3

        self.min_full5x5_r9_EB_high_r9 = 0.85
        self.min_full5x5_r9_EE_high_r9 = 0.9
        self.min_full5x5_r9_EB_low_r9 = 0.5
        self.min_full5x5_r9_EE_low_r9 = 0.8
        self.max_trkSumPtHollowConeDR03_EB_low_r9 = (
            6.0  # for v11, we cut on Photon_pfChargedIsoPFPV
        )
        self.max_trkSumPtHollowConeDR03_EE_low_r9 = 6.0  # Leaving the names of the preselection cut variables the same to change as little as possible
        self.max_sieie_EB_low_r9 = 0.015
        self.max_sieie_EE_low_r9 = 0.035
        self.max_pho_iso_EB_low_r9 = 4.0
        self.max_pho_iso_EE_low_r9 = 4.0

        self.eta_rho_corr = 1.5
        self.low_eta_rho_corr = 0.16544
        self.high_eta_rho_corr = 0.13212
        # EA values for Run3 from Egamma
        self.EA1_EB1 = 0.102056
        self.EA2_EB1 = -0.000398112
        self.EA1_EB2 = 0.0820317
        self.EA2_EB2 = -0.000286224
        self.EA1_EE1 = 0.0564915
        self.EA2_EE1 = -0.000248591
        self.EA1_EE2 = 0.0428606
        self.EA2_EE2 = -0.000171541
        self.EA1_EE3 = 0.0395282
        self.EA2_EE3 = -0.000121398
        self.EA1_EE4 = 0.0369761
        self.EA2_EE4 = -8.10369e-05
        self.EA1_EE5 = 0.0369417
        self.EA2_EE5 = -2.76885e-05

        logger.debug(f"Setting up processor with metaconditions: {self.meta}")

        if (self.bjet_mva != "deepJet") and (self.nano_version < 12):
            logger.error(f"\n {self.bjet_mva} is only supported for nanoAOD v12 and above. Please change the bjet_mva to deepJet. Exiting...\n")
            exit()

        self.taggers = []
        if taggers is not None:
            self.taggers = taggers
            self.taggers.sort(key=lambda x: x.priority)

        self.prefixes = {"pho_lead": "lead", "pho_sublead": "sublead"}

        if not self.doDeco:
            logger.info("Skipping Mass resolution decorrelation as required")
        else:
            logger.info("Performing Mass resolution decorrelation as required")

        # build the chained quantile regressions
        if self.applyCQR:
            try:
                self.chained_quantile: Optional[
                    ChainedQuantileRegression
                ] = ChainedQuantileRegression(**self.meta["PhoIdInputCorrections"])
            except Exception as e:
                warnings.warn(f"Could not instantiate ChainedQuantileRegression: {e}")
                self.chained_quantile = None
        else:
            logger.info("Skipping CQR as required")
            self.chained_quantile = None

        # initialize photonid_mva
        photon_id_mva_dir = os.path.dirname(photon_id_mva_weights.__file__)
        try:
            logger.debug(
                f"Looking for {self.meta['flashggPhotons']['photonIdMVAweightfile_EB']} in {photon_id_mva_dir}"
            )
            self.photonid_mva_EB = load_photonid_mva(
                os.path.join(
                    photon_id_mva_dir,
                    self.meta["flashggPhotons"]["photonIdMVAweightfile_EB"],
                )
            )
            self.photonid_mva_EE = load_photonid_mva(
                os.path.join(
                    photon_id_mva_dir,
                    self.meta["flashggPhotons"]["photonIdMVAweightfile_EE"],
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate PhotonID MVA on the fly: {e}")
            self.photonid_mva_EB = None
            self.photonid_mva_EE = None

        # initialize diphoton mva
        diphoton_weights_dir = os.path.dirname(diphoton_mva_dir.__file__)
        logger.debug(
            f"Base path to look for IDMVA weight files: {diphoton_weights_dir}"
        )

        try:
            self.diphoton_mva = load_bdt(
                os.path.join(
                    diphoton_weights_dir, self.meta["flashggDiPhotonMVA"]["weightFile"]
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate diphoton MVA: {e}")
            self.diphoton_mva = None

    def process_extra(self, events: ak.Array) -> ak.Array:
        raise NotImplementedError

    def apply_filters_and_triggers(self, events: ak.Array) -> ak.Array:
        # met filters
        met_filters = self.meta["flashggMetFilters"][self.data_kind]
        filtered = functools.reduce(
            operator.and_,
            (events.Flag[metfilter.split("_")[-1]] for metfilter in met_filters),
        )

        triggered = ak.ones_like(filtered)
        if self.apply_trigger:
            trigger_names = []
            triggers = self.meta["TriggerPaths"][self.trigger_group][self.analysis]
            hlt = events.HLT
            for trigger in triggers:
                actual_trigger = trigger.replace("HLT_", "").replace("*", "")
                for field in hlt.fields:
                    if field.startswith(actual_trigger):
                        trigger_names.append(field)
            triggered = functools.reduce(
                operator.or_, (hlt[trigger_name] for trigger_name in trigger_names)
            )

        return events[filtered & triggered]

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        # here we start recording possible coffea accumulators
        # most likely histograms, could be counters, arrays, ...
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(
                ak.num(events.genWeight, axis=0)
            )
            histos_etc[dataset_name]["nPos"] = int(ak.sum(events.genWeight > 0))
            histos_etc[dataset_name]["nNeg"] = int(ak.sum(events.genWeight < 0))
            histos_etc[dataset_name]["nEff"] = int(
                histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]
            )
            histos_etc[dataset_name]["genWeightSum"] = float(
                ak.sum(events.genWeight)
            )
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(ak.sum(events.genWeight))
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        # add veto EE leak branch for photons, could also be used for electrons
        if (
            self.year[dataset_name][0] == "2022EE"
            or self.year[dataset_name][0] == "2022postEE"
        ):
            events.Photon = veto_EEleak_flag(self, events.Photon)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        # If --Smear-sigma_m == True and no Smearing correction in .json for MC throws an error, since the pt spectrum need to be smeared in order to properly calculate the smeared sigma_m_m
        if (
            self.data_kind == "mc"
            and self.Smear_sigma_m
            and ("Smearing_Trad" not in correction_names and "Smearing_IJazZ" not in correction_names and "Smearing2G_IJazZ" not in correction_names)
        ):
            warnings.warn(
                "Smearing_Trad or  Smearing_IJazZ or Smearing2G_IJazZ should be specified in the corrections field in .json in order to smear the mass!"
            )
            sys.exit(0)

        # save raw pt if we use scale/smearing corrections
        # These needs to be before the smearing of the mass resolution in order to have the raw pt for the function
        s_or_s_applied = False
        for correction in correction_names:
            if "scale" or "smearing" in correction.lower():
                s_or_s_applied = True
        if s_or_s_applied:
            events.Photon["pt_raw"] = ak.copy(events.Photon.pt)

        # Since now we are applying Smearing term to the sigma_m_over_m i added this portion of code
        # specially for the estimation of smearing terms for the data events [data pt/energy] are not smeared!
        if self.data_kind == "data" and self.Smear_sigma_m:
            if "Scale_Trad" in correction_names:
                correction_name = "Smearing_Trad"
            elif "Scale_IJazZ" in correction_names:
                correction_name = "Smearing_IJazZ"
            elif "Scale2G_IJazZ" in correction_names:
                correction_name = "Smearing2G_IJazZ"
            else:
                logger.info('Specify a scale correction for the data in the corrections field in .json in order to smear the mass!')
                sys.exit(0)

            logger.info(
                f"""
                Applying correction {correction_name} to dataset {dataset_name}\n
                This is only for the addition of the smearing term to the sigma_m_over_m in data\n
                """
            )
            varying_function = available_object_corrections[correction_name]
            events = varying_function(events=events, year=self.year[dataset_name][0])

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(
                    events=events, year=self.year[dataset_name][0]
                )
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                logger.warning(f"Could not process correction {correction_name}.")
                continue

        # apply jetvetomap: only retain events that without any jets in the veto region
        if not self.skipJetVetoMap:
            events = jetvetomap(
                self, events, logger, dataset_name, year=self.year[dataset_name][0]
            )

        original_photons = events.Photon
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet

        # Computing the normalizing flow correction
        if self.data_kind == "mc" and self.doFlow_corrections:
            original_photons = apply_flow_corrections_to_photons(
                original_photons,
                events,
                self.meta,
                self.year[dataset_name][0],
                self.add_photonid_mva_run3,
                logger
            )

        # Add additional collections if object systematics should be applied
        collections = {
            "Photon": original_photons,
        }

        # Apply the systematic variations.
        collections = apply_systematic_variations_object_level(
            systematic_names,
            events,
            self.year[dataset_name][0],
            logger,
            available_object_systematics,
            available_weight_systematics,
            collections
        )

        original_photons = collections["Photon"]

        # Write systematic variations to dictss
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(
                    original_photons.systematics[systematic][variation]
                )

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        # Flatten
        variations_flattened = sum(variations_combined, [])  # Begin with empty list and keep concatenating
        # Attach _down and _up
        variations = [item + suffix for item in variations_flattened for suffix in ['_down', '_up']]
        # Add nominal to the list
        variations.append('nominal')
        logger.debug(f"[systematics variations] {variations}")

        for variation in variations:
            photons, jets = photons_dct["nominal"], events.Jet
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
            do_variation = variation  # We can also simplify this a bit but for now it works

            if self.chained_quantile is not None:
                photons = self.chained_quantile.apply(photons, events)
            # recompute photonid_mva on the fly
            if self.photonid_mva_EB and self.photonid_mva_EE:
                photons = self.add_photonid_mva(photons, events)

            # photon preselection
            photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0])

            diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)

            # Apply the fiducial cut at detector level with helper function
            diphotons = apply_fiducial_cut_det_level(self, diphotons)

            if self.data_kind == "mc":
                # Add the fiducial flags for particle level
                diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
                diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')

                GenPTH, GenYH, GenPhiH = get_higgs_gen_attributes(events)

                GenPTH = ak.fill_none(GenPTH, -999.0)
                diphotons['GenPTH'] = GenPTH

                genJets = get_genJets(self, events, pt_cut=30., eta_cut=2.5)
                diphotons['GenNJ'] = ak.num(genJets)
                GenPTJ0 = choose_jet(genJets.pt, 0, -999.0)  # Choose zero (leading) jet and pad with -999 if none
                diphotons['GenPTJ0'] = GenPTJ0

                gen_first_jet_eta = choose_jet(genJets.eta, 0, -999.0)
                gen_first_jet_mass = choose_jet(genJets.mass, 0, -999.0)
                gen_first_jet_phi = choose_jet(genJets.phi, 0, -999.0)

                gen_first_jet_pz = GenPTJ0 * numpy.sinh(gen_first_jet_eta)
                gen_first_jet_energy = numpy.sqrt((GenPTJ0**2 * numpy.cosh(gen_first_jet_eta)**2) + gen_first_jet_mass**2)

                # B-Jets
                # Following the recommendations of https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools for hadronFlavour
                # and the Run 2 recommendations for the bjets
                genJetCondition = (genJets.pt > 30) & (numpy.abs(genJets.eta) < 2.5)
                genBJetCondition = genJetCondition & (genJets.hadronFlavour == 5)
                genJets = ak.with_field(genJets, genBJetCondition, "GenIsBJet")
                num_bjets = ak.sum(genJets["GenIsBJet"], axis=-1)
                diphotons["GenNBJet"] = num_bjets

                gen_first_bjet_pt = choose_jet(genJets[genJets["GenIsBJet"] == True].pt, 0, -999.0)
                diphotons["GenPTbJ1"] = gen_first_bjet_pt

                gen_first_jet_hFlav = choose_jet(genJets.hadronFlavour, 0, -999.0)
                diphotons["GenJ1hFlav"] = gen_first_jet_hFlav

                with numpy.errstate(divide='ignore', invalid='ignore'):
                    GenYJ0 = 0.5 * numpy.log((gen_first_jet_energy + gen_first_jet_pz) / (gen_first_jet_energy - gen_first_jet_pz))

                GenYJ0 = ak.fill_none(GenYJ0, -999)
                GenYJ0 = ak.where(numpy.isnan(GenYJ0), -999, GenYJ0)
                diphotons['GenYJ0'] = GenYJ0

                GenYH = ak.fill_none(GenYH, -999)
                GenYH = ak.where(numpy.isnan(GenYH), -999, GenYH)
                diphotons['GenYH'] = GenYH

                GenAbsPhiHJ0 = numpy.abs(gen_first_jet_phi - GenPhiH)

                # Set all entries above 2*pi to -999
                GenAbsPhiHJ0 = ak.where(
                    GenAbsPhiHJ0 > 2 * numpy.pi,
                    -999,
                    GenAbsPhiHJ0
                )
                GenAbsPhiHJ0_pi_array = ak.full_like(GenAbsPhiHJ0, 2 * numpy.pi)

                # Select the smallest angle
                GenAbsPhiHJ0 = ak.where(
                    GenAbsPhiHJ0 > numpy.pi,
                    GenAbsPhiHJ0_pi_array - GenAbsPhiHJ0,
                    GenAbsPhiHJ0
                )
                GenAbsPhiHJ0 = ak.fill_none(GenAbsPhiHJ0, -999.0)

                diphotons["GenDPhiHJ0"] = GenAbsPhiHJ0

                GenAbsYHJ0 = numpy.abs(GenYJ0 - GenYH)

                # Set all entries above 500 to -999
                GenAbsYHJ0 = ak.where(
                    GenAbsYHJ0 > 500,
                    -999,
                    GenAbsYHJ0
                )

                diphotons["GenDYHJ0"] = GenAbsYHJ0

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                diphotons = self.add_diphoton_mva(diphotons, events)

            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            btagMVA_selection = {
                "deepJet": {"btagDeepFlavB": jets.btagDeepFlavB},  # Always available
                "particleNet": {"btagPNetB": jets.btagPNetB} if self.nano_version >= 12 else {},
                "robustParticleTransformer": {"btagRobustParTAK4B": jets.btagRobustParTAK4B} if self.nano_version >= 12 else {},
            }

            # jet_variables
            jets = ak.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": ak.zeros_like(
                        jets.pt
                    ),
                    **btagMVA_selection.get(self.bjet_mva, {}),
                    "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.zeros_like(jets.pt),
                    "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                    "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                    "btagDeepFlav_QG": jets.btagDeepFlavQG,
                    "jetId": jets.jetId,
                    **(
                        {"neHEF": jets.neHEF, "neEmEF": jets.neEmEF, "chEmEF": jets.chEmEF, "muEF": jets.muEF} if self.nano_version == 12 else {}
                    ),
                    **(
                        {"neHEF": jets.neHEF, "neEmEF": jets.neEmEF, "chMultiplicity": jets.chMultiplicity, "neMultiplicity": jets.neMultiplicity, "chEmEF": jets.chEmEF, "chHEF": jets.chHEF, "muEF": jets.muEF} if self.nano_version == 13 else {}
                    ),
                }
            )
            jets = ak.with_name(jets, "PtEtaPhiMCandidate")

            electrons = ak.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.charge,
                    "cutBased": events.Electron.cutBased,
                    "mvaIso_WP90": events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": events.Electron.mvaIso_WP80,
                }
            )
            electrons = ak.with_name(electrons, "PtEtaPhiMCandidate")

            # Special cut for base workflow to replicate iso cut for electrons also for muons
            events['Muon'] = events.Muon[events.Muon.pfRelIso03_all < 0.2]

            muons = ak.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.charge,
                    "tightId": events.Muon.tightId,
                    "mediumId": events.Muon.mediumId,
                    "looseId": events.Muon.looseId,
                    "isGlobal": events.Muon.isGlobal,
                    "pfIsoId": events.Muon.pfIsoId
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[
                select_electrons(self, electrons, diphotons)
            ]
            sel_muons = muons[select_muons(self, muons, diphotons)]

            # jet selection and pt ordering
            jets = jets[
                select_jets(self, jets, diphotons, sel_muons, sel_electrons)
            ]
            jets = jets[ak.argsort(jets.pt, ascending=False)]

            # adding selected jets to events to be used in ctagging SF calculation
            events["sel_jets"] = jets
            n_jets = ak.num(jets)
            Njets2p5 = ak.num(jets[(jets.pt > 30) & (numpy.abs(jets.eta) < 2.5)])

            # B-Jets
            btag_WP = getBTagMVACut(mva_name=self.bjet_mva,
                                    mva_wp=self.bjet_wp,
                                    year=self.year[dataset_name][0])

            btag_mva_column = list(btagMVA_selection[self.bjet_mva].keys())[0]

            bJetCondition = (jets.pt > 30) & (abs(jets.eta) < 2.5) & (jets[btag_mva_column] >= btag_WP)
            jets = ak.with_field(jets, bJetCondition, f"{self.bjet_mva}_IsBJet")
            num_bjets = ak.sum(jets[f"{self.bjet_mva}_IsBJet"], axis=-1)
            diphotons[f"{self.bjet_mva}_NBJet"] = num_bjets

            first_bjet_pt = choose_jet(jets[jets[f"{self.bjet_mva}_IsBJet"] == True].pt, 0, -999.0)
            diphotons[f"{self.bjet_mva}_PTbJ1"] = first_bjet_pt

            first_jet_pt = choose_jet(jets.pt, 0, -999.0)
            first_jet_eta = choose_jet(jets.eta, 0, -999.0)
            first_jet_phi = choose_jet(jets.phi, 0, -999.0)
            first_jet_mass = choose_jet(jets.mass, 0, -999.0)
            first_jet_charge = choose_jet(jets.charge, 0, -999.0)

            second_jet_pt = choose_jet(jets.pt, 1, -999.0)
            second_jet_eta = choose_jet(jets.eta, 1, -999.0)
            second_jet_phi = choose_jet(jets.phi, 1, -999.0)
            second_jet_mass = choose_jet(jets.mass, 1, -999.0)
            second_jet_charge = choose_jet(jets.charge, 1, -999.0)

            diphotons["PTJ0"] = first_jet_pt
            diphotons["first_jet_eta"] = first_jet_eta
            diphotons["first_jet_phi"] = first_jet_phi
            diphotons["first_jet_mass"] = first_jet_mass
            diphotons["first_jet_charge"] = first_jet_charge

            diphotons["PTJ1"] = second_jet_pt
            diphotons["second_jet_eta"] = second_jet_eta
            diphotons["second_jet_phi"] = second_jet_phi
            diphotons["second_jet_mass"] = second_jet_mass
            diphotons["second_jet_charge"] = second_jet_charge

            diphotons["n_jets"] = n_jets
            diphotons["NJ"] = Njets2p5

            first_jet_pz = first_jet_pt * numpy.sinh(first_jet_eta)
            first_jet_energy = numpy.sqrt((first_jet_pt**2 * numpy.cosh(first_jet_eta)**2) + first_jet_mass**2)

            first_jet_y = 0.5 * numpy.log((first_jet_energy + first_jet_pz) / (first_jet_energy - first_jet_pz))
            first_jet_y = ak.fill_none(first_jet_y, -999)
            first_jet_y = ak.where(numpy.isnan(first_jet_y), -999, first_jet_y)
            diphotons["YJ0"] = first_jet_y

            AbsPhiHJ0 = numpy.abs(first_jet_phi - diphotons["phi"])

            AbsPhiHJ0_pi_array = ak.full_like(AbsPhiHJ0, 2 * numpy.pi)

            # Select the smallest angle
            AbsPhiHJ0 = ak.where(
                AbsPhiHJ0 > numpy.pi,
                AbsPhiHJ0_pi_array - AbsPhiHJ0,
                AbsPhiHJ0
            )
            AbsPhiHJ0 = ak.where(
                AbsPhiHJ0 > 2 * numpy.pi,
                -999,
                ak.where(
                    AbsPhiHJ0 < 0,
                    -999,
                    AbsPhiHJ0
                )
            )
            diphotons["DPhiHJ0"] = AbsPhiHJ0

            AbsYHJ0 = numpy.abs(first_jet_y - diphotons["rapidity"])

            # Set all entries above 500 to -999
            AbsYHJ0 = ak.where(
                AbsYHJ0 > 500,
                -999,
                AbsYHJ0
            )

            diphotons["DYHJ0"] = AbsYHJ0

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, diphotons
                )  # creates new column in diphotons - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = ak.num(diphotons.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        ak.flatten(
                            diphotons[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = ak.from_regular(
                    ak.unflatten(flat_tags, counts), axis=2
                )
                winner = ak.min(tags[tags != 0], axis=2)
                diphotons["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = ak.argsort(diphotons.best_tag, stable=True)
                diphotons = diphotons[sorted]

            diphotons = ak.firsts(diphotons)
            # set diphotons as part of the event record
            events[f"diphotons_{do_variation}"] = diphotons
            # annotate diphotons with event information
            diphotons["event"] = events.event
            diphotons["lumi"] = events.luminosityBlock
            diphotons["run"] = events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = events.PV.npvs
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = ak.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    ak.is_none(diphotons)
                    | ak.is_none(diphotons.best_tag)
                )
                diphotons = diphotons[selection_mask]
            else:
                selection_mask = ~ak.is_none(diphotons)
                diphotons = diphotons[selection_mask]

            bTagFixedWP_present = any("bTagFixedWP" in item for item in systematic_names) + any("bTagFixedWP" in item for item in correction_names)
            PNet_present = any("bTagFixedWP_PNet" in item for item in systematic_names) + any("bTagFixedWP_PNet" in item for item in correction_names)

            if PNet_present and (self.nano_version < 12):
                logger.error("\n B-Tagging systematics and corrections using Particle Net are only available for NanoAOD v12 or higher. Exiting! \n")
                exit()

            # return if there is no surviving events
            if len(diphotons) == 0:
                logger.debug("No surviving events in this run, return now!")
                return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]),storeIndividual=True)
                # set weights to generator weights
                event_weights._weight = ak.to_numpy(events["genWeight"][selection_mask])

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        common_args = {
                            "events": events[selection_mask],
                            "photons": events[f"diphotons_{do_variation}"][selection_mask],
                            "weights": event_weights,
                            "dataset_name": dataset_name,
                            "year": self.year[dataset_name][0],
                        }

                        if any("bTagFixedWP" in item for item in correction_names):
                            common_args["bTagEffFileName"] = self.bTagEffFileName

                        varying_function = available_weight_corrections[correction_name]
                        event_weights = varying_function(**common_args)

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    diphotons["nweight_LHEScale"] = ak.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        ak.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                common_args = {
                                    "events": events[selection_mask],
                                    "photons": events[f"diphotons_{do_variation}"][selection_mask],
                                    "weights": event_weights,
                                    "dataset_name": dataset_name,
                                    "year": self.year[dataset_name][0],
                                }

                                if any("bTagFixedWP" in item for item in systematic_names):
                                    common_args["bTagEffFileName"] = self.bTagEffFileName

                                varying_function = available_weight_systematics[systematic_name]
                                event_weights = varying_function(**common_args)

                diphotons["weight"] = event_weights.weight() / (
                    event_weights.partial_weight(include=["bTagFixedWP"])
                    if bTagFixedWP_present
                    else 1
                )
                diphotons["weight_central"] = event_weights.weight() / (
                    (event_weights.partial_weight(include=["bTagFixedWP"]) * events["genWeight"][selection_mask])
                    if bTagFixedWP_present
                    else events["genWeight"][selection_mask]
                )

                if bTagFixedWP_present:
                    diphotons["weight_bTagFixedWP"] = event_weights.partial_weight(include=["bTagFixedWP"])

                metadata["sum_weight_central"] = str(
                    ak.sum(
                        event_weights.weight()
                        / (
                            event_weights.partial_weight(include=["bTagFixedWP"])
                            if bTagFixedWP_present
                            else 1
                        )
                    )
                )
                metadata["sum_weight_central_wo_bTagSF"] = str(
                    ak.sum(
                        event_weights.weight()
                        / (
                            (event_weights.partial_weight(include=["bTagSF"]) * event_weights.partial_weight(include=["bTagFixedWP"]))
                            if bTagFixedWP_present
                            else event_weights.partial_weight(include=["bTagSF"])
                        )
                    )
                )

                # Handle variations
                if do_variation == "nominal":
                    if event_weights.variations:
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        diphotons["weight_" + modifier] = event_weights.weight(modifier=modifier)
                        if "bTagSF" in modifier:
                            metadata["sum_weight_" + modifier] = str(
                                ak.sum(event_weights.weight(modifier=modifier))
                                / (
                                    event_weights.partial_weight(include=["bTagFixedWP"])
                                    if bTagFixedWP_present
                                    else 1
                                )
                            )

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = ak.ones_like(
                    diphotons["event"]
                )
                diphotons["weight"] = ak.ones_like(diphotons["event"])

            # Compute and store the different variations of sigma_m_over_m
            diphotons = compute_sigma_m(diphotons, processor='base', flow_corrections=self.doFlow_corrections, smear=self.Smear_sigma_m, IsData=(self.data_kind == "data"))

            # Decorrelating the mass resolution - Still need to supress the decorrelator noises
            if self.doDeco:

                # Decorrelate nominal sigma_m_over_m
                diphotons["sigma_m_over_m_nominal_decorr"] = decorrelate_mass_resolution(diphotons, type="nominal", year=self.year[dataset_name][0])

                # decorrelate smeared nominal sigma_m_overm_m
                if (self.Smear_sigma_m):
                    diphotons["sigma_m_over_m_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="smeared", year=self.year[dataset_name][0])

                # decorrelate flow corrected sigma_m_over_m
                if (self.doFlow_corrections):
                    diphotons["sigma_m_over_m_corr_decorr"] = decorrelate_mass_resolution(diphotons, type="corr", year=self.year[dataset_name][0])

                # decorrelate flow corrected smeared sigma_m_over_m
                if (self.doFlow_corrections and self.Smear_sigma_m):
                    if self.data_kind == "data" and ("Scale_IJazZ" in correction_names or "Scale2G_IJazZ" in correction_names):
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    elif self.data_kind == "mc" and ("Smearing2G_IJazZ" in correction_names or "Smearing_IJazZ" in correction_names):
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    else:
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)

                # Instead of the nominal sigma_m_over_m, we will use the smeared version of it -> (https://indico.cern.ch/event/1319585/#169-update-on-the-run-3-mass-r)
                # else:
                #    warnings.warn("Smeamering need to be applied in order to decorrelate the (Smeared) mass resolution. -- Exiting!")
                #    sys.exit(0)

            if self.output_location is not None:
                if self.output_format == "root":
                    df = diphoton_list_to_pandas(self, diphotons)
                else:
                    akarr = diphoton_ak_array(self, diphotons)

                    # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                    akarr = akarr[
                        [
                            field
                            for field in akarr.fields
                            if "lead_fixedGridRhoAll" not in field
                        ]
                    ]

                fname = (
                    events.behavior[
                        "__events_factory__"
                    ]._partition_key.replace("/", "_")
                    + ".%s" % self.output_format
                )
                fname = (fname.replace("%2F","")).replace("%3B1","")
                subdirs = []
                if "dataset" in events.metadata:
                    subdirs.append(events.metadata["dataset"])
                subdirs.append(do_variation)
                if self.output_format == "root":
                    dump_pandas(self, df, fname, self.output_location, subdirs)
                else:
                    dump_ak_array(
                        self, akarr, fname, self.output_location, metadata, subdirs,
                    )

        return histos_etc

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        raise NotImplementedError

    def add_diphoton_mva(
        self, diphotons: ak.Array, events: ak.Array
    ) -> ak.Array:
        return calculate_diphoton_mva(
            self,
            (self.diphoton_mva, self.meta["HiggsDNA_DiPhotonMVA"]["inputs"]),
            diphotons,
            events,
        )

    def add_photonid_mva(
        self, photons: ak.Array, events: ak.Array
    ) -> ak.Array:
        photons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll * ak.ones_like(
            photons.pt
        )
        counts = ak.num(photons, axis=-1)
        photons = ak.flatten(photons)
        isEB = ak.to_numpy(numpy.abs(photons.eta) < 1.5)
        mva_EB = calculate_photonid_mva(
            (self.photonid_mva_EB, self.meta["flashggPhotons"]["inputs_EB"]), photons
        )
        mva_EE = calculate_photonid_mva(
            (self.photonid_mva_EE, self.meta["flashggPhotons"]["inputs_EE"]), photons
        )
        mva = ak.where(isEB, mva_EB, mva_EE)
        photons["mvaID"] = mva

        return ak.unflatten(photons, counts)

    def add_photonid_mva_run3(
        self, photons: ak.Array, events: ak.Array
    ) -> ak.Array:

        preliminary_path = os.path.join(os.path.dirname(__file__), '../tools/flows/run3_mvaID_models/')
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        rho = events.Rho.fixedGridRhoAll * ak.ones_like(photons.pt)
        rho = ak.flatten(rho)

        photons = ak.flatten(photons)

        isEB = ak.to_numpy(numpy.abs(photons.eta) < 1.5)
        mva_EB = calculate_photonid_mva_run3(
            [photonid_mva_EB, self.meta["flashggPhotons"]["inputs_EB"]], photons , rho
        )
        mva_EE = calculate_photonid_mva_run3(
            [photonid_mva_EE, self.meta["flashggPhotons"]["inputs_EE"]], photons, rho
        )
        mva = ak.where(isEB, mva_EB, mva_EE)
        photons["mvaID_run3"] = mva

        return mva

    def add_corr_photonid_mva_run3(
        self, photons: ak.Array, events: ak.Array
    ) -> ak.Array:

        preliminary_path = os.path.join(os.path.dirname(__file__), '../tools/flows/run3_mvaID_models/')
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        rho = events.Rho.fixedGridRhoAll * ak.ones_like(photons.pt)
        rho = ak.flatten(rho)

        photons = ak.flatten(photons)

        # Now calculating the corrected mvaID
        isEB = ak.to_numpy(numpy.abs(photons.eta) < 1.5)
        corr_mva_EB = calculate_photonid_mva_run3(
            [photonid_mva_EB, self.meta["flashggPhotons"]["inputs_EB_corr"]], photons, rho
        )
        corr_mva_EE = calculate_photonid_mva_run3(
            [photonid_mva_EE, self.meta["flashggPhotons"]["inputs_EE_corr"]], photons, rho
        )
        corr_mva = ak.where(isEB, corr_mva_EB, corr_mva_EE)

        return corr_mva
