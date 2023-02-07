# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset_v2x import EarlyFusionDatasetV2X
from opencood.data_utils.datasets.early_fusion_dataset_dair import EarlyFusionDatasetDAIR
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_delay import IntermediateFusionDatasetAsync
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2x import IntermediateFusionDatasetV2X
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR
from opencood.data_utils.datasets.late_fusion_dataset_v2x import LateFusionDatasetV2X
from opencood.data_utils.datasets.late_fusion_dataset_dair import LateFusionDatasetDAIR
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2_v2x import IntermediateFusionDatasetV2V2X
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2_dair import IntermediateFusionDatasetV2DAIR
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_delay import IntermediateFusionDatasetDAIRAsync
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_multisweep import IntermediateFusionDatasetDAIRMultisweep
from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_multisweep import IntermediateFusionDatasetMultisweep
from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_irregular import IntermediateFusionDatasetIrregular

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'EarlyFusionDatasetV2X': EarlyFusionDatasetV2X,
    'EarlyFusionDatasetDAIR': EarlyFusionDatasetDAIR,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'IntermediateFusionDatasetAsync': IntermediateFusionDatasetAsync, 
    'IntermediateFusionDatasetV2X': IntermediateFusionDatasetV2X,
    'IntermediateFusionDatasetDAIR': IntermediateFusionDatasetDAIR,
    'LateFusionDatasetV2X': LateFusionDatasetV2X,
    'LateFusionDatasetDAIR': LateFusionDatasetDAIR,
    'IntermediateFusionDatasetV2': IntermediateFusionDatasetV2,
    'IntermediateFusionDatasetV2V2X': IntermediateFusionDatasetV2V2X,
    'IntermediateFusionDatasetV2DAIR': IntermediateFusionDatasetV2DAIR,
    'IntermediateFusionDatasetDAIRAsync': IntermediateFusionDatasetDAIRAsync,
    'IntermediateFusionDatasetMultisweep': IntermediateFusionDatasetMultisweep,
    'IntermediateFusionDatasetIrregular': IntermediateFusionDatasetIrregular
}

# the final range for evaluation
GT_RANGE_OPV2V = [-140, -40, -3, 140, 40, 1]
GT_RANGE_V2XSIM = [-32, -32, -3, 32, 32, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
