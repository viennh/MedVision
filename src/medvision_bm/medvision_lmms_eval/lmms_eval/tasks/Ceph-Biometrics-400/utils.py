from lmms_eval.tasks.medvision.medvision_utils import (
    aggregate_results_avgMAE,
    aggregate_results_avgMRE,
    aggregate_results_MAE,
    aggregate_results_MRE,
    aggregate_results_SuccessRate,
    create_doc_to_text_BiometricsFromLandmarks,
    create_doc_to_text_BiometricsFromLandmarks_CoT,
    create_doc_to_text_BiometricsFromLandmarks_CoT_woInstruct,
    doc_to_target_BiometricsFromLandmarks,
    doc_to_visual,
    process_results_BiometricsFromLandmarks,
)
from medvision_ds.datasets.Ceph_Biometrics_400 import preprocess_biometry

doc_to_text_BiometricsFromLandmarks = create_doc_to_text_BiometricsFromLandmarks(preprocess_biometry)
doc_to_text_BiometricsFromLandmarks_CoT = create_doc_to_text_BiometricsFromLandmarks_CoT(preprocess_biometry)
doc_to_text_BiometricsFromLandmarks_CoT_woInstruct = create_doc_to_text_BiometricsFromLandmarks_CoT_woInstruct(preprocess_biometry)
