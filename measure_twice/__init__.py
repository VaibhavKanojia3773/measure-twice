"""Measure Twice: calibrated error bars for metric spatial questions from video.

Pipeline (GPU):  grounding -> reconstruction (several models x frame subsets) -> scene_memory -> programs
Analysis (CPU):  calibration (scene-level split conformal, selective answering, decisions), mllm_compare,
                 degradation_eval
"""
__version__ = "1.0.0"
