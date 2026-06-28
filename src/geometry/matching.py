from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .metrics import intersection_over_union as iou_func
from .core import Ellipse


@dataclass
class MatchingResult:
    """
    Класс для хранения результатов мэтчинга (сопоставления) истинных эллипсов 
    и предсказанных
    """
    true_matched: list[Ellipse]
    pred_matched: list[Ellipse]

    true_unmatched: list[Ellipse]
    pred_unmatched: list[Ellipse]


def match(
        true_ellipses: list[Ellipse], 
        pred_ellipses: list[Ellipse],
        threshold: float = 0.5
    ):
    """
    Функция для мэтчинга (сопоставления) истинных эллипсов и предсказанных

    Args:
        true_ellipses (list): список истинных эллипсов на изображении
        pred_ellipses (list): список обнаруженных эллипсов на изображении
    
    Notes:
        1) Размеры списков могут отличаться

    """
    true_ellipses, pred_ellipses = deepcopy(true_ellipses), deepcopy(pred_ellipses)

    true_ellipses_matched = []
    pred_ellipses_matched = []
    true_ellipses_unmatched = []
    pred_ellipses_unmatched = []

    for pred_e in pred_ellipses:
        if len(true_ellipses) == 0:
            pred_ellipses_unmatched.append(pred_e)
            continue

        true_ellipses = sorted(
            true_ellipses, key=lambda true_e: iou_func(true_e, pred_e), reverse=True
        )
        e_true_best = true_ellipses[0]

        if iou_func(e_true_best, pred_e) >= threshold:
            true_ellipses.pop(0)
            true_ellipses_matched.append(e_true_best)
            pred_ellipses_matched.append(pred_e)
        else:
            pred_ellipses_unmatched.append(pred_e)

    true_ellipses_unmatched = true_ellipses

    result = MatchingResult(
        true_matched=true_ellipses_matched,
        pred_matched=pred_ellipses_matched,
        true_unmatched=true_ellipses_unmatched,
        pred_unmatched=pred_ellipses_unmatched
    )

    return result


def matching_recall(matching_result: MatchingResult):
    """
    Вычислить recall мэтчинга
    """
    tp = len(matching_result.true_matched)
    fn = len(matching_result.true_unmatched)
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan


def matching_precision(matching_result: MatchingResult):
    """
    Вычислить precision мэтчинга
    """
    tp = len(matching_result.pred_matched)
    fp = len(matching_result.pred_unmatched)
    return tp / (tp + fp) if (tp + fp) > 0 else np.nan

