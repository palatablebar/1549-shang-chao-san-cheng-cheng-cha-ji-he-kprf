"""basket backtest."""

from .contracts import (
    Verification,
    Assessment,
    CalibrationRow,
    ModelReport,
    CalibrationDecision,
    CalibrationSelections,
    CohortCalibration,
    ZoneCalibration,
    CalibrationChoice,
    CohortCalibrationPlan,
    ZoneCalibrationPlan,
    CalibrationPlan,
)
from .service import (
    prepare_verification,
    validate_predictions,
    performance,
    calibration_evidence,
    check_point,
    select_cohort_calibrations,
    select_cohort_calibration,
    plan_cohort_calibrations,
    choose_cohort_calibrations,
    choose_cohort_calibration,
    cohort_calibration_plan,
    calibration_choices,
    calibration_for_cohort,
    cohort_calibration,
    zone_calibration,
    fit_cohort_verifications,
    cohort_verifications,
    evaluate_cohort_choice,
    validate_with_selections,
    calibration_gate,
    group_by_cohort,
)

__all__ = (
    'Verification', 'Assessment', 'CalibrationRow', 'ModelReport',
    'CalibrationDecision', 'CalibrationSelections', 'CohortCalibration',
    'ZoneCalibration', 'CalibrationChoice', 'CohortCalibrationPlan',
    'ZoneCalibrationPlan', 'CalibrationPlan',
    'prepare_verification', 'validate_predictions', 'performance', 'calibration_evidence',
    'check_point', 'select_cohort_calibrations', 'select_cohort_calibration',
    'plan_cohort_calibrations', 'choose_cohort_calibrations', 'choose_cohort_calibration',
    'cohort_calibration_plan', 'calibration_choices', 'calibration_for_cohort',
    'cohort_calibration', 'zone_calibration', 'fit_cohort_verifications',
    'cohort_verifications', 'evaluate_cohort_choice', 'validate_with_selections',
    'calibration_gate', 'group_by_cohort',
)

# 兼容从包根直接引用各 service/contracts 公共名的调用方式
from . import service as _service  # noqa: E402
from . import contracts as _contracts  # noqa: E402

for _module in (_service, _contracts):
    for _name in dir(_module):
        if not _name.startswith('_'):
            globals().setdefault(_name, getattr(_module, _name))
            if _name not in __all__:
                __all__ += (_name,)
