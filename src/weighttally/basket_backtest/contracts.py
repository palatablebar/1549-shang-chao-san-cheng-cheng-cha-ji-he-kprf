from dataclasses import dataclass

from ..tare_deviation.contracts import Prediction

RISK_MODE = True

# 校准选用范围：全场校准 / 同作业区分区校准
GLOBAL_SCOPE = 'global'
COHORT_SCOPE = 'cohort'
SCOPE_LABELS = {GLOBAL_SCOPE: '全场校准', COHORT_SCOPE: '分区校准'}


@dataclass(frozen=True, slots=True)
class Verification:
    method: str
    parameters: tuple
    size: int
    positives: int | None
    cohort: str | None = None

    def __post_init__(self):
        if self.size < 0 or self.positives is not None and not 0 <= self.positives <= self.size:
            raise ValueError('校准回查计数失衡')


@dataclass(frozen=True, slots=True)
class Assessment:
    raw: Prediction
    estimate: float
    lower: float | None
    upper: float | None
    uncertain: bool


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    index: int
    predicted: float | None
    observed: float | None
    count: int


@dataclass(frozen=True, slots=True)
class ModelReport:
    status: str
    count: int
    mae: float | None
    error: float | None
    buckets: tuple[CalibrationRow, ...]
    auxiliary: int | float | None


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    """单个作业区当班采用哪套校准的回放结论。

    adopted=True 时 selected 为仅用同区校准期实绩拟合的 candidate；
    否则 selected 与 baseline（全场校准）一致。
    """

    cohort: str
    adopted: bool
    code: str
    reason: str
    baseline: Verification
    selected: Verification
    candidate: Verification | None = None
    cutoff: float = 0.6
    calibration_size: int = 0
    replay_size: int = 0
    normal_count: int = 0
    positive_count: int = 0
    old_caught: tuple[str, ...] = ()
    new_caught: tuple[str, ...] = ()
    lost: tuple[str, ...] = ()
    old_false_alarms: tuple[str, ...] = ()
    new_false_alarms: tuple[str, ...] = ()

    # ---- 派生计数 ----
    @property
    def scope(self):
        return COHORT_SCOPE if self.adopted else GLOBAL_SCOPE

    @property
    def scheme(self):
        return SCOPE_LABELS[self.scope]

    @property
    def choice(self):
        return '启用分区校准' if self.adopted else '沿用全场校准'

    @property
    def caught_before(self):
        return len(self.old_caught)

    @property
    def caught_after(self):
        return len(self.new_caught)

    @property
    def missed_before(self):
        return self.positive_count - len(self.old_caught)

    @property
    def missed_after(self):
        return self.positive_count - len(self.new_caught)

    @property
    def false_before(self):
        return len(self.old_false_alarms)

    @property
    def false_after(self):
        return len(self.new_false_alarms)

    @property
    def false_alarm_relief(self):
        # 分区校准让正常签少报的张数（腾出的复秤位）
        return len(self.old_false_alarms) - len(self.new_false_alarms)

    # ---- 兼容不同命名习惯的只读别名 ----
    @property
    def use_cohort(self):
        return self.adopted

    @property
    def use_zone(self):
        return self.adopted

    @property
    def zone_adopted(self):
        return self.adopted

    @property
    def is_cohort(self):
        return self.adopted

    @property
    def selected_verification(self):
        return self.selected

    @property
    def chosen(self):
        return self.selected

    @property
    def global_verification(self):
        return self.baseline

    @property
    def global_model(self):
        return self.baseline

    @property
    def baseline_verification(self):
        return self.baseline

    @property
    def cohort_verification(self):
        return self.candidate

    @property
    def zone_verification(self):
        return self.candidate

    @property
    def candidate_verification(self):
        return self.candidate

    @property
    def threshold(self):
        return self.cutoff

    @property
    def replay_normals(self):
        return self.normal_count

    @property
    def replay_positives(self):
        return self.positive_count

    @property
    def reason_code(self):
        return self.code

    # 决策可直接当作所选校准锚点传给 check_point（只需 parameters）
    @property
    def parameters(self):
        return self.selected.parameters

    @property
    def method(self):
        return self.selected.method

    @property
    def size(self):
        return self.selected.size

    @property
    def positives(self):
        return self.selected.positives

    def __getitem__(self, key):
        if hasattr(self, key):
            value = getattr(self, key)
            if not key.startswith('_'):
                return value
        aliases = _DECISION_KEY_ALIASES.get(key)
        if aliases is not None:
            return aliases(self)
        evidence = self.evidence()
        if key in evidence:
            return evidence[key]
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def evidence(self):
        """工作台“称差回放”环节渲染的中文判定行。"""
        return {
            '作业区': self.cohort,
            '校准选择': self.scheme,
            '当班校准': self.scheme,
            '结论': self.choice,
            '是否启用分区校准': self.adopted,
            '原因码': self.code,
            '缘由': self.reason,
            '判定缘由': self.reason,
            '校准期实绩': self.calibration_size,
            '回放实绩': self.replay_size,
            '回放正常签': self.normal_count,
            '回放异常签': self.positive_count,
            '旧版抓住异常': len(self.old_caught),
            '本区抓住异常': len(self.new_caught),
            '旧版漏查': self.missed_before,
            '本区漏查': self.missed_after,
            '丢失旧版命中': len(self.lost),
            '旧版正常签误报': len(self.old_false_alarms),
            '本区正常签误报': len(self.new_false_alarms),
            '正常签少报': self.false_alarm_relief,
            '腾出复秤位': self.false_alarm_relief,
            '旧版命中单': list(self.old_caught),
            '本区命中单': list(self.new_caught),
            '丢失异常单': list(self.lost),
            '旧版误报单': list(self.old_false_alarms),
            '本区误报单': list(self.new_false_alarms),
        }

    def to_dict(self):
        data = dict(self.evidence())
        data.update({
            'cohort': self.cohort,
            'adopted': self.adopted,
            'scope': self.scope,
            'code': self.code,
            'reason': self.reason,
            'cutoff': self.cutoff,
            'calibration_method': self.selected.method,
            'calibration_parameters': list(self.selected.parameters),
            'selected_size': self.selected.size,
            'baseline_parameters': list(self.baseline.parameters),
            'candidate_parameters': list(self.candidate.parameters) if self.candidate is not None else None,
            'candidate_method': self.candidate.method if self.candidate is not None else None,
        })
        return data


_DECISION_KEY_ALIASES = {
    'scope': CalibrationDecision.scope.fget,
    'scheme': CalibrationDecision.scheme.fget,
    'choice': CalibrationDecision.choice.fget,
    'false_before': CalibrationDecision.false_before.fget,
    'false_after': CalibrationDecision.false_after.fget,
    'relief': CalibrationDecision.false_alarm_relief.fget,
    'caught_before': CalibrationDecision.caught_before.fget,
    'caught_after': CalibrationDecision.caught_after.fget,
}


class CalibrationSelections(list):
    """各作业区当班校准结论的有序集合。

    直接迭代/下标得到 CalibrationDecision；也可用作业区名取单条结论
    （plan['熟食台']）。未知作业区通过 verification_for 回退全场校准。
    """

    def __init__(self, baseline, decisions=(), cutoff=0.6):
        ordered = tuple(sorted(decisions, key=lambda item: item.cohort))
        super().__init__(ordered)
        self.baseline = baseline
        self.cutoff = cutoff
        self._by_cohort = {item.cohort: item for item in ordered}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._by_cohort[key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        if isinstance(key, str):
            return self._by_cohort.get(key, default)
        return default

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._by_cohort
        return list.__contains__(self, key)

    @property
    def decisions(self):
        return tuple(self)

    def rows(self):
        return tuple(self)

    def keys(self):
        return self._by_cohort.keys()

    def values(self):
        return self._by_cohort.values()

    def items(self):
        return self._by_cohort.items()

    def as_dict(self):
        return dict(self._by_cohort)

    @property
    def global_verification(self):
        return self.baseline

    @property
    def adopted_cohorts(self):
        return tuple(item.cohort for item in self if item.adopted)

    def verification_for(self, cohort):
        decision = self._by_cohort.get(cohort)
        if decision is not None:
            return decision.selected
        return self.baseline

    def assess(self, points):
        from .service import check_point

        scored = []
        for item in points:
            raw = getattr(item, 'raw', item)
            scored.append(check_point(raw, self.verification_for(getattr(raw, 'cohort', None))))
        return tuple(scored)

    def evidence(self):
        return [item.evidence() for item in self]

    def to_list(self):
        return [item.to_dict() for item in self]


# ---- 近义类型名（不同命名习惯的兼容别名） ----
CohortCalibration = CalibrationDecision
CohortCalibrationDecision = CalibrationDecision
ZoneCalibration = CalibrationDecision
ZoneCalibrationDecision = CalibrationDecision
CalibrationChoice = CalibrationDecision
CalibrationVerdict = CalibrationDecision

CohortCalibrationPlan = CalibrationSelections
ZoneCalibrationPlan = CalibrationSelections
CalibrationPlan = CalibrationSelections
CalibrationAssignment = CalibrationSelections
CohortCalibrationSet = CalibrationSelections
