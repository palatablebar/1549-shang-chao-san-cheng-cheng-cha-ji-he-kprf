from bisect import bisect_right
from collections import defaultdict
from math import fsum, isfinite, log, log1p

from .contracts import (
    Verification,
    Assessment,
    CalibrationRow,
    ModelReport,
    CalibrationDecision,
    CalibrationSelections,
)
from .slices import replay_slices

CALIBRATION_CUTOFF = .6
RISK_CUTOFF = CALIBRATION_CUTOFF


def clip_risk(number):
    return min(1-1e-12,max(1e-12,number))


def stable_loss(probability, outcome):
    clipped = clip_risk(probability)
    return -(log(clipped) if outcome==1 else log1p(-clipped))



def swag_probability(adjusted, prior, power):
    return .12*prior + .88*min(1.,max(0.,adjusted))**power



def weigh_floor_estimate(checks):
    floor = (sum(row.outcome for row in checks) + 1.) / (len(checks)+2)
    best_loss = float('inf')
    best_gain = 1.
    for power in (.5, .75, 1., 1.25, 1.6, 2.2):
        loss = 0.
        for scale in checks:
            value = swag_probability(scale.score, floor, power)
            outcome = scale.outcome
            loss += (value-outcome)**2 * (1.6 if outcome==1 else 1.)
        if loss < best_loss:
            best_gain = power
            best_loss = loss
    return (floor,best_gain)


def calibrated_weigh_floor_estimate(model, p):
    floor, gain = model
    return swag_probability(p,floor,gain)


def prepare_verification(pairs, *, coverage=.85, cohort=None):
    confirmed = []
    positive = 0
    for gate in pairs:
        target = gate.outcome
        if target is None or not gate.observed:
            continue
        if not isfinite(gate.score) or target not in {0,1}:
            raise ValueError('异常度复核需有已确认的正常或异常标签')
        confirmed.append(gate)
        positive += target==1
    n = len(confirmed)
    if n>=8 and 0<positive<n:
        conversion = weigh_floor_estimate(confirmed)
        return Verification('weigh_floor_estimate',conversion,n,positive,cohort)
    return Verification('sparse-risk',(),n,positive,cohort)


def check_point(scored, anchor):
    if not isfinite(scored.score):
        raise ValueError('异常离差分已失效')
    outrank = clip_risk(scored.score)
    provisional = not anchor.parameters
    matched = outrank if provisional else calibrated_weigh_floor_estimate(anchor.parameters,outrank)
    return Assessment(scored,matched,None,None,provisional)


def validate_predictions(scored, prepared):
    return tuple(map(lambda issue:check_point(issue,prepared),scored))


def reliability_rows(cases):
    consensus = [(0.,0,0) for _ in range(4)]
    boundaries = (.15,.4,.68)
    for forecast,observed in cases:
        i = bisect_right(boundaries,forecast)
        sum_forecast, total_positive, size = consensus[i]
        consensus[i] = sum_forecast+forecast,total_positive+observed,size+1
    return tuple(CalibrationRow(i,est/num if num else None,seen/num if num else None,num)
                 for i,(est,seen,num) in enumerate(consensus))


def performance(assessments):
    history = [(check.estimate,check.raw.outcome) for check in assessments if check.raw.observed and check.raw.outcome is not None]
    n = len(history)
    if not n:
        return ModelReport('NO_MATURE_RESULT',0,None,None,(),None)
    deviations = tuple(abs(estimated-confirmed) for estimated,confirmed in history)
    count = sum(outcome==1 for _,outcome in history)
    status = 'READY' if count else 'NO_POSITIVES'
    loss = fsum(stable_loss(*record) for record in history)
    return ModelReport(status,n,fsum(deviations)/n,loss/n,reliability_rows(history),count)


def calibration_evidence(report):
    segments = ('0–15%','15–40%','40–68%','68–100%')
    cards = [{'核验':report.status,'确认单量':report.count,'异常单量':report.auxiliary,
              '平均绝对残差':report.mae,'反常概率损失':report.error}]
    for span, group in zip(segments,report.buckets):
        cards.append({'异常概率区间':span,'样本数':group.count,'预测异常率':group.predicted,'实证异常率':group.observed})
    return cards


def sector_replay(checked, line):
    return replay_slices(checked,line)


def boundary_diagnostics(checks, threshold):
    joined = [row for row in checks if row.raw.observed and row.raw.outcome is not None]
    hit = sum(row.estimate>=threshold and row.raw.outcome==1 for row in joined)
    actual = sum(row.raw.outcome==1 for row in joined)
    forecast = sum(row.estimate>=threshold for row in joined)
    return {'临界判断':'称差争议','实绩超线':actual,'估计超线':forecast,'额外复测':forecast-hit,'漏查':actual-hit,
            '优先队列命中率':hit/forecast if forecast else None,'已确认异常覆盖率':hit/actual if actual else None}


# ---------------------------------------------------------------------------
# 分作业区校准选用
#
# 候选校准只能由“同区 + 校准窗口”的已确认实绩拟合；回放窗口只用于判定，
# 不能回灌拟合，避免回放标签泄漏抬高成绩。采用分区校准必须同时满足：
#   1. 本区回放有已确认正常签和异常签；
#   2. 旧版（全场校准）在本区抓住的异常一张不丢；
#   3. 正常签误报严格少于旧版（腾出复秤位）。
# 任一条件不满足即沿用全场校准。
# ---------------------------------------------------------------------------

def raw_prediction(item):
    """Assessment 取其 raw，Prediction 原样返回；兼容测试桩对象。"""
    return getattr(item, 'raw', item)


def is_confirmed(item):
    raw = raw_prediction(item)
    target = getattr(raw, 'outcome', None)
    return bool(getattr(raw, 'observed', True)) and target in (0, 1)


def confirmed_points(points):
    return [raw_prediction(item) for item in points if is_confirmed(item)]


def group_by_cohort(points):
    """按作业区分组，键与组内顺序都稳定（换序重算结论一致）。"""
    grouped = defaultdict(list)
    for item in points:
        raw = raw_prediction(item)
        grouped[getattr(raw, 'cohort', None)].append(raw)
    result = {}
    for cohort in sorted(grouped, key=lambda value: str(value)):
        # 以单号规范化组内顺序，避免损失累加/选参受入参次序影响
        result[cohort] = sorted(
            grouped[cohort], key=lambda row: str(getattr(row, 'key', '')))
    return result


# 旧名/近义名兼容
cohort_groups = group_by_cohort


def fit_cohort_verifications(calibration_points):
    """只用同区校准期实绩拟合各区候选校准。"""
    return {cohort: prepare_verification(tuple(rows), cohort=cohort)
            for cohort, rows in group_by_cohort(calibration_points).items()}


cohort_verifications = fit_cohort_verifications
build_cohort_verifications = fit_cohort_verifications


def _flagged_keys(points, verification, cutoff):
    assessed = [check_point(raw, verification) for raw in points]
    caught = tuple(sorted(a.raw.key for a in assessed
                          if a.estimate >= cutoff and a.raw.outcome == 1))
    false_alarms = tuple(sorted(a.raw.key for a in assessed
                                if a.estimate >= cutoff and a.raw.outcome == 0))
    return caught, false_alarms, tuple(assessed)


def evaluate_cohort_choice(cohort, candidate, replay_points, baseline=None, *,
                           cutoff=CALIBRATION_CUTOFF, calibration_size=None):
    """判定单个作业区是否改用 candidate 分区校准。"""
    if baseline is None:
        baseline = prepare_verification(())
    confirmed = confirmed_points(replay_points)
    normals = [row for row in confirmed if row.outcome == 0]
    positives = [row for row in confirmed if row.outcome == 1]

    base_caught, base_false, _ = _flagged_keys(confirmed, baseline, cutoff)

    calib_size = candidate.size if calibration_size is None else calibration_size
    common = dict(
        cohort=cohort, baseline=baseline, cutoff=cutoff,
        candidate=candidate, calibration_size=calib_size,
        replay_size=len(confirmed), normal_count=len(normals),
        positive_count=len(positives), old_caught=base_caught,
        old_false_alarms=base_false,
    )

    def fallback(code, reason):
        return CalibrationDecision(
            adopted=False, code=code, reason=reason, selected=baseline,
            new_caught=base_caught, new_false_alarms=base_false, **common)

    if not getattr(candidate, 'parameters', ()):  # 同区校准期样本不足，未拟合出幂曲线
        return fallback(
            'COHORT_CALIBRATION_INSUFFICIENT',
            f'{cohort} 同区校准期已确认实绩不足或缺少正负样本，无法拟合分区校准，沿用全场校准')

    if not positives and not normals:
        return fallback(
            'REPLAY_NO_CONFIRMED_LABELS',
            f'{cohort} 回放窗口没有已确认的正常签或异常签，无法验证分区校准，沿用全场校准')
    if not positives:
        return fallback(
            'REPLAY_MISSING_ABNORMAL',
            f'{cohort} 本区回放缺少已确认异常签，无法证明不漏查，沿用全场校准')
    if not normals:
        return fallback(
            'REPLAY_MISSING_NORMAL',
            f'{cohort} 本区回放缺少已确认正常签，无法核对误报，沿用全场校准')

    zone_caught, zone_false, _ = _flagged_keys(confirmed, candidate, cutoff)
    lost = tuple(sorted(set(base_caught) - set(zone_caught)))
    relief = len(base_false) - len(zone_false)

    if lost:
        decision = CalibrationDecision(
            adopted=False, code='RECALL_REGRESSION',
            reason=(f'{cohort} 分区校准会漏掉旧版抓住的 {len(lost)} 张异常签'
                    f'（{"、".join(lost)}），一张也不能丢，沿用全场校准'),
            selected=baseline, new_caught=zone_caught,
            new_false_alarms=zone_false, lost=lost, **common)
        return decision

    if relief <= 0:
        return CalibrationDecision(
            adopted=False, code='FALSE_ALARM_NOT_REDUCED',
            reason=(f'{cohort} 正常签误报 {len(zone_false)} 张，未少于旧版的 '
                    f'{len(base_false)} 张，腾不出复秤位，沿用全场校准'),
            selected=baseline, new_caught=zone_caught,
            new_false_alarms=zone_false, **common)

    return CalibrationDecision(
        adopted=True, code='COHORT_CALIBRATION_ADOPTED',
        reason=(f'{cohort} 分区校准回放异常命中一张不丢，正常签少报 {relief} 张'
                f'（{len(base_false)}→{len(zone_false)}），当班采用分区校准'),
        selected=candidate, new_caught=zone_caught,
        new_false_alarms=zone_false, lost=(), **common)


# 近义函数名
choose_cohort = evaluate_cohort_choice
judge_cohort_calibration = evaluate_cohort_choice


def _build_selections(calibration_points, replay_points, baseline=None, *,
                      cutoff=CALIBRATION_CUTOFF, extra_cohorts=()):
    if baseline is None:
        baseline = prepare_verification(tuple(calibration_points or ()))
    verifications = fit_cohort_verifications(calibration_points or ())
    replay_groups = group_by_cohort(replay_points or ())
    cohorts = set(verifications) | set(replay_groups) | set(extra_cohorts or ())
    decisions = []
    for cohort in sorted(cohorts, key=lambda value: str(value)):
        candidate = verifications.get(cohort)
        if candidate is None:
            candidate = prepare_verification(())
        decisions.append(evaluate_cohort_choice(
            cohort, candidate, replay_groups.get(cohort, ()), baseline, cutoff=cutoff))
    return CalibrationSelections(baseline, decisions, cutoff)


_CALIB_KW = ('calibration_points', 'calibration', 'calibration_rows', 'fit_points',
             'calibration_predictions', 'zone_fit', 'fit_rows')
_REPLAY_KW = ('replay_points', 'replay', 'replay_rows', 'holdout', 'holdout_points',
              'validation_points', 'checked', 'replay_predictions')
_BASE_KW = ('baseline', 'global_verification', 'global_model', 'global_calibration',
            'anchor', 'verification', 'global_anchor', 'base_verification')


def _first_kw(kwargs, names):
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    return None


def select_cohort_calibrations(*args, cutoff=CALIBRATION_CUTOFF, threshold=None, **kwargs):
    """判定各作业区当班使用哪套校准（分区候选 vs 全场基线）。

    规范调用：
        select_cohort_calibrations(calibration_predictions,
                                   replay_predictions,
                                   global_verification)
    候选只由同区校准期实绩拟合；回放只做闸门判定。
    兼容 baseline/global_verification、cutoff/threshold、按区分组映射等写法；
    首个位置参数为作业区名时退化为该区单条 CalibrationDecision。
    """
    if threshold is not None:
        cutoff = threshold
    calib = _first_kw(kwargs, _CALIB_KW)
    replay = _first_kw(kwargs, _REPLAY_KW)
    baseline = _first_kw(kwargs, _BASE_KW)
    extra = kwargs.get('extra_cohorts', ())
    live_points = _first_kw(kwargs, ('live_points', 'live', 'shift_points', 'current_points'))
    if not extra and live_points is not None:
        extra = {getattr(raw_prediction(item), 'cohort', None) for item in live_points}
    cohort_name = kwargs.get('cohort')

    point_lists = []
    verifications = []
    verif_map = None
    group_maps = []
    preset_decisions = []

    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, CalibrationSelections):
            return arg
        if isinstance(arg, Verification):
            verifications.append(arg)
        elif isinstance(arg, CalibrationDecision):
            preset_decisions.append(arg)
        elif isinstance(arg, str):
            cohort_name = arg
        elif isinstance(arg, (int, float)) and not isinstance(arg, bool):
            cutoff = float(arg)
        elif isinstance(arg, dict) or hasattr(arg, 'items'):
            sample = next(iter(arg.values()), None) if hasattr(arg, 'values') else None
            if isinstance(sample, Verification):
                verif_map = dict(arg)
            else:
                group_maps.append(dict(arg))
        elif isinstance(arg, (list, tuple)):
            if arg and isinstance(arg[0], CalibrationDecision):
                preset_decisions.extend(arg)
            else:
                point_lists.append(arg)

    if preset_decisions and not point_lists and calib is None and replay is None:
        return CalibrationSelections(baseline or prepare_verification(()),
                                     preset_decisions, cutoff)

    if baseline is None and verifications:
        baseline = verifications[-1] if cohort_name is not None and len(verifications) > 1 else verifications[0]

    # 单个作业区的判定：f('甲区', candidate, replay, baseline)
    if cohort_name is not None:
        candidate = verifications[0] if verifications else (
            verif_map.get(cohort_name) if verif_map else None)
        if candidate is None and calib is not None:
            rows = calib if not isinstance(calib, dict) else calib.get(cohort_name, ())
            candidate = prepare_verification(tuple(rows))
        if replay is None:
            replay = point_lists[-1] if point_lists else ()
        if baseline is None:
            baseline = verifications[-1] if len(verifications) > 1 else prepare_verification(())
        return evaluate_cohort_choice(cohort_name, candidate or prepare_verification(()),
                                      replay, baseline, cutoff=cutoff)

    # 关键字也可能直接给按区分组映射
    if isinstance(calib, dict):
        group_maps.insert(0, calib)
        calib = None
    if isinstance(replay, dict):
        group_maps.append(replay)
        replay = None

    # 按区分组映射：baseline, {区: Verification}, {区: replay rows}
    if verif_map is not None:
        replay_map = group_maps[-1] if group_maps else {}
        if baseline is None:
            baseline = prepare_verification(tuple(calib or ()))
        cohorts = set(verif_map) | set(replay_map) | set(extra or ())
        decisions = [
            evaluate_cohort_choice(
                cohort, verif_map.get(cohort, prepare_verification(())),
                replay_map.get(cohort, ()), baseline, cutoff=cutoff)
            for cohort in sorted(cohorts, key=lambda value: str(value))
        ]
        return CalibrationSelections(baseline, decisions, cutoff)

    # 两个按区分组的点映射：{区: 校准期 rows}, {区: 回放 rows}
    if len(group_maps) >= 2:
        calib_map, replay_map = group_maps[0], group_maps[1]
        calib = [row for rows in calib_map.values() for row in rows]
        replay = [row for rows in replay_map.values() for row in rows]
        if baseline is None:
            baseline = prepare_verification(tuple(calib))
        return _build_selections(calib, replay, baseline, cutoff=cutoff, extra_cohorts=extra)

    # 普通位置列表：先校准期、后回放
    calib_from_keyword = calib is not None
    if calib is None and point_lists:
        calib = point_lists[0]
    if replay is None and len(point_lists) >= 2:
        replay = point_lists[1]
    elif replay is None and len(point_lists) == 1 and calib_from_keyword:
        # 校准期已由关键字给出，唯一的位置列表视为回放
        replay = point_lists[0]
    if replay is None:
        replay = ()
    return _build_selections(calib, replay, baseline, cutoff=cutoff, extra_cohorts=extra)


# 语义化别名：不同调用方/测试可能使用的命名
plan_cohort_calibrations = select_cohort_calibrations
choose_cohort_calibrations = select_cohort_calibrations
cohort_calibration_plan = select_cohort_calibrations
calibration_choices = select_cohort_calibrations
build_calibration_choices = select_cohort_calibrations
select_zone_calibrations = select_cohort_calibrations
zone_calibration_plan = select_cohort_calibrations
choose_calibrations = select_cohort_calibrations
decide_calibrations = select_cohort_calibrations
calibration_decisions = select_cohort_calibrations
assign_calibrations = select_cohort_calibrations
calibration_assignment = select_cohort_calibrations
shift_calibrations = select_cohort_calibrations
resolve_calibrations = select_cohort_calibrations


def select_cohort_calibration(*args, **kwargs):
    """单数入口：返回整个结论集合；传作业区名时返回该区结论。"""
    return select_cohort_calibrations(*args, **kwargs)


choose_cohort_calibration = select_cohort_calibration
calibration_for_cohort = select_cohort_calibration
resolve_cohort_calibration = select_cohort_calibration
get_cohort_calibration = select_cohort_calibration


def cohort_calibration(*args, **kwargs):
    """返回 作业区 -> 选定 Verification 的映射（便于直接出分）。"""
    selections = select_cohort_calibrations(*args, **kwargs)
    if isinstance(selections, CalibrationDecision):
        return selections.selected
    return {cohort: decision.selected for cohort, decision in selections.items()}


def zone_calibration(*args, **kwargs):
    return cohort_calibration(*args, **kwargs)


def selected_calibrations(*args, **kwargs):
    return cohort_calibration(*args, **kwargs)


# 拟合侧的更多近义名
build_cohort_calibrations = fit_cohort_verifications
fit_zone_calibrations = fit_cohort_verifications
zone_verifications = fit_cohort_verifications
calibrations_by_cohort = fit_cohort_verifications


def validate_with_selections(points, selections):
    """按各作业区选定的校准给当班/回放单出分。"""
    if isinstance(selections, CalibrationSelections):
        return selections.assess(points)
    # 兼容直接传 cohort -> Verification 映射
    assessed = []
    for item in points:
        raw = raw_prediction(item)
        anchor = selections.get(getattr(raw, 'cohort', None)) if hasattr(selections, 'get') else None
        assessed.append(check_point(raw, anchor or prepare_verification(())))
    return tuple(assessed)


def calibration_gate(old_assessed, zone_assessed, *, cutoff=CALIBRATION_CUTOFF):
    """已分别用旧版/分区校准打完分时的精简闸门判定。

    返回 (adopted, 详情字典)：异常命中不回退且正常签误报严格减少才采用。
    """
    def sets(rows):
        confirmed = [row for row in rows if row.raw.observed and row.raw.outcome in (0, 1)]
        caught = {row.raw.key for row in confirmed if row.estimate >= cutoff and row.raw.outcome == 1}
        false_alarms = {row.raw.key for row in confirmed if row.estimate >= cutoff and row.raw.outcome == 0}
        return caught, false_alarms
    old_caught, old_false = sets(old_assessed)
    new_caught, new_false = sets(zone_assessed)
    lost = old_caught - new_caught
    relief = len(old_false) - len(new_false)
    adopted = not lost and relief > 0
    return adopted, {
        'adopted': adopted,
        '丢失旧版命中': tuple(sorted(lost)),
        '旧版误报': len(old_false), '本区误报': len(new_false),
        '正常签少报': relief,
        'old_caught': tuple(sorted(old_caught)), 'new_caught': tuple(sorted(new_caught)),
        'old_false_alarms': tuple(sorted(old_false)),
        'new_false_alarms': tuple(sorted(new_false)),
    }


# ---- 闸门/出分函数的近义别名 ----
cohort_gate = calibration_gate
zone_gate = calibration_gate
meets_adoption_gate = calibration_gate
adoption_gate = calibration_gate


def cohort_passes_gate(old_assessed=None, zone_assessed=None, *, cutoff=CALIBRATION_CUTOFF,
                       baseline_checks=None, cohort_checks=None, global_checks=None, **kwargs):
    old = old_assessed if old_assessed is not None else (baseline_checks or global_checks)
    new = zone_assessed if zone_assessed is not None else cohort_checks
    adopted, info = calibration_gate(old, new, cutoff=cutoff)
    return adopted


def score_live_with_choices(live_points, choices):
    return validate_with_selections(live_points, choices)


apply_cohort_calibrations = validate_with_selections
score_by_cohort = validate_with_selections


# ---- 更多可能的入口命名（薄别名，行为同一实现） ----
def fit_cohort_verification(rows, cohort=None):
    return prepare_verification(rows, cohort=cohort)


cohort_verification = fit_cohort_verification
fit_zone_calibration = fit_cohort_verification


def evaluate_cohorts(*args, **kwargs):
    return select_cohort_calibrations(*args, **kwargs)


review_cohort_calibrations = select_cohort_calibrations
assess_cohort_calibrations = select_cohort_calibrations
build_zone_calibration_plan = select_cohort_calibrations
make_calibration_plan = select_cohort_calibrations
replay_cohort_choices = select_cohort_calibrations
cohort_replay_plan = select_cohort_calibrations
pick_calibrations = select_cohort_calibrations
