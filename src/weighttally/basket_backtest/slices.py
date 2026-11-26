from collections import defaultdict
from statistics import median


def cohort_groups(points):
    """按作业区（cohort）归集回放点，返回 {cohort: [point, ...]}。"""
    grouped = defaultdict(list)
    for measure in points:
        raw = getattr(measure, 'raw', measure)
        grouped[getattr(raw, 'cohort', None)].append(measure)
    return {cohort: grouped[cohort] for cohort in sorted(grouped, key=lambda v: str(v))}


# 近义名
group_by_cohort = cohort_groups
cohort_slices_grouped = cohort_groups


def cohort_slices(points, threshold=.6):
    """各作业区回放的命中/误报/漏查计数，键顺序稳定。"""
    rows = []
    for cohort, measures in cohort_groups(points).items():
        confirmed = [m for m in measures
                     if getattr(m.raw, 'observed', True) and m.raw.outcome in (0, 1)]
        positives = [m for m in confirmed if m.raw.outcome == 1]
        normals = [m for m in confirmed if m.raw.outcome == 0]
        caught = [m for m in positives if m.estimate >= threshold]
        false_alarms = [m for m in normals if m.estimate >= threshold]
        rows.append({
            '作业区': cohort,
            '回放实绩': len(confirmed),
            '正常签': len(normals),
            '异常签': len(positives),
            '抓住异常': len(caught),
            '漏查': len(positives) - len(caught),
            '正常签误报': len(false_alarms),
            '命中单': tuple(sorted(m.raw.key for m in caught)),
            '误报单': tuple(sorted(m.raw.key for m in false_alarms)),
        })
    return rows


replay_by_cohort = cohort_slices
cohort_replay = cohort_slices


def groups_for_basket_backtest(raw):
    ticket_g, actual_g, tare_g, temp_c, bins, return_g = raw
    if ticket_g<420:
        yield '轻称签'
    if tare_g>=85:
        yield '高扣皮档'
    if return_g>=60:
        yield '退秤非零'
    if temp_c>=30:
        yield '高温巡档'
    if bins>=14:
        yield '多格分拣台'

def replay_slices(points, threshold, *, risk_model=False):
    confirmed = defaultdict(list)
    awaiting = defaultdict(int)
    for measure in points:
        truth = measure.raw.outcome
        bands = groups_for_basket_backtest(measure.raw.input_values)
        for label in bands:
            if not measure.raw.observed or truth is None:
                awaiting[label] += 1
            else:
                confirmed[label].append((measure.estimate, truth))
    statements = []
    for domain in ('轻称签', '高扣皮档', '退秤非零', '高温巡档', '多格分拣台'):
        readings = confirmed[domain]
        if not readings:
            statements.append({'难点':domain,'实绩数':0,'等待见证':awaiting[domain],'状态':'暂无核实称差或复点结果'})
            continue
        top = sorted(readings,key=lambda pair:(-pair[0],-pair[1]))
        n = max(1,len(top)//5)
        high = sum(outcome>=threshold for _,outcome in top[:n])
        scores, facts = zip(*top,strict=True)
        centre = median(scores)
        baseline = median(facts)
        spread = median(abs(value-centre) for value in scores)
        error = median(abs(a-b) for a,b in readings)
        statements.append({'难点':domain,'实绩数':len(readings),'前段可复核数':n,'前段实绩超线':high,
                           '估计中位偏离':round(centre-baseline,4),'估计绝对离差':round(spread,4),
                           '中位绝对误差':round(error,4),'等待见证':awaiting[domain]})
    return statements
