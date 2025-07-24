import time

PERFORMANCE_LOG = {}
LABEL_SEQUENCE = []

USE_LOG = True

def log_performance(label):
    if not USE_LOG:
        return

    if label not in PERFORMANCE_LOG:
        PERFORMANCE_LOG[label] = []
        LABEL_SEQUENCE.append(label)
    PERFORMANCE_LOG[label].append(time.time())
    
def print_performance():
    if not USE_LOG:
        return
    
    for i in range(len(LABEL_SEQUENCE) - 1):
        label_before = LABEL_SEQUENCE[i]
        label_after = LABEL_SEQUENCE[i + 1]
        times_before = PERFORMANCE_LOG[label_before]
        times_after = PERFORMANCE_LOG[label_after]
        if len(times_before) == 0 or len(times_after) == 0:
            print(f"{label_before} -> {label_after}: 기록 없음")
            continue
        # 두 구간의 짝수 개수만큼 평균 계산
        pair_count = min(len(times_before), len(times_after))
        durations = [times_after[j] - times_before[j] for j in range(pair_count)]
        avg_duration = sum(durations) / len(durations) if durations else 0
        print(f"{avg_duration * 1000:.5f}ms (sample: {pair_count})\t ({label_before} -> {label_after})")
        
def clear_performance():
    PERFORMANCE_LOG.clear()
    LABEL_SEQUENCE.clear()