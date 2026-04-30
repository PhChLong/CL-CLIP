METRICS = {}

def metric(fn):
    METRICS[fn.__name__] = fn
    return fn

@metric
def AVG(results):
    # Average accuracy over seen tasks after each training step.
    # At step i, only tasks 0..i are treated as learned/seen.
    avg = 0.0
    for i, row in enumerate(results):
        avg += sum(row[:i+1]) / (i+1)
    return avg / len(results)

@metric
def Last(results):
    # Final average accuracy over all tasks after the last training step.
    last_row = results[-1]
    seen = len(results)
    return sum(last_row[:seen]) / seen

@metric
def BWT(results):
    # Backward transfer: final accuracy change on old tasks after later training.
    # Positive means later tasks improved old tasks; negative means forgetting.
    T = len(results)
    if T <= 1:
        return 0.0
    last_row = results[-1]
    bwt = 0.0
    for i in range(T - 1):
        bwt += last_row[i] - results[i][i]
    return bwt / (T - 1)

@metric
def Transfer(results):
    # Zero-shot transfer before each task is trained.
    # Uses the upper-right triangle: results[i][j] where j > i.
    T = len(results)
    if T <= 1:
        return 0.0
    
    # Upper-right triangle: results[i][j] với j > i
    # results[i][j] = accuracy task j sau khi train xong task i
    # Với j > i: task j chưa được fine-tune → đây là zero-shot transfer
    
    col_avgs = []
    for j in range(1, T):          # mỗi task j (trừ task 0)
        col_vals = []
        for i in range(j):         # tất cả timestamps i < j
            col_vals.append(results[i][j])
        col_avgs.append(sum(col_vals) / len(col_vals))
    
    return sum(col_avgs) / len(col_avgs)

def compute_all_metrics(results):
    metric_scores = {
        metric_name: metric_fn(results)
        for metric_name, metric_fn in METRICS.items()
    }
    return metric_scores
