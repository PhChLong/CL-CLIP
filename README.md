# CL-CLIP

Project nay dung de thu nghiem continual learning tren CLIP voi LoRA.

Muc tieu: train CLIP qua nhieu task lien tiep, sau moi task danh gia lai tren toan bo task sequence, roi tinh cac metric continual learning nhu AVG, Last, BWT, Transfer, ZSD va Forget.

## Dataset

Task sequence hien tai nam trong `src/data/get_data.py`:

1. `flowers102`
2. `cars`
3. `eurosat`
4. `cifar100`
5. `dtd`

Dataset duoc download tu HuggingFace lan dau, sau do cache vao:

```text
src/data/dataset_cache/
```

Neu chay voi `--test-pipeline`, moi dataset chi lay subset rat nho de test pipeline nhanh.

## Cai dat

Tao virtual environment va cai dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement-cpu.txt
```

Neu dung GPU:

```powershell
pip install -r requirement-gpu.txt
```

## Methods

### FineTune

FineTune train LoRA tren task hien tai bang cross entropy:

```text
loss = CE(logits, labels)
```

### LwF_LoRA

LwF_LoRA luu LoRA cua task truoc, tinh logits cua model cu va model hien tai, roi dung KL divergence de giu hanh vi cu:

```text
loss = CE(logits, labels) + lambda_old * KD(old_logits, new_logits)
```

## Metrics

`results_matrix[i][j]` la accuracy tren task `j` sau khi train xong task `i`.

- `AVG`: accuracy trung binh tren cac task da thay sau moi training step.
- `Last`: accuracy trung binh tren tat ca task sau training step cuoi.
- `BWT`: backward transfer, do accuracy cua task cu thay doi sau khi hoc task moi.
- `Transfer`: zero-shot transfer tren cac task chua train, dung upper-right triangle cua matrix.
- `ZSD`: zero-shot degradation, do muc giam zero-shot score cua task tuong lai truoc khi task do duoc train.
- `Forget`: forgetting, do muc giam tu best accuracy cua moi task cu den final accuracy.

## Output

Sau khi train, project luu ket qua vao:

```text
results/<method>/
```

Moi run tao:

- `log_<run_id>.txt`: log train/valid loss.
- `results_<run_id>.json`: metric, results matrix va training history.

## Ghi chu

- Lan chay dau co the cham vi phai download CLIP va dataset.
- CLIP model duoc cache vao `src/models/model_cache/`.
- Dataset duoc cache vao `src/data/dataset_cache/`.
- Dung `--test-pipeline` truoc de kiem tra code chay dung, roi moi chay full dataset.
