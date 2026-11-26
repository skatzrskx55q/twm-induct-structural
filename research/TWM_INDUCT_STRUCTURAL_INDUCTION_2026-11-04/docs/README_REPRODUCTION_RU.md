# TWM INDUCT — воспроизведение структурного pilot

**Статус:** исследовательский pilot, официальная E-версия не создавалась. Открытые N2 dev_L2/dev_L3, закрытых test-меток в пакете **нет**. В архиве вся документация и всё нужное для повторения.

## Среда проверки

Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, pytest 9.0.2. Проверялось на CPU с `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` — это также предотвращает замеченные ранее зависания при многопоточном линейном решателе.

## Запуск из корня распакованного пакета

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests/test_structural.py
python code/structural_experiment.py --steps 850 --seeds 17,29,43 --modes conditioned,static
python code/structural_rollout.py --seeds 17,29,43 --modes conditioned,static,ablate,E36,random
python code/analyze_structural.py
python code/rollout_dev.py --model CART --seeds 17
python code/verify_package.py
```

**Осторожно:** повторное обучение записывает checkpoint по тем же именам. При необходимости сохраните копию исходных checkpoint перед запуском. В `results/RETRAIN_REPRODUCIBILITY.json` сохранён аудит повтора seed 17.

Файлы `reference/N2_*` сокращены **только** до train/dev_L2/dev_L3: 1 600 эпизодов. Исходные SHA-256 полных файлов и факт фильтрации описаны в `protocol/ARCHIVE_DATA_MINIMIZATION.json`. `reference/OPEN_LAW_SPLIT_AUDIT_ONLY.json` предназначен исключительно для аудита сплитов, не для инференса.

Программа `code/structural_induction.py` является чистым нейронным runtime-ядром и импортирует только PyTorch. Универсальный вычислитель средового закона `lawbench_v0.py` **никогда не вызывается** из нейронного runtime; исходники E34 нужны лишь для общих признаков и алгоритмического контроля. Корректность именно инференса проверяется тестами.

**Пути:** `checkpoints/E36_seed*_N2_train.pt` — замороженные рёберные сети; `checkpoints/Structural_...` — новые веса; `results/structural_*_H1.json` — одношаговые оценки; `results/rollout_*_traces.jsonl` — собственные траектории; `results/STATISTICAL_AND_MECHANISM_AUDIT.json` — главные результаты.

Реализованные методики не подтверждают обобщение на никогда не открытые test-подмножества; их метки намеренно отсутствуют в архиве.
