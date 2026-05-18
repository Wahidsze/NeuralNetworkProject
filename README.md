# Детекция и распознавание показаний водяных и электрических счётчиков с помощью **YOLOv5s**

Модель находит каждую цифру на снимке счётчика отдельным боксом и классифицирует её среди 12 классов: цифры `0–9`, десятичная точка и доли единиц измерения `liters` (значения, которые не учитываются при подаче показаний).

---

## Результаты

| Метрика           | Целевое значение | Достигнуто |
| ----------------- | ---------------- | ---------- |
| **Precision**     | ≥ 0.90           | **0.952**  |
| **Recall**        | -          | **0.915**  |
| **mAP @ 0.5**     | -          | **0.948**  |
| **mAP @ 0.5:0.95**| -          | **0.655**  |
| Размер весов      | Малый            | ~14 МБ     |
| Inference         | < 100 мс          | ~8 мс      |

---

## Структура репозитория

```
NeuralNetworkProject/
├── yolov5_meter_v2.ipynb          # основной ноутбук: подготовка данных + обучение
├── training/                      # артефакты обучения (200 эпох)
│   ├── results.csv                # лог метрик по эпохам
│   ├── results.png                # графики train/val losses + P/R/mAP
│   ├── confusion_matrix.png
│   ├── PR_curve.png · F1_curve.png · P_curve.png · R_curve.png
│   ├── labels.jpg · labels_correlogram.jpg
│   ├── train_batch{0,1,2}.jpg     # примеры тренировочных батчей
│   ├── val_batch{0,1,2}_labels.jpg / _pred.jpg
│   ├── hyp.yaml                   # гиперпараметры
│   └── opt.yaml                   # параметры запуска train.py
├── validation/                    # независимый прогон val.py
│   ├── confusion_matrix.png
│   ├── PR_curve.png · F1_curve.png · P_curve.png · R_curve.png
│   └── val_batch{0,1,2}_labels.jpg / _pred.jpg
└── test_detect/                   # запуск на собственных тестовых фото
    ├── test_photos/               # исходные снимки
    └── detect/meter_test_v2/      # результаты с боксами и подписями
        └── labels/                # txt-файлы с координатами в YOLO-формате
```

---

## Данные

**Датасет** `combined_dataset_v2` — сборный, объединён из различных наборов цифр со счётчиков и дополнен собственными снимками приборов учета

| Параметр       | Значение                       |
| -------------- | ------------------------------ |
| Изображений    | ~9 000                         |
| Боксов         | ~80 000                        |
| Классов        | 12 (`0–9`, `.`, `liters`)      |
| Split          | 80 / 15 / 5 (train / test / val) |
| Формат         | YOLO (txt-файлы `class x y w h confidence`) |

Распределение классов несбалансированное: класс `0` доминирует (≈ 19k боксов), `liters` — самый редкий (≈ 2.5k). Компенсируется оверсэмплингом.

---

## Архитектура и обучение

- **Модель:** YOLOv5s (Ultralytics), ≈ 7.2M параметров
- **Init:** `yolov5s.pt` (предобученные веса COCO)
- **Input size:** 416 × 416
- **Optimizer:** SGD
- **Эпохи:** 200, batch 64, patience 30
- **Аугментации:** mosaic 1.0, mixup 0.1, copy-paste 0.1, rotate ±5°, scale 0.7
- **Flip отключён** намеренно: горизонтальное зеркалирование меняет значения цифр (6 = 9)
- **Среда:** Google Colab

Полная конфигурация — в [`training/opt.yaml`](training/opt.yaml) и [`training/hyp.yaml`](training/hyp.yaml).

---

## Воспроизведение

### 1. Клонировать YOLOv5 и установить зависимости

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 2. Подготовить датасет

`data.yaml` (12 классов):

```yaml
train: path/to/combined_dataset_v2/images/train
val:   path/to/combined_dataset_v2/images/val

nc: 12
names: ['0','1','2','3','4','5','6','7','8','9','.','liters']
```

### 3. Запустить обучение

```bash
python train.py \
  --weights yolov5s.pt \
  --data combined_dataset_v2/data.yaml \
  --img 416 \
  --batch 64 \
  --epochs 200 \
  --optimizer SGD \
  --patience 30 \
  --cache ram \
  --name meter_combined_v2
```

### 4. Валидация

```bash
python val.py \
  --weights runs/train/meter_combined_v2/weights/best.pt \
  --data combined_dataset_v2/data.yaml \
  --img 416
```

### 5. Инференс на своих фото

```bash
python detect.py \
  --weights runs/train/meter_combined_v2/weights/best.pt \
  --source path/to/photos/ \
  --img 416 \
  --conf 0.40 \
  --iou 0.45 \
  --save-txt --save-conf \
  --name meter_test_v2
```

Результаты — папки `runs/detect/meter_test_v2/`.

---

## Стек

- PyTorch
- Ultralytics YOLOv5
- Google Colab
- OpenCV, NumPy, Matplotlib
