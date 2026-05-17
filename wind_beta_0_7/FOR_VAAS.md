# Краткое резюме подхода

Мы начали с одного XGBoost-регрессора, который оказался сильнее более сложных ансамблей. Затем через серию проверок на validation leaderboard выявили систематические ошибки модели: недооценка части ramp-zone, ошибки по направлениям ветра, high-shear режимам, давлению, температуре и облачности.

Главная физическая идея: выработка ВЭС не определяется только скоростью ветра. Важны режимы пограничного слоя, вертикальный shear, direction/wake-like effects, плотность воздуха и погодные условия. Поэтому в production-модель добавлены:

- hub-height 84 m wind proxy;
- power-curve признаки;
- wind shear / ratios / direction sectors;
- vector-like direction interactions;
- pressure / temperature / air-density;
- cloud / precipitation flags;
- time-aware weather lags;
- conservative deterministic correction-map по найденным физическим режимам.

Validation-calibration дала лучший диагностический файл около 8.01, но для production мы не используем teacher CSV. В beta 0.7 модель полностью воспроизводима из train: обучается несколько XGBoost-моделей старого сильного семейства, затем берётся небольшой ансамбль и физически интерпретируемая корректировка.
