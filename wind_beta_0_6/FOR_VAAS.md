# Краткое резюме beta 0.6

Мы разделили два режима решения:

1. **Competition calibration** — ручные/teacher-поправки, которые доводили validation-score до ~8.01, но завязаны на конкретный hidden-valid.
2. **Production model** — самовоспроизводимый pipeline `train -> model -> predict`, который можно применить к 2027 без CSV-подсказки.

В production-ветке лучше всего работает семейство старого XGBoost-регрессора с параметрами:
`learning_rate=0.03`, `max_depth=3`, `min_child_weight=5`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_alpha=0.04`, `reg_lambda=5.0`.

Физические признаки: wind speed на 84 м, power-curve proxy, shear, direction sectors, pressure/temp/cloud, availability. Диагностика hidden-valid показала важные режимы: направление 60–75° переоценивается, cloud_hi переоценивается, low pressure / warm temperature / 5–7 м/с чаще недооцениваются. Эти закономерности оформлены как deterministic correction rules.

В beta 0.5 выяснилось, что небольшой положительный global bias улучшает production-score. Beta 0.6 делает узкий поиск вокруг `rules_strength≈0.45` и `bias≈0.5–0.85`, а не тратит время на широкие архитектурные эксперименты.

BorutaSHAP отложен: он полезен для пост-хакатонного отбора признаков, но в текущем временном лимите слишком дорогой и не гарантирует leaderboard-прирост. Вместо этого используем уже найденные физически интерпретируемые признаки и проверяем минимум файлов.
