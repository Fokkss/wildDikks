# Кратко для README / защиты

Мы начали с одного XGBoost-регрессора, который оказался сильнее сложных ансамблей и нейросетевых попыток. После аудита выяснилось, что ключевая ошибка не в архитектуре, а в форме power curve и режимных смещениях: ramp-zone, направление ветра, shear, облачность и давление/температура.

Дальше мы разделили работу на две линии:

1. **Диагностическая leaderboard-calibration.** Через малые probe-сдвиги находили знак ошибки в сегментах: `dir60_75`, `dir90_135`, `cloud_hi`, `pressure_low`, `ws5_7`, `shear180q4 × pred50_70`. Это дало validation artifact около 8.01, но он считается подогнанным к текущему valid.

2. **Production model.** Для воспроизводимого решения отказались от teacher CSV и вернулись к train-only модели. Beta 0.5 использует сильные старые параметры XGBoost, ансамбль shallow/depth4 моделей, физические фичи 84 м, direction degrees/1000, density, shear, pressure/temp, cloud, availability и консервативную rule-correction map.

Beta 0.5 также проверяет гипотезу старого хорошего запуска: `n_estimators=10000` + `early_stopping_rounds=300`, затем refit на всём train с найденным числом деревьев.

Для 2027 основной режим: `features_2027 → saved ensemble → conservative rules → prediction`, без использования validation submission CSV.
