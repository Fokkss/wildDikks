# Кратко о beta 0.9

Мы остановили teacher/CSV-подгонку и делаем воспроизводимую production-модель: train → XGBoost ensemble → deterministic physics correction → prediction.

Лучший production-профиль перед beta 0.9: blend `legacy1700 + depth4`, вес legacy около `0.75`, rules около `0.45`, bias около `0.70–0.75`, плюс физический профиль `pressure/temp/ws`.

Физическая интерпретация correction-layer:
- давление ниже нормы и температура выше 5°C меняют плотность воздуха и режим пограничного слоя;
- облачность и направления ветра отражают wake/terrain-like систематические ошибки;
- `ws5_7` и `ws10_11` — зоны чувствительной ramp/power-curve;
- bias компенсирует обнаруженную систематическую недооценку production-модели на Q1-like горизонте.

Beta 0.9 делает узкий line-search, чтобы не тратить много сабмитов.
