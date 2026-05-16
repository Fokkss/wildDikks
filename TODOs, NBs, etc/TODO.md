
## General

## Data + Phys

1. Проанализировать уже прикрученные фичи (колонок +~200)
2. Посмотреть как лучше отсекать выработку: по 90МВ, или по доступной мощности с ремонтом (это прям неочев) (строчки где это надо поменять найдете с помощью гпт)
4. Снизу же у нас есть подрезка по мощности? (вроде cut-in)
5. битыми данными посмотреть


## Prog:
1. Подумать как добавить потом исторический контекст для 18.05
2. Надо на текущей версии поиграться с ансамблями и попробовать RMSE метрику
3. Там конечно можно порефакторить но это гиблое дело. Жптшка нормально справляется с добавлением. Там блоки разделены логически довольно чисто

## Prettifying
1. Вероятно перед выгрузкой гита надо посидеть будет и сделать чуть более человекочитаемо base.py (в остальных особо нечего рефакторить)
   2. Оставить или переписать комментарии
   3. Вынести общий код в кучу вспомогательных функций в base.py
   4. Все константы в config добавить
2. Readme
3. Файлик по хакатону


## GPT feature suggestions bank

Идеи из старого проекта. Пока НЕ внедрены в основной pipeline.

### Уже безопасно можно вернуть потом

- hour_sin / hour_cos
- month_sin / month_cos
- dayofyear_sin / dayofyear_cos
- available_turbines
- availability_ratio
- available_capacity_mw
- ws80_sq
- ws80_cube
- power_curve_proxy
- expected_power_proxy
- air_density
- wind_power_density

### Осторожно, но потенциально полезно

- wind_shear_80_10
- wind_shear_ratio_80_10
- alpha_80_10
- wind_shear_120_80
- wind_shear_ratio_120_80
- wind_shear_180_80
- wind_shear_ratio_180_80

### Опасно до проверки порядка строк

- hub_ws_lag1
- hub_ws_lag2
- hub_ws_lag24
- hub_ws_diff1
- hub_ws_roll_mean_3

Причина: если train отсортирован по времени, а valid_features.csv идёт в обратном порядке,
lag1 начинает означать не прошлый час, а следующий час.

### FOLLOW RIGHT PROJECT STRUCTURE

- RUN_INSTRUCTION.md should be duplicated to main README