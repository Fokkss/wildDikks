# config.py
FARM_CAPACITY_MW = 90.09
N_TURBINES = 26
TURBINE_CAPACITY_MW = FARM_CAPACITY_MW / N_TURBINES
HUB_HEIGHT_M = 80
R_DRY_AIR = 287.05  # J/(kg*K)

# Сюда физик может дописывать свои идеи
FEATURE_LIST = [
    'wind_speed_80m',
    'wind_speed_cube',  # Вот так легко будет добавить новую фичу
    'air_density',
    'hour_sin',
    'hour_cos'
]
