# config.py
FARM_CAPACITY_MW = 90.09
N_TURBINES = 26
TURBINE_CAPACITY_MW = FARM_CAPACITY_MW / N_TURBINES
HUB_HEIGHT_M = 80
R_DRY_AIR = 287.05  # J/(kg*K)


FEATURE_LIST = [
    'wind_speed_80m',
    'wind_speed_cube',
    'air_density',
    'hour_sin',
    'hour_cos'
]

COLUMN_MAP = {
    "METEOFORECASTHOUR_OPENM_Datetime": "datetime",
    "month": "month",
    "hour_of_day": "hour",
    "Выработка.Результирующий расчет": "target",
    "wind_speed_10m": "ws_10m",
    "wind_speed_80m": "ws_80m",
    "wind_speed_120m": "ws_120m",
    "wind_speed_180m": "ws_180m",
    "wind_direction_10m": "wd_10m",
    "wind_direction_80m": "wd_80m",
    "wind_direction_120m": "wd_120m",
    "wind_direction_180m": "wd_180m",
    "wind_gusts_10m": "gusts_10m",
    "temperature_80m": "temp_k_80",
    "temperature_120m": "temp_k_120",
    "pressure_msl": "pressure",
    "rain": "precip_rain",
    "showers": "precip_showers",
    "snowfall": "precip_snow",
    "cloud_cover_low": "cloud",
    "Кол-во_ВЭУ_в_ремонте": "repair_count"
}
