
#:NOTE: KEYWORDS (for what reason?)
#:ASSUMTION: Maybe all this stuff should be annihalated.
#            moreover there are many funcs in features_enginering.py
#            connected with searching same names - it's useless

KIND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wind_speed": (
        "wind_speed",
        "windspeed",
        "wind speed",
        "ws",
        "speed",
        "скорость",
        "ветер",
    ),
    "wind_dir": (
        "wind_dir",
        "wind direction",
        "direction",
        "dir",
        "wd",
        "направ",
        "азимут",
        "румб",
    ),
    "temp": (
        "temperature",
        "temp",
        "t2m",
        "температура",
        "темп",
    ),
    "pressure": (
        "pressure",
        "press",
        "msl",
        "sp",
        "давление",
    ),
    "precip": (
        "precip",
        "rain",
        "snow",
        "осад",
        "дожд",
        "снег",
    ),
    "cloud": (
        "cloud",
        "облач",
    ),
    "repair": (
        "repair",
        "ремонт",
        "turbines_off",
        "unavailable",
        "offline",
        "outage",
        "maintenance",
        "неработ",
        "отключ",
    ),
}

DATE_KEYWORDS = (
    "datetime",
    "timestamp",
    "date",
    "time",
    "dt",
    "дата",
    "время",
)

TARGET_KEYWORDS = (
    "результирующий",
    "выработка",
    "generation",
    "power",
    "target",
    "fact",
    "actual",
)