"""
District Historical Weather Data Fetcher
=========================================
Fetches 10 years (2014-2024) of daily weather data for every Indian district
from the Open-Meteo Historical API and stores by-district parquet files.

Usage:
    # Fetch all Indian districts (~45-60 min, ~1.8 GB)
    cd agri_crop_recommendation
    python scripts/fetch_district_weather.py

    # Fetch a sample of 5 districts (fast, for testing)
    python scripts/fetch_district_weather.py --sample 5

Output structure:
    data/weather/district/<DISTRICT_ID>/<YEAR>.parquet
    Each parquet has daily rows: date, temp_max, temp_min, rainfall, humidity, wind_speed
"""

import json
import sys
import time
import argparse
import requests
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime


class RateLimitError(Exception):
    """Raised when the API responds with HTTP 429 (daily limit exhausted)."""
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Open-Meteo Historical API ────────────────────────────────────────────────
HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 30          # seconds per request
RETRY_LIMIT     = 3           # retries on network failure
RETRY_DELAY     = 5           # seconds between retries

START_YEAR = 2014
END_YEAR   = 2024

# ── District lat/lon lookup (representative city coords per district) ────────
# Derived from standard Indian district centroid database
# Format: "REGION_ID": (latitude, longitude)
DISTRICT_COORDS = {
    # Andhra Pradesh
    "AP_VISAK":  (17.6868, 83.2185), "AP_VIJAY":  (16.5062, 80.6480),
    "AP_GUNTUR": (16.3067, 80.4365), "AP_KRISH":  (16.5088, 80.6478),
    "AP_KURNL":  (15.8281, 78.0373), "AP_KADAPA": (14.4674, 78.8241),
    "AP_ANANT":  (14.6819, 77.6006), "AP_CHITTR": (13.6288, 79.4192),
    "AP_NELR":   (14.4426, 79.9865), "AP_PRKAS":  (15.5057, 79.5109),
    "AP_SRIKA":  (14.3415, 80.0267), "AP_ELURU":  (16.7107, 81.1029),
    "AP_BAPATL": (15.9070, 80.4670), "AP_KONASE": (15.3647, 79.9842),
    "AP_TIRUP":  (13.6288, 79.4192), "AP_CHITTA": (13.2100, 79.1008),
    "AP_POTTI":  (14.7004, 77.8116), "AP_NANDYL": (15.4786, 78.4836),
    "AP_NRASAP": (16.4320, 81.6706), "AP_ALURI":  (18.0297, 82.8028),
    "AP_ALLA":   (16.2335, 81.1486), "AP_KAKIN":  (16.9891, 82.2475),
    "AP_EAST_G": (17.1200, 82.0070), "AP_WEST_G": (16.9167, 81.6833),
    "AP_VIZIAN": (18.1066, 83.3955),
    # Telangana
    "TG_HYD":   (17.3850, 78.4867), "TG_RANG":   (17.4065, 78.4772),
    "TG_MEDAK":  (18.0531, 78.2605), "TG_NIZMA":  (16.7688, 77.9939),
    "TG_MAHB":   (16.7396, 77.9837), "TG_NARAY":  (16.8500, 77.9000),
    "TG_WARANG": (17.9784, 79.5941), "TG_KHAMMA": (17.2474, 80.1514),
    "TG_NIZAMB": (18.6705, 78.0943), "TG_KARIMNJ":(18.4386, 79.1288),
    "TG_ADILB":  (19.6730, 79.5276), "TG_KOMRAM": (19.7360, 79.6040),
    "TG_MANCHE": (18.8830, 78.5330), "TG_JAGTIAL":(18.7953, 78.9148),
    "TG_RAJANN": (18.4561, 78.4390), "TG_SIRICI": (18.3852, 78.8460),
    "TG_PEDDAP": (18.6130, 79.3750), "TG_JAYASH": (17.9950, 79.5670),
    "TG_BHUPAL": (17.9640, 79.8990), "TG_MULUG":  (18.1920, 80.0480),
    "TG_BHADRA": (17.6100, 80.8860), "TG_MAHABUB":(16.7400, 79.4700),
    "TG_NAGARK": (16.4824, 79.1015), "TG_WANAP":  (17.6900, 78.5600),
    "TG_MEDCHL": (17.6800, 78.5200), "TG_VIKARA": (17.3300, 78.6500),
    "TG_SANGARE":(17.6200, 78.0800), "TG_YADADR": (17.2360, 78.9170),
    "TG_NALGON": (17.0566, 79.2676), "TG_SURYAP": (17.1430, 79.6220),
    "TG_JANGAO": (17.7270, 79.1550), "TG_HANAMK": (18.4110, 77.6880),
    # Maharashtra
    "MH_PUNE":   (18.5204, 73.8567), "MH_NASHIK": (19.9975, 73.7898),
    "MH_NAGPUR": (21.1458, 79.0882), "MH_MUMCITY":(19.0760, 72.8777),
    "MH_THANE":  (19.2183, 72.9781), "MH_AURANG": (19.8762, 75.3433),
    "MH_SOLAP":  (17.6599, 75.9064), "MH_KOLHAP": (16.7050, 74.2433),
    "MH_AMRAV":  (20.9320, 77.7523), "MH_AKOLA":  (20.7002, 77.0082),
    "MH_LATUR":  (18.4088, 76.5604), "MH_SATARA": (17.6805, 74.0183),
    "MH_SANGLI": (16.8524, 74.5815), "MH_JALGAO": (21.0077, 75.5626),
    "MH_AHMED":  (19.0948, 74.7480), "MH_NANDED": (19.1383, 77.3210),
    "MH_BEED":   (18.9890, 75.7601), "MH_WARDHA": (20.7452, 78.6022),
    "MH_YAVATM": (20.3888, 78.1204), "MH_BULDHA": (20.5292, 76.1842),
    "MH_WASHIM": (20.1120, 77.1450), "MH_HINGOL": (19.7160, 77.1490),
    "MH_PARBHA": (19.2706, 76.7745), "MH_OSMANA": (18.1839, 76.0430),
    "MH_RAIGAD": (18.5158, 73.1800), "MH_RATNAG": (16.9902, 73.3120),
    "MH_SINDHU": (16.7080, 73.8280), "MH_GONDIY": (21.4600, 80.2000),
    "MH_GADCHI": (20.1800, 80.0040), "MH_BHANDA": (20.6500, 79.6500),
    "MH_CHANDR": (19.9615, 79.2961), "MH_DHULE":  (20.9042, 74.7749),
    "MH_NANDUR": (20.4914, 74.2433), "MH_MUMBSUB":(19.1880, 72.9780),
    # Karnataka
    "KA_BANG":   (12.9716, 77.5946), "KA_MYSORE": (12.2958, 76.6394),
    "KA_HUB":    (15.3647, 75.1240), "KA_MANGAL": (12.9141, 74.8560),
    "KA_BELGAU": (15.8497, 74.4977), "KA_GULB":   (17.3297, 76.8343),
    "KA_BIJAPU": (16.8302, 75.7100), "KA_DHARW":  (15.4589, 75.0078),
    "KA_SHIV":   (14.4438, 76.9214), "KA_HAVE":   (14.7955, 75.7139),
    "KA_TUMKUR": (13.3409, 77.1010), "KA_DAKSHI": (12.8438, 75.2479),
    "KA_UTTARK": (14.8000, 74.1333), "KA_CHIKMG": (13.9149, 75.5694),
    "KA_KODAGU": (12.4244, 75.7382), "KA_HASSAN": (13.0070, 76.0998),
    "KA_CHITRA": (14.2295, 76.4016), "KA_GADAG":  (15.4166, 75.6272),
    "KA_KOPPAL": (15.3548, 76.1547), "KA_YADGI":  (16.7600, 77.1350),
    "KA_CHAMR":  (11.9261, 77.1427), "KA_MANDYA": (12.5218, 76.8951),
    "KA_CHIKBAL":(13.4355, 77.7268), "KA_KOLAR":  (13.1360, 78.1294),
    "KA_RAMAN":  (12.7157, 77.4926), "KA_BANGRU": (13.3353, 77.4969),
    "KA_RAICHUR":(16.2120, 77.3566), "KA_BIDAR":  (17.9104, 77.5199),
    "KA_KALLAGI":(17.3330, 76.8200),
    # Tamil Nadu
    "TN_CHEN":   (13.0827, 80.2707), "TN_COIM":   (11.0168, 76.9558),
    "TN_MADURAI":(9.9252, 78.1198),  "TN_SALEM":  (11.6643, 78.1460),
    "TN_TIRUP":  (8.7139, 77.7567),  "TN_TIRUCH": (10.7905, 78.7047),
    "TN_VELLOR": (12.9165, 79.1325), "TN_ERODE":  (11.3410, 77.7172),
    "TN_TIRUNL": (8.7242, 77.6972),  "TN_THANJAS":(10.7867, 79.1378),
    "TN_DINDGL": (10.3673, 77.9803), "TN_VIRUD":  (9.5889, 77.9641),
    "TN_KANNIY": (8.0883, 77.5385),  "TN_NILGIR": (11.4916, 76.7337),
    "TN_KRISHN": (12.1292, 79.9067), "TN_NAGAP":  (10.7666, 79.8483),
    "TN_CUDDAL": (11.7447, 79.7563), "TN_VILLUP": (12.9342, 79.4985),
    "TN_KANCHEEP":(12.8342, 79.7047),"TN_RAMANTH":(9.3762, 78.8302),
    "TN_SIVAGAN":(9.8470, 78.4867),  "TN_ARIYALR":(11.1434, 79.0784),
    "TN_TIRUVAR":(10.7740, 79.6339), "TN_PUDUKKOT":(10.3833, 78.8001),
    "TN_TENKASI":(8.9597, 77.3152),  "TN_RANIPET":(12.9270, 79.3319),
    "TN_TIRUPATH":(12.4979, 78.5626),"TN_KALLAK": (11.7354, 78.1425),
    "TN_CHENGAL":(12.6819, 80.0029), "TN_TIRUVAL":(13.1304, 79.9064),
    "TN_THOOTU": (8.8042, 78.1480),
    # Kerala
    "KL_THIRU":  (8.5241, 76.9366),  "KL_KOCHI":  (9.9312, 76.2673),
    "KL_KOZHI":  (11.2588, 75.7804), "KL_THRISSUR":(10.5276, 76.2144),
    "KL_MALAPP": (11.0733, 76.0740), "KL_ALAPPY": (9.4981, 76.3388),
    "KL_PALAK":  (10.7867, 76.6548), "KL_KOLLAM": (8.8932, 76.6141),
    "KL_KANNUR": (11.8745, 75.3704), "KL_KASARG": (12.4996, 74.9869),
    "KL_KOTTAY": (9.5916, 76.5222),  "KL_WAYANAD":(11.6854, 76.1320),
    "KL_IDUKKI": (9.8496, 76.9741),  "KL_PATHANM":(9.2648, 76.7870),
    # Gujarat
    "GJ_AHMED":  (23.0225, 72.5714), "GJ_SURAT":  (21.1702, 72.8311),
    "GJ_VADOD":  (22.3072, 73.1812), "GJ_RAJKOT": (22.3039, 70.8022),
    "GJ_BHAVNA": (21.7645, 72.1519), "GJ_JAMNAG": (22.4707, 70.0577),
    "GJ_JUNAG":  (21.5222, 70.4579), "GJ_GANDHIN":(23.2156, 72.6369),
    "GJ_ANAND":  (22.5645, 72.9289), "GJ_KHEDA":  (22.7500, 72.6800),
    "GJ_MEHSANA":(23.5880, 72.3693), "GJ_PATAN":  (23.8493, 72.1266),
    "GJ_BANAS":  (24.1731, 72.4367), "GJ_SABAN":  (23.0000, 73.0200),
    "GJ_ARAVEL": (21.3330, 73.3000), "GJ_NAVSARI":(20.9467, 72.9520),
    "GJ_VALSAD": (20.6139, 72.9281), "GJ_DANG":   (20.7000, 73.7200),
    "GJ_TAPI":   (21.1167, 73.5167), "GJ_BHARUCH":(21.7051, 72.9959),
    "GJ_NARMAD": (21.8700, 73.5000), "GJ_CHOTA":  (22.6000, 74.0200),
    "GJ_PANCHMH":(22.7000, 73.5200), "GJ_DAHOD":  (22.8400, 74.2500),
    "GJ_MORBI":  (22.8170, 70.8380), "GJ_DEVBHUM":(22.3584, 69.6493),
    "GJ_GIRMHAL":(21.6937, 70.8547), "GJ_AMRELI": (21.6010, 71.2214),
    "GJ_BHAVAL": (22.1500, 71.5300), "GJ_BOTAD":  (22.1680, 71.6650),
    "GJ_CHHOTA": (23.8793, 73.5500), "GJ_MAHISG": (23.0911, 73.0780),
    "GJ_KATCH":  (23.7337, 69.8597), "GJ_SUREN":  (22.7280, 72.0333),
    # Rajasthan
    "RJ_JAIPUR": (26.9124, 75.7873), "RJ_JODHPUR":(26.2389, 73.0243),
    "RJ_KOTA":   (25.2138, 75.8648), "RJ_BIKANER":(28.0229, 73.3119),
    "RJ_AJMER":  (26.4499, 74.6399), "RJ_UDAIPUR":(24.5854, 73.7125),
    "RJ_ALWAR":  (27.5530, 76.6346), "RJ_BHILWAR":(25.3407, 74.6313),
    "RJ_BARMER": (25.7451, 71.3922), "RJ_SIKAR":  (27.6094, 75.1399),
    "RJ_CHURU":  (28.2929, 74.9668), "RJ_JHUNJH": (28.1289, 75.3955),
    "RJ_NAGAUR": (27.2030, 73.7344), "RJ_PALI":   (25.7731, 73.3234),
    "RJ_JALORE": (25.3500, 72.6167), "RJ_SIROHI": (24.8867, 72.8600),
    "RJ_JAISALM":(26.9157, 70.9083), "RJ_BANSWAR":(23.5409, 74.4534),
    "RJ_DUNGAR": (23.8421, 73.7149), "RJ_PRATAP": (24.0337, 74.7749),
    "RJ_RAJASMN":(25.0500, 73.8800), "RJ_BUNDI":  (25.4360, 75.6391),
    "RJ_SAWAI":  (26.9800, 76.7300), "RJ_DHOLPUR":(26.7015, 77.8941),
    "RJ_GANGANR":(29.9139, 73.8769), "RJ_HANUMANM":(29.5920, 74.3270),
    "RJ_SRIGANG":(29.9040, 73.8770), "RJ_BIKANER2":(27.9925, 72.5000),
    "RJ_TONK":   (26.1679, 75.7899), "RJ_SAWAIMD":(26.9800, 76.7300),
    "RJ_KARAUL": (26.5070, 77.0237), "RJ_BHARATP":(27.2152, 77.5030),
    # Uttar Pradesh
    "UP_LUCK":   (26.8467, 80.9462), "UP_KANPUR": (26.4499, 80.3319),
    "UP_AGRA":   (27.1767, 78.0081), "UP_VARANA": (25.3176, 82.9739),
    "UP_MEERUT": (28.9845, 77.7064), "UP_ALLAHAB":(25.4358, 81.8463),
    "UP_BAREILL":(28.3670, 79.4304), "UP_MORAD":  (28.8386, 78.7733),
    "UP_GHAZIAB":(28.6692, 77.4538), "UP_ALIGARH":(27.8974, 78.0880),
    "UP_SAHARAR":(29.9642, 77.5449), "UP_GORAKH": (26.7606, 83.3732),
    "UP_FAIZAB": (26.7719, 82.1427), "UP_MUDAFAR":(29.0279, 77.7225),
    "UP_JHANSI": (25.4484, 78.5685), "UP_MATHURA":(27.4924, 77.6737),
    "UP_FIROZAB":(27.1490, 78.3946), "UP_ETAWAH": (26.7856, 79.0197),
    "UP_MAINPUR":(27.2402, 79.0303), "UP_AURAIY": (26.4659, 79.5160),
    "UP_KANNAUJ":(27.0544, 79.9161), "UP_UNNAO":  (26.5477, 80.4931),
    "UP_RAIT":   (26.2394, 80.2052), "UP_LUCKNOW":(26.8467, 80.9462),
    "UP_HARDOI": (27.4159, 80.1286), "UP_SITAPUR":(27.5653, 80.6822),
    "UP_BAHRAICH":(27.5742, 81.5970),"UP_GONDA":  (27.1339, 81.9610),
    "UP_BASTI":  (26.7940, 82.7220), "UP_SIDDHAR":(27.1511, 83.0720),
    "UP_MAHARJ": (28.1207, 81.6680), "UP_SHRAVS": (27.4400, 82.0200),
    "UP_BALRAM": (27.6756, 81.5831), "UP_AMBEDKR":(26.4688, 80.3319),
    "UP_PRAYAG": (25.4358, 81.8463), "UP_KAUSHAMB":(25.5000, 81.3800),
    "UP_FATEHPUR":(25.9302, 80.8136),"UP_PRATAP": (25.8990, 81.9870),
    "UP_BANDA":  (25.4820, 80.3359), "UP_CHITRAK":(25.1968, 80.8457),
    "UP_HAMIRPUR":(25.9636, 80.1473),"UP_JALAUN": (26.1500, 79.3500),
    "UP_LALITPUR":(24.6864, 78.4106),"UP_MAHOBA": (25.2927, 79.8706),
    "UP_MIRZAPUR":(25.1337, 82.5650),"UP_SONAPUR":(24.6829, 82.2981),
    "UP_BHADOHI":(25.3990, 82.5649), "UP_SANT_RI":(24.8340, 82.9910),
    "UP_BALLIA": (25.7580, 84.1464), "UP_GAZIPUR":(25.5840, 83.5780),
    "UP_MAU":    (25.9440, 83.5610), "UP_AZAMGAR":(26.0620, 83.1840),
    "UP_DEORIA": (26.5040, 83.7800), "UP_CUSHINA":(26.7420, 84.1180),
    "UP_MAHAJG": (28.9440, 83.6200), "UP_BARELI": (28.3670, 79.4304),
    # Punjab
    "PB_AMRIT":  (31.6340, 74.8723), "PB_LUDHI":  (30.9010, 75.8573),
    "PB_JALLAND":(31.3260, 75.5762), "PB_PATIALA":(30.3398, 76.3869),
    "PB_BATHIND":(30.2110, 74.9455), "PB_MOHALI": (30.7046, 76.7179),
    "PB_HOSHIAR":(31.5143, 75.9112), "PB_GURARSP":(32.2740, 74.8790),
    "PB_FATEHGR":(30.6419, 76.3883), "PB_MOGA":   (30.8185, 75.1688),
    "PB_FIROZPU":(30.9351, 74.6083), "PB_FAZILKA":(30.4032, 74.0258),
    "PB_MUKTSAR":(30.4739, 74.5154), "PB_FARIDKOT":(30.6745, 74.7578),
    "PB_BARNA":  (30.3749, 75.5495), "PB_MANSA":  (29.9983, 75.3980),
    "PB_ROPAR":  (30.9651, 76.5278), "PB_NAWASH": (31.1208, 76.1167),
    "PB_SAS_NGR":(30.7046, 76.7179), "PB_SHAHID": (31.7200, 76.0200),
    "PB_TARN_TR":(31.4518, 74.9278), "PB_KAPURTH":(31.3795, 75.3859),
    # Haryana
    "HR_GURUGR": (28.4595, 77.0266), "HR_FARIDAB":(28.4089, 77.3178),
    "HR_AMBALA": (30.3782, 76.7767), "HR_SONIPAT":(28.9931, 77.0151),
    "HR_KARNAL": (29.6857, 76.9905), "HR_HISAR":  (29.1492, 75.7217),
    "HR_ROHTAK": (28.8955, 76.6066), "HR_PANIPAT":(29.3909, 76.9635),
    "HR_BHIWANI":(28.7929, 75.9977), "HR_JIND":   (29.3160, 76.3003),
    "HR_KAITHAL":(29.8010, 76.3997), "HR_KURUKSH":(29.9695, 76.8783),
    "HR_YAMUNAN":(30.1299, 77.2674), "HR_PANCHK": (30.7343, 76.8606),
    "HR_FATEHAB":(29.5183, 75.4550), "HR_SRE":    (29.6500, 75.0200),
    "HR_SIRSA":  (29.5350, 75.0300), "HR_PEHOWA": (29.9820, 76.5890),
    "HR_CHARKHI":(28.5939, 76.2657), "HR_NUHAN":  (27.5520, 77.1940),
    "HR_REWARI": (28.1960, 76.6169), "HR_MAHEND": (28.2818, 76.9530),
    # Madhya Pradesh
    "MP_BHOPAL": (23.2599, 77.4126), "MP_INDORE": (22.7196, 75.8577),
    "MP_GWALIOR":(26.2183, 78.1828), "MP_JABALP": (23.1815, 79.9864),
    "MP_UJJAIN": (23.1765, 75.7885), "MP_SAGAR":  (23.8388, 78.7378),
    "MP_REWA":   (24.5365, 81.2962), "MP_SATNA":  (24.5703, 80.8322),
    "MP_RAISEN": (23.3282, 77.7869), "MP_VIDISHA":(23.5237, 77.8151),
    "MP_RAJG":   (24.0234, 77.7250), "MP_DAMOH":  (23.8354, 79.4419),
    "MP_SAGR":   (23.8388, 78.7378), "MP_NARS":   (22.9535, 79.1940),
    "MP_MANDSAU":(24.0782, 75.0697), "MP_RATLAM": (23.3315, 75.0367),
    "MP_SHAJAPUR":(23.4264, 76.2814),"MP_DEWAS":  (22.9623, 76.0505),
    "MP_DHAR":   (22.5984, 75.2996), "MP_JHABUA": (22.7676, 74.5889),
    "MP_ALIRAJ": (22.1693, 74.3543), "MP_KHARG":  (22.0439, 76.5539),
    "MP_BURHAN": (22.0700, 77.5900), "MP_BETUL":  (21.9026, 77.8958),
    "MP_HARDA":  (22.3400, 77.0900), "MP_HOSHANG":(22.7478, 77.7278),
    "MP_CHHIND": (22.0573, 78.9352), "MP_SEONI":  (22.0875, 79.5458),
    "MP_KHAND":  (21.8255, 76.3486), "MP_BALAGHA":(21.8126, 80.1870),
    "MP_MANDLA": (22.5980, 80.3743), "MP_DINDR":  (22.9432, 81.0712),
    "MP_SHEO":   (22.1897, 80.0050), "MP_KATNI":  (23.8226, 80.3917),
    "MP_UMARIY": (23.5234, 80.8409), "MP_SIDHI":  (24.4155, 81.8790),
    "MP_SINGRA": (24.1994, 81.9000), "MP_PANNA":  (24.7179, 80.1863),
    "MP_TIKAMG": (24.7443, 78.8318), "MP_CHATARPUR":(24.9088, 79.5853),
    "MP_MORENA": (26.4971, 77.9980), "MP_BHIND":  (26.5616, 78.7874),
    "MP_DATIA":  (25.6602, 78.4601), "MP_SHIVPURI":(25.4231, 77.6586),
    "MP_GUNA":   (24.6474, 77.3152), "MP_ASHOKNAG":(24.5742, 77.7298),
    "MP_AGAR":   (23.7180, 76.0180), "MP_NIWARI": (24.9660, 78.7330),
    # Bihar
    "BR_PATNA":  (25.5941, 85.1376), "BR_MUZAFF": (26.1197, 85.3910),
    "BR_GAYA":   (24.7955, 85.0002), "BR_BHAGAL": (25.2414, 86.9826),
    "BR_DARBHAN":(26.1542, 85.8818), "BR_PURNIA": (25.7772, 87.4754),
    "BR_ARARIA": (26.1536, 87.4704), "BR_BEGUS":  (25.4189, 86.1272),
    "BR_BUXAR":  (25.5614, 83.9733), "BR_ROHTAS": (24.9892, 84.1014),
    "BR_ARWAL":  (25.2499, 84.6824), "BR_JEHANAB":(25.2180, 84.9938),
    "BR_NALANDA":(25.1386, 85.4478), "BR_SHEIKHP":(25.1389, 84.8308),
    "BR_NAWADA": (24.8891, 85.5434), "BR_LAKHISA":(25.1547, 86.0969),
    "BR_MUNGER": (25.3767, 86.4733), "BR_SHEOHAR":(26.5129, 85.2989),
    "BR_SITAMAR":(26.5903, 85.4933), "BR_VAISHALI":(25.6796, 85.2005),
    "BR_SARAN":  (25.9106, 84.7497), "BR_SIWAN":  (26.2192, 84.3602),
    "BR_GOPALGANJ":(26.4687, 84.4356),"BR_CHAPRA":(25.7813, 84.7332),
    "BR_SUPAUL": (26.1235, 86.5791), "BR_MADHEP": (25.9210, 87.0200),
    "BR_SAHARSA":(25.8766, 86.5960), "BR_KATIHAR":(25.5389, 87.5756),
    "BR_MDHUBANI":(26.3467, 86.0793),"BR_DARBHANG":(26.1542, 85.8818),
    "BR_WEST_CH":(26.5490, 84.7500), "BR_EAST_CH":(26.5440, 85.0600),
    "BR_SITAPUR":(26.5903, 85.4933), "BR_JAMUI":  (24.9221, 86.2248),
    "BR_KHAGARIA":(25.4949, 86.4640),"BR_BANKA":  (24.8747, 86.9296),
    "BR_BHOJPUR":(25.5614, 84.3036),
    # Chhattisgarh
    "CG_RAIPUR": (21.2514, 81.6296), "CG_BILASPUR":(22.0796, 82.1391),
    "CG_DURG":   (21.1905, 81.2849), "CG_RAJNAND":(21.0972, 81.0293),
    "CG_KORBA":  (22.3595, 82.7501), "CG_JANJG":  (22.0100, 82.5900),
    "CG_SURGUJA":(23.1048, 83.1989), "CG_RAIGAR": (21.8969, 81.9953),
    "CG_KANKER": (20.2748, 81.4888), "CG_KONDAGN":(20.9740, 81.6680),
    "CG_BASTER": (19.2989, 81.8344), "CG_DANTEW": (18.8990, 81.3500),
    "CG_SUKMA":  (18.3840, 81.6620), "CG_BIJAPUR":(18.8380, 80.7980),
    "CG_NARAYN": (20.0570, 80.4930), "CG_GARIYAB":(22.7100, 82.0600),
    "CG_MUNGELI":(22.0700, 81.6800), "CG_BEMETARA":(21.7200, 81.5400),
    "CG_BALODA": (20.7290, 81.4180), "CG_BALODAB":(20.7290, 81.4180),
    "CG_BALAMAR":(21.6200, 82.8700), "CG_SARANGG":(22.1080, 84.0200),
    "CG_SURAJPUR":(23.2177, 83.0005),
    # Odisha
    "OD_BHUBAN": (20.2961, 85.8245), "OD_CUTTACK":(20.4625, 85.8828),
    "OD_ROURK":  (22.2270, 84.8640), "OD_SAMBHAL":(21.4669, 83.9812),
    "OD_PURI":   (19.8106, 85.8314), "OD_BALSOR": (21.4853, 86.9319),
    "OD_MAYURB": (21.9320, 86.4300), "OD_KEONJH": (21.6297, 85.5813),
    "OD_SUNDAR": (19.8013, 83.9000), "OD_GANJAM": (19.3876, 84.6800),
    "OD_KORAPUT":(18.8135, 82.7157), "OD_RAYAGAD":(19.1472, 83.0040),
    "OD_MALKANG":(18.3514, 83.5755), "OD_NAWARDP":(19.2340, 82.5424),
    "OD_KALAHAN":(20.1380, 83.1690), "OD_BOLANGR":(20.6867, 83.7760),
    "OD_DEOGARH": (21.5330, 84.7270), "OD_JHARSUGUDA":(21.8543, 84.0074),
    "OD_ANGUL":  (20.8397, 85.0983), "OD_DHENKANAL":(20.4500, 84.5840),
    "OD_KANDHAMAL":(20.4700, 83.1700), "OD_BOUDH":  (20.8380, 84.3360),
    "OD_NAYAGARH":(20.1260, 84.3000), "OD_KHURDA": (20.1820, 85.6160),
    "OD_JAGATSINGHPUR":(19.9310, 84.3976), "OD_GAJAPATI":(18.8200, 84.1000),
    "OD_BARGARH":(21.3295, 83.6073), "OD_SAMBALPUR":(20.0594, 85.0948),
    "OD_BHADRAK":(21.0579, 86.4927),
    # West Bengal
    "WB_KOLKATA": (22.5726, 88.3639), "WB_MURSHIDABAD":(24.1822, 88.2701),
    "WB_BURDWAN":  (23.2324, 87.8615), "WB_MIDNAPORE":(22.4248, 87.3199),
    "WB_HOOGHLY":(22.9000, 88.3900), "WB_HOWRAH": (22.5887, 88.2663),
    "WB_NORTH_24_PARGANAS":(22.7200, 88.5400), "WB_SOUTH_24_PARGANAS":(22.1500, 88.3500),
    "WB_NADIA":  (23.4652, 88.5502), "WB_JALPAIGURI":   (26.5163, 88.7180),
    "WB_COOCH_BEHAR":  (26.3204, 89.4457), "WB_DARJEELING": (27.0360, 88.2627),
    "WB_MALDA":  (25.0108, 88.1420), "WB_NORTH_DINAJPUR":(25.6230, 88.6420),
    "WB_SOUTH_DINAJPUR":(25.3300, 88.6000), "WB_PURULIA":(23.3302, 86.3647),
    "WB_BANKURA": (23.2330, 87.0739), "WB_BISHNUPUR":   (22.9989, 86.9928),
    "WB_JHARGRAM":  (22.4520, 86.9924), "WB_BIRBHUM":(23.8876, 87.5300),
    # Jharkhand
    "JH_RANCHI": (23.3441, 85.3096), "JH_EAST_SINGHBHUM":(22.8046, 86.2029),
    "JH_DHANBAD":(23.7957, 86.4304), "JH_BOKARO": (23.6693, 85.9875),
    "JH_HAZARIBAGH":(23.9921, 85.3636), "JH_GIRIDIH":(24.1881, 86.2994),
    "JH_DEOGHAR":(24.4830, 86.6940), "JH_DUMKA":  (24.2682, 87.2501),
    "JH_PAKUR":  (24.6365, 87.8441), "JH_GODDA":  (24.8282, 87.2103),
    "JH_SAHEBGANJ": (25.2433, 87.6397), "JH_JAMTARA":(23.9608, 86.8048),
    "JH_SIMDEGA":(22.6063, 84.5027), "JH_WEST_SINGHBHUM":(22.1530, 84.4300),
    "JH_GUMLA":  (23.0448, 84.5393), "JH_GARHWA": (24.1552, 83.8090),
    "JH_PALAMU":(23.9921, 84.0760), "JH_LATEHAR":(23.7440, 84.5054),
    "JH_WEST_SINGHBHUM":(22.5555, 85.8050), "JH_SERAIKELA":(22.5853, 85.9371),
    "JH_EAST_SINGHBHUM":(22.9218, 85.4760), "JH_RAMGARH":(23.6300, 85.5200),
    "JH_KODERMA":  (23.8600, 85.0000),
    # Assam
    "AS_KAMRUP_METRO":(26.1445, 91.7362), "AS_DIBRUGARH":(27.4728, 94.9120),
    "AS_JORHAT": (26.7465, 94.2026), "AS_CACHAR": (24.8333, 92.7789),
    "AS_NAGAON": (26.3483, 92.6850), "AS_KAMRUP":  (26.0700, 91.6000),
    "AS_SONITPUR":(26.6300, 92.8000),"AS_TINSUKIA": (27.4891, 95.3499),
    "AS_LAKHIMPUR":(27.2381, 94.1000), "AS_DHEMAJI": (27.4800, 94.5700),
    "AS_BONGAIGAON":(26.4771, 90.5584), "AS_KOKRAJHAR": (26.4000, 90.0200),
    "AS_UDALGURI":(26.7500, 90.8000), "AS_DHUBRI":  (26.0220, 89.9740),
    "AS_GOALPARA":(25.9840, 90.6260),"AS_BARPETA": (26.3190, 91.0040),
    "AS_NALBARI":(26.4440, 91.4380), "AS_MORIGAON":(26.2530, 92.3360),
    "AS_KARBI_ANGLONG":  (25.9920, 93.0330), "AS_DIMA_HASAO":(25.5780, 93.0218),
    "AS_CACHAR": (24.8089, 92.8574), "AS_HAILAKANDI":(24.3330, 92.6670),
    "AS_KARIMGANJ":(24.8650, 92.3610),
    # Himachal Pradesh
    "HP_SHIMLA": (31.1048, 77.1734), "HP_MANDI":  (31.7089, 76.9317),
    "HP_KANGRA": (32.1000, 76.2700), "HP_SOLAN":  (30.9045, 77.0967),
    "HP_KULLU":  (31.9581, 77.1089), "HP_HAMIRPUR":(31.6862, 76.5215),
    "HP_UNA":    (31.4686, 76.2705), "HP_BILASPUR":(31.3413, 76.7601),
    "HP_SIRMAUR":(30.5600, 77.4700), "HP_KINNAUR":(31.5882, 78.3665),
    "HP_LAHAUL_SPITI":  (32.5600, 77.4000), "HP_CHAMBA": (32.5519, 76.1249),
    # Uttarakhand
    "UK_DEHRADUN": (30.3165, 78.0322), "UK_HARIDWAR":(29.9457, 78.1642),
    "UK_NAINITAL":(29.3909, 79.4570),"UK_UDHAM_SINGH_NAGAR":  (28.9738, 79.5117),
    "UK_ALMORA": (29.5892, 79.6467), "UK_CHAMPAWAT":(29.3356, 80.0907),
    "UK_PITHORAGARH": (29.5820, 80.2080), "UK_BAGESHWAR": (29.8370, 79.7700),
    "UK_CHAMOLI":(30.4000, 79.3200), "UK_RUDRAPRAYAG": (30.2899, 78.9763),
    "UK_TEHRI_GARHWAL":  (30.3781, 78.4804), "UK_UTTARKASHI":(30.6929, 78.4305),
    "UK_PAURI_GARHWAL":  (29.8633, 78.8547),
    # Delhi
    "DL_DELHI": (28.6139, 77.2090),
    # Goa
    "GA_NORTH_GOA":  (15.4909, 73.8278), "GA_SOUTH_GOA":  (15.2993, 74.1240),
    # Northeast states (key districts)
    "MN_IMPHAL_WEST": (24.8170, 93.9368), "ML_EAST_KHASI":(25.5788, 91.8933),
    "NL_KOHIMA": (25.6747, 94.1086), "TR_WEST_TRIPURA":(23.8315, 91.2868),
    "AR_PAPUM_PARE": (27.0844, 93.6053), "SK_EAST_SIKKIM":(27.3314, 88.6138),
    "MZ_AIZAWL": (23.7271, 92.7176),
    # Jammu & Kashmir / Ladakh
    "JK_SRINAGAR":(34.0837, 74.7973),"JK_JAMMU":  (32.7266, 74.8570),
    "JK_ANANTNAG":(33.7311, 75.1487),"JK_BARAMULLA":(34.1992, 74.3629),
    "LA_LEH":    (34.1526, 77.5771),
}


def fetch_year_for_district(
    region_id: str,
    lat: float,
    lon: float,
    year: int,
    output_dir: Path
) -> bool:
    """Fetch one year of daily weather for a district from Open-Meteo Historical API."""
    out_path = output_dir / region_id / f"{year}.parquet"
    if out_path.exists():
        logger.debug(f"  Skipping {region_id}/{year} (already exists)")
        return True

    url = HISTORICAL_API
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date":   f"{year}-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_max",
            "wind_speed_10m_max",
        ],
        "timezone": "Asia/Kolkata",
    }

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

            # Detect daily rate-limit BEFORE raise_for_status so we can surface a
            # helpful message and stop immediately rather than silently retrying.
            if resp.status_code == 429:
                raise RateLimitError(
                    "Open-Meteo daily API limit reached. "
                    "Re-run tomorrow — already-downloaded files will be skipped."
                )

            resp.raise_for_status()
            data = resp.json()["daily"]

            df = pd.DataFrame({
                "date":       data["time"],
                "temp_max":   data["temperature_2m_max"],
                "temp_min":   data["temperature_2m_min"],
                "rainfall":   data["precipitation_sum"],
                "humidity":   data["relative_humidity_2m_max"],
                "wind_speed": data["wind_speed_10m_max"],
            })

            # Basic cleaning
            df["date"]       = pd.to_datetime(df["date"])
            df["rainfall"]   = df["rainfall"].fillna(0.0)
            df["temp_max"]   = df["temp_max"].interpolate().bfill().fillna(30.0)
            df["temp_min"]   = df["temp_min"].interpolate().bfill().fillna(18.0)
            df["humidity"]   = df["humidity"].interpolate().bfill().fillna(60.0)
            df["wind_speed"] = df["wind_speed"].interpolate().bfill().fillna(10.0)

            df["region_id"] = region_id

            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            return True

        except RateLimitError:
            raise  # propagate immediately — no retries
        except Exception as e:
            logger.warning(f"  Attempt {attempt}/{RETRY_LIMIT} failed for {region_id}/{year}: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

    logger.error(f"  FAILED to fetch {region_id}/{year} after {RETRY_LIMIT} attempts")
    return False


def fetch_all_districts(output_dir: str = "data/weather/district", sample: int = None):
    """
    Fetch historical weather for all districts.

    Args:
        output_dir: Directory to write parquet files to
        sample:     If set, only fetch this many districts (for testing)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    coords = DISTRICT_COORDS
    if sample:
        items = list(coords.items())[:sample]
        coords = dict(items)
        logger.info(f"SAMPLE MODE: fetching {sample} districts")
    else:
        logger.info(f"Fetching {len(coords)} districts × {END_YEAR - START_YEAR + 1} years")

    total = len(coords) * (END_YEAR - START_YEAR + 1)
    done = 0
    failed = []

    start_time = datetime.now()

    for region_id, (lat, lon) in coords.items():
        for year in range(START_YEAR, END_YEAR + 1):
            try:
                success = fetch_year_for_district(region_id, lat, lon, year, output_path)
            except RateLimitError as e:
                logger.error(f"\n🚫 {e}")
                logger.info(f"   Stopped at {region_id}/{year}. "
                            f"{done} records processed, {done - len(failed)} saved.")
                sys.exit(0)  # clean exit – tomorrow re-run will resume
            done += 1
            if not success:
                failed.append(f"{region_id}/{year}")

            # Rate limit: ~1 req/sec to be polite
            time.sleep(0.5)

            if done % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = done / elapsed
                remaining = (total - done) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {done}/{total} ({100*done//total}%) "
                    f"| ETA: {int(remaining//60)}m {int(remaining%60)}s"
                )

    logger.info(f"\n✅ Done — {done - len(failed)}/{total} files fetched successfully")
    if failed:
        logger.warning(f"❌ {len(failed)} files failed: {failed[:10]}{'...' if len(failed)>10 else ''}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical district weather data")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Number of districts to fetch (default: all). Use small values for testing."
    )
    parser.add_argument(
        "--output", type=str, default="data/weather/district",
        help="Output directory (default: data/weather/district)"
    )
    args = parser.parse_args()

    fetch_all_districts(output_dir=args.output, sample=args.sample)
