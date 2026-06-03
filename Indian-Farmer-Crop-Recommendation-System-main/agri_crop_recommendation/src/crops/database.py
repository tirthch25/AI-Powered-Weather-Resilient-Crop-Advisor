"""
Comprehensive Crop Knowledge Base for Indian Agriculture

Contains short-duration crops (15-90 days) with detailed information including
temperature requirements, water needs, soil compatibility, and regional suitability
for all major Indian agricultural regions.
"""

from typing import Dict, List, Optional
from src.crops.models import CropInfo
from src.crops.soil import SoilInfo, calculate_soil_compatibility_score, get_soil_amendment_suggestions
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Regional Enrichment (Gemini-generated per-district crop suitability) ──────
# Loaded once at startup from data/reference/regional_crops.json.
# If the file doesn't exist yet (first run), the dict is empty and
# the system falls back to the static zone-based scores as before.
_REGIONAL_ENRICHMENT: Dict[str, Dict] = {}


def _load_regional_enrichment() -> None:
    """Load Gemini-generated per-district crop suitability from JSON file."""
    global _REGIONAL_ENRICHMENT
    # Look for the file relative to this module's location
    base = Path(__file__).resolve().parent.parent.parent  # agri_crop_recommendation/
    json_path = base / "data" / "reference" / "regional_crops.json"
    if not json_path.exists():
        logger.debug("regional_crops.json not found — using static zone-based scores")
        return
    try:
        _REGIONAL_ENRICHMENT = json.loads(json_path.read_text(encoding="utf-8"))
        logger.info(f"Loaded regional enrichment for {len(_REGIONAL_ENRICHMENT)} districts")
    except Exception as e:
        logger.warning(f"Failed to load regional_crops.json: {e}")


# Load enrichment at module import time
_load_regional_enrichment()


def get_regional_enrichment(region_id: str) -> Optional[Dict]:
    """
    Return Gemini-generated enrichment data for a district, or None if not available.

    Returns a dict with keys:
        'approved'  -> {crop_id: suitability_score (0.50–1.00)}
        'excluded'  -> [crop_id, ...]
        'source'    -> 'gemini-2.0-flash'
    """
    return _REGIONAL_ENRICHMENT.get(region_id)


# ── Zone → Region ID mapping (for nationwide suitability coverage) ──────────
# Each zone maps to ALL region IDs from regions.json for that agro-climatic zone.
# Suitability 0.75 is assigned whenever a crop's zone matches.
# Regions not in this dict get the recommender's 0.50 fallback (penalized).
ZONE_REGIONS = {
    "North": [
        # Uttar Pradesh (complete)
        "UP_LUCKNOW","UP_VARANASI","UP_AGRA","UP_KANPUR","UP_ALLAHABAD",
        "UP_MEERUT","UP_BAREILLY","UP_GORAKHPUR","UP_MORADABAD","UP_ALIGARH",
        "UP_MATHURA","UP_FIROZABAD","UP_MAINPURI","UP_ETAWAH","UP_FARRUKHABAD",
        "UP_HARDOI","UP_SITAPUR","UP_LAKHIMPUR_KHERI","UP_BARABANKI","UP_RAEBARELI",
        "UP_SULTANPUR","UP_PRATAPGARH","UP_AZAMGARH","UP_MAU","UP_BALLIA",
        "UP_GHAZIPUR","UP_JAUNPUR","UP_MIRZAPUR","UP_SONBHADRA","UP_CHANDAULI",
        "UP_BHADOHI","UP_BANDA","UP_CHITRAKOOT","UP_HAMIRPUR","UP_MAHOBA",
        "UP_LALITPUR","UP_JHANSI","UP_JALAUN","UP_FATEHPUR","UP_PRAYAGRAJ",
        "UP_PILIBHIT","UP_SHAHJAHANPUR","UP_RAMPUR","UP_SAMBHAL","UP_AMROHA",
        "UP_BULANDSHAHR","UP_HAPUR","UP_GAUTAM_BUDDH_NAGAR","UP_GHAZIABAD",
        "UP_MUZAFFARNAGAR","UP_SHAMLI","UP_BIJNOR","UP_SAHARANPUR",
        "UP_UNNAO","UP_KANPUR_DEHAT","UP_ETAH","UP_KASGANJ","UP_BUDAUN",
        "UP_BAHRAICH","UP_SHRAWASTI","UP_BALRAMPUR","UP_GONDA","UP_BASTI",
        "UP_SANT_KABIR_NAGAR","UP_SIDDHARTHNAGAR","UP_MAHARAJGANJ","UP_KUSHINAGAR",
        "UP_DEORIA","UP_AMBEDKAR_NAGAR","UP_AMETHI","UP_AYODHYA","UP_LAKHIMPUR",
        # Punjab (complete)
        "PB_LUDHIANA","PB_AMRITSAR","PB_JALANDHAR","PB_PATIALA","PB_GURDASPUR",
        "PB_HOSHIARPUR","PB_NAWANSHAHR","PB_RUPNAGAR","PB_FATEHGARH_SAHIB",
        "PB_MOHALI","PB_KAPURTHALA","PB_FIROZPUR","PB_FAZILKA","PB_MUKTSAR",
        "PB_MOGA","PB_BARNALA","PB_SANGRUR","PB_MANSA","PB_BATHINDA",
        "PB_FARIDKOT","PB_TARN_TARAN","PB_PATHANKOT",
        # Haryana (complete)
        "HR_AMBALA","HR_HISAR","HR_KARNAL","HR_ROHTAK","HR_FARIDABAD",
        "HR_GURUGRAM","HR_MEWAT","HR_PALWAL","HR_REWARI","HR_MAHENDRAGARH",
        "HR_BHIWANI","HR_CHARKHI_DADRI","HR_JHAJJAR","HR_SONIPAT","HR_PANIPAT",
        "HR_KAITHAL","HR_KURUKSHETRA","HR_YAMUNANAGAR","HR_PANCHKULA",
        "HR_SIRSA","HR_FATEHABAD","HR_JIND",
        # Rajasthan (complete)
        "RJ_JAIPUR","RJ_JODHPUR","RJ_UDAIPUR","RJ_AJMER","RJ_BIKANER",
        "RJ_ALWAR","RJ_BHARATPUR","RJ_DAUSA","RJ_DHOLPUR","RJ_KARAULI",
        "RJ_SAWAI_MADHOPUR","RJ_TONK","RJ_BUNDI","RJ_KOTA","RJ_BARAN",
        "RJ_JHALAWAR","RJ_CHITTORGARH","RJ_RAJSAMAND","RJ_BHILWARA",
        "RJ_SIKAR","RJ_JHUNJHUNU","RJ_CHURU","RJ_HANUMANGARH","RJ_GANGANAGAR",
        "RJ_NAGAUR","RJ_PALI","RJ_SIROHI","RJ_JALORE","RJ_BARMER",
        "RJ_JAISALMER","RJ_DUNGARPUR","RJ_BANSWARA","RJ_PRATAPGARH",
        # Himachal Pradesh (complete)
        "HP_SHIMLA","HP_KANGRA","HP_MANDI","HP_SOLAN","HP_SIRMAUR","HP_BILASPUR",
        "HP_UNA","HP_HAMIRPUR","HP_KULLU","HP_LAHAUL_SPITI","HP_CHAMBA","HP_KINNAUR",
        # Uttarakhand (complete)
        "UK_DEHRADUN","UK_HARIDWAR","UK_NAINITAL","UK_UDHAM_SINGH_NAGAR",
        "UK_ALMORA","UK_PITHORAGARH","UK_CHAMPAWAT","UK_BAGESHWAR",
        "UK_CHAMOLI","UK_RUDRAPRAYAG","UK_TEHRI_GARHWAL","UK_PAURI_GARHWAL","UK_UTTARKASHI",
        # Delhi & J&K
        "DL_NEW_DELHI",
        "JK_JAMMU","JK_SRINAGAR","JK_BARAMULLA","JK_ANANTNAG","JK_PULWAMA",
        "JK_BUDGAM","JK_KULGAM","JK_SHOPIAN","JK_GANDERBAL","JK_BANDIPORA",
        "JK_KUPWARA","JK_KATHUA","JK_UDHAMPUR","JK_DODA","JK_KISHTWAR",
        "JK_RAMBAN","JK_REASI","JK_RAJOURI","JK_POONCH","JK_SAMBA",
    ],

    "South": [
        # Karnataka (complete)
        "KA_BENGALURU","KA_MYSURU","KA_HUBLI","KA_DHARWAD","KA_BELAGAVI",
        "KA_BALLARI","KA_RAICHUR","KA_KOPPAL","KA_GADAG","KA_HAVERI",
        "KA_UTTARA_KANNADA","KA_SHIVAMOGGA","KA_CHIKKAMAGALURU","KA_HASSAN",
        "KA_KODAGU","KA_MANDYA","KA_CHAMARAJANAGAR","KA_TUMKURU",
        "KA_CHITRADURGA","KA_DAVANAGERE","KA_KOLAR","KA_CHIKKABALLAPURA",
        "KA_RAMANAGARA","KA_BENGALURU_RURAL","KA_YADGIR","KA_BIDAR",
        "KA_KALABURAGI","KA_VIJAYAPURA","KA_BAGALKOT","KA_DAKSHINA_KANNADA","KA_UDUPI",
        # Tamil Nadu (complete)
        "TN_CHENNAI","TN_COIMBATORE","TN_MADURAI","TN_TIRUNELVELI","TN_SALEM",
        "TN_ERODE","TN_NAMAKKAL","TN_DHARMAPURI","TN_KRISHNAGIRI","TN_VELLORE",
        "TN_TIRUVANNAMALAI","TN_VILLUPURAM","TN_CUDDALORE","TN_NAGAPATTINAM",
        "TN_THANJAVUR","TN_TIRUVARUR","TN_PUDUKKOTTAI","TN_DINDIGUL","TN_THENI",
        "TN_VIRUDHUNAGAR","TN_RAMANATHAPURAM","TN_THOOTHUKUDI","TN_KANYAKUMARI",
        "TN_TENKASI","TN_TRICHIRAPPALLI","TN_PERAMBALUR","TN_ARIYALUR","TN_KARUR",
        "TN_SIVAGANGA","TN_NILGIRIS","TN_KANCHEEPURAM","TN_TIRUVALLUR",
        "TN_RANIPET","TN_TIRUPATTUR","TN_KALLAKURICHI","TN_CHENGALPATTU","TN_MAYILADUTHURAI",
        # Andhra Pradesh (complete)
        "AP_VISAKHAPATNAM","AP_SRIKAKULAM","AP_VIZIANAGARAM","AP_KRISHNA",
        "AP_GUNTUR","AP_KURNOOL","AP_ANANTAPUR","AP_CHITTOOR","AP_EAST_GODAVARI",
        "AP_WEST_GODAVARI","AP_PRAKASAM","AP_NELLORE","AP_KADAPA",
        "AP_PARVATHIPURAM_MANYAM","AP_ALLURI_SITHARAMA_RAJU","AP_ANAKAPALLI",
        "AP_KAKINADA","AP_KONASEEMA","AP_ELURU","AP_NTR","AP_BAPATLA",
        "AP_PALNADU","AP_NANDYAL","AP_SRI_SATHYA_SAI","AP_ANNAMAYYA","AP_TIRUPATI",
        # Telangana (complete)
        "TS_HYDERABAD","TS_WARANGAL","TS_NIZAMABAD","TS_KARIMNAGAR",
        "TS_ADILABAD","TS_KOMARAM_BHEEM","TS_MANCHERIAL","TS_NIRMAL",
        "TS_JAGTIAL","TS_PEDDAPALLI","TS_JAYASHANKAR","TS_RAJANNA_SIRCILLA",
        "TS_MEDCHAL_MALKAJGIRI","TS_YADADRI_BHUVANAGIRI","TS_JANGAON",
        "TS_HANAMKONDA","TS_MULUGU","TS_BHADRADRI_KOTHAGUDEM","TS_KHAMMAM",
        "TS_MAHABUBABAD","TS_SURYAPET","TS_NALGONDA","TS_NAGARKURNOOL",
        "TS_WANAPARTHY","TS_GADWAL","TS_RANGAREDDY","TS_VIKARABAD",
        "TS_SANGAREDDY","TS_MEDAK","TS_SIDDIPET","TS_KAMAREDDY","TS_MAHABUBNAGAR",
        # Kerala (complete)
        "KL_THIRUVANANTHAPURAM","KL_KOLLAM","KL_PATHANAMTHITTA","KL_ALAPPUZHA",
        "KL_KOTTAYAM","KL_IDUKKI","KL_ERNAKULAM","KL_THRISSUR","KL_PALAKKAD",
        "KL_MALAPPURAM","KL_KOZHIKODE","KL_WAYANAD","KL_KANNUR","KL_KASARAGOD",
        # Union territories in South
        "PY_PUDUCHERRY","AN_PORT_BLAIR","GA_NORTH_GOA","GA_SOUTH_GOA",
    ],

    "West": [
        # Maharashtra (complete — all 33 districts)
        "MH_PUNE","MH_NASHIK","MH_CHHATRAPATI_SAMBHAJINAGAR","MH_SOLAPUR",
        "MH_KOLHAPUR","MH_NAGPUR","MH_AHMEDNAGAR","MH_LATUR","MH_JALGAON",
        "MH_SATARA","MH_NANDED","MH_OSMANABAD","MH_PARBHANI","MH_WASHIM",
        "MH_YAVATMAL","MH_BULDHANA","MH_AKOLA","MH_AMRAVATI","MH_WARDHA",
        "MH_CHANDRAPUR","MH_GADCHIROLI","MH_GONDIA","MH_BHANDARA","MH_DHULE",
        "MH_NANDURBAR","MH_RAIGAD","MH_RATNAGIRI","MH_SINDHUDURG","MH_THANE",
        "MH_PALGHAR","MH_HINGOLI","MH_SANGLI",
        # Gujarat (complete)
        "GJ_AHMEDABAD","GJ_SURAT","GJ_VADODARA","GJ_RAJKOT","GJ_ANAND",
        "GJ_MEHSANA","GJ_JUNAGADH","GJ_BHARUCH","GJ_KUTCH","GJ_BANASKANTHA",
        "GJ_PATAN","GJ_SABARKANTHA","GJ_GANDHINAGAR","GJ_SURENDRANAGAR",
        "GJ_MORBI","GJ_JAMNAGAR","GJ_DEVBHUMI_DWARKA","GJ_PORBANDAR",
        "GJ_GIR_SOMNATH","GJ_AMRELI","GJ_BOTAD","GJ_BHAVNAGAR","GJ_KHEDA",
        "GJ_ARAVALLI","GJ_CHHOTA_UDAIPUR","GJ_PANCHMAHALS","GJ_DAHOD",
        "GJ_NARMADA","GJ_TAPI","GJ_NAVSARI","GJ_VALSAD","GJ_DANG",
        "GJ_MAHESANA",
    ],

    "East": [
        # West Bengal (complete)
        "WB_KOLKATA","WB_HOWRAH","WB_BURDWAN","WB_NADIA","WB_MURSHIDABAD",
        "WB_MALDA","WB_PURBA_MEDINIPUR","WB_PASCHIM_MEDINIPUR","WB_HOOGHLY",
        "WB_MIDNAPORE","WB_COOCH_BEHAR","WB_ALIPURDUAR","WB_JALPAIGURI",
        "WB_DARJEELING","WB_KALIMPONG","WB_JHARGRAM","WB_PASCHIM_BARDHAMAN",
        "WB_PURULIA","WB_BANKURA","WB_BIRBHUM","WB_UTTAR_DINAJPUR",
        "WB_DAKSHIN_DINAJPUR","WB_SOUTH_24_PARGANAS","WB_NORTH_24_PARGANAS",
        "WB_HUGLI","WB_KOCH_BIHAR",
        # Bihar (complete)
        "BR_PATNA","BR_MUZAFFARPUR","BR_GAYA","BR_BHAGALPUR","BR_DARBHANGA",
        "BR_SARAN","BR_SIWAN","BR_GOPALGANJ","BR_EAST_CHAMPARAN","BR_WEST_CHAMPARAN",
        "BR_SITAMARHI","BR_SHEOHAR","BR_VAISHALI","BR_SAMASTIPUR","BR_BEGUSARAI",
        "BR_KHAGARIA","BR_BANKA","BR_MUNGER","BR_LAKHISARAI","BR_SHEIKHPURA",
        "BR_NALANDA","BR_ARWAL","BR_JEHANABAD","BR_AURANGABAD","BR_NAWADA",
        "BR_JAMUI","BR_ROHTAS","BR_KAIMUR","BR_BUXAR","BR_BHOJPUR",
        "BR_MADHUBANI","BR_SUPAUL","BR_SAHARSA","BR_MADHEPURA",
        "BR_PURNEA","BR_KISHANGANJ","BR_ARARIA","BR_KATIHAR",
        # Odisha (complete)
        "OD_BHUBANESWAR","OD_CUTTACK","OD_PURI","OD_SAMBALPUR","OD_ANGUL",
        "OD_BALANGIR","OD_BALASORE","OD_BARGARH","OD_BHADRAK","OD_BOUDH",
        "OD_DEBAGARH","OD_DHENKANAL","OD_GAJAPATI","OD_GANJAM","OD_JAGATSINGHPUR",
        "OD_JAJPUR","OD_JHARSUGUDA","OD_KALAHANDI","OD_KANDHAMAL","OD_KENDRAPARA",
        "OD_KENDUJHAR","OD_KHORDHA","OD_KORAPUT","OD_MALKANGIRI","OD_MAYURBHANJ",
        "OD_NABARANGPUR","OD_NAYAGARH","OD_NUAPADA","OD_RAYAGADA","OD_SONEPUR","OD_SUNDARGARH",
        # Jharkhand (complete)
        "JH_RANCHI","JH_DHANBAD","JH_JAMSHEDPUR","JH_BOKARO","JH_CHATRA",
        "JH_DEOGHAR","JH_DUMKA","JH_EAST_SINGHBHUM","JH_GARHWA","JH_GIRIDIH",
        "JH_GODDA","JH_GUMLA","JH_HAZARIBAGH","JH_JAMTARA","JH_KHUNTI",
        "JH_KODERMA","JH_LATEHAR","JH_LOHARDAGA","JH_PAKUR","JH_PALAMU",
        "JH_RAMGARH","JH_SAHEBGANJ","JH_SARAIKELA","JH_SIMDEGA","JH_WEST_SINGHBHUM",
    ],

    "Central": [
        # Madhya Pradesh (complete)
        "MP_BHOPAL","MP_INDORE","MP_GWALIOR","MP_JABALPUR","MP_UJJAIN",
        "MP_AGAR_MALWA","MP_ALIRAJPUR","MP_ANUPPUR","MP_ASHOKNAGAR","MP_BALAGHAT",
        "MP_BARWANI","MP_BETUL","MP_BHIND","MP_BURHANPUR","MP_CHHATARPUR",
        "MP_CHHINDWARA","MP_DAMOH","MP_DATIA","MP_DEWAS","MP_DHAR","MP_DINDORI",
        "MP_GUNA","MP_HARDA","MP_HOSHANGABAD","MP_KATNI","MP_KHANDWA",
        "MP_KHARGONE","MP_MANDLA","MP_MANDSAUR","MP_MORENA","MP_NARSINGHPUR",
        "MP_NEEMUCH","MP_NIWARI","MP_PANNA","MP_RAISEN","MP_RAJGARH","MP_RATLAM",
        "MP_REWA","MP_SAGAR","MP_SATNA","MP_SEHORE","MP_SEONI","MP_SHAHDOL",
        "MP_SHAJAPUR","MP_SHEOPUR","MP_SHIVPURI","MP_SIDHI","MP_SINGRAULI",
        "MP_TIKAMGARH","MP_UMARIA","MP_VIDISHA","MP_MURAINA",
        # Chhattisgarh (complete)
        "CG_RAIPUR","CG_BILASPUR","CG_DURG","CG_RAJNANDGAON","CG_BALOD",
        "CG_BALODA_BAZAR","CG_BALRAMPUR","CG_BASTAR","CG_BEMETARA","CG_BIJAPUR",
        "CG_DANTEWADA","CG_DHAMTARI","CG_GARIABAND","CG_JANJGIR_CHAMPA",
        "CG_JASHPUR","CG_KABIRDHAM","CG_KANKER","CG_KONDAGAON","CG_KORBA",
        "CG_KORIYA","CG_MAHASAMUND","CG_MUNGELI","CG_NARAYANPUR","CG_RAIGARH",
        "CG_SAKTI","CG_SUKMA","CG_SURAJPUR","CG_SURGUJA",
        # Vidarbha (Maharashtra) also Central agro-climatic zone
        "MH_NAGPUR","MH_WARDHA","MH_CHANDRAPUR","MH_GADCHIROLI",
        "MH_GONDIA","MH_BHANDARA","MH_AMRAVATI","MH_BULDHANA","MH_AKOLA","MH_WASHIM",
    ],

    "Northeast": [
        # Assam (complete)
        "AS_KAMRUP","AS_NAGAON","AS_DIBRUGARH","AS_JORHAT","AS_BARPETA",
        "AS_BISWANATH","AS_BONGAIGAON","AS_CACHAR","AS_CHARAIDEO","AS_CHIRANG",
        "AS_DARRANG","AS_DHEMAJI","AS_DHUBRI","AS_DIMA_HASAO","AS_GOALPARA",
        "AS_GOLAGHAT","AS_HAILAKANDI","AS_HOJAI","AS_KAMRUP_METRO","AS_KARBI_ANGLONG",
        "AS_KARIMGANJ","AS_KOKRAJHAR","AS_LAKHIMPUR","AS_MAJULI","AS_MORIGAON",
        "AS_NALBARI","AS_SIVASAGAR","AS_SONITPUR","AS_SOUTH_SALMARA","AS_TAMULPUR",
        "AS_TINSUKIA","AS_UDALGURI","AS_WEST_KARBI_ANGLONG",
        # Arunachal Pradesh
        "AR_PAPUM_PARE","AR_LOHIT","AR_CHANGLANG","AR_DIBANG_VALLEY",
        "AR_EAST_KAMENG","AR_EAST_SIANG","AR_KURUNG_KUMEY","AR_LONGDING",
        "AR_LOWER_DIBANG_VALLEY","AR_LOWER_SIANG","AR_LOWER_SUBANSIRI","AR_NAMSAI",
        "AR_SIANG","AR_TAWANG","AR_TIRAP","AR_UPPER_DIBANG_VALLEY",
        "AR_UPPER_SIANG","AR_UPPER_SUBANSIRI","AR_WEST_KAMENG","AR_WEST_SIANG",
        # Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura
        "MN_BISHNUPUR","MN_CHANDEL","MN_CHURACHANDPUR","MN_IMPHAL_EAST",
        "MN_IMPHAL_WEST","MN_SENAPATI","MN_TAMENGLONG","MN_THOUBAL","MN_UKHRUL",
        "ML_EAST_GARO_HILLS","ML_EAST_JAINTIA_HILLS","ML_EAST_KHASI_HILLS",
        "ML_NORTH_GARO_HILLS","ML_RI_BHOI","ML_SOUTH_GARO_HILLS",
        "ML_WEST_GARO_HILLS","ML_WEST_JAINTIA_HILLS","ML_WEST_KHASI_HILLS",
        "MZ_AIZAWL","MZ_CHAMPHAI","MZ_KOLASIB","MZ_LAWNGTLAI","MZ_LUNGLEI",
        "MZ_MAMIT","MZ_SAIHA","MZ_SERCHHIP",
        "NL_DIMAPUR","NL_KOHIMA","NL_MOKOKCHUNG","NL_MON","NL_PHEK",
        "NL_TUENSANG","NL_WOKHA","NL_ZUNHEBOTO",
        "SK_EAST_SIKKIM","SK_NORTH_SIKKIM","SK_SOUTH_SIKKIM","SK_WEST_SIKKIM",
        "TR_DHALAI","TR_GOMATI","TR_KHOWAI","TR_NORTH_TRIPURA","TR_SEPAHIJALA",
        "TR_SOUTH_TRIPURA","TR_UNAKOTI","TR_WEST_TRIPURA",
        # Lakshadweep (island)
        "LD_LAKSHADWEEP",
    ],

    # ── Alternate / Variant IDs ───────────────────────────────────────────────
    # Some districts in regions.json use alternate spellings or prefixes.
    # Mapped here so they get zone scores instead of the 0.50 fallback.
    "North_Alt": [
        # Delhi alternate
        "DL_DELHI","CH_CHANDIGARH",
        # UP alternate spellings
        "UP_AURAIYA","UP_BAGHPAT","UP_GAUTAM_BUDDHA_NAGAR","UP_HATHRAS",
        "UP_KANNAUJ","UP_KAUSHAMBI","UP_RAE_BARELI","UP_SHRAVASTI",
        # Punjab alternate
        "PB_FEROZEPUR","PB_ROPAR","PB_SAS_NAGAR_MOHALI",
        # Haryana alternate
        "HR_NUH",
        # Rajasthan alternate
        "RJ_SRI_GANGANAGAR",
        # Ladakh
        "LA_LEH",
    ],
    "South_Alt": [
        # Telangana uses TL_ prefix in some records
        "TL_ADILABAD","TL_BHADRADRI_KOTHAGUDEM","TL_GADWAL","TL_HANAMKONDA",
        "TL_HYDERABAD","TL_JAGTIAL","TL_JANGAON","TL_JAYASHANKAR",
        "TL_JOGULAMBA","TL_KAMAREDDY","TL_KARIMNAGAR","TL_KHAMMAM",
        "TL_KUMURAM_BHEEM","TL_MAHABUBABAD","TL_MAHABUBNAGAR","TL_MANCHERIAL",
        "TL_MEDAK","TL_MEDCHAL","TL_MULUGU","TL_NAGARKURNOOL","TL_NALGONDA",
        "TL_NARAYANPET","TL_NIRMAL","TL_NIZAMABAD","TL_PEDDAPALLI",
        "TL_RAJANNA_SIRCILLA","TL_RANGAREDDY","TL_SANGAREDDY","TL_SIDDIPET",
        "TL_SURYAPET","TL_VIKARABAD","TL_WANAPARTHY","TL_WARANGAL","TL_YADADRI",
        # Karnataka alternate spellings
        "KA_BANGALORE_RURAL","KA_CHIKBALLAPUR","KA_MYSORE","KA_TUMKUR",
        # Tamil Nadu alternate
        "TN_KANCHIPURAM","TN_TIRUCHIRAPPALLI","TN_TIRUPATHUR","TN_TIRUPPUR",
        # Puducherry
        "PY_PUDUCHERRY",
    ],
    "West_Alt": [
        # Maharashtra alternate / missing districts
        "MH_BEED","MH_JALNA",
        # Gujarat alternate spellings
        "GJ_DEVBHOOMI_DWARKA","GJ_MAHISAGAR","GJ_PANCHMAHAL",
    ],
    "East_Alt": [
        # Bihar with doubled-district-name IDs (data inconsistency in regions.json)
        "BR_ARARIA_ARARIA","BR_ARWAL_ARWAL","BR_AURANGABAD_BR_AURANGABAD",
        "BR_BANKA_BANKA","BR_BEGUSARAI_BEGUSARAI","BR_BHABUA_BHABUA",
        "BR_BUXAR_BUXAR","BR_EAST_CHAMPARAN_EAST_CHAMPARAN",
        "BR_GOPALGANJ_GOPALGANJ","BR_JAMUI_JAMUI","BR_JEHANABAD_JEHANABAD",
        "BR_KAIMUR_KAIMUR","BR_KATIHAR_KATIHAR","BR_KHAGARIA_KHAGARIA",
        "BR_KISHANGANJ_KISHANGANJ","BR_LAKHISARAI_LAKHISARAI",
        "BR_MADHEPURA_MADHEPURA","BR_MADHUBANI_MADHUBANI","BR_MUNGER_MUNGER",
        "BR_NALANDA_NALANDA","BR_NAWADA_NAWADA","BR_PURNIA_PURNIA",
        "BR_ROHTAS_ROHTAS","BR_SARAN_SARAN","BR_SHEIKHPURA_SHEIKHPURA",
        "BR_SHEOHAR_SHEOHAR","BR_SITAMARHI_SITAMARHI","BR_SIWAN_SIWAN",
        "BR_SUPAUL_SUPAUL","BR_VAISHALI_VAISHALI","BR_WEST_CHAMPARAN_WEST_CHAMPARAN",
        # Jharkhand alternate
        "JH_SERAIKELA",
        # Odisha alternate
        "OD_BOLANGIR","OD_DEOGARH",
    ],
    "Central_Alt": [
        # MP alternate
        "MP_NARMADAPURAM",
    ],
    "Northeast_Alt": [
        # Arunachal alternate
        "AR_KAMLE",
        # Meghalaya alternate
        "ML_EAST_KHASI",
        # Sikkim new districts
        "SK_PAKYONG","SK_SORENG",
    ],
}


def _zone_suitability(zones: list) -> dict:
    """Build regional_suitability dict from zone list at 0.75 score."""
    result = {}
    for zone in zones:
        for region_id in ZONE_REGIONS.get(zone, []):
            result[region_id] = 0.75
    return result


# Comprehensive crop database with 50+ short-duration crops (15-90 days)
CROPS_DATA = [

    # Millets
    CropInfo(
        crop_id="BAJRA_01",
        common_name="Bajra (Pearl Millet)",
        scientific_name="Pennisetum glaucum",
        duration_days=75,
        duration_range=(70, 85),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=42,
        water_requirement_mm=400,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=8.0,
        suitable_soil_textures=["Sandy", "Sandy-Loam", "Loam"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["North", "West", "Central", "South"]),
            # Marathwada & Vidarbha — premium Bajra belt
            "MH_SOLAPUR": 0.92, "MH_LATUR": 0.90, "MH_CHHATRAPATI_SAMBHAJINAGAR": 0.88,
            "MH_AHMEDNAGAR": 0.86, "MH_PUNE": 0.85, "MH_SANGLI": 0.82,
            "MH_NASHIK": 0.80, "MH_SATARA": 0.78, "MH_JALGAON": 0.76,
            # North India Bajra belt
            "RJ_JODHPUR": 0.90, "RJ_JAIPUR": 0.88, "RJ_BIKANER": 0.85,
            "HR_HISAR": 0.88, "HR_BHIWANI": 0.85, "GJ_SABARKANTHA": 0.82,
        },
        successful_regions=["MH_SOLAPUR", "MH_LATUR", "MH_CHHATRAPATI_SAMBHAJINAGAR", "MH_AHMEDNAGAR"],
        seasons=["Kharif"],
        varieties=["GHB-558", "GHB-732", "ICMH-356"],
        typical_yield_kg_per_ha=1500,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="JOWAR_01",
        common_name="Jowar (Sorghum)",
        scientific_name="Sorghum bicolor",
        duration_days=85,
        duration_range=(75, 90),
        temp_min=18, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=450,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=8.5,
        suitable_soil_textures=["Clay", "Clay-Loam", "Loam"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "Low"},
        regional_suitability={
            **_zone_suitability(["West", "Central", "South", "North"]),
            # Premium Jowar belt — Marathwada & Deccan
            "MH_SOLAPUR": 0.92, "MH_LATUR": 0.92, "MH_CHHATRAPATI_SAMBHAJINAGAR": 0.90,
            "MH_AHMEDNAGAR": 0.90, "MH_PUNE": 0.88, "MH_SANGLI": 0.88,
            "MH_SATARA": 0.85, "MH_NASHIK": 0.85, "MH_JALGAON": 0.82,
            # Karnataka Jowar belt
            "KA_KALABURAGI": 0.88, "KA_VIJAYAPURA": 0.86, "KA_RAICHUR": 0.84,
            "KA_BALLARI": 0.82, "TS_NIZAMABAD": 0.80,
        },
        successful_regions=["MH_SOLAPUR", "MH_AHMEDNAGAR", "MH_CHHATRAPATI_SAMBHAJINAGAR", "MH_LATUR"],
        seasons=["Kharif", "Rabi"],
        varieties=["CSH-16", "CSV-15", "M-35-1"],
        typical_yield_kg_per_ha=1800,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="RAGI_01",
        common_name="Ragi (Finger Millet)",
        scientific_name="Eleusine coracana",
        duration_days=80,
        duration_range=(75, 85),
        temp_min=18, temp_optimal_min=22, temp_optimal_max=30, temp_max=35,
        water_requirement_mm=500,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=5.0, soil_ph_max=8.2,
        suitable_soil_textures=["Sandy-Loam", "Loam", "Clay-Loam"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["South", "West"]),
            # Karnataka & TN — primary Ragi belt
            "KA_BENGALURU": 0.90, "KA_MYSURU": 0.88, "KA_HASSAN": 0.88,
            "KA_TUMKURU": 0.86, "KA_DAVANAGERE": 0.85, "KA_CHITRADURGA": 0.84,
            "KA_MANDYA": 0.83, "TN_NAMAKKAL": 0.85, "TN_DHARMAPURI": 0.83,
            # Maharashtra secondary Ragi belt
            "MH_KOLHAPUR": 0.85, "MH_SATARA": 0.82, "MH_NASHIK": 0.80,
            "MH_PUNE": 0.76, "MH_SANGLI": 0.76, "MH_SOLAPUR": 0.70,
        },
        successful_regions=["KA_BENGALURU", "KA_MYSURU", "MH_KOLHAPUR", "MH_SATARA"],
        seasons=["Kharif"],
        varieties=["GPU-28", "ML-365", "VL-149"],
        typical_yield_kg_per_ha=1200,
        market_demand="Moderate"
    ),
    
    CropInfo(
        crop_id="FOXTAIL_01",
        common_name="Foxtail Millet",
        scientific_name="Setaria italica",
        duration_days=70,
        duration_range=(65, 75),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=32, temp_max=38,
        water_requirement_mm=350,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy", "Sandy-Loam", "Loam"],
        nutrient_requirements={"N": "Low", "P": "Low", "K": "Low"},
        regional_suitability={
            **_zone_suitability(["South", "Central", "West"]),
            # Andhra & Telangana — primary Foxtail belt
            "AP_ANANTAPUR": 0.88, "AP_KURNOOL": 0.85, "AP_PRAKASAM": 0.83,
            "TS_NIZAMABAD": 0.85, "TS_MEDAK": 0.82, "TS_KARIMNAGAR": 0.80,
            # Maharashtra secondary
            "MH_SOLAPUR": 0.78, "MH_LATUR": 0.76, "MH_CHHATRAPATI_SAMBHAJINAGAR": 0.76,
        },
        successful_regions=["AP_ANANTAPUR", "TS_NIZAMABAD", "MH_SOLAPUR", "MH_LATUR"],
        seasons=["Kharif"],
        varieties=["SiA-3156", "Prasad", "Lepakshi"],
        typical_yield_kg_per_ha=1000,
        market_demand="Moderate"
    ),
    
    # Pulses
    CropInfo(
        crop_id="MOONG_01",
        common_name="Green Gram (Moong)",
        scientific_name="Vigna radiata",
        duration_days=70,
        duration_range=(65, 75),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=350,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy-Loam", "Loam", "Clay-Loam"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["West", "North", "South", "Central", "East"]),
            # Marathwada — premium Moong belt
            "MH_SOLAPUR": 0.88, "MH_LATUR": 0.86, "MH_CHHATRAPATI_SAMBHAJINAGAR": 0.86,
            "MH_AHMEDNAGAR": 0.84, "MH_JALGAON": 0.83, "MH_PUNE": 0.82,
            "MH_NASHIK": 0.80, "MH_SANGLI": 0.80, "MH_SATARA": 0.78,
            # North India Moong belt
            "RJ_JAIPUR": 0.86, "RJ_AJMER": 0.84, "MP_INDORE": 0.82,
            "UP_AGRA": 0.80, "HR_HISAR": 0.80,
        },
        successful_regions=["MH_SOLAPUR", "MH_LATUR", "MH_CHHATRAPATI_SAMBHAJINAGAR", "MH_AHMEDNAGAR"],
        seasons=["Kharif", "Rabi"],
        varieties=["Pusa-105", "SML-668", "IPM-02-3"],
        typical_yield_kg_per_ha=800,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="URAD_01",
        common_name="Black Gram (Urad)",
        scientific_name="Vigna mungo",
        duration_days=75,
        duration_range=(70, 80),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=400,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.5, soil_ph_max=7.5,
        suitable_soil_textures=["Loam", "Clay-Loam", "Sandy-Loam"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["West", "South", "Central"]),
            # Marathwada Urad belt
            "MH_SOLAPUR": 0.82, "MH_LATUR": 0.82, "MH_CHHATRAPATI_SAMBHAJINAGAR": 0.80,
            "MH_AHMEDNAGAR": 0.79, "MH_PUNE": 0.78, "MH_NASHIK": 0.76,
            # Telangana & AP Urad belt
            "TS_WARANGAL": 0.82, "TS_KARIMNAGAR": 0.80, "AP_KURNOOL": 0.80,
            "AP_PRAKASAM": 0.78, "MP_INDORE": 0.78,
        },
        successful_regions=["MH_SOLAPUR", "MH_LATUR", "TS_WARANGAL", "MH_CHHATRAPATI_SAMBHAJINAGAR"],
        seasons=["Kharif"],
        varieties=["TAU-1", "PU-31", "LBG-752"],
        typical_yield_kg_per_ha=700,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="COWPEA_01",
        common_name="Cowpea",
        scientific_name="Vigna unguiculata",
        duration_days=75,
        duration_range=(70, 80),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=400,
        drought_tolerance="High",
        waterlogging_tolerance="Moderate",
        soil_ph_min=5.5, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy", "Sandy-Loam", "Loam", "Clay-Loam"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["South", "West", "Central"]),
            # South India primary Cowpea belt
            "KA_BENGALURU": 0.86, "KA_TUMKURU": 0.84, "TN_COIMBATORE": 0.86,
            "TN_SALEM": 0.84, "AP_ANANTAPUR": 0.84, "TS_ADILABAD": 0.82,
            # Maharashtra belt
            "MH_SOLAPUR": 0.84, "MH_LATUR": 0.83, "MH_CHHATRAPATI_SAMBHAJINAGAR": 0.82,
            "MH_JALGAON": 0.81, "MH_AHMEDNAGAR": 0.80,
        },
        successful_regions=["MH_SOLAPUR", "MH_LATUR", "MH_CHHATRAPATI_SAMBHAJINAGAR", "MH_JALGAON"],
        seasons=["Kharif"],
        varieties=["Pusa-578", "Arka-Garima", "Kashi-Kanchan"],
        typical_yield_kg_per_ha=900,
        market_demand="Moderate"
    ),
    
    CropInfo(
        crop_id="GUAR_01",
        common_name="Cluster Bean (Guar)",
        scientific_name="Cyamopsis tetragonoloba",
        duration_days=85,
        duration_range=(80, 90),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=42,
        water_requirement_mm=350,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=8.5,
        suitable_soil_textures=["Sandy", "Sandy-Loam", "Loam"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["North", "West"]),
            # Rajasthan — premier Guar belt
            "RJ_JODHPUR": 0.92, "RJ_BIKANER": 0.90, "RJ_JAIPUR": 0.88,
            "RJ_NAGAUR": 0.88, "RJ_BARMER": 0.86, "RJ_CHURU": 0.85,
            "RJ_JAISALMER": 0.84,
            # Haryana secondary Guar belt
            "HR_HISAR": 0.86, "HR_SIRSA": 0.84, "HR_FATEHABAD": 0.82,
            # Maharashtra drier zones
            "MH_SOLAPUR": 0.85, "MH_LATUR": 0.85, "MH_CHHATRAPATI_SAMBHAJINAGAR": 0.82,
            "MH_AHMEDNAGAR": 0.78,
        },
        successful_regions=["RJ_JODHPUR", "RJ_BIKANER", "MH_SOLAPUR", "MH_LATUR"],
        seasons=["Kharif"],
        varieties=["RGC-1066", "HG-563", "Pusa-Navbahar"],
        typical_yield_kg_per_ha=1200,
        market_demand="Moderate"
    ),
    
    # Oilseeds
    CropInfo(
        crop_id="SESAME_01",
        common_name="Sesame (Til)",
        scientific_name="Sesamum indicum",
        duration_days=85,
        duration_range=(80, 90),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=400,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=8.0,
        suitable_soil_textures=["Sandy-Loam", "Loam", "Clay-Loam"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["West", "South", "Central"]),
            # Gujarat & Rajasthan — primary Sesame belt
            "GJ_JUNAGADH": 0.88, "GJ_RAJKOT": 0.86, "GJ_SURENDRANAGAR": 0.85,
            "RJ_BARMER": 0.86, "RJ_JODHPUR": 0.84,
            # Andhra & Maharashtra secondary
            "AP_GUNTUR": 0.84, "AP_KURNOOL": 0.82, "TS_NIZAMABAD": 0.80,
            "MH_SOLAPUR": 0.80, "MH_LATUR": 0.79, "MH_CHHATRAPATI_SAMBHAJINAGAR": 0.78,
            "MH_AHMEDNAGAR": 0.76,
        },
        successful_regions=["GJ_JUNAGADH", "MH_SOLAPUR", "MH_LATUR", "MH_CHHATRAPATI_SAMBHAJINAGAR"],
        seasons=["Kharif", "Rabi"],
        varieties=["Phule-Til", "N-32", "TKG-22"],
        typical_yield_kg_per_ha=600,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="SUNFLOWER_01",
        common_name="Sunflower (Short-duration)",
        scientific_name="Helianthus annuus",
        duration_days=85,
        duration_range=(80, 90),
        temp_min=15, temp_optimal_min=20, temp_optimal_max=30, temp_max=35,
        water_requirement_mm=500,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam", "Clay-Loam", "Sandy-Loam"],
        nutrient_requirements={"N": "High", "P": "High", "K": "High"},
        regional_suitability={
            **_zone_suitability(["South", "West", "Central"]),
            # Karnataka — premier Sunflower belt
            "KA_BALLARI": 0.90, "KA_RAICHUR": 0.88, "KA_KOPPAL": 0.88,
            "KA_DAVANAGERE": 0.87, "KA_HAVERI": 0.86, "KA_GADAG": 0.85,
            "KA_DHARWAD": 0.84, "KA_CHITRADURGA": 0.84,
            # Maharashtra secondary
            "MH_NASHIK": 0.83, "MH_KOLHAPUR": 0.82, "MH_SATARA": 0.81,
            "MH_PUNE": 0.80, "MH_AHMEDNAGAR": 0.80, "MH_JALGAON": 0.79,
        },
        successful_regions=["KA_BALLARI", "KA_DHARWAD", "MH_NASHIK", "MH_KOLHAPUR"],
        seasons=["Kharif", "Rabi"],
        varieties=["KBSH-44", "Phule-Bhaskar", "DRSH-1"],
        typical_yield_kg_per_ha=1500,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="SOYBEAN_01",
        common_name="Soybean (Early variety)",
        scientific_name="Glycine max",
        duration_days=85,
        duration_range=(80, 90),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=30, temp_max=35,
        water_requirement_mm=500,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Moderate",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam", "Clay-Loam", "Sandy-Loam"],
        nutrient_requirements={"N": "Low", "P": "High", "K": "High"},
        regional_suitability={
            **_zone_suitability(["West", "Central"]),
            # Madhya Pradesh — India's Soybean capital
            "MP_INDORE": 0.92, "MP_UJJAIN": 0.90, "MP_DEWAS": 0.90,
            "MP_SEHORE": 0.88, "MP_BHOPAL": 0.87, "MP_HOSHANGABAD": 0.87,
            "MP_RAISEN": 0.86, "MP_SHAJAPUR": 0.85, "MP_MANDSAUR": 0.84,
            # Maharashtra Soybean belt
            "MH_JALGAON": 0.84, "MH_NASHIK": 0.82, "MH_KOLHAPUR": 0.82,
            "MH_AHMEDNAGAR": 0.80, "MH_SATARA": 0.80, "MH_PUNE": 0.79,
        },
        successful_regions=["MP_INDORE", "MP_UJJAIN", "MH_JALGAON", "MH_NASHIK"],
        seasons=["Kharif"],
        varieties=["JS-335", "MAUS-71", "Phule-Kalyani"],
        typical_yield_kg_per_ha=1800,
        market_demand="High"
    ),
    
    # Vegetables
    CropInfo(
        crop_id="TOMATO_01",
        common_name="Tomato (Short-duration)",
        scientific_name="Solanum lycopersicum",
        duration_days=75,
        duration_range=(70, 80),
        temp_min=15, temp_optimal_min=20, temp_optimal_max=28, temp_max=35,
        water_requirement_mm=600,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.0,
        suitable_soil_textures=["Loam", "Sandy-Loam", "Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "High", "K": "High"},
        regional_suitability={
            **_zone_suitability(["South", "West", "North", "Central", "East"]),
            # Maharashtra premier Tomato belt
            "MH_NASHIK": 0.92, "MH_PUNE": 0.88, "MH_SATARA": 0.86,
            "MH_KOLHAPUR": 0.88, "MH_AHMEDNAGAR": 0.84, "MH_JALGAON": 0.83,
            "MH_SANGLI": 0.80,
            # South India Tomato belt
            "KA_KOLAR": 0.90, "KA_CHIKKABALLAPURA": 0.88, "KA_BENGALURU_RURAL": 0.87,
            "AP_CHITTOOR": 0.88, "AP_KURNOOL": 0.85,
            # North India Tomato belt
            "HR_KARNAL": 0.85, "UP_AGRA": 0.82, "PB_LUDHIANA": 0.82,
        },
        successful_regions=["MH_NASHIK", "MH_PUNE", "MH_KOLHAPUR", "KA_KOLAR"],
        seasons=["Kharif", "Rabi"],
        varieties=["Abhinav", "Pusa-Ruby", "Arka-Vikas"],
        typical_yield_kg_per_ha=25000,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="BRINJAL_01",
        common_name="Brinjal (Eggplant)",
        scientific_name="Solanum melongena",
        duration_days=80,
        duration_range=(75, 85),
        temp_min=18, temp_optimal_min=22, temp_optimal_max=30, temp_max=38,
        water_requirement_mm=550,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=7.0,
        suitable_soil_textures=["Loam", "Sandy-Loam", "Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "High", "K": "High"},
        regional_suitability={
            **_zone_suitability(["South", "West", "East", "Central"]),
            # Maharashtra Brinjal belt
            "MH_NASHIK": 0.88, "MH_KOLHAPUR": 0.87, "MH_SATARA": 0.85,
            "MH_PUNE": 0.84, "MH_JALGAON": 0.83, "MH_AHMEDNAGAR": 0.81,
            "MH_SANGLI": 0.81,
            # South India Brinjal belt
            "KA_BENGALURU": 0.88, "KA_TUMKURU": 0.85, "AP_GUNTUR": 0.86,
            "TN_COIMBATORE": 0.85, "WB_BURDWAN": 0.84, "WB_NADIA": 0.82,
        },
        successful_regions=["MH_NASHIK", "MH_KOLHAPUR", "KA_BENGALURU", "AP_GUNTUR"],
        seasons=["Kharif", "Rabi"],
        varieties=["Pusa-Purple-Long", "Arka-Shirish", "Phule-Prakash"],
        typical_yield_kg_per_ha=20000,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="OKRA_01",
        common_name="Okra (Bhindi)",
        scientific_name="Abelmoschus esculentus",
        duration_days=70,
        duration_range=(65, 75),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=500,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam", "Sandy-Loam", "Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["South", "West", "Central", "North", "East"]),
            # Maharashtra Okra belt
            "MH_NASHIK": 0.85, "MH_JALGAON": 0.84, "MH_PUNE": 0.82,
            "MH_KOLHAPUR": 0.83, "MH_AHMEDNAGAR": 0.81, "MH_SATARA": 0.81,
            "MH_SOLAPUR": 0.79,
            # South India Okra belt
            "KA_BENGALURU": 0.86, "KA_TUMKURU": 0.84, "AP_GUNTUR": 0.85,
            "TN_COIMBATORE": 0.86, "TN_ERODE": 0.84,
            # North India Okra belt
            "UP_LUCKNOW": 0.82, "UP_AGRA": 0.80, "HR_KARNAL": 0.82,
        },
        successful_regions=["MH_NASHIK", "MH_JALGAON", "MH_KOLHAPUR", "KA_BENGALURU"],
        seasons=["Kharif", "Rabi"],
        varieties=["Pusa-Sawani", "Arka-Anamika", "Phule-Utkarsha"],
        typical_yield_kg_per_ha=12000,
        market_demand="High"
    ),
    
    CropInfo(
        crop_id="BOTTLEGOURD_01",
        common_name="Bottle Gourd (Lauki)",
        scientific_name="Lagenaria siceraria",
        duration_days=75,
        duration_range=(70, 80),
        temp_min=18, temp_optimal_min=22, temp_optimal_max=32, temp_max=38,
        water_requirement_mm=550,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Moderate",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam", "Sandy-Loam", "Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "Medium"},
        regional_suitability={
            **_zone_suitability(["North", "West", "Central", "East", "South"]),
            # Maharashtra Bottle Gourd belt
            "MH_NASHIK": 0.83, "MH_JALGAON": 0.82, "MH_PUNE": 0.80,
            "MH_KOLHAPUR": 0.82, "MH_AHMEDNAGAR": 0.79, "MH_SATARA": 0.80,
            # North India Bottle Gourd belt
            "UP_LUCKNOW": 0.84, "UP_AGRA": 0.82, "UP_VARANASI": 0.82,
            "HR_KARNAL": 0.83, "HR_KURUKSHETRA": 0.82,
            "PB_LUDHIANA": 0.82, "RJ_JAIPUR": 0.80,
        },
        successful_regions=["MH_NASHIK", "MH_JALGAON", "UP_LUCKNOW", "HR_KARNAL"],
        seasons=["Kharif", "Rabi"],
        varieties=["Pusa-Summer-Prolific-Long", "Arka-Bahar", "Samrat"],
        typical_yield_kg_per_ha=18000,
        market_demand="Moderate"
    ),

    # ── SHORT-CYCLE CROPS (15–90 days) ─────────────────────────────────────

    # ── Leafy Greens (15–45 days) ──
    CropInfo(
        crop_id="SPINACH_01",
        common_name="Spinach",
        scientific_name="Spinacia oleracea",
        duration_days=30,
        duration_range=(20, 40),
        temp_min=5, temp_optimal_min=15, temp_optimal_max=20, temp_max=30,
        water_requirement_mm=300,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam", "Clay", "Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South"]),
        successful_regions=["UP_LUCKNOW","PB_LUDHIANA","HR_AMBALA","RJ_JAIPUR"],
        seasons=["Rabi","Kharif","Zaid"],
        varieties=["Pusa-Bharati","Palak-No-1","Virginia-Savoy"],
        typical_yield_kg_per_ha=9000,
        market_demand="High",
        growing_tip="Sow seeds 1–2 cm deep in rows 20 cm apart. Irrigate every 5–7 days. Best in cool weather (15–20°C)."
    ),

    CropInfo(
        crop_id="FENUGREEK_01",
        common_name="Fenugreek (Methi)",
        scientific_name="Trigonella foenum-graecum",
        duration_days=28,
        duration_range=(20, 35),
        temp_min=10, temp_optimal_min=15, temp_optimal_max=25, temp_max=35,
        water_requirement_mm=200,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=8.0,
        suitable_soil_textures=["Loam","Sandy","Clay","Sandy-Loam"],
        nutrient_requirements={"N": "Low", "P": "Low", "K": "Low"},
        regional_suitability=_zone_suitability(["North","Central","West","South"]),
        successful_regions=["RJ_JAIPUR","GJ_AHMEDABAD","MP_INDORE","HR_HISAR"],
        seasons=["Rabi","Kharif","Zaid"],
        varieties=["Pusa-Early-Bunching","Kasuri-Methi","RMT-1"],
        typical_yield_kg_per_ha=7000,
        market_demand="High",
        growing_tip="Broadcast seeds and mix lightly with soil. First cutting in 20–25 days. Needs less water than most greens."
    ),

    CropInfo(
        crop_id="CORIANDER_01",
        common_name="Coriander (Dhaniya)",
        scientific_name="Coriandrum sativum",
        duration_days=35,
        duration_range=(25, 45),
        temp_min=10, temp_optimal_min=15, temp_optimal_max=25, temp_max=30,
        water_requirement_mm=200,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=8.0,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "Low", "P": "Low", "K": "Low"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["RJ_JAIPUR","MP_INDORE","GJ_AHMEDABAD","AP_GUNTUR"],
        seasons=["Rabi","Kharif","Zaid"],
        varieties=["CS-6","RCr-41","Pant-Haritima"],
        typical_yield_kg_per_ha=6000,
        market_demand="High",
        growing_tip="Split seeds before sowing for better germination. Sow in rows 20 cm apart. Avoid waterlogging."
    ),

    CropInfo(
        crop_id="AMARANTH_01",
        common_name="Amaranth (Chaulai)",
        scientific_name="Amaranthus tricolor",
        duration_days=32,
        duration_range=(20, 45),
        temp_min=15, temp_optimal_min=25, temp_optimal_max=35, temp_max=42,
        water_requirement_mm=200,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=8.0,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay"],
        nutrient_requirements={"N": "Low", "P": "Low", "K": "Low"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["UP_LUCKNOW","MP_BHOPAL","KA_BENGALURU","WB_KOLKATA"],
        seasons=["Kharif","Zaid","Rabi"],
        varieties=["Pusa-Kiran","Pusa-Lal-Chaulai","Suvarna"],
        typical_yield_kg_per_ha=10000,
        market_demand="Moderate",
        growing_tip="Very hardy crop. Broadcast seeds thinly. Grows well in hot weather (25–35°C). First harvest in 20 days."
    ),

    CropInfo(
        crop_id="MUSTARD_GREENS_01",
        common_name="Mustard Greens (Sarson Saag)",
        scientific_name="Brassica juncea",
        duration_days=32,
        duration_range=(25, 40),
        temp_min=5, temp_optimal_min=10, temp_optimal_max=25, temp_max=30,
        water_requirement_mm=350,
        drought_tolerance="Low",
        waterlogging_tolerance="Moderate",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Clay","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","East"]),
        successful_regions=["PB_LUDHIANA","HR_AMBALA","UP_LUCKNOW","BR_PATNA"],
        seasons=["Rabi"],
        varieties=["Pusa-Saag-1","LC-1","PBM-1"],
        typical_yield_kg_per_ha=12000,
        market_demand="High",
        growing_tip="Sow in October–November. Grows best at 10–25°C. Very popular in Punjab, Haryana, UP."
    ),

    CropInfo(
        crop_id="LETTUCE_01",
        common_name="Lettuce",
        scientific_name="Lactuca sativa",
        duration_days=38,
        duration_range=(30, 45),
        temp_min=7, temp_optimal_min=12, temp_optimal_max=20, temp_max=28,
        water_requirement_mm=300,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.0,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","South","West","Northeast"]),
        successful_regions=["HP_SHIMLA","UK_DEHRADUN","KA_BENGALURU","TN_OOTY"],
        seasons=["Rabi","Zaid"],
        varieties=["Iceberg","Romaine","Pusa-Rohini"],
        typical_yield_kg_per_ha=8000,
        market_demand="Moderate",
        growing_tip="Transplant seedlings 25 cm apart. Prefers cool weather. Harvest outer leaves for continuous yield."
    ),

    # ── Vegetables (25–90 days) ──
    CropInfo(
        crop_id="RADISH_01",
        common_name="Radish (Mooli)",
        scientific_name="Raphanus sativus",
        duration_days=35,
        duration_range=(25, 45),
        temp_min=5, temp_optimal_min=10, temp_optimal_max=22, temp_max=30,
        water_requirement_mm=300,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy","Sandy-Loam","Loam"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "High"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South"]),
        successful_regions=["UP_LUCKNOW","PB_LUDHIANA","HR_AMBALA","MP_INDORE"],
        seasons=["Rabi","Kharif","Zaid"],
        varieties=["Pusa-Desi","Pusa-Chetki","Japanese-White"],
        typical_yield_kg_per_ha=22000,
        market_demand="High",
        growing_tip="Sow seeds 1 cm deep in rows 30 cm apart. Thin to 8 cm spacing. Harvest when roots are 15–20 cm long."
    ),

    CropInfo(
        crop_id="SPRING_ONION_01",
        common_name="Green Onion (Spring Onion)",
        scientific_name="Allium fistulosum",
        duration_days=40,
        duration_range=(30, 50),
        temp_min=8, temp_optimal_min=15, temp_optimal_max=25, temp_max=35,
        water_requirement_mm=350,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South"]),
        successful_regions=["MH_NASHIK","GJ_AHMEDABAD","UP_LUCKNOW","WB_NADIA"],
        seasons=["Rabi","Kharif","Zaid"],
        varieties=["Improved-Japanese-Bunching","White-Lisboa","Pusa-White-Round"],
        typical_yield_kg_per_ha=13000,
        market_demand="High",
        growing_tip="Plant sets or seedlings 10 cm apart. Harvest when tops are 15–20 cm tall. Grows in most Indian regions."
    ),

    CropInfo(
        crop_id="CARROT_01",
        common_name="Carrot (Gajar)",
        scientific_name="Daucus carota",
        duration_days=75,
        duration_range=(60, 90),
        temp_min=7, temp_optimal_min=15, temp_optimal_max=22, temp_max=30,
        water_requirement_mm=400,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy","Sandy-Loam","Loam"],
        nutrient_requirements={"N": "Medium", "P": "High", "K": "High"},
        regional_suitability=_zone_suitability(["North","Central","East","South"]),
        successful_regions=["PB_LUDHIANA","HP_SHIMLA","UP_LUCKNOW","KA_BENGALURU"],
        seasons=["Rabi"],
        varieties=["Pusa-Kesar","Pusa-Meghali","Nantes"],
        typical_yield_kg_per_ha=25000,
        market_demand="High",
        growing_tip="Sow seeds in well-drained, deep, loose soil. Thin seedlings to 5 cm apart. Avoid heavy clay soil."
    ),

    CropInfo(
        crop_id="TURNIP_01",
        common_name="Turnip (Shalgam)",
        scientific_name="Brassica rapa",
        duration_days=60,
        duration_range=(45, 75),
        temp_min=5, temp_optimal_min=10, temp_optimal_max=20, temp_max=28,
        water_requirement_mm=350,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Clay"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","East"]),
        successful_regions=["PB_LUDHIANA","UP_LUCKNOW","HR_AMBALA","BR_PATNA"],
        seasons=["Rabi"],
        varieties=["Purple-Top-White-Globe","Snowball","PTSWG"],
        typical_yield_kg_per_ha=17000,
        market_demand="Moderate",
        growing_tip="Sow in October–November. Harvest when roots are 5–8 cm in diameter. Grows best in cool spells."
    ),

    CropInfo(
        crop_id="BEETROOT_01",
        common_name="Beetroot (Chukandar)",
        scientific_name="Beta vulgaris",
        duration_days=62,
        duration_range=(55, 70),
        temp_min=8, temp_optimal_min=15, temp_optimal_max=25, temp_max=32,
        water_requirement_mm=400,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.5, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "High"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South"]),
        successful_regions=["MH_PUNE","KA_BENGALURU","HP_SHIMLA","UP_LUCKNOW"],
        seasons=["Rabi","Zaid"],
        varieties=["Detroit-Dark-Red","Crimson-Globe","Pusa-Madhuram"],
        typical_yield_kg_per_ha=22000,
        market_demand="Moderate",
        growing_tip="Soak seeds overnight before sowing. Thin to 10 cm apart. Prefers slightly alkaline soil (pH 6.5–7.5)."
    ),

    CropInfo(
        crop_id="CUCUMBER_01",
        common_name="Cucumber (Kheera)",
        scientific_name="Cucumis sativus",
        duration_days=50,
        duration_range=(40, 60),
        temp_min=18, temp_optimal_min=20, temp_optimal_max=30, temp_max=38,
        water_requirement_mm=600,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "High", "K": "High"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["UP_LUCKNOW","MH_PUNE","WB_KOLKATA","KA_BENGALURU"],
        seasons=["Kharif","Zaid"],
        varieties=["Pusa-Uday","Pant-Shankar-Kheera-1","Arka-Sheetal"],
        typical_yield_kg_per_ha=17000,
        market_demand="High",
        growing_tip="Sow 2–3 seeds per hill, 60 cm apart. Needs warm weather (20–30°C). Provide support for vine growth."
    ),

    CropInfo(
        crop_id="RIDGE_GOURD_01",
        common_name="Ridge Gourd (Torai)",
        scientific_name="Luffa acutangula",
        duration_days=55,
        duration_range=(45, 65),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=450,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["WB_KOLKATA","UP_LUCKNOW","MH_PUNE","KA_BENGALURU"],
        seasons=["Kharif","Zaid"],
        varieties=["Pusa-Nasdar","Arka-Sumeet","CO-1"],
        typical_yield_kg_per_ha=13000,
        market_demand="Moderate",
        growing_tip="Sow on raised beds with trellis support. Harvest young fruits when ridges are distinct for best taste."
    ),

    CropInfo(
        crop_id="BITTER_GOURD_01",
        common_name="Bitter Gourd (Karela)",
        scientific_name="Momordica charantia",
        duration_days=60,
        duration_range=(50, 70),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=450,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "Medium", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["UP_LUCKNOW","MH_PUNE","WB_KOLKATA","AP_GUNTUR"],
        seasons=["Kharif","Zaid"],
        varieties=["Pusa-Do-Mausami","Arka-Harit","Priya"],
        typical_yield_kg_per_ha=11000,
        market_demand="High",
        growing_tip="Soak seeds 24 hours before sowing. Grow on trellis. Harvest when fruits are green and firm."
    ),

    CropInfo(
        crop_id="FRENCH_BEANS_01",
        common_name="French Beans",
        scientific_name="Phaseolus vulgaris",
        duration_days=52,
        duration_range=(45, 60),
        temp_min=12, temp_optimal_min=15, temp_optimal_max=25, temp_max=32,
        water_requirement_mm=400,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam"],
        nutrient_requirements={"N": "Low", "P": "High", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","South","West","Northeast"]),
        successful_regions=["HP_SHIMLA","UK_DEHRADUN","KA_BENGALURU","MH_PUNE"],
        seasons=["Kharif","Rabi"],
        varieties=["Contender","Arka-Komal","Pusa-Parvati"],
        typical_yield_kg_per_ha=9000,
        market_demand="High",
        growing_tip="Sow seeds 5 cm deep, 10 cm apart in rows 45 cm apart. Harvest when pods snap cleanly."
    ),

    CropInfo(
        crop_id="CLUSTER_BEANS_01",
        common_name="Cluster Beans (Gwar — vegetable)",
        scientific_name="Cyamopsis tetragonoloba",
        duration_days=52,
        duration_range=(45, 60),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=42,
        water_requirement_mm=250,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=8.5,
        suitable_soil_textures=["Sandy","Sandy-Loam","Loam"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","West","Central"]),
        successful_regions=["RJ_JAIPUR","GJ_AHMEDABAD","HR_HISAR","UP_AGRA"],
        seasons=["Kharif"],
        varieties=["Pusa-Navbahar","HG-563","RGC-936"],
        typical_yield_kg_per_ha=6000,
        market_demand="Moderate",
        growing_tip="Drought-resistant crop. Ideal for Rajasthan, Gujarat, Haryana. Sow with monsoon onset."
    ),

    CropInfo(
        crop_id="CAPSICUM_01",
        common_name="Capsicum (Bell Pepper)",
        scientific_name="Capsicum annuum var. grossum",
        duration_days=75,
        duration_range=(60, 90),
        temp_min=15, temp_optimal_min=18, temp_optimal_max=28, temp_max=35,
        water_requirement_mm=500,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.0,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "High", "K": "High"},
        regional_suitability=_zone_suitability(["North","South","West","Central"]),
        successful_regions=["HP_SHIMLA","KA_BENGALURU","MH_PUNE","TN_COIMBATORE"],
        seasons=["Kharif","Rabi"],
        varieties=["California-Wonder","Arka-Gaurav","Bombay"],
        typical_yield_kg_per_ha=20000,
        market_demand="High",
        growing_tip="Transplant 5–6 week seedlings. Mulch to retain moisture. Harvest when fruits are firm and full-sized."
    ),

    CropInfo(
        crop_id="GREEN_CHILLI_01",
        common_name="Green Chilli",
        scientific_name="Capsicum annuum",
        duration_days=62,
        duration_range=(50, 75),
        temp_min=15, temp_optimal_min=20, temp_optimal_max=30, temp_max=38,
        water_requirement_mm=450,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "High"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["AP_GUNTUR","TS_HYDERABAD","KA_BENGALURU","MH_NASHIK"],
        seasons=["Kharif","Rabi","Zaid"],
        varieties=["Pusa-Jwala","NP-46","Arka-Meghna"],
        typical_yield_kg_per_ha=11000,
        market_demand="High",
        growing_tip="Transplant 40-day seedlings. Space 45×30 cm. Multiple pickings possible. Major crop in Andhra, Telangana."
    ),

    CropInfo(
        crop_id="SPONGE_GOURD_01",
        common_name="Sponge Gourd (Nenua)",
        scientific_name="Luffa cylindrica",
        duration_days=52,
        duration_range=(45, 60),
        temp_min=18, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=450,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South"]),
        successful_regions=["UP_LUCKNOW","WB_KOLKATA","MH_PUNE","KA_BENGALURU"],
        seasons=["Kharif","Zaid"],
        varieties=["Pusa-Sneha","Arka-Sumeet","CO-1"],
        typical_yield_kg_per_ha=12000,
        market_demand="Moderate",
        growing_tip="Sow on raised beds. Provide trellis support. Harvest young tender fruits. Very popular summer vegetable."
    ),

    CropInfo(
        crop_id="PUMPKIN_01",
        common_name="Pumpkin (Kaddu)",
        scientific_name="Cucurbita moschata",
        duration_days=82,
        duration_range=(75, 90),
        temp_min=18, temp_optimal_min=22, temp_optimal_max=32, temp_max=38,
        water_requirement_mm=500,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "High", "K": "High"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South"]),
        successful_regions=["UP_LUCKNOW","WB_KOLKATA","MH_PUNE","OD_BHUBANESWAR"],
        seasons=["Kharif","Zaid"],
        varieties=["Pusa-Alankar","Arka-Suryamukhi","IARI-Selection"],
        typical_yield_kg_per_ha=25000,
        market_demand="Moderate",
        growing_tip="Sow on raised beds, 2 m apart. Train vines. Harvest when fruit sounds hollow when tapped."
    ),

    # ── Pulses & Legumes (45–90 days) ──
    CropInfo(
        crop_id="MOONG_DAL_01",
        common_name="Moong Dal (Green Gram — quick)",
        scientific_name="Vigna radiata",
        duration_days=67,
        duration_range=(60, 75),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=350,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South"]),
        successful_regions=["UP_LUCKNOW","MP_BHOPAL","RJ_JAIPUR","AP_GUNTUR"],
        seasons=["Kharif","Zaid"],
        varieties=["Pusa-105","SML-668","IPM-02-3"],
        typical_yield_kg_per_ha=1000,
        market_demand="High",
        growing_tip="Sow at 30×10 cm spacing. Drought-tolerant. Fixes nitrogen in soil. Excellent intercrop."
    ),

    CropInfo(
        crop_id="URAD_DAL_01",
        common_name="Urad Dal (Black Gram — quick)",
        scientific_name="Vigna mungo",
        duration_days=77,
        duration_range=(65, 90),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=350,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.5, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Clay-Loam","Sandy-Loam","Clay"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","East","South"]),
        successful_regions=["UP_LUCKNOW","MP_BHOPAL","BR_PATNA","AP_GUNTUR"],
        seasons=["Kharif"],
        varieties=["TAU-1","PU-31","LBG-752"],
        typical_yield_kg_per_ha=750,
        market_demand="High",
        growing_tip="Sow with onset of monsoon. Avoid waterlogged fields. Good rotation crop with cereals."
    ),

    CropInfo(
        crop_id="COWPEA_VEG_01",
        common_name="Cowpea (Lobia — vegetable pods)",
        scientific_name="Vigna unguiculata",
        duration_days=75,
        duration_range=(60, 90),
        temp_min=20, temp_optimal_min=25, temp_optimal_max=35, temp_max=40,
        water_requirement_mm=350,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy","Loam","Sandy-Loam"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["RJ_JAIPUR","GJ_AHMEDABAD","UP_LUCKNOW","WB_KOLKATA"],
        seasons=["Kharif","Zaid"],
        varieties=["Pusa-578","Arka-Garima","Kashi-Kanchan"],
        typical_yield_kg_per_ha=1200,
        market_demand="Moderate",
        growing_tip="Heat and drought tolerant. Grows in poor soils. Green pods ready in 60 days, dry grain in 90 days."
    ),

    CropInfo(
        crop_id="MASOOR_01",
        common_name="Masoor (Red Lentil)",
        scientific_name="Lens culinaris",
        duration_days=85,
        duration_range=(80, 90),
        temp_min=7, temp_optimal_min=15, temp_optimal_max=25, temp_max=30,
        water_requirement_mm=250,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Clay","Clay-Loam"],
        nutrient_requirements={"N": "Low", "P": "Medium", "K": "Low"},
        regional_suitability=_zone_suitability(["North","Central","East"]),
        successful_regions=["UP_VARANASI","MP_BHOPAL","BR_PATNA","UP_LUCKNOW"],
        seasons=["Rabi"],
        varieties=["Pant-L-406","IPL-81","K-75"],
        typical_yield_kg_per_ha=1000,
        market_demand="High",
        growing_tip="Sow in October–November. Needs cool dry weather. Minimal irrigation needed. Good for UP, MP, Bihar."
    ),

    # ── Quick Herbs & Spices ──
    CropInfo(
        crop_id="MINT_01",
        common_name="Mint (Pudina)",
        scientific_name="Mentha spicata",
        duration_days=22,
        duration_range=(15, 30),
        temp_min=10, temp_optimal_min=18, temp_optimal_max=28, temp_max=35,
        water_requirement_mm=600,
        drought_tolerance="Low",
        waterlogging_tolerance="Moderate",
        soil_ph_min=6.0, soil_ph_max=7.0,
        suitable_soil_textures=["Loam","Sandy","Clay","Sandy-Loam"],
        nutrient_requirements={"N": "High", "P": "Medium", "K": "Medium"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["UP_LUCKNOW","PB_LUDHIANA","HR_AMBALA","MP_INDORE"],
        seasons=["Kharif","Rabi","Zaid"],
        varieties=["Kosi","Saksham","MAS-1"],
        typical_yield_kg_per_ha=22000,
        market_demand="High",
        growing_tip="Plant root cuttings or runners. Spreads quickly. Keep soil moist. Multiple harvests per season."
    ),

    CropInfo(
        crop_id="DILL_01",
        common_name="Dill (Sowa / Suva)",
        scientific_name="Anethum graveolens",
        duration_days=32,
        duration_range=(25, 40),
        temp_min=8, temp_optimal_min=15, temp_optimal_max=25, temp_max=32,
        water_requirement_mm=200,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=5.8, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy","Loam","Sandy-Loam"],
        nutrient_requirements={"N": "Low", "P": "Low", "K": "Low"},
        regional_suitability=_zone_suitability(["North","Central","West","South"]),
        successful_regions=["GJ_AHMEDABAD","RJ_JAIPUR","MP_INDORE","HR_KARNAL"],
        seasons=["Rabi","Zaid"],
        varieties=["Sowa","Suva","Local"],
        typical_yield_kg_per_ha=6000,
        market_demand="Moderate",
        growing_tip="Broadcast seeds and cover lightly. Grows fast in cool weather. Popular in Gujarat, Rajasthan."
    ),

    CropInfo(
        crop_id="CURRY_LEAVES_01",
        common_name="Curry Leaves (Kari Patta)",
        scientific_name="Murraya koenigii",
        duration_days=45,
        duration_range=(30, 60),
        temp_min=15, temp_optimal_min=20, temp_optimal_max=30, temp_max=38,
        water_requirement_mm=250,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy","Sandy-Loam","Loam"],
        nutrient_requirements={"N": "Low", "P": "Low", "K": "Low"},
        regional_suitability=_zone_suitability(["South","West","Central","East"]),
        successful_regions=["TN_CHENNAI","KA_BENGALURU","MH_PUNE","AP_GUNTUR"],
        seasons=["Kharif","Rabi","Zaid"],
        varieties=["Gamthi","Senkaambu","Regular"],
        typical_yield_kg_per_ha=5000,
        market_demand="High",
        growing_tip="Plant stem cuttings. Once established, harvest leaves repeatedly. Thrives in South India's climate."
    ),

    # ── Quick Root & Tuber ──
    CropInfo(
        crop_id="BABY_POTATO_01",
        common_name="Baby Potato (Early variety)",
        scientific_name="Solanum tuberosum",
        duration_days=75,
        duration_range=(60, 90),
        temp_min=7, temp_optimal_min=15, temp_optimal_max=22, temp_max=28,
        water_requirement_mm=500,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=7.0,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam"],
        nutrient_requirements={"N": "High", "P": "High", "K": "High"},
        regional_suitability=_zone_suitability(["North","Central","East","Northeast"]),
        successful_regions=["UP_AGRA","PB_JALANDHAR","HP_SHIMLA","WB_BURDWAN"],
        seasons=["Rabi"],
        varieties=["Kufri-Pukhraj","Kufri-Jyoti","Kufri-Chipsona"],
        typical_yield_kg_per_ha=20000,
        market_demand="High",
        growing_tip="Plant Kufri Pukhraj or Kufri Jyoti for early harvest. Ridge planting. Harvest when plant starts yellowing."
    ),

    # ── Quick Cereals & Others ──
    CropInfo(
        crop_id="BABY_CORN_01",
        common_name="Baby Corn",
        scientific_name="Zea mays (baby corn type)",
        duration_days=57,
        duration_range=(50, 65),
        temp_min=18, temp_optimal_min=22, temp_optimal_max=32, temp_max=40,
        water_requirement_mm=450,
        drought_tolerance="Moderate",
        waterlogging_tolerance="Low",
        soil_ph_min=6.0, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy-Loam","Clay-Loam"],
        nutrient_requirements={"N": "High", "P": "High", "K": "High"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South"]),
        successful_regions=["PB_LUDHIANA","HR_AMBALA","UP_LUCKNOW","KA_BENGALURU"],
        seasons=["Kharif","Zaid","Rabi"],
        varieties=["HM-4","VL-42","Pusa-HM-4"],
        typical_yield_kg_per_ha=8000,
        market_demand="High",
        growing_tip="Use high-density planting (75,000 plants/ha). Harvest within 1–3 days of silk emergence. High market value."
    ),

    CropInfo(
        crop_id="MICROGREENS_01",
        common_name="Microgreens / Sprouts",
        scientific_name="Various (tray-grown)",
        duration_days=11,
        duration_range=(7, 15),
        temp_min=15, temp_optimal_min=18, temp_optimal_max=24, temp_max=30,
        water_requirement_mm=50,
        drought_tolerance="Low",
        waterlogging_tolerance="Low",
        soil_ph_min=5.5, soil_ph_max=7.5,
        suitable_soil_textures=["Loam","Sandy","Sandy-Loam","Clay","Clay-Loam"],
        nutrient_requirements={"N": "Low", "P": "Low", "K": "Low"},
        regional_suitability=_zone_suitability(["North","Central","West","East","South","Northeast"]),
        successful_regions=["DL_NEW_DELHI","MH_PUNE","KA_BENGALURU","TN_CHENNAI"],
        seasons=["Kharif","Rabi","Zaid"],
        varieties=["Radish","Sunflower","Pea","Fenugreek"],
        typical_yield_kg_per_ha=0,
        market_demand="High",
        growing_tip="Grow on trays with coco peat. Harvest in 7–14 days. Very high value in urban markets. Minimal space needed."
    ),

    CropInfo(
        crop_id="DRUMSTICK_LEAVES_01",
        common_name="Drumstick Leaves (Moringa)",
        scientific_name="Moringa oleifera",
        duration_days=40,
        duration_range=(20, 60),
        temp_min=15, temp_optimal_min=25, temp_optimal_max=35, temp_max=42,
        water_requirement_mm=250,
        drought_tolerance="High",
        waterlogging_tolerance="Low",
        soil_ph_min=6.3, soil_ph_max=7.5,
        suitable_soil_textures=["Sandy","Sandy-Loam","Loam"],
        nutrient_requirements={"N": "Low", "P": "Low", "K": "Low"},
        regional_suitability=_zone_suitability(["South","West","Central","East"]),
        successful_regions=["TN_COIMBATORE","KA_BENGALURU","AP_GUNTUR","MH_PUNE"],
        seasons=["Kharif","Rabi","Zaid"],
        varieties=["PKM-1","PKM-2","Dhanraj"],
        typical_yield_kg_per_ha=7000,
        market_demand="High",
        growing_tip="Plant stem cuttings for quick growth. Leaves harvestable from 20 days. Extremely nutritious. Drought tolerant."
    ),
]



class CropDatabase:
    """
    Manages crop information and provides querying functionality.
    """
    
    def __init__(self):
        """Initialize crop database."""
        self.crops: Dict[str, CropInfo] = {}
        self._load_crops()
    
    def _load_crops(self) -> None:
        """Load crops into database."""
        for crop in CROPS_DATA:
            self.crops[crop.crop_id] = crop
        logger.info(f"Loaded {len(self.crops)} crops into database")
    
    def get_crop(self, crop_id: str) -> Optional[CropInfo]:
        """Get crop by ID."""
        return self.crops.get(crop_id)
    
    def get_all_crops(self) -> List[CropInfo]:
        """Get all crops."""
        return list(self.crops.values())
    
    def get_crops_by_season(self, season: str) -> List[CropInfo]:
        """Get crops suitable for a season."""
        return [crop for crop in self.crops.values() if crop.is_suitable_for_season(season)]
    
    def get_crops_by_region(self, region_id: str, threshold: float = 0.3) -> List[CropInfo]:
        """Get crops suitable for a region."""
        return [crop for crop in self.crops.values() if crop.is_suitable_for_region(region_id, threshold)]
    
    def filter_by_soil(self, crops: List[CropInfo], soil: SoilInfo, min_score: float = 50.0) -> List[CropInfo]:
        """
        Filter crops by soil compatibility.
        
        Args:
            crops: List of crops to filter
            soil: Soil information
            min_score: Minimum compatibility score (0-100)
            
        Returns:
            List of compatible crops
        """
        compatible_crops = []
        for crop in crops:
            score = calculate_soil_compatibility_score(crop, soil)
            if score >= min_score:
                compatible_crops.append(crop)
        return compatible_crops
    
    def get_crops_with_soil_scores(self, crops: List[CropInfo], soil: SoilInfo) -> List[tuple]:
        """
        Get crops with their soil compatibility scores.
        
        Args:
            crops: List of crops
            soil: Soil information
            
        Returns:
            List of tuples (crop, score, amendments)
        """
        results = []
        for crop in crops:
            score = calculate_soil_compatibility_score(crop, soil)
            amendments = get_soil_amendment_suggestions(crop, soil)
            results.append((crop, score, amendments))
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_short_duration_crops(self, min_days: int = 70, max_days: int = 90) -> List[CropInfo]:
        """Get short-duration crops (70-90 days)."""
        return [crop for crop in self.crops.values() if crop.is_short_duration(min_days, max_days)]
    
    def get_crop_count(self) -> int:
        """Get total number of crops."""
        return len(self.crops)


# Global crop database instance
crop_db = CropDatabase()
