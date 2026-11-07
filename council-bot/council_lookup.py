import re
import requests
from typing import Optional, Tuple, Dict

# Standard Singapore 2-digit postal sectors mapped to Town Councils
POSTAL_SECTOR_MAP = {
    # 01-08: City, Raffles Place, Anson, Tanjong Pagar
    "01": "Tanjong Pagar Town Council",
    "02": "Tanjong Pagar Town Council",
    "03": "Tanjong Pagar Town Council",
    "04": "Tanjong Pagar Town Council",
    "05": "Tanjong Pagar Town Council",
    "06": "Tanjong Pagar Town Council",
    "07": "Tanjong Pagar Town Council",
    "08": "Tanjong Pagar Town Council",
    # 09-10: Harbourfront, Telok Blangah
    "09": "Tanjong Pagar Town Council",
    "10": "Tanjong Pagar Town Council",
    # 11-13: Pasir Panjang, Clementi, West Coast
    "11": "West Coast Town Council",
    "12": "West Coast Town Council",
    "13": "West Coast Town Council",
    # 14-16: Queenstown, Tiong Bahru, Bukit Merah
    "14": "Tanjong Pagar Town Council",
    "15": "Tanjong Pagar Town Council",
    "16": "Tanjong Pagar Town Council",
    # 17-21: High Street, Beach Road, Middle Road, Golden Mile, Little India
    "17": "Jalan Besar Town Council",
    "18": "Jalan Besar Town Council",
    "19": "Jalan Besar Town Council",
    "20": "Jalan Besar Town Council",
    "21": "Jalan Besar Town Council",
    # 22-23: Orchard, Cairnhill, River Valley
    "22": "Tanjong Pagar Town Council",
    "23": "Tanjong Pagar Town Council",
    # 24-27: Tanglin, Bukit Timah, Holland
    "24": "Holland-Bukit Panjang Town Council",
    "25": "Holland-Bukit Panjang Town Council",
    "26": "Holland-Bukit Panjang Town Council",
    "27": "Holland-Bukit Panjang Town Council",
    # 28-30: Watten Estate, Novena, Thomson
    "28": "Bishan-Toa Payoh Town Council",
    "29": "Bishan-Toa Payoh Town Council",
    "30": "Bishan-Toa Payoh Town Council",
    # 31-33: Balestier, Toa Payoh
    "31": "Bishan-Toa Payoh Town Council",
    "32": "Bishan-Toa Payoh Town Council",
    "33": "Bishan-Toa Payoh Town Council",
    # 34-37: Macpherson, Braddell, Potong Pasir
    "34": "Jalan Besar Town Council",
    "35": "Jalan Besar Town Council",
    "36": "Marine Parade Town Council",
    "37": "Marine Parade Town Council",
    # 38-41: Geylang, Eunos, Aljunied
    "38": "Marine Parade Town Council",
    "39": "Marine Parade Town Council",
    "40": "Marine Parade Town Council",
    "41": "Aljunied-Hougang Town Council",
    # 42-45: Katong, Joo Chiat, Marine Parade
    "42": "Marine Parade Town Council",
    "43": "Marine Parade Town Council",
    "44": "Marine Parade Town Council",
    "45": "Marine Parade Town Council",
    # 46-48: Bedok, Upper East Coast
    "46": "East Coast Town Council",
    "47": "East Coast Town Council",
    "48": "East Coast Town Council",
    # 49-50: Loyang, Changi
    "49": "Pasir Ris-Punggol Town Council",
    "50": "East Coast Town Council",
    # 51-52: Simei, Tampines
    "51": "Tampines Town Council",
    "52": "Tampines Town Council",
    # 53-55: Serangoon Garden, Hougang, Kovan
    "53": "Aljunied-Hougang Town Council",
    "54": "Sengkang Town Council",
    "55": "Aljunied-Hougang Town Council",
    # 56-57: Bishan, Ang Mo Kio
    "56": "Ang Mo Kio Town Council",
    "57": "Bishan-Toa Payoh Town Council",
    # 58-59: Upper Bukit Timah, Ulu Pandan
    "58": "Holland-Bukit Panjang Town Council",
    "59": "Holland-Bukit Panjang Town Council",
    # 60-64: Jurong, Boon Lay
    "60": "Jurong-Clementi Town Council",
    "61": "Jurong-Clementi Town Council",
    "62": "West Coast Town Council",
    "63": "West Coast Town Council",
    "64": "Jurong-Clementi Town Council",
    # 65-68: Hillview, Bukit Panjang, Choa Chu Kang
    "65": "Jurong-Clementi Town Council",
    "66": "Holland-Bukit Panjang Town Council",
    "67": "Holland-Bukit Panjang Town Council",
    "68": "Chua Chu Kang Town Council",
    # 69-71: Lim Chu Kang, Tengah
    "69": "Chua Chu Kang Town Council",
    "70": "Chua Chu Kang Town Council",
    "71": "Chua Chu Kang Town Council",
    # 72-73: Kranji, Woodlands
    "72": "Marsiling-Yew Tee Town Council",
    "73": "Marsiling-Yew Tee Town Council",
    # 75-76: Yishun, Sembawang
    "75": "Sembawang Town Council",
    "76": "Nee Soon Town Council",
    # 77-78: Upper Thomson, Springleaf
    "77": "Nee Soon Town Council",
    "78": "Ang Mo Kio Town Council",
    # 79-80: Seletar, Yio Chu Kang
    "79": "Ang Mo Kio Town Council",
    "80": "Ang Mo Kio Town Council",
    # 82: Punggol
    "82": "Pasir Ris-Punggol Town Council",
}

# Common keywords in Singapore road & town names
TOWN_NAME_KEYWORDS = [
    ("ANG MO KIO", "Ang Mo Kio Town Council"),
    ("AMK", "Ang Mo Kio Town Council"),
    ("KEBUN BARU", "Ang Mo Kio Town Council"),
    ("YIO CHU KANG", "Ang Mo Kio Town Council"),
    ("BEDOK", "East Coast Town Council"),
    ("EAST COAST", "East Coast Town Council"),
    ("SIMEI", "Tampines Town Council"),
    ("TAMPINES", "Tampines Town Council"),
    ("PASIR RIS", "Pasir Ris-Punggol Town Council"),
    ("PUNGGOL", "Pasir Ris-Punggol Town Council"),
    ("SENGKANG", "Sengkang Town Council"),
    ("HOUGANG", "Aljunied-Hougang Town Council"),
    ("ALJUNIED", "Aljunied-Hougang Town Council"),
    ("BISHAN", "Bishan-Toa Payoh Town Council"),
    ("TOA PAYOH", "Bishan-Toa Payoh Town Council"),
    ("MARYMOUNT", "Bishan-Toa Payoh Town Council"),
    ("POTONG PASIR", "Jalan Besar Town Council"),
    ("JALAN BESAR", "Jalan Besar Town Council"),
    ("KALLANG", "Jalan Besar Town Council"),
    ("WHAMPOA", "Jalan Besar Town Council"),
    ("MARINE PARADE", "Marine Parade Town Council"),
    ("MACPHERSON", "Marine Parade Town Council"),
    ("GEYLANG", "Marine Parade Town Council"),
    ("MOUNTBATTEN", "Marine Parade Town Council"),
    ("TANJONG PAGAR", "Tanjong Pagar Town Council"),
    ("QUEENSTOWN", "Tanjong Pagar Town Council"),
    ("TIONG BAHRU", "Tanjong Pagar Town Council"),
    ("TELOK BLANGAH", "Tanjong Pagar Town Council"),
    ("RADIN MAS", "Tanjong Pagar Town Council"),
    ("BUKIT MERAH", "Tanjong Pagar Town Council"),
    ("HOLLAND", "Holland-Bukit Panjang Town Council"),
    ("BUKIT TIMAH", "Holland-Bukit Panjang Town Council"),
    ("BUKIT PANJANG", "Holland-Bukit Panjang Town Council"),
    ("CHOA CHU KANG", "Chua Chu Kang Town Council"),
    ("CHUA CHU KANG", "Chua Chu Kang Town Council"),
    ("CCK", "Chua Chu Kang Town Council"),
    ("HONG KAH", "Chua Chu Kang Town Council"),
    ("TENGAH", "Chua Chu Kang Town Council"),
    ("JURONG", "Jurong-Clementi Town Council"),
    ("CLEMENTI", "Jurong-Clementi Town Council"),
    ("BUKIT BATOK", "Jurong-Clementi Town Council"),
    ("YUHUA", "Jurong-Clementi Town Council"),
    ("WEST COAST", "West Coast Town Council"),
    ("PIONEER", "West Coast Town Council"),
    ("BOON LAY", "West Coast Town Council"),
    ("WOODLANDS", "Marsiling-Yew Tee Town Council"),
    ("MARSILING", "Marsiling-Yew Tee Town Council"),
    ("YEW TEE", "Marsiling-Yew Tee Town Council"),
    ("SEMBAWANG", "Sembawang Town Council"),
    ("CANBERRA", "Sembawang Town Council"),
    ("ADMIRALTY", "Sembawang Town Council"),
    ("YISHUN", "Nee Soon Town Council"),
    ("NEE SOON", "Nee Soon Town Council"),
    ("KHATIB", "Nee Soon Town Council"),
]


def extract_postal_code(text: str) -> Optional[str]:
    """Finds a 6-digit Singapore postal code in input text."""
    match = re.search(r"\b(\d{6})\b", text)
    return match.group(1) if match else None


def check_statutory_agency(text: str) -> Optional[str]:
    """Checks if the user's issue belongs to a Singapore statutory board instead of Town Council."""
    t = text.lower()
    if re.search(r"\b(bus stop|traffic light|zebra crossing|road sign|mrt|bus lane|expressway|pie|aye|cte|tpe|kpe|bke|sle)\b", t):
        return (
            "🚌 **Land Transport Authority (LTA) Jurisdiction**\n\n"
            "Bus stops, traffic lights, expressways, and road infrastructure are managed by **LTA**, not Town Councils.\n\n"
            "• Submit directly via the **OneService App** or call the **LTA hotline at 1800-2255-582**."
        )
    if re.search(r"\b(wild boar|monkey|monkeys|snake|python|monitor lizard|bee hive|hornet|wasp nest)\b", t):
        return (
            "🐒 **Wildlife / Animal Management (NParks)**\n\n"
            "Wild animals (monkeys, boars, snakes) and bee/wasp nests in trees are handled by the **National Parks Board (NParks)**.\n\n"
            "• **NParks 24-hr Helpline:** 1800-476-1600\n"
            "• If in immediate physical danger, call the **Police at 999**."
        )
    if re.search(r"\b(hawker center|hawker centre|food stall|coffee shop stall|food poisoning|dirty food|food hygiene)\b", t):
        return (
            "🍲 **Food Hygiene / Stall Cleanliness (SFA)**\n\n"
            "Hygiene of individual food stalls and coffee shops falls under the **Singapore Food Agency (SFA)**.\n\n"
            "• You can lodge feedback online at: [SFA Feedback Portal](https://www.sfa.gov.sg/feedback)"
        )
    if re.search(r"\b(inside my flat|ceiling leak inside|toilet choke inside|bedroom floor|master bedroom)\b", t):
        return (
            "🏠 **Internal Flat Maintenance (HDB)**\n\n"
            "Town Councils manage **common property** (corridors, lifts, void decks, parks, public lighting).\n"
            "Defects inside your own flat are the responsibility of the owner or **HDB**.\n\n"
            "• Contact your local HDB Branch at: **1800-225-5432**"
        )
    return None



def query_onemap_search(query: str) -> Optional[Dict]:
    """Search address using OneMap search API."""
    try:
        url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={query}&returnGeom=Y&getAddrDetails=Y"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                return results[0]
    except Exception:
        pass
    return None


COUNCIL_DEFAULT_COORDS = {
    "Ang Mo Kio Town Council": (1.3691, 103.8454),
    "Bishan-Toa Payoh Town Council": (1.3508, 103.8482),
    "Tampines Town Council": (1.3533, 103.9452),
    "Jurong-Clementi Town Council": (1.3496, 103.7142),
    "Aljunied-Hougang Town Council": (1.3713, 103.8925),
    "East Coast Town Council": (1.3236, 103.9273),
    "Holland-Bukit Panjang Town Council": (1.3774, 103.7719),
    "Jalan Besar Town Council": (1.3164, 103.8568),
    "Marine Parade Town Council": (1.3024, 103.9073),
    "Marsiling-Yew Tee Town Council": (1.4382, 103.7790),
    "Nee Soon Town Council": (1.4304, 103.8354),
    "Pasir Ris-Punggol Town Council": (1.3721, 103.9474),
    "Sembawang Town Council": (1.4491, 103.8185),
    "Sengkang Town Council": (1.3916, 103.8944),
    "Tanjong Pagar Town Council": (1.2764, 103.8440),
    "West Coast Town Council": (1.2988, 103.7663),
    "Chua Chu Kang Town Council": (1.3840, 103.7470),
}


def resolve_town_council(raw_input: str, lat: Optional[float] = None, lon: Optional[float] = None) -> Tuple[str, Optional[str], Optional[float], Optional[float]]:
    """
    Given a user's location input (postal code, text address, or coordinates),
    resolves:
      1) Normalized address / location string
      2) Tagged Singapore Town Council name (e.g. 'Ang Mo Kio Town Council')
      3) Latitude
      4) Longitude
    """
    postal = extract_postal_code(raw_input)
    address_resolved = None
    council = None
    out_lat, out_lng = lat, lon

    # 1. If postal code found, query OneMap
    if postal:
        info = query_onemap_search(postal)
        if info:
            address_resolved = info.get("ADDRESS", "").title()
            if not address_resolved:
                blk = info.get("BLK_NO", "")
                road = info.get("ROAD_NAME", "").title()
                address_resolved = f"Blk {blk} {road}, Singapore {postal}".strip()
            try:
                out_lat = float(info.get("LATITUDE"))
                out_lng = float(info.get("LONGITUDE"))
            except Exception:
                pass

        # Sector mapping
        sector = postal[:2]
        council = POSTAL_SECTOR_MAP.get(sector)

    # 2. If no postal code or council not determined yet, check keywords in text/resolved address
    lookup_text = (address_resolved or "") + " " + raw_input
    lookup_upper = lookup_text.upper()

    for kw, c_name in TOWN_NAME_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", lookup_upper):
            # Prioritize specific name matches if sector didn't match or for better precision
            council = c_name
            break

    # 3. Handle GPS coordinates if provided
    if not address_resolved and lat is not None and lon is not None:
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "json"},
                headers={"User-Agent": "TownCouncilClassifierBot/1.0"},
                timeout=4,
            )
            if r.status_code == 200:
                geo_data = r.json()
                display_name = geo_data.get("display_name", "")
                addr = geo_data.get("address", {})
                postcode = addr.get("postcode")
                if postcode and len(postcode) == 6:
                    postal = postcode
                    sector = postal[:2]
                    council = POSTAL_SECTOR_MAP.get(sector, council)
                address_resolved = display_name
                for kw, c_name in TOWN_NAME_KEYWORDS:
                    if kw in display_name.upper():
                        council = c_name
                        break
        except Exception:
            pass

    # Final fallback for location description
    if not address_resolved:
        address_resolved = raw_input.strip()

    if not council:
        council = "Unassigned / General Town Council"

    # Default fallback coordinates based on council center if not found
    if (out_lat is None or out_lng is None) and council in COUNCIL_DEFAULT_COORDS:
        out_lat, out_lng = COUNCIL_DEFAULT_COORDS[council]

    return address_resolved, council, out_lat, out_lng
