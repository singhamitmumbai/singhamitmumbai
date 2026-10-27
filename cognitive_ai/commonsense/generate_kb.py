"""Generate a large curated common-sense KB from taxonomies (non-LLM).

Produces thousands of explicit facts + reusable rules without scraping
untrusted web content. Facts are deterministic and auditable.
"""

from __future__ import annotations

from typing import Iterator

from ..knowledge import Rule, Triple

# --- taxonomies ---------------------------------------------------------------

ANIMALS = {
    "bird": ["sparrow", "eagle", "parrot", "crow", "pigeon", "owl", "duck", "swan", "canary", "hawk"],
    "fish": ["salmon", "tuna", "goldfish", "cod", "shark", "trout", "carp", "bass"],
    "mammal": ["dog", "cat", "horse", "cow", "elephant", "lion", "tiger", "bear", "monkey", "whale", "dolphin", "sheep", "goat", "pig", "rabbit"],
    "insect": ["ant", "bee", "butterfly", "mosquito", "beetle", "fly", "cricket"],
    "reptile": ["snake", "lizard", "crocodile", "turtle", "iguana"],
}

ANIMAL_PROPS = {
    "bird": [("can", "fly"), ("has", "wings"), ("has", "feathers"), ("lays", "eggs")],
    "fish": [("can", "swim"), ("lives_in", "water"), ("has", "gills"), ("has", "fins")],
    "mammal": [("has", "hair_or_fur"), ("is", "warm_blooded"), ("breathes", "air")],
    "insect": [("has", "exoskeleton"), ("has", "six_legs")],
    "reptile": [("is", "cold_blooded"), ("has", "scales")],
}

EXCEPTIONS = [
    ("penguin", "is_a", "bird"),
    ("penguin", "cannot", "fly"),
    ("penguin", "can", "swim"),
    ("ostrich", "is_a", "bird"),
    ("ostrich", "cannot", "fly"),
    ("kiwi", "is_a", "bird"),
    ("kiwi", "cannot", "fly"),
    ("bat", "is_a", "mammal"),
    ("bat", "can", "fly"),
    ("whale", "is_a", "mammal"),
    ("whale", "lives_in", "water"),
    ("dolphin", "is_a", "mammal"),
    ("dolphin", "lives_in", "water"),
]

SPECIES_PROPS = [
    ("dog", "can", "bark"),
    ("dog", "can", "fetch"),
    ("cat", "can", "meow"),
    ("cat", "can", "purr"),
    ("lion", "can", "roar"),
    ("horse", "can", "gallop"),
    ("bee", "can", "sting"),
    ("snake", "can", "slither"),
    ("eagle", "can", "hunt"),
    ("cow", "can", "moo"),
]

TOOLS = {
    "knife": "cutting",
    "scissors": "cutting",
    "saw": "cutting",
    "axe": "chopping",
    "hammer": "hitting_nails",
    "screwdriver": "turning_screws",
    "wrench": "tightening_bolts",
    "key": "opening_lock",
    "lock": "securing",
    "umbrella": "rain_protection",
    "raincoat": "rain_protection",
    "flashlight": "illumination",
    "lamp": "illumination",
    "candle": "illumination",
    "phone": "communication",
    "radio": "communication",
    "email": "communication",
    "book": "reading",
    "newspaper": "reading",
    "pen": "writing",
    "pencil": "writing",
    "brush": "painting",
    "broom": "cleaning",
    "soap": "cleaning",
    "towel": "drying",
    "cup": "drinking",
    "plate": "eating",
    "spoon": "eating",
    "fork": "eating",
    "bed": "sleeping",
    "chair": "sitting",
    "table": "supporting_objects",
    "bag": "carrying",
    "backpack": "carrying",
    "car": "transport",
    "bus": "transport",
    "train": "transport",
    "bicycle": "transport",
    "boat": "water_transport",
    "plane": "air_transport",
    "clock": "telling_time",
    "calendar": "tracking_dates",
    "map": "navigation",
    "compass": "navigation",
    "thermometer": "measuring_temperature",
    "scale": "measuring_weight",
    "ruler": "measuring_length",
}

MATERIALS = {
    "wood": [("can", "burn"), ("is", "organic"), ("comes_from", "trees")],
    "metal": [("can", "conduct_electricity"), ("is", "durable"), ("can", "melt")],
    "glass": [("is", "fragile"), ("is", "transparent"), ("can", "shatter")],
    "plastic": [("is", "synthetic"), ("can", "melt"), ("is_often", "recyclable")],
    "water": [("is", "liquid_at_room_temp"), ("freezes_at_c", "0"), ("boils_at_c", "100")],
    "ice": [("is", "frozen_water"), ("melts_to", "water"), ("is", "cold")],
    "stone": [("is", "hard"), ("is", "heavy")],
    "paper": [("can", "burn"), ("is", "flammable"), ("used_for", "writing")],
    "rubber": [("is", "elastic"), ("used_for", "tires")],
    "cotton": [("is", "fabric_source"), ("used_for", "clothing")],
}

FOODS = [
    "rice", "wheat", "bread", "milk", "egg", "apple", "banana", "orange", "mango",
    "potato", "tomato", "onion", "carrot", "spinach", "chicken", "fish_food", "dal",
    "yogurt", "cheese", "butter", "sugar", "salt", "tea", "coffee", "water_drink",
]

CITIES = {
    "mumbai": "india",
    "delhi": "india",
    "bengaluru": "india",
    "chennai": "india",
    "kolkata": "india",
    "hyderabad": "india",
    "pune": "india",
    "ahmedabad": "india",
    "jaipur": "india",
    "lucknow": "india",
    "london": "uk",
    "paris": "france",
    "berlin": "germany",
    "tokyo": "japan",
    "beijing": "china",
    "new_york": "usa",
    "chicago": "usa",
    "toronto": "canada",
    "sydney": "australia",
    "dubai": "uae",
}

COUNTRIES = {
    "india": "asia",
    "china": "asia",
    "japan": "asia",
    "uk": "europe",
    "france": "europe",
    "germany": "europe",
    "usa": "north_america",
    "canada": "north_america",
    "australia": "oceania",
    "uae": "asia",
    "brazil": "south_america",
    "egypt": "africa",
    "nigeria": "africa",
    "south_africa": "africa",
}

PROFESSIONS = {
    "doctor": "patients",
    "nurse": "patients",
    "teacher": "students",
    "professor": "students",
    "engineer": "systems",
    "farmer": "crops",
    "chef": "meals",
    "pilot": "aircraft",
    "driver": "vehicles",
    "police_officer": "public_safety",
    "firefighter": "fires",
    "judge": "legal_cases",
    "lawyer": "clients",
    "accountant": "finances",
    "scientist": "research",
}

CAUSES = [
    ("rain", "can_cause", "wet_ground"),
    ("rain", "can_cause", "flooding"),
    ("drought", "can_cause", "crop_failure"),
    ("fire", "can_cause", "smoke"),
    ("fire", "needs", "oxygen"),
    ("fire", "because", "combustion"),
    ("smoke", "can_cause", "coughing"),
    ("sleep", "reduces", "tiredness"),
    ("sleep", "because", "body_restoration"),
    ("food", "reduces", "hunger"),
    ("food", "because", "energy_intake"),
    ("water_drink", "reduces", "thirst"),
    ("exercise", "can_improve", "health"),
    ("exercise", "can_cause", "fitness"),
    ("study", "can_improve", "knowledge"),
    ("practice", "can_improve", "skill"),
    ("vaccine", "can_reduce", "disease_risk"),
    ("seatbelt", "can_reduce", "injury_risk"),
    ("helmet", "can_reduce", "head_injury_risk"),
    ("sun", "provides", "light"),
    ("sun", "provides", "heat"),
    ("sun", "can_cause", "daylight"),
    ("night", "has", "darkness"),
    ("winter", "is", "cold_season"),
    ("winter", "can_cause", "cold_weather"),
    ("summer", "is", "hot_season"),
    ("summer", "can_cause", "heat"),
    ("monsoon", "brings", "rain"),
    ("monsoon", "can_cause", "flooding"),
    ("overdue_invoice", "can_cause", "cashflow_pressure"),
    ("low_stock", "can_cause", "project_delay"),
    ("missing_receipt", "can_cause", "delivery_uncertainty"),
]

SOCIAL = [
    ("promise", "should_be", "kept"),
    ("lying", "is_generally", "wrong"),
    ("stealing", "is_generally", "wrong"),
    ("thank_you", "is", "polite"),
    ("please", "is", "polite"),
    ("queue", "requires", "waiting_turn"),
    ("apology", "expresses", "regret"),
    ("gift", "expresses", "goodwill"),
    ("vote", "is_part_of", "democracy"),
    ("law", "should_be", "followed"),
    ("contract", "should_be", "honored"),
    ("privacy", "should_be", "respected"),
    ("consent", "is_required_for", "personal_actions"),
]

TIME = [
    ("week", "has_days", "7"),
    ("year", "has_months", "12"),
    ("day", "has_hours", "24"),
    ("hour", "has_minutes", "60"),
    ("minute", "has_seconds", "60"),
    ("yesterday", "is_before", "today"),
    ("tomorrow", "is_after", "today"),
    ("morning", "is_before", "afternoon"),
    ("afternoon", "is_before", "evening"),
    ("evening", "is_before", "night"),
]

COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "brown", "pink"]
SHAPES = ["circle", "square", "triangle", "rectangle", "sphere", "cube", "cylinder"]

BODY_PARTS = {
    "eye": "seeing",
    "ear": "hearing",
    "nose": "smelling",
    "tongue": "tasting",
    "hand": "grasping",
    "foot": "walking",
    "leg": "walking",
    "heart": "pumping_blood",
    "lung": "breathing",
    "brain": "thinking",
    "tooth": "chewing",
    "skin": "protecting_body",
}


def iter_facts() -> Iterator[tuple[str, str, str]]:
    # category membership + inherited props as explicit instance props for robustness
    for category, members in ANIMALS.items():
        yield (category, "is_a", "animal")
        for prop in ANIMAL_PROPS.get(category, []):
            yield (category, prop[0], prop[1])
        for m in members:
            yield (m, "is_a", category)
            yield (m, "is_a", "animal")
            for prop in ANIMAL_PROPS.get(category, []):
                yield (m, prop[0], prop[1])

    for t in EXCEPTIONS:
        yield t
    for t in SPECIES_PROPS:
        yield t

    for tool, purpose in TOOLS.items():
        yield (tool, "is_a", "tool")
        yield (tool, "used_for", purpose)

    for mat, props in MATERIALS.items():
        yield (mat, "is_a", "material")
        for p in props:
            yield (mat, p[0], p[1])

    for food in FOODS:
        yield (food, "is_a", "food")
        yield (food, "can_reduce", "hunger")

    for city, country in CITIES.items():
        yield (city, "is_a", "city")
        yield (city, "is_in", country)

    for country, continent in COUNTRIES.items():
        yield (country, "is_a", "country")
        yield (country, "is_in", continent)

    for job, helps in PROFESSIONS.items():
        yield (job, "is_a", "profession")
        yield (job, "helps", helps)

    for t in CAUSES:
        yield t
    for t in SOCIAL:
        yield t
    for t in TIME:
        yield t

    for c in COLORS:
        yield (c, "is_a", "color")
    for s in SHAPES:
        yield (s, "is_a", "shape")

    for part, fn in BODY_PARTS.items():
        yield (part, "is_a", "body_part")
        yield (part, "used_for", fn)

    # Combinatorial but still curated: every tool material pairing as "can_be_made_of" candidates
    for tool in list(TOOLS)[:25]:
        for mat in ("metal", "wood", "plastic"):
            yield (tool, "can_be_made_of", mat)

    # City–country–continent chain explicit facts for retrieval without inference
    for city, country in CITIES.items():
        continent = COUNTRIES.get(country)
        if continent:
            yield (city, "is_in_continent", continent)

    # Number / calendar expansions
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, d in enumerate(days):
        yield (d, "is_a", "weekday" if i < 5 else "weekend_day")
        yield (d, "day_index", str(i))
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    for i, m in enumerate(months, 1):
        yield (m, "is_a", "month")
        yield (m, "month_index", str(i))

    # Opposites
    opposites = [
        ("hot", "cold"), ("up", "down"), ("in", "out"), ("open", "closed"),
        ("full", "empty"), ("wet", "dry"), ("light", "dark"), ("day", "night"),
        ("start", "end"), ("buy", "sell"), ("give", "take"), ("win", "lose"),
        ("true", "false"), ("yes", "no"), ("left", "right"), ("north", "south"),
        ("east", "west"), ("push", "pull"), ("increase", "decrease"),
    ]
    for a, b in opposites:
        yield (a, "opposite_of", b)
        yield (b, "opposite_of", a)

    # Human basics
    yield ("human", "is_a", "mammal")
    yield ("human", "can", "think")
    yield ("human", "can", "speak")
    yield ("human", "needs", "food")
    yield ("human", "needs", "water_drink")
    yield ("human", "needs", "sleep")
    yield ("child", "is_a", "human")
    yield ("adult", "is_a", "human")

    # Expand to thousands via curated combinatorial templates
    actions = [
        "walk", "run", "sit", "stand", "eat", "drink", "sleep", "read", "write",
        "speak", "listen", "look", "carry", "open", "close", "buy", "sell",
        "cook", "clean", "build", "repair", "measure", "count", "pay", "sign",
    ]
    actors = ["human", "adult", "child", "worker", "farmer", "teacher", "doctor", "engineer"]
    for actor in actors:
        yield (actor, "is_a", "human") if actor != "human" else ("human", "is_a", "mammal")
        for act in actions:
            yield (actor, "can_try", act)

    places = [
        "home", "school", "hospital", "office", "market", "farm", "factory",
        "station", "airport", "temple", "park", "library", "warehouse", "bank",
        "court", "kitchen", "bedroom", "bathroom", "garden", "road",
    ]
    for place in places:
        yield (place, "is_a", "place")
        yield (place, "can_be", "visited")

    weather = ["sunny", "rainy", "cloudy", "windy", "foggy", "stormy", "humid", "dry"]
    for w in weather:
        yield (w, "is_a", "weather_condition")

    # Numbers 1..100 as quantity facts (useful for NLU grounding demos)
    for n in range(1, 101):
        yield (f"number_{n}", "is_a", "integer")
        yield (f"number_{n}", "value", str(n))
        if n % 2 == 0:
            yield (f"number_{n}", "parity", "even")
        else:
            yield (f"number_{n}", "parity", "odd")

    # Unit conversions (explicit, not inferred)
    units = [
        ("km", "meters", "1000"),
        ("meter", "centimeters", "100"),
        ("kg", "grams", "1000"),
        ("liter", "milliliters", "1000"),
        ("hour", "minutes", "60"),
        ("day", "hours", "24"),
        ("week", "days", "7"),
        ("lakh", "units", "100000"),
        ("crore", "units", "10000000"),
    ]
    for a, b, factor in units:
        yield (a, "equals_in", b)
        yield (a, "conversion_factor", factor)

    # Business entities for Finderbay-adjacent common sense
    biz = [
        ("invoice", "records", "amount_owed"),
        ("payment", "settles", "invoice"),
        ("contract", "binds", "parties"),
        ("delivery", "moves", "goods"),
        ("warehouse", "stores", "goods"),
        ("supplier", "provides", "goods"),
        ("customer", "receives", "goods"),
        ("tax", "is_added_to", "price"),
        ("receipt", "proves", "delivery_or_payment"),
        ("deadline", "constrains", "schedule"),
    ]
    for t in biz:
        yield t

    # Hypernym chains for tools
    for tool in TOOLS:
        yield (tool, "is_a", "artifact")
        yield (tool, "is_a", "physical_object")

    # Extra curated expansions to reach multi-thousand fact coverage
    vehicles = {
        "car": ["road", "fuel"],
        "bus": ["road", "passengers"],
        "train": ["rail", "passengers"],
        "bicycle": ["road", "pedals"],
        "motorcycle": ["road", "fuel"],
        "truck": ["road", "cargo"],
        "boat": ["water", "cargo"],
        "ship": ["water", "cargo"],
        "plane": ["air", "passengers"],
        "helicopter": ["air", "passengers"],
        "rickshaw": ["road", "passengers"],
        "scooter": ["road", "fuel"],
        "metro": ["rail", "passengers"],
        "tram": ["rail", "passengers"],
        "ferry": ["water", "passengers"],
    }
    for v, props in vehicles.items():
        yield (v, "is_a", "vehicle")
        yield (v, "is_a", "artifact")
        yield (v, "travels_on", props[0])
        yield (v, "typically_carries", props[1])

    appliances = {
        "fridge": "cooling_food",
        "oven": "baking",
        "stove": "cooking",
        "microwave": "heating_food",
        "washer": "washing_clothes",
        "dryer": "drying_clothes",
        "fan": "air_circulation",
        "ac": "cooling_room",
        "heater": "heating_room",
        "vacuum": "cleaning_floors",
        "blender": "mixing_food",
        "toaster": "toasting_bread",
        "kettle": "boiling_water",
        "iron": "pressing_clothes",
        "router": "network_access",
    }
    for a, purpose in appliances.items():
        yield (a, "is_a", "appliance")
        yield (a, "is_a", "artifact")
        yield (a, "used_for", purpose)
        yield (a, "needs", "electricity")

    indian_foods = [
        "roti", "chapati", "paratha", "idli", "dosa", "sambar", "rasam", "biryani",
        "pulao", "paneer", "ghee", "lassi", "chai", "jalebi", "samosa", "pakora",
        "chutney", "pickle", "curd", "raita", "khichdi", "upma", "poha", "vada",
    ]
    for food in indian_foods:
        yield (food, "is_a", "food")
        yield (food, "is_a", "indian_food")
        yield (food, "can_reduce", "hunger")
        yield (food, "common_in", "india")

    more_cities = {
        "nagpur": "india", "indore": "india", "bhopal": "india", "patna": "india",
        "chandigarh": "india", "kochi": "india", "coimbatore": "india", "surat": "india",
        "vadodara": "india", "rajkot": "india", "mysuru": "india", "thane": "india",
        "noida": "india", "gurgaon": "india", "visakhapatnam": "india", "kanpur": "india",
        "manchester": "uk", "birmingham": "uk", "lyon": "france", "munich": "germany",
        "osaka": "japan", "shanghai": "china", "los_angeles": "usa", "houston": "usa",
        "vancouver": "canada", "melbourne": "australia", "abu_dhabi": "uae",
        "cairo": "egypt", "lagos": "nigeria", "cape_town": "south_africa",
        "sao_paulo": "brazil", "rio_de_janeiro": "brazil",
    }
    for city, country in more_cities.items():
        yield (city, "is_a", "city")
        yield (city, "is_in", country)
        continent = COUNTRIES.get(country)
        if continent:
            yield (city, "is_in_continent", continent)

    sports = {
        "cricket": "bat_and_ball",
        "football": "ball",
        "hockey": "stick_and_ball",
        "tennis": "racket_and_ball",
        "badminton": "racket_and_shuttle",
        "basketball": "ball_and_hoop",
        "volleyball": "ball_and_net",
        "chess": "board_and_pieces",
        "kabaddi": "contact",
        "wrestling": "contact",
        "swimming": "water",
        "athletics": "track",
        "boxing": "contact",
        "table_tennis": "paddle_and_ball",
        "golf": "club_and_ball",
    }
    for sport, equip in sports.items():
        yield (sport, "is_a", "sport")
        yield (sport, "uses", equip)
        yield (sport, "can_be", "played")
        yield (sport, "can_improve", "fitness")

    # Relational templates: every place can host common activities
    place_acts = ["meet", "work", "rest", "talk", "wait", "learn", "trade"]
    for place in places:
        for act in place_acts:
            yield (place, "can_support", act)

    # Material × property explicit grid for retrieval demos
    mat_props = ["hard", "soft", "heavy", "light", "flexible", "rigid", "opaque", "smooth"]
    for mat in MATERIALS:
        for prop in mat_props:
            yield (mat, "may_be", prop)

    # Extra animal species for broader coverage
    more_animals = {
        "bird": ["peacock", "flamingo", "pelican", "vulture", "woodpecker", "kingfisher"],
        "fish": ["rohu", "hilsa", "mackerel", "sardine", "catfish"],
        "mammal": ["deer", "buffalo", "camel", "fox", "wolf", "leopard", "giraffe", "zebra"],
        "insect": ["wasp", "termite", "dragonfly", "moth", "grasshopper"],
        "reptile": ["gecko", "python", "cobra", "alligator"],
    }
    for category, members in more_animals.items():
        for m in members:
            yield (m, "is_a", category)
            yield (m, "is_a", "animal")
            for prop in ANIMAL_PROPS.get(category, []):
                yield (m, prop[0], prop[1])

    # Safety / health common sense
    safety = [
        ("medicine", "should_be", "used_as_directed"),
        ("fire", "should_not_be", "left_unattended"),
        ("electricity", "can_cause", "shock"),
        ("wet_floor", "can_cause", "slipping"),
        ("smoke_alarm", "warns_about", "fire"),
        ("first_aid", "helps", "injury"),
        ("handwashing", "reduces", "infection_risk"),
        ("mask", "can_reduce", "airborne_spread"),
        ("password", "should_be", "kept_secret"),
        ("backup", "protects", "data"),
        ("helmet", "should_be_worn_on", "motorcycle"),
        ("life_jacket", "helps", "water_safety"),
        ("crosswalk", "is_for", "pedestrians"),
        ("red_light", "means", "stop"),
        ("green_light", "means", "go"),
        ("yellow_light", "means", "caution"),
    ]
    for t in safety:
        yield t

    # Language / communication basics (EN + HI labels as entities)
    langs = [
        ("english", "is_a", "language"),
        ("hindi", "is_a", "language"),
        ("marathi", "is_a", "language"),
        ("tamil", "is_a", "language"),
        ("telugu", "is_a", "language"),
        ("bengali", "is_a", "language"),
        ("gujarati", "is_a", "language"),
        ("kannada", "is_a", "language"),
        ("malayalam", "is_a", "language"),
        ("punjabi", "is_a", "language"),
        ("hindi", "spoken_in", "india"),
        ("english", "spoken_in", "india"),
        ("english", "spoken_in", "uk"),
        ("english", "spoken_in", "usa"),
    ]
    for t in langs:
        yield t


def build_rules() -> list[Rule]:
    specs = [
        ("inherit_is_a", [("?x", "is_a", "?y"), ("?y", "is_a", "?z")], [("?x", "is_a", "?z")]),
        ("inherit_can", [("?x", "is_a", "?y"), ("?y", "can", "?a")], [("?x", "can", "?a")]),
        ("inherit_has", [("?x", "is_a", "?y"), ("?y", "has", "?a")], [("?x", "has", "?a")]),
        ("inherit_lives_in", [("?x", "is_a", "?y"), ("?y", "lives_in", "?a")], [("?x", "lives_in", "?a")]),
        ("tool_affordance", [("?t", "used_for", "?p")], [("?t", "enables", "?p")]),
        ("city_in_country", [("?c", "is_in", "?k"), ("?k", "is_a", "country")], [("?c", "located_in_country", "?k")]),
        ("food_satisfies", [("?f", "is_a", "food")], [("?f", "can_be_eaten", "true")]),
        ("profession_role", [("?p", "is_a", "profession"), ("?p", "helps", "?h")], [("?p", "serves", "?h")]),
        ("material_instance", [("?x", "can_be_made_of", "?m"), ("?m", "is_a", "material")], [("?x", "has_material_option", "?m")]),
        ("opposite_symmetric_check", [("?a", "opposite_of", "?b")], [("?a", "contrasts_with", "?b")]),
    ]
    rules = []
    for name, ants, cons in specs:
        rules.append(
            Rule(
                name=name,
                antecedents=[Triple(*a) for a in ants],
                consequents=[Triple(*c) for c in cons],
            )
        )
    return rules


def materialize() -> tuple[list[tuple[str, str, str]], list[Rule]]:
    # dedupe while preserving order
    seen = set()
    facts: list[tuple[str, str, str]] = []
    for f in iter_facts():
        if f not in seen:
            seen.add(f)
            facts.append(f)
    return facts, build_rules()
