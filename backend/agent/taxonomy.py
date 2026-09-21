"""Species taxonomy used by the routing agent.

These sets decide one thing: whether an image that ResNet50 has just
classified should be re-run through the high-capacity YOLOv8x detector rather
than YOLOv8n, and whether a generic COCO label should be refined to the
species name. They were extracted from ``auto_selector.py``, where 333 lines
of literal data sat in front of 60 lines of routing logic.

Membership is tested case-insensitively via :func:`is_animal_image`.
"""

from __future__ import annotations

# ==================== MAMMAL CLASSES ====================
MAMMAL_CLASSES = {
    # --------- BIG CATS ---------
    "lion", "tiger", "cheetah", "leopard", "snow leopard", "jaguar", "cougar",
    "puma", "panther", "black panther", "lynx", "bobcat", "ocelot", "serval",
    "caracal", "clouded leopard", "flat-headed cat",

    # --------- BEARS ---------
    "brown bear", "American black bear", "ice bear", "polar bear", "sloth bear",
    "sun bear", "panda bear", "giant panda", "red panda",

    # --------- ELEPHANTS ---------
    "African elephant", "Indian elephant", "Asian elephant",

    # --------- HOOFED ANIMALS (UNGULATES) ---------
    "zebra", "giraffe", "gazelle", "impala", "hartebeest", "ibex", "bighorn",
    "ram", "sheep", "goat", "moose", "elk", "reindeer", "deer", "red deer",
    "fallow deer", "mule deer", "white-tailed deer", "axis deer", "antelope",
    "wildebeest", "gnu", "buffalo", "bison", "water buffalo", "carabao",
    "yak", "musk ox", "pronghorn", "kudu", "eland", "gemsbok", "oryx",

    # --------- CANINES ---------
    "timber wolf", "gray wolf", "white wolf", "red wolf", "coyote", "dingo",
    "dog", "wolf", "African wild dog", "painted dog", "dhhole",

    # --------- FELINES (WILD CATS) ---------
    "hyena", "spotted hyena", "striped hyena", "brown hyena",

    # --------- FOXES ---------
    "red fox", "kit fox", "Arctic fox", "grey fox", "swift fox", "fennec fox",
    "corsac fox", "Tibetan fox",

    # --------- PRIMATES ---------
    "gorilla", "orangutan", "chimpanzee", "bonobo", "baboon", "mandrill",
    "macaque", "rhesus monkey", "golden monkey", "monkey", "ape", "lemur",
    "gibbon", "marmoset", "tamarin", "capuchin", "spider monkey",

    # --------- LARGE HERBIVORES ---------
    "hippopotamus", "hippo", "rhinoceros", "rhino", "black rhino", "white rhino",
    "warthog", "wild boar", "hog", "anteater", "aardvark",

    # --------- MARINE MAMMALS ---------
    "sea lion", "seal", "fur seal", "walrus", "dugong", "manatee",
    "grey whale", "killer whale", "orca", "dolphin", "porpoise",
    "sperm whale", "humpback whale", "blue whale", "right whale",

    # --------- REPTILES (LARGE) ---------
    "African crocodile", "Nile crocodile", "American alligator", "saltwater crocodile",
    "crocodile", "alligator", "caiman", "gharial",
    "Komodo dragon", "monitor lizard", "python", "anaconda",

    # --------- SMALL CARNIVORES ---------
    "mongoose", "meerkat", "badger", "otter", "sea otter", "weasel", "mink",
    "ferret", "stoat", "polecat", "wolverine", "skunk", "raccoon",
    "coati", "kinkajou", "civet", "genet", "honey badger",

    # --------- RODENTS & SMALL MAMMALS ---------
    "porcupine", "hedgehog", "echidna", "platypus", "beaver", "groundhog",
    "prairie dog", "squirrel", "flying squirrel", "chipmunk", "marmot",
    "mouse", "rat", "guinea pig", "nutria", "pika", "hyrax",

    # --------- UNIQUE MAMMALS ---------
    "sloth", "armadillo", "pangolin"
}

# ==================== BIRD CLASSES ====================
BIRD_CLASSES = {
    # --------- RAPTORS (EAGLES, HAWKS, FALCONS) ---------
    "bald eagle", "golden eagle", "steller's eagle", "harpy eagle", "hawk",
    "red-tailed hawk", "red-shouldered hawk", "sharp-shinned hawk", "cooper's hawk",
    "kestrel", "american kestrel", "merlin", "falcon", "peregrine falcon",
    "gyrfalcon", "lanner falcon", "saker falcon", "vulture", "griffon vulture",
    "lappet-faced vulture", "white-backed vulture", "palm-nut vulture",
    "kite", "red kite", "black kite", "osprey", "fish eagle",
    "buzzard", "common buzzard", "honey buzzard", "harrier", "marsh harrier",
    "northern harrier", "montagu's harrier",

    # --------- OWLS ---------
    "great grey owl", "great gray owl", "barn owl", "snowy owl", "tawny owl",
    "screech owl", "eastern screech owl", "western screech owl",
    "spotted owl", "mexican spotted owl", "barred owl",
    "great horned owl", "long-eared owl", "short-eared owl",
    "elf owl", "burrowing owl", "boreal owl", "pygmy owl", "little owl",
    "pearl-spotted owlet", # --------- LARGE FLIGHTLESS BIRDS ---------
    "ostrich", "emu", "cassowary", "rhea", "kiwi", "tinamou",

    # --------- CRANES, HERONS & STORKS ---------
    "crane", "whooping crane", "sandhill crane", "demoiselle crane", "hooded crane",
    "heron", "great blue heron", "grey heron", "purple heron", "little heron",
    "night heron", "bittern", "american bittern", "eurasian bittern",
    "stork", "white stork", "black stork", "marabou stork", "saddle-billed stork",
    "jabiru", "shoebill",

    # --------- WATER BIRDS ---------
    "flamingo", "greater flamingo", "lesser flamingo", "chilean flamingo",
    "pelican", "great white pelican", "brown pelican", "australian pelican",
    "swan", "mute swan", "whooper swan", "trumpeter swan", "black swan",
    "goose", "canada goose", "greylag goose", "snow goose", "barnacle goose",
    "duck", "mallard", "pintail", "teal", "wigeon", "shoveler", "wood duck",
    "mandarin duck", "harlequin duck", "king eider",
    "ibis", "glossy ibis", "scarlet ibis", "white ibis", "sacred ibis",
    "cormorant", "great cormorant", "double-crested cormorant",
    "loon", "common loon", "red-throated loon", "great northern diver",
    "grebe", "great crested grebe", "western grebe", "pied-billed grebe",
    "albatross", "black-browed albatross", "wandering albatross",
    "gannet", "northern gannet", "masked booby", "brown booby",
    "petrel", "storm petrel", "giant petrel", "shearwater",
    "frigatebird", "great frigatebird",

    # --------- GAME BIRDS ---------
    "pheasant", "ring-necked pheasant", "golden pheasant", "lady amherst's pheasant",
    "grouse", "red grouse", "willow grouse", "rock ptarmigan", "white-tailed ptarmigan",
    "partridge", "red-legged partridge", "grey partridge", "hungarian partridge",
    "quail", "california quail", "bobwhite quail", "japanese quail",
    "turkey", "wild turkey", "ocellated turkey",
    "peacock", "indian peafowl", "green peafowl", "congo peafowl",
    "guinea fowl", "helmet guinea fowl",

    # --------- PIGEONS & DOVES ---------
    "pigeon", "rock dove", "rock pigeon", "wood pigeon", "band-tailed pigeon",
    "dove", "mourning dove", "diamond dove", "collared dove", "ringed dove",
    "turtle dove", "spotted dove", "laughing dove", "palm dove",

    # --------- CUCKOOS & ROADRUNNERS ---------
    "cuckoo", "common cuckoo", "roadrunner", "greater roadrunner",

    # --------- PARROTS & PARAKEETS ---------
    "parrot", "african grey", "grey parrot", "macaw", "blue-and-yellow macaw",
    "scarlet macaw", "green-winged macaw", "cockatoo", "sulphur-crested cockatoo",
    "moluccan cockatoo", "umbrella cockatoo", "palm cockatoo",
    "lovebird", "peach-faced lovebird", "masked lovebird",
    "budgie", "budgerigar", "parakeet", "ring-necked parakeet",
    "lorikeet", "rainbow lorikeet", "musk lorikeet",
    "conure", "green-winged conure", "jenday conure",
    "amazon parrot", "yellow-crowned amazon", "green-winged amazon",
    "eclectus", "eclectus parrot", "galah", "cockatiel",

    # --------- WOODPECKERS ---------
    "woodpecker", "great spotted woodpecker", "green woodpecker",
    "pileated woodpecker", "downy woodpecker", "hairy woodpecker",
    "acorn woodpecker", "ivory-billed woodpecker",

    # --------- CROWS, JAYS & MAGPIES ---------
    "jay", "blue jay", "steller's jay", "gray jay", "scrub jay",
    "crow", "american crow", "carrion crow", "hooded crow",
    "raven", "common raven", "thick-billed raven",
    "magpie", "eurasian magpie", "black-billed magpie",
    "nutcracker", "eurasian nutcracker",

    # --------- SONGBIRDS & SMALL BIRDS ---------
    "sparrow", "house sparrow", "tree sparrow", "white-crowned sparrow",
    "robin", "american robin", "european robin",
    "thrush", "wood thrush", "hermit thrush", "eastern bluebird",
    "finch", "house finch", "purple finch", "goldfinch", "american goldfinch",
    "siskin", "european siskin", "canary", "wild canary",
    "lark", "skylark", "woodlark", "crested lark",
    "swallow", "barn swallow", "cliff swallow", "tree swallow", "sand martin",
    "swift", "common swift", "alpine swift", "pallid swift",
    "hummingbird", "ruby-throated hummingbird", "anna's hummingbird",
    "rufous hummingbird", "giant hummingbird", "bee hummingbird",
    "nightingale", "common nightingale", "thrush nightingale",
    "kingfisher", "belted kingfisher", "common kingfisher", "green kingfisher",
    "bee-eater", "european bee-eater", "carmine bee-eater",
    "roller", "european roller", "lilac-breasted roller",
    "hoopoe", "eurasian hoopoe",
    "flycatcher", "spotted flycatcher", "pied flycatcher",
    "wren", "winter wren", "carolina wren",
    "tit", "great tit", "blue tit", "coal tit", "marsh tit",
    "nuthatch", "eurasian nuthatch", "white-breasted nuthatch",
    "creeper", "eurasian treecreeper",
    "warbler", "sedge warbler", "reed warbler", "great reed warbler",
    "waxwing", "bohemian waxwing", "cedar waxwing",
    "starling", "common starling", "rosy starling",

    # --------- EXOTIC & TROPICAL BIRDS ---------
    "toucan", "toco toucan", "keel-billed toucan",
    "hornbill", "great hornbill", "pied hornbill", "ground hornbill",
    "turaco", "great blue turaco", "livingstone's turaco",
    "crowned crane", "east african crowned crane", "west african crowned crane",
    "secretary bird", "sunbird", "amethyst sunbird", "scarlet sunbird",
    "quetzal", "resplendent quetzal",
    "hoatzin", # --------- DOMESTIC & COMMON BIRDS ---------
    "chicken", "rooster", "red junglefowl", "hen",
    "peacowl", "peafowl", "pea hen",
    "owl", "common owl",

    # --------- RAILS & COOTS ---------
    "rail", "corncrake", "water rail",
    "coot", "common coot", "american coot",
    "moorhen", "common moorhen", "purple swamphen",

    # --------- BUSTARDS ---------
    "bustard", "great bustard", "kori bustard", "little bustard",

    # --------- SANDPIPERS & PLOVERS ---------
    "sandpiper", "common sandpiper", "green sandpiper", "wood sandpiper",
    "plover", "golden plover", "grey plover", "ringed plover",
    "lapwing", "northern lapwing",
}

# ==================== REPTILE CLASSES ====================
REPTILE_CLASSES = {
    # --------- CROCODILIANS ---------
    "crocodile", "african crocodile", "nile crocodile", "saltwater crocodile",
    "american crocodile", "estuarine crocodile", "siamese crocodile",
    "alligator", "american alligator", "chinese alligator",
    "caiman", "spectacled caiman", "black caiman", "yacare caiman",
    "gharial", "false gharial",

    # --------- SNAKES ---------
    "python", "ball python", "burmese python", "african python",
    "cobra", "king cobra", "indian cobra", "egyptian cobra",
    "viper", "puff adder", "mamba", "black mamba", "green mamba",
    "anaconda", "green anaconda", "yellow anaconda",
    "boa", "emerald boa", "rainbow boa", "boa constrictor",
    "rattlesnake", "eastern diamondback", "western diamondback",
    "coral snake", "eastern coral snake", "texas coral snake",
    "sea snake", "boomslang", "vine snake", "whip snake",
    "garter snake", "grass snake", "water snake",

    # --------- LIZARDS ---------
    "lizard", "monitor lizard", "komodo dragon", "nile monitor",
    "iguana", "green iguana", "rhinoceros iguana",
    "chameleon", "panther chameleon", "veiled chameleon", "jackson's chameleon",
    "bearded dragon", "frilled lizard", "basilisk",
    "geckos", "leopard gecko", "crested gecko", "tokay gecko",
    "skink", "blue-tailed skink", "sand skink",
    "anole", "green anole", "brown anole",
    "agama", "painted agama", "collared lizard",

    # --------- TURTLES & TORTOISES ---------
    "turtle", "sea turtle", "green sea turtle", "loggerhead turtle",
    "hawksbill turtle", "leatherback turtle", "olive ridley turtle",
    "tortoise", "giant tortoise", "galapagos tortoise", "aldabra tortoise",
    "box turtle", "painted turtle", "red-eared slider",
    "snapping turtle", "alligator snapping turtle",
    "softshell turtle", "leatherback", "mud turtle",

    # --------- TUATARA ---------
    "tuatara",
}

# ==================== AMPHIBIAN CLASSES ====================
AMPHIBIAN_CLASSES = {
    # --------- FROGS & TOADS ---------
    "frog", "tree frog", "red-eyed tree frog", "white's tree frog",
    "poison dart frog", "golden poison dart frog", "blue poison dart frog",
    "bullfrog", "american bullfrog", "green frog", "leopard frog",
    "toad", "american toad", "cane toad", "colorado river toad",
    "arroyo toad", "boreal toad", "marine toad",
    "true frog", "common frog", "agile frog", "marsh frog",
    "spadefoot toad", "asian giant toad", "surinam toad",

    # --------- SALAMANDERS & NEWTS ---------
    "salamander", "tiger salamander", "spotted salamander",
    "newt", "red-spotted newt", "alpine newt", "great crested newt",
    "giant salamander", "chinese giant salamander", "japanese giant salamander",
    "axolotl", "lungless salamander", "two-lined salamander",

    # --------- CAECILIANS ---------
    "caecilian",
}

# ==================== FISH CLASSES ====================
FISH_CLASSES = {
    # --------- SHARKS & RAYS ---------
    "shark", "great white shark", "tiger shark", "bull shark",
    "hammerhead shark", "whale shark", "nurse shark", "lemon shark",
    "mako shark", "basking shark", "thresher shark",
    "ray", "manta ray", "eagle ray", "stingray", "electric ray",
    "sawfish", "guitarfish", "skate",

    # --------- SALMON & TROUT ---------
    "salmon", "atlantic salmon", "chinook salmon", "coho salmon",
    "sockeye salmon", "pink salmon", "chum salmon",
    "trout", "brown trout", "rainbow trout", "lake trout",
    "steelhead", "cutthroat trout", "brook trout",

    # --------- CATFISH ---------
    "catfish", "channel catfish", "blue catfish", "flathead catfish",
    "walking catfish", "armored catfish",

    # --------- BASS & PERCH ---------
    "bass", "largemouth bass", "smallmouth bass", "striped bass",
    "perch", "walleye", "yellow perch", "european perch",

    # --------- CARP & MINNOWS ---------
    "carp", "common carp", "koi", "grass carp", "bighead carp",
    "minnow", "goldfish", "danio", "tetra", "barb",

    # --------- CICHLIDS ---------
    "cichlid", "tilapia", "oscar fish", "angelfish", "discus fish",

    # --------- EELS ---------
    "eel", "electric eel", "garden eel", "moray eel", "conger eel",
    "freshwater eel", "gulper eel",

    # --------- SEAHORSES & PIPEFISH ---------
    "seahorse", "pipefish", "seadragon", "leafy seadragon",

    # --------- TUNA & MACKEREL ---------
    "tuna", "yellowfin tuna", "bluefin tuna", "skipjack tuna",
    "mackerel", "atlantic mackerel", "king mackerel", "spanish mackerel",

    # --------- FLATFISH ---------
    "flatfish", "flounder", "halibut", "plaice", "sole",
    "turbot", "dab",

    # --------- PUFFERFISH & RELATIVES ---------
    "pufferfish", "blowfish", "boxfish", "triggerfish", "filefish",
    "porcupinefish",

    # --------- COD & RELATED ---------
    "cod", "atlantic cod", "pacific cod", "pollock",
    "haddock", "whiting", "hake",

    # --------- GROUPER & SNAPPER ---------
    "grouper", "goliath grouper", "red snapper",

    # --------- EXOTIC & ORNAMENTAL FISH ---------
    "betta", "siamese fighting fish", "guppy", "molly",
    "platy", "swordtail", "corydoras", "loach",
    "arowana", "piranha", "gourami", "hatchetfish",

    # --------- JELLYFISH (Technically not fish, but aquatic) ---------
    "jellyfish", "sea anemone", "coral",
}

# All animal classes (for quick lookup)


# Every recognised species, across all five groups.
ANIMAL_CLASSES: frozenset[str] = frozenset(
    label.lower()
    for group in (
        MAMMAL_CLASSES,
        BIRD_CLASSES,
        REPTILE_CLASSES,
        AMPHIBIAN_CLASSES,
        FISH_CLASSES,
    )
    for label in group
)

SPECIES_COUNT = len(ANIMAL_CLASSES)


def is_animal_image(classification_label: str | None) -> bool:
    """True when a classification label names an animal we route specially.

    ImageNet class names are comma-separated synonym lists, for example
    "African elephant, Loxodonta africana". Any synonym matching is enough,
    so the gate works whether it is handed the raw class name or a cleaned
    common name.
    """
    if not classification_label:
        return False

    label = classification_label.strip().lower()
    if label in ANIMAL_CLASSES:
        return True

    return any(
        part.strip() in ANIMAL_CLASSES
        for part in label.split(",")
        if part.strip()
    )
