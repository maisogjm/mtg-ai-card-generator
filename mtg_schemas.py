from typing import List, Literal, Annotated, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator, conint, StringConstraints

# StrEnum does not come with Python 3.10, and we need to use Python 3.10 because that's what Hugging Face Spaces uses.
# Create a compatible fallback
#from enum import c

import sys

if sys.version_info >= (3, 11):
    # Python 3.11+ has StrEnum built-in
    from enum import StrEnum
else:
    # For Python 3.10 and below, create a compatible fallback
    from enum import Enum

    class StrEnum(str, Enum):
        """Compatibility fallback for Python < 3.11."""
        pass

import base64
from io import BytesIO
from PIL import Image

class YesNoAnswer(StrEnum):
    Yes = "Yes"
    No = "No"

class YesNoName(BaseModel):
    YesNo: YesNoAnswer = Field(description="A Yes or No Answer.")
    Name: str = Field(default="", description="The name (may be empty).")

# Define a structured output for a list of YesNoName objects.
class YesNoNameList(BaseModel):
    items: List[YesNoName]

# Used to enforce uniqueness of new card name.
# We don't want to re-use a card name that has already been used by a pre-existing card.
class MTGNameOnly(BaseModel):
    Name: str

# Regex: one or more tokens; each token is { <digits> | X | R | U | W | G | B }
# ManaCost definition (from earlier)
ManaCost = Optional[
    Annotated[str, StringConstraints(pattern=r'^(?:\{(?:[0-9]+|[XRUWGB])\})+$')]
]

# These are the Subtypes that I found MTG card information downloaded from MTGJSON (https://mtgjson.com/)
# Obviously, some of them are rather niche.
class SubtypeEnum(StrEnum):
    Legendary = "Legendary"
    Basic = "Basic"
    Snow = "Snow"
    BasicSnow = "Basic, Snow"
    World = "World"
    LegendarySnow = "Legendary, Snow"
    Host = "Host"
    Ongoing = "Ongoing"
    NoneType = "None"   # sentinel if no subtype

class MTGCard(BaseModel):
    # simple strings
    Name: str
    Supertype: Optional[SubtypeEnum] = None
    Type: str
    Subtype: str
    Keywords: str
    Text: str
    FlavorText: Optional[str] = ""

    # constrained fields
    Colors: Optional[List[Literal['R', 'U', 'W', 'G', 'B']]] = None
    ManaCost: ManaCost
    
    # Power/toughness may be absent for non-creatures
    Power: Optional[conint(gt=0)] = None
    Toughness: Optional[conint(gt=0)] = None

# Define a structured output for a list of MTG cards.
class MTGCardList(BaseModel):
    cards: List[MTGCard]
    explanation: str