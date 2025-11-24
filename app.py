#!/usr/bin/env python
# coding: utf-8

import os
import json
import base64
from io import BytesIO

from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# Pydantic schemas (your "original" structured outputs)
from mtg_schemas import MTGCard, YesNoName

# ============================================================================
# ENV + CLIENT SETUP
# ============================================================================

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeWarning("No usable OpenAI API key found in environment.")
openai_client = OpenAI(api_key=openai_api_key)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openai_api_key:
    raise RuntimeWarning("No usable OpenRouter API key found in environment.")

openrouter_url = "https://openrouter.ai/api/v1"

clients = {
    "openai": openai_client,
    "openrouter": OpenAI(api_key=openrouter_api_key, base_url=openrouter_url) if openrouter_api_key else None,
}

# Model mappings
card_models = {
    "gpt-5.1": "openai",
    "gpt-4.1": "openai",
    "gpt-4.1-nano-2025-04-14": "openai",
    "gpt-4o-mini": "openai",
    "ft:gpt-4.1-nano-2025-04-14:personal::CR0Okw3U": "openai",
    "x-ai/grok-4-fast": "openrouter",
    "deepseek/deepseek-chat-v3.1": "openrouter",
    "meta-llama/llama-3.2-3b-instruct": "openrouter",
    "qwen/qwen3-vl-30b-a3b-instruct": "openrouter"
}

extract_models = card_models.copy()

image_models = {
    "dall-e-3": "openai",
    "gpt-image-1": "openai",
}

system_prompt = (
    """You are a creative and imaginative designer of cards for the collectible/trading card game
    Magic: The Gathering. Respond only with a single JSON object that matches the schema.
    If the card has a non-null mana cost, try to match the mana cost with the potency of the card.
    I.e., creatures with high Power and/or Toughness should tend to cost more; and instants that
    cause more damage should tend to cost more. Keep in mind that Lands typically do not cost mana.
    Most (82%) MTG cards have a NaN (missing) Supertype value; the most common non-missing Supertype value is 'Legendary',
    accounting for 14% of all cards. It is OK to generate a card with a missing/None Supertype value!
    In fact, if the card is a common and/or low-powered creature or artifact, or if it isn't a creature or artifact to begin with,
    it might be best to just have Supertype with a value of None (missing).
    The top six most common Type values are (in decreasing order): Creature, Land, Instant, Sorcery, Enchantment, and Artifact.
    Creatures are the most common Type value, accounting for about 44% of all cards.
    Land cards are the next most common Type.
    A large proportion of (38%) cards have a missing Subtype."""
)

# ============================================================================
# CARD NAME DATABASE
# ============================================================================

try:
    CARD_NAMES_FILE = "cardnames.txt"
    with open(CARD_NAMES_FILE, "r", encoding="utf-8", errors="replace") as f:
        card_names = set(f.read().splitlines())
    print(f"✓ Loaded {len(card_names)} existing card names")
except FileNotFoundError:
    print("⚠ Card names file not found, starting with empty set")
    card_names = set()

# ============================================================================
# HELPER FUNCTIONS (ported from mtg_gradio_v9, adapted to Streamlit)
# ============================================================================

def get_client(model_name: str, model_dict: dict) -> OpenAI:
    """Get the appropriate client for a given model."""
    provider = model_dict.get(model_name)
    if provider is None:
        raise ValueError(f"Unknown model: {model_name}")
    client = clients.get(provider)
    if client is None:
        raise ValueError(
            f"Client not configured for provider: {provider}. "
            "Check that the corresponding API key is set."
        )
    return client


def ExtractNameIfAny(txt: str, extract_model: str) -> str:
    """Extract card name from user text if specified."""
    client = get_client(extract_model, extract_models)

    msg = f"""Here is some text.
<TEXT>
{txt}
</TEXT>
If the text includes a request to specify the name of an item (e.g., a card), respond with 'Yes', and also give the specified name.
Otherwise, respond with 'No', and leave the name unspecified.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]

    completion = client.beta.chat.completions.parse(
        model=extract_model,
        messages=messages,
        response_format=YesNoName,
        temperature=0.2,
    )

    yesnoname = completion.choices[0].message.content
    return yesnoname


def GenerateCommaSeparatedColorString(parsed_colors):
    """Convert color codes to comma-separated color names."""
    if parsed_colors is None or len(parsed_colors) == 0:
        return ""

    color_map = {"W": "white", "U": "blue", "B": "black", "G": "green", "R": "red"}
    color_list = [color_map[c] for c in parsed_colors if c in color_map]
    return ",".join(color_list)


def GenerateCardImage(parsed: MTGCard, msg: str, image_model: str) -> Image.Image:
    """Generate an image for the MTG card."""
    client = get_client(image_model, image_models)

    Name = parsed.Name
    FlavorText = parsed.FlavorText or ""
    Type = parsed.Type
    Subtype = parsed.Subtype

    # Clean flavor text
    FlavorText = FlavorText.replace('"', "").replace("'", "").replace("—", "")

    # Handle colors
    color_string = GenerateCommaSeparatedColorString(parsed.Colors)
    preamble = f"Use a palette with a moderate emphasis on the color {color_string}, " if color_string else ""

    # Check for flying
    has_flying = False
    if parsed.Keywords:
        has_flying = "flying" in parsed.Keywords.lower()
    if not has_flying and parsed.Text:
        has_flying = "flying" in parsed.Text.lower()
    flying_prefix = "flying " if has_flying else ""

    image_prompt = f"""{preamble}generate a picture illustrating: {flying_prefix}{Subtype}, {Type}, {FlavorText}, {Name}, {msg}

[STYLE & MEDIUM]
Art style: epic fantasy.
Medium: digital matte painting

[RENDERING DETAILS]
Clarity: highly detailed, sharp focus
Resolution intent: designed for 256×256, suitable for a playing card size.
No artifacts, clean edges, refined image quality.
Not too dark, so that the artwork is easily visible.

[NEGATIVE PROMPTS]
Do NOT include:
- No text, typography, labels, symbols, numbers, or visible writing.
- No copyrighted characters.
- No watermarks, signatures, logos.
- No warped hands.
- No borders along the edges.

[TECHNICAL OVERRIDES]
This is a standalone image prompt. Follow the constraints strictly.
"""

    try:
        if image_model == "gpt-image-1":
            image_response = client.images.generate(
                model=image_model,
                prompt=image_prompt,
                size="1024x1024",
                n=1
            )
        elif image_model == "dall-e-3":
            image_response = client.images.generate(
                model=image_model,
                prompt=f"""{image_prompt}.
                           High fantasy epic oil painting style, with NO text, NO visible signs, NO writing, NO labels, NO lettering in the image.""",
                size="1024x1024",
                n=1,
                response_format="b64_json",
                quality="hd"
            )
        else:
            raise ValueError("Unknown image generator:", image_model)
            
    except Exception as e:
        msg = str(e)
        if "safety system" in msg.lower() or "content_policy_violation" in msg.lower():
            # Raise a special exception that CreateCard() will catch
            raise RuntimeError("IMAGE_SAFETY_BLOCKED")
        raise        # Other exceptions re-raise normally

        
    # Extract base64-encoded image data
    image_base64 = image_response.data[0].b64_json
    if image_base64 is None:
        raise RuntimeError("IMAGE_RESPONSE_EMPTY")
    
    # Decode it and display
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def generate_unique_name_for_card(parsed_card, used_names, extract_model):
    """
    Ask the LLM to generate a new, unique card name,
    consistent with the card's other attributes.
    """

    client = get_client(extract_model, extract_models)

    card_info = json.dumps(parsed_card.model_dump(), indent=2)

    prompt = f"""
You must generate a NEW, UNIQUE name for this Magic: The Gathering card.

Here are all of the card's attributes except the name:
<card>
{card_info}
</card>

Requirements:
- Do NOT reuse any name in the following list:
{list(used_names)}
- The new name MUST NOT match any existing card name.
- The new name MUST match the style, color identity, type, subtype, flavor,
  and general theme of the provided card.
- Respond ONLY with a single JSON object containing the field "Name".
"""

    completion = client.beta.chat.completions.parse(
        model=extract_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=MTGNameOnly,
        temperature=0.4,     # low temperature is best for names
    )

    return completion.choices[0].message.parsed.Name


def CreateCard(msg: str, card_model: str, extract_model: str, image_model: str, temp: float):
    """Main function to create an MTG card."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]

    # Check if name already exists
    try:
        yesnoname = ExtractNameIfAny(msg, extract_model)
        yesnoname_dict = json.loads(yesnoname)
        if yesnoname_dict.get("YesNo") == "Yes":
            card_name = yesnoname_dict.get("Name", "")
            if card_name and card_name in card_names:
                return None, (
                    f"❌ Sorry, the name '{card_name}' has already been used. "
                    "Please select another name or leave it unspecified."
                )
    except Exception as e:
        # Non-fatal; just log to console
        print(f"Warning: Name extraction failed: {e}")

    # Try to create card (with retries for duplicate names)
    max_card_attempts = 5     # regenerate card up to 5 times
    max_image_attempts = 3    # retry image generation 3 times per card
    for attempt in range(max_card_attempts):
        try:
            client = get_client(card_model, card_models)

            completion = client.beta.chat.completions.parse(
                model=card_model,
                messages=messages,
                response_format=MTGCard,
                temperature=temp,
            )

            parsed: MTGCard = completion.choices[0].message.parsed
            
            # If name already used, regenerate ONLY the name
            if parsed.Name in card_names:
                try:
                    new_name = generate_unique_name_for_card(
                        parsed_card=parsed,
                        used_names=card_names,
                        extract_model=extract_model,
                    )
                    parsed.Name = new_name
                except Exception as e:
                    print(f"⚠ Failed to generate replacement name: {e}")
                    continue  # fallback: try regenerating whole card


            # Try generating the image (may need retries)
            image = None
            for img_attempt in range(max_image_attempts):
                try:
                    image = GenerateCardImage(parsed, msg, image_model)
                    break
                except RuntimeError as e:
                    if "IMAGE_SAFETY_BLOCKED" in str(e):
                        print("⚠ Image safety block. Regenerating full card...")
                        break   # break image loop → regenerate card
                    else:
                        # Other image errors: retry image only
                        print(f"⚠ Image error (attempt {img_attempt+1}): {e}")
                        continue
            if image is None:
                # A safety block triggered — regenerate entire card
                continue

            # Success
            card_info = json.dumps(parsed.model_dump(), indent=4, ensure_ascii=False)
            output_text = card_info
            return image, output_text

        except ValidationError as ve:
            return None, f"❌ Validation Error: {ve}"

        except Exception as e:
            print(f"❌ Unexpected error while generating card: {e}")
            continue

    return None, "❌ Failed to generate a safe card after several attempts."

# pretty-printing function
def format_card_info(raw_json: str) -> str:
    """
    Transform the raw JSON dump into a nicer human-readable block:
    - Remove quotes and commas
    - Rename keys (OriginalText → Original Text, etc.)
    - Flatten Colors list into comma-separated string
    - Convert None → None
    """
    try:
        data = json.loads(raw_json)
    except Exception:
        return raw_json  # fallback
    
    # Key renaming map
    rename = {
        "OriginalText": "Original Text",
        "FlavorText": "Flavor Text",
        "ManaCost": "Mana Cost",
    }

    # Build formatted lines
    lines = []

    for key, value in data.items():
        # Rename key if applicable
        pretty_key = rename.get(key, key)

        # Process Colors list
        if key == "Colors":
            if isinstance(value, list):
                value_str = ", ".join(value)
            else:
                value_str = "None"
        # Process None
        elif value is None:
            value_str = "None"
        else:
            value_str = str(value)

        # Remove quotes from value_str (clean but safe)
        value_str = value_str.replace('"', "")

        lines.append(f"{pretty_key}: {value_str}")

    return "\n".join(lines)


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="MTG Card Generator", layout="wide")

st.title("🎴 Magic: The Gathering Card Generator")
st.markdown(
    "Generate custom MTG cards with **Pydantic-validated structured output** and AI-generated artwork."
)

# Initialize session state for image and card info
if "card_image" not in st.session_state:
    st.session_state["card_image"] = None
if "card_info" not in st.session_state:
    st.session_state["card_info"] = ""

# Top: large image area
st.markdown("### Generated Card")

col_img, col_info = st.columns([1, 1])

with col_img:
    st.markdown("#### A.I. Image")
    if st.session_state["card_image"] is not None:
        st.image(
            st.session_state["card_image"],
            caption="Generated MTG Card Artwork",
            use_column_width=False,
            width=512
        )
    else:
        st.info("No card generated yet. Enter a description below and click **Generate Card**.")

with col_info:
    st.markdown("#### Card Information")
    if st.session_state["card_info"]:
        raw = st.session_state["card_info"]

        # Remove optional prefix like "✓ Generated Card:" 
        if raw.startswith("✓ Generated Card:"):
            # Split on the first '{'
            _, json_part = raw.split("{", 1)
            raw_json = "{" + json_part.strip()
        else:
            raw_json = raw

        pretty = format_card_info(raw_json)
        st.text_area(
            "Formatted Card Info",
            value=pretty,
            height=500
        )
    else:
        st.info("Card details will appear here after generation.")


# Input + Submit row
st.markdown("---")
st.markdown("#### User Prompt")
col_prompt, col_button = st.columns([4, 1])

with col_prompt:
    user_prompt = st.text_area(
        "Card Description",
        value="Please generate a new MTG card.",
        height=120,
    )

with col_button:
    st.write("")  # spacing
    st.write("")
    generate_btn = st.button("Generate Card", type="primary")

# Error message placeholder (show errors above Settings)
error_placeholder = st.empty()

# Settings section
st.markdown("---")
st.markdown("#### ⚙️ Settings")
col1, col2 = st.columns(2)
with col1:
    card_model_choice = st.selectbox(
        "Card Generation Model",
        options=list(card_models.keys()),
        index=list(card_models.keys()).index("gpt-4o-mini"),
    )
    extract_model_choice = st.selectbox(
        "Name Extraction Model",
        options=list(extract_models.keys()),
        index=list(extract_models.keys()).index("gpt-4.1-nano-2025-04-14"),
    )
with col2:
    image_model_choice = st.selectbox(
        "Image Generation Model",
        options=list(image_models.keys()),
        index=list(image_models.keys()).index("gpt-image-1"),
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1)

# On submit
if generate_btn:
    if not user_prompt.strip():
        error_placeholder.warning("Please enter a description for the card.")
    else:
        try:
            with st.spinner("Generating card..."):
                image, card_info = CreateCard(
                    msg=user_prompt.strip(),
                    card_model=card_model_choice,
                    extract_model=extract_model_choice,
                    image_model=image_model_choice,
                    temp=temperature,
                )

            if image is None:
                error_placeholder.error(card_info or "Unknown error occurred.")
            else:
                st.session_state["card_image"] = image
                st.session_state["card_info"] = card_info
                st.success("Card generated successfully!")
                st.rerun()   # ← Force UI to update immediately
        except Exception as e:
            error_placeholder.error(f"Unexpected error: {e}")

# Card info output
# st.markdown("### Card Information")
# if st.session_state["card_info"]:
    # st.text_area(
        # "Structured Card JSON",
        # value=st.session_state["card_info"],
        # height=300,
    # )
# else:
    # st.info("Card details will appear here after generation.")
