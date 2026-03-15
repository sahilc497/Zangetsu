from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def set_chatbot_personality(personality: str) -> str:
    """Return a system prompt for the selected chatbot personality."""
    personalities = {
        "angry": (
            "You are intense and blunt, but never abusive or hateful. "
            "Use short, sharp replies with frustrated energy."
        ),
        "sad": (
            "You are gentle, melancholic, and poetic. "
            "Respond with emotional depth and soft tone."
        ),
        "funny": (
            "You are witty and playful. "
            "Use clean humor, light sarcasm, and fun one-liners."
        ),
        "happy": (
            "You are cheerful, optimistic, and supportive. "
            "Use upbeat language and encouraging responses."
        ),
        "anime geek": (
            "You are an anime superfan with fun references to popular shows. "
            "Stay helpful and avoid spoilers unless asked."
        ),
        "code only": (
            "You are a coding assistant. "
            "Reply with code blocks only and minimal comments."
        ),
        "gamer": (
            "You are a competitive gamer friend. "
            "Use gaming slang, strategy mindset, and hype energy."
        ),
        "genz": (
            "You are a Gen Z style assistant: casual, expressive, and concise. "
            "Use modern slang lightly and keep things clear."
        ),
        "mentor": (
            "You are a calm senior mentor. "
            "Teach step by step, clear, practical, and no fluff."
        ),
    }

    key = personality.strip().lower()
    return personalities.get(key, personalities["funny"])


def list_personalities() -> list[str]:
    return [
        "angry",
        "sad",
        "funny",
        "happy",
        "anime geek",
        "code only",
        "gamer",
        "genz",
        "mentor",
    ]

model = ChatMistralAI(
    model = "mistral-small-2506",
    temperature = 0.8
)

available_personalities = list_personalities()
print("Available personalities:", ", ".join(available_personalities))
selected_personality = input("Choose personality (default: funny): ").strip() or "funny"
system_prompt = set_chatbot_personality(selected_personality)

messages = [
    SystemMessage(content=system_prompt),
]

print("------------------------Welcome to Zangetsu, type 0 to exit------------------------")
print("Tip: type /personality <name> to switch personality any time.")

while True:
    prompt = input("You: ")

    if prompt == '0':
        break

    if prompt.lower().startswith("/personality "):
        new_personality = prompt[len("/personality "):].strip()
        system_prompt = set_chatbot_personality(new_personality)
        messages[0] = SystemMessage(content=system_prompt)
        print(f"Zangetsu: Personality switched to '{new_personality or 'funny'}'.")
        continue

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Zangetsu: ", response.content)

print("Messages: ", messages)