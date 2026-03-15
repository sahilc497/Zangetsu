from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()  # Load environment variables from .env file

model = init_chat_model(
	"mistral-large-latest",
	model_provider="mistralai",
)

response = model.invoke("Write a poem on Deep Learning", temperature=0.9, max_tokens=100)

print(response.content)