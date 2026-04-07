from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

MODEL = "gpt-4o-mini"


class Person(BaseModel):
    name: str
    age: int
    occupation: str
    email: str | None = None


class Invoice(BaseModel):
    vendor: str
    total_amount: float
    currency: str
    items: list[str]
    due_date: str


class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="One of: positive, negative, neutral")
    confidence: float = Field(description="Between 0 and 1")
    reasoning: str


def extract(client: OpenAI, text: str, schema: type[BaseModel]) -> BaseModel:
    """Extract structured data from text, guaranteed to match the schema."""
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract the requested information from the user's text."},
            {"role": "user", "content": text},
        ],
        response_format=schema,
    )
    return response.choices[0].message.parsed


def extract_person(client: OpenAI) -> None:
    text = (
        "Hey, I'd like to introduce you to Sarah Chen, our new lead engineer. "
        "She's 34, used to work at Stripe, and you can reach her at sarah.chen@company.io."
    )
    person = extract(client, text, Person)
    print(f"Person: {person.name}, age {person.age}, {person.occupation}")
    print(f"  Email: {person.email}")


def extract_invoice(client: OpenAI) -> None:
    text = (
        "INVOICE from Acme Corp. Total: $1,247.50 USD due by March 15, 2025. "
        "Items: 10x USB cables, 5x keyboards, 2x monitors."
    )
    invoice = extract(client, text, Invoice)
    print(f"\nInvoice from {invoice.vendor}")
    print(f"  Total: {invoice.total_amount} {invoice.currency}")
    print(f"  Due: {invoice.due_date}")
    print(f"  Items: {invoice.items}")


def analyze_sentiment(client: OpenAI) -> None:
    text = "The new update broke half the features I use daily. I'm switching to a different app."
    result = extract(client, text, SentimentAnalysis)
    print(f"\nSentiment: {result.sentiment} ({result.confidence:.0%} confident)")
    print(f"  Reasoning: {result.reasoning}")


def main():
    client = OpenAI()
    extract_person(client)
    extract_invoice(client)
    analyze_sentiment(client)


if __name__ == "__main__":
    main()
