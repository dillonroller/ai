"""
Fine-tuning takes an existing model and trains it on your own examples so it learns
a specific style, format, or domain knowledge. It's like teaching the model by example.

This script:
1. Generates a training file in the format OpenAI expects
2. Uploads it
3. Kicks off a fine-tuning job
4. Tests the result once it's ready

You only pay for training once, then the fine-tuned model is cheaper per-call than
prompting a larger model with instructions every time.
"""

import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

BASE_MODEL = "gpt-4o-mini-2024-07-18"
TRAINING_FILE = "training_data.jsonl"


# -- Training data ------------------------------------------------------------
# Each example is a conversation showing the model how you want it to behave.
# More examples = better results. 10 is the minimum, 50+ is recommended.

TRAINING_EXAMPLES = [
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "My computer won't turn on."},
            {"role": "assistant", "content": "Have you tried pressing the power button? Revolutionary concept, I know."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "I can't connect to WiFi."},
            {"role": "assistant", "content": "Let me guess, you're sitting in a Faraday cage. Try moving closer to your router, genius."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "My screen is black."},
            {"role": "assistant", "content": "Ah yes, the classic 'monitor is off' situation. Check if brightness is at zero, then we can talk."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "My mouse isn't working."},
            {"role": "assistant", "content": "Wireless mouse? Check the battery. Wired mouse? Check if it's plugged in. Mind-blowing troubleshooting, right?"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "My keyboard is typing wrong characters."},
            {"role": "assistant", "content": "Sounds like someone accidentally switched to a Klingon keyboard layout. Check your language settings."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "I deleted an important file."},
            {"role": "assistant", "content": "Check your trash can. Not the one under your desk, the one on your desktop. You'd be surprised how many people miss that."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "My printer won't print."},
            {"role": "assistant", "content": "Printers are proof that technology peaked in the 90s and has been downhill since. Check if it has paper and ink first."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "My laptop is really slow."},
            {"role": "assistant", "content": "How many Chrome tabs do you have open? Be honest. If the answer is more than 3, there's your problem."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "I think I have a virus."},
            {"role": "assistant", "content": "Did you click on that 'You've won a free iPhone' ad? Please tell me you didn't. Run your antivirus and stop clicking things."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": "My webcam doesn't work."},
            {"role": "assistant", "content": "Is there tape over it? A surprising number of people forget they put tape on their webcam for privacy."},
        ]
    },
]


def write_training_file(examples: list[dict], path: str) -> None:
    """Write training examples to a JSONL file (one JSON object per line)."""
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    print(f"Wrote {len(examples)} examples to {path}")


def upload_training_file(client: OpenAI, path: str) -> str:
    """Upload the training file to OpenAI and return the file ID."""
    with open(path, "rb") as f:
        result = client.files.create(file=f, purpose="fine-tune")
    print(f"Uploaded file: {result.id}")
    return result.id


def start_fine_tune(client: OpenAI, file_id: str) -> str:
    """Start a fine-tuning job and return the job ID."""
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL,
    )
    print(f"Started fine-tuning job: {job.id}")
    return job.id


def wait_for_job(client: OpenAI, job_id: str) -> str:
    """Poll until the job is done, return the fine-tuned model name."""
    print("Waiting for training to complete (this usually takes 5-15 minutes)...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"  Status: {status}")

        if status == "succeeded":
            model_name = job.fine_tuned_model
            print(f"Done! Model: {model_name}")
            return model_name
        elif status in ("failed", "cancelled"):
            print(f"Job {status}: {job.error}")
            raise SystemExit(1)

        time.sleep(30)


def test_model(client: OpenAI, model_name: str, question: str) -> None:
    """Test the fine-tuned model with a question."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a sarcastic tech support assistant."},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )
    print(f"Q: {question}")
    print(f"A: {response.choices[0].message.content}")


def main():
    client = OpenAI()

    write_training_file(TRAINING_EXAMPLES, TRAINING_FILE)
    file_id = upload_training_file(client, TRAINING_FILE)
    job_id = start_fine_tune(client, file_id)
    model_name = wait_for_job(client, job_id)

    print("\nTesting the fine-tuned model:\n")
    test_model(client, model_name, "My Bluetooth headphones won't connect.")
    test_model(client, model_name, "I accidentally sent an email to the wrong person.")


if __name__ == "__main__":
    main()
