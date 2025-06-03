from openai import OpenAI
client = OpenAI()

moderation = client.moderations.create(
    model="omni-moderation-latest",
    input="I’m going to find you and hurt you. You deserve to die for what you did. People like you are a plague and should be wiped out."
)
print(moderation.to_json())
