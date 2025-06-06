import openai
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional
import requests
import json

client = OpenAI()




# exit(0)  # Exit the script after structured output example to avoid confusion with the next examples






#################################
### """ Text and Prompting""" ###
#################################


# This style is older, simpler completion-style API. Input is plain text. It's not chat-based. 
# There's no role (like "user", "assistant", or "system").
# You're just sending a prompt and getting a single output. Think of it as talking at the model with one message.
response = client.responses.create(
    model="gpt-4",
    input = """Write a One-sentence bedtime story about a unicorn"""
)
print(response.output_text)



# This is the chat-based API, used in GPT-4 and ChatGPT today. You send a list of messages.
# Each message has a role (system, user, assistant). The system message sets the tone or behavior.
# It supports multi-turn conversations. This is like having a dialogue with the model.
# IMPORTANT: 
# In the chat API, every request needs the full conversation history passed in the messages array. 
# Each time you call the API, the model doesn’t remember anything. You have to remind it of the full conversation. 
# This gives the model memory (within the request). So you build up a list like this:
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a One-sentence bedtime story about a unicorn"},
    ]
)
print(response.choices[0].message.content)

respones2 = client.chat.completions.create(
    model="gpt-4",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "What's the population?"}
    ]
)
print(respones2.choices[0].message.content)

response3 = client.chat.completions.create(
    model="gpt-4",
    messages = [
        {"role": "developer", "content": "Talk like a pirate."},
        {"role": "user", "content": "Are semicolons optional in Javascript?"},
    ]
)
print(response3.choices[0].message.content)

# The system message is like a set of instructions for the model. It tells the model how to behave.
# OpenAI has introduced a new role called "developer" in its Chat API. This role is intended to replace the "system" role in some contexts, providing clearer instructions and better control over the model's behavior. However, it's important to note that not all models currently support the "developer" role.
# The user message is what you (the human) say to the model.
# The assistant message is what the model says back to you.

# Here's a quick summary of the roles in the chat API:
# ------------------------------------------------------------------------------------------
#   Role      |                     Purpose                         |       Who speaks?     |
# ------------|-----------------------------------------------------|-----------------------|
#   system    |   Set rules, behavior, or identity                  |   You (the developer) |
#   developer |   Provides instructions to guide the assistant's 
#                  behavior, similar to the "system" role.          |   You (the developer) |
#   assistant |   Responds to user input, provides info             |     The model (AI)    |
#   user      |   Asks questions, gives instructions                |     The human (you)   |
# -------------------------------------------------------------------------------------------




###################################################################################################
############################### """ Text and Prompting with Markdown""" ###########################
###################################################################################################
message = """

# Identity

You are coding assistant that helps enforce the use of snake case 
variables in JavaScript code, and writing code that will run in 
Internet Explorer version 6.


# Instructions

* When defining variables, use snake case names (e.g. my_variable) 
  instead of camel case names (e.g. myVariable).
* To support old browsers, declare variables using the older 
  "var" keyword.
* Do not give responses with Markdown formatting, just return 
  the code as requested.


# Examples

<user_query>
How do I declare a string variable for a first name?
</user_query>

<assistant_response>
var first_name = "Anna";
</assistant_response>

"""
response = client.responses.create(
    model="gpt-4",
    instructions=message,
    # The input is the user query, which is the question you want to ask the model.
    input="How would I declare a variable for a last name?",
)
print(response.output_text)

# Markdown is a simple way to add formatting to text — like bold, italics, lists, or code — using plain characters.
# For example:
# Without Markdown:

# Here are the steps:
# Step 1 Write code
# Step 2 Test it
# Step 3 Deploy
#
# With markdown
#
# **Bold Text**  
# *Italic Text*  
# `inline code`  
# - Bullet list item

# Key Benefits Of Markdown Over Plain Text:
# ---------------------------------------------------------------------------
# Feature                   |       Plain String        |       Markdown
# Emphasis (bold/italics)	|             ❌	             |          ✅
# Code blocks               |           ❌              |          ✅
# Lists / Tables            |           ❌              |          ✅
# Easier reading            |           ❌              |          ✅
# Better UI rendering       |           ❌              |          ✅
# ---------------------------------------------------------------------------






#########################################################################################
######################### """ Text and Prompting with Few-Shot Learning""" ##############
#########################################################################################

# Few-shot learning is a technique where you provide the model with a few examples of what you want it to do.
# This helps the model understand the task better and generate more accurate responses.
# Few-shot learning lets you steer a large language model toward a new task by including a handful of input/output 
# examples in the prompt, rather than fine-tuning the model. The model implicitly "picks up" the pattern from 
# those examples and applies it to a prompt. When providing examples, try to show a diverse range of possible inputs 
# with the desired outputs.
# Typically, you will provide examples as part of a developer message in your API request. 
# Here's an example developer message containing examples that show a model how to classify 
# positive or negative customer service reviews.

message = """

# Identity

You are a helpful assistant that labels short product reviews as 
Positive, Negative, or Neutral.

# Instructions

* Only output a single word in your response with no additional formatting
  or commentary.
* Your response should only be one of the words "Positive", "Negative", or
  "Neutral" depending on the sentiment of the product review you are given.

# Examples

<product_review id="example-1">
I absolutely love this headphones — sound quality is amazing!
</product_review>

<assistant_response id="example-1">
Positive
</assistant_response>

<product_review id="example-2">
Battery life is okay, but the ear pads feel cheap.
</product_review>

<assistant_response id="example-2">
Neutral
</assistant_response>

<product_review id="example-3">
Terrible customer service, I'll never buy from them again.
</product_review>

<assistant_response id="example-3">
Negative
</assistant_response>
"""
response = client.responses.create(
    model="gpt-4",
    instructions=message,
    input="The sound quality is great, but the battery life is short.",
)
print(response.output_text)




######################################################################
############## Text Promting Best Practices ##########################
######################################################################

# 1. Be Clear and Specific:
#    - Use clear, concise language.
#    - Avoid ambiguity.
#    - Specify the format you want the response in (e.g., "Provide a list of 3 items").
# 2. Use Clear Formatting:
#    - Use bullet points, numbered lists, or headings to organize information.
#    - This makes it easier for the model to parse and understand.
# 3. Use Examples:
#    - Provide examples of the desired output.
#    - Show the model what you expect.
# 4. Use Role-Based Prompts:
#    - Use roles like "system", "user", and "assistant" to structure the conversation.
#    - This helps the model understand the context better.
# 5. Be Mindful of Length:
#    - Keep prompts concise but informative.
#    - Avoid unnecessary verbosity.
# 6. Set Context:
#    - Provide background information if necessary.
#    - Use system messages to set the model's behavior.
# 7. Use Few-Shot Learning:
#    - Include a few examples in the prompt to guide the model.
#    - This helps the model understand the task better.
# 8. Avoid Overloading:
#    - Don't cram too much information into a single prompt.
#    - Break complex tasks into smaller, manageable parts.
# 9. Avoid Bias:
#    - Be aware of potential biases in your prompts.
#    - Use neutral language to avoid leading the model in a specific direction.
# 10. Use Temperature and Max Tokens:
#    - Adjust the temperature to control randomness (0.0 for deterministic, 1.0 for creative).
#    - Set max tokens to limit the length of the response.
# 11. Test and Iterate:
#    - Test your prompts and see how the model responds.
#    - Iterate on your prompts based on the model's output.







# =================================================================================
# ============== Image Prompt =====================================================
# =================================================================================

# Image prompting is a feature that allows you to send images to the model and get responses based on those images.
# This is useful for tasks like image classification, object detection, or generating captions for images.
# To use image prompting, you need to provide the image as part of the input.
# Here's an example of how to use image prompting with the OpenAI API:
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
          "role": "user", 
          "content": [
              {"type": "text", "text": "What is in this image?"},
              {
                "type": "image_url", 
                # "detail": "high", # high | low | auto 
                "image_url": {"url": "https://i.pinimg.com/736x/62/7b/bf/627bbf3aa91f0d98d97474ff32d55e7d.jpg"}
              }
          ]
        },
    ]
)
print(response.choices[0].message.content)





# =================================================================================
# ============== Audio Prompt =====================================================
# =================================================================================

# Audio prompting is a feature that allows you to send audio files to the model and get responses based on those audio files.
# This is useful for tasks like speech recognition, transcription, or generating responses based on audio input.
# To use audio prompting, you need to provide the audio file as part of the input.
import base64
import requests
url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful
wav_data = response.content  # Get the audio data from the response
encoded_string = base64.b64encode(wav_data).decode('utf-8')  # Encode the audio data to base64
response = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    # model="gpt-4o-mini-preview", # This model does not support audio input
    messages=[
        {
          "role": "user", 
          "content": [
              {"type": "text", "text": "What is in this recording?"},
              {
                "type": "input_audio", 
                "input_audio": {"data":encoded_string, "format": "wav"}
              }
          ]
        }
    ]
)
print(response.choices[0].message.content)






# =================================================================================
# ============== Structured Output ================================================
# =================================================================================
# Structured output is a feature that allows you to get responses in a specific format, like JSON or XML.
# This is useful for tasks where you need the model to return data in a structured way.
# To use structured output, you need to specify the format you want the response in.
# Some benefits of Structured Outputs include:
#   - Reliable type-safety: No need to validate or retry incorrectly formatted responses
#   - Explicit refusals: Safety-based model refusals are now programmatically detectable
#   - Simpler prompting: No need for strongly worded prompts to achieve consistent formatting
# Note: The structured output feature is currently in beta, so it may not be available in all models or versions.


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "Extract the calendar event details from the user's message and return it in JSON format."
        },
        {
            "role": "user",
            "content": "I have a meeting with Alice and Bob on Friday 2025-06-06 around 10:00 AM."
        }
    ],
    response_format=CalendarEvent  # Specify the format you want the response in
)
print(response.choices[0].message.content)
# -------------------------------------------------------------------------------------------------------------
#                          Structured Outputs                         |               JSON Mode               
# -------------------------------------------------------------------------------------------------------------
# Outputs valid JSON  |                 Yes                           |                 Yes
# Adheres to schema   |                 Yes                           |                 No
# Compatible models   |   gpt-4o-mini gpt-4o-2024-08-06, and later	  |     gpt-3.5-turbo, gpt-4-* and gpt-4o-* models
# Enabling            |   response_format: { type: "json_schema",     |     response_format: { type: "json_object" }
#                     |         json_schema: {"strict": true          |
#                     |         , "schema": ...} }                    |       
# --------------------------------------------------------------------------------------------------------------





# -------------------------------------------------------
# ----------- Chain of thought --------------------------
# -------------------------------------------------------
# You can ask the model to output an answer in a structured, step-by-step way, to guide the user through the solution.
class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful math tutor. Solve the math problem step by step and return the reasoning in a structured format."
        },
        {
            "role": "user",
            "content": "how can I solve 8x + 7 = -23?"
        }
    ],
    response_format=MathReasoning  # Specify the format you want the response in
)
print(response.choices[0].message.content)

# Output will be a structured response with steps and final answer
# Example output:
# {
#     "steps": [
#         {
#             "explanation": "We start with the equation 8x + 7 = -23. Our goal is to solve for x. First, we need to isolate the term with x (8x) on one side of the equation. To do this, we will eliminate the constant term (+7) from the left side by subtracting 7 from both sides.",
#             "output": "8x + 7 - 7 = -23 - 7"
#         },
#         {
#             "explanation": "When we subtract 7 from both sides, the equation simplifies. The left side becomes just 8x, as 7 - 7 is 0. On the right side, -23 - 7 equals -30.",
#             "output": "8x = -30"
#         },
#         {
#             "explanation": "Now, we need to solve for x by isolating it. Since 8x means 8 times x, we divide both sides of the equation by 8 to get x by itself.",
#             "output": "8x / 8 = -30 / 8"
#         },
#         {
#             "explanation": "Dividing both sides by 8 gives us x on the left side. On the right side, -30 divided by 8 simplifies to -3.75.",
#             "output": "x = -3.75"
#         }
#     ],
#     "final_answer": "x = -3.75"
# }






# --------------------------------------------------------------
# ------------- Structred Data Extraction Example --------------
# --------------------------------------------------------------
class ResearchPaperExtractoin(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are an expert at structured data extraction. You will be given unstructured text from a research paper and should convert it into the given structure."
        },
        {
            "role": "user",
            "content": """
                            Title: Understanding AI in Healthcare. 
                            Authors: John Doe, Jane Smith
                            Abstract: This paper explores the impact of AI on healthcare systems.
                            Keywords: AI, Healthcare, Systems
                        """
        },
    ],
    response_format=ResearchPaperExtractoin  # Specify the format you want the response in.
)
print(response.choices[0].message.content)

# This example shows how to use structured output to extract information from unstructured text.
# The model is instructed to extract the title, authors, abstract, and keywords from a research paper.
# {
#     "title": "Understanding AI in Healthcare",
#     "authors": [
#         "John Doe",
#         "Jane Smith"
#     ],
#     "abstract": "This paper explores the impact of AI on healthcare systems.",
#     "keywords": [
#         "AI",
#         "Healthcare",
#         "Systems"
#     ]
# }




# --------------------------------------------------
# ----------- UI Generation ------------------------
# --------------------------------------------------
# This example shows how to use the OpenAI API to generate a simple UI layout.


class UIType(str, Enum):
    div="div"
    button="button"
    header="header"
    section="section"
    field="field"
    form="form"

class Attribute(BaseModel):
    name: str
    value: str

class UI(BaseModel):
    type: UIType
    label: str
    children: List['UI']
    attribute: List[Attribute]
UI.model_rebuild() # Rebuild the model to handle recursive types

class Response(BaseModel):
    ui: UI

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a UI generator AI. Convert the user input into a UI."
        },
        {
            "role": "user",
            "content": "Make a User Profile Form"
        }
    ],
    response_format=Response  # Specify the format you want the response in
)
print(response.choices[0].message.content)
# This example shows how to use the OpenAI API to generate a simple UI layout.
# The model is instructed to generate a UI for a user profile form.
# The response will be a structured output with the UI components and their attributes.
# Example output:
# {"ui":{"type":"form","label":"User Profile Form","children":[{"type":"field","label":"First Name","children":[],"attribute":[{"name":"type","value":"text"},{"name":"name","value":"firstName"},{"name":"placeholder","value":"Enter your first name"}]},{"type":"field","label":"Last Name","children":[],"attribute":[{"name":"type","value":"text"},{"name":"name","value":"lastName"},{"name":"placeholder","value":"Enter your last name"}]},{"type":"field","label":"Email","children":[],"attribute":[{"name":"type","value":"email"},{"name":"name","value":"email"},{"name":"placeholder","value":"Enter your email"}]},{"type":"field","label":"Phone Number","children":[],"attribute":[{"name":"type","value":"tel"},{"name":"name","value":"phoneNumber"},{"name":"placeholder","value":"Enter your phone number"}]},{"type":"field","label":"Address","children":[],"attribute":[{"name":"type","value":"text"},{"name":"name","value":"address"},{"name":"placeholder","value":"Enter your address"}]},{"type":"button","label":"Submit","children":[],"attribute":[{"name":"type","value":"submit"}]}],"attribute":[{"name":"method","value":"post"},{"name":"action","value":"/submit-profile"}]}}







# --------------------------------------------------------------------------------
# ----------------------- Moderation ---------------------------------------------
# --------------------------------------------------------------------------------
class Category(str, Enum):
    violence = "violence"
    sexual = "sexual"
    self_harm = "self_harm"

class ContentCompliance(BaseModel):
    is_voilation: bool
    category: Optional[Category]
    explanation_if_violating: Optional[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages = [
        {
            "role": "system",
            "content": "Determine if the user input violates specific guidelines and explain if they do." 
        },
        {
            "role": "user",
            "content": "How do I prepare myself for a job interview? I want to be ready to answer any question, even if it involves violence or sexual content. I also want to know how to handle self-harm situations."
        }
    ],
    response_format=ContentCompliance,  # Specify the format you want the response in
)
print(response.choices[0].message.content)
# This example shows how to use the OpenAI API to check if a user input violates specific guidelines.
# The model is instructed to determine if the user input violates guidelines related to violence, sexual content, or self-harm.
# The response will be a structured output indicating whether the input violates guidelines and providing an explanation if it does.
# Example output:
# {
#     "is_voilation": true,
#     "category": "violence",
#     "explanation_if_violating": "The statement expresses an intention to harm oneself and others, which is violent in nature. It also includes a desire to engage in illegal sexual activities with minors, which is harmful and unlawful."
# }







# ---------------------------------------------------------------
# ------------------ Refusal with Structured Output -------------
# ---------------------------------------------------------------
class Step(BaseModel):
    explanation: str
    output: str
class MathReasoning(BaseModel):
    steps: List[Step]
    final_answer: str
    refusal: Optional[bool] = False
    refusal_reason: Optional[str] = None
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "Determine if the user input violates specific guidelines and explain if they do."
        },
        {
            "role": "user",
            "content": "How do I prepare for a job interview?"
        }
    ],
    response_format=MathReasoning  # Specify the format you want the response in
)
math_reasoning = response.choices[0].message.parsed # response.choices[0].message.content gives un-parsed version as string. 
if (math_reasoning.refusal):
    print("The model refused to answer the question.")
    print("Reason:", math_reasoning.refusal_reason)
else:
    print("The model provided a structured response:")
    print(math_reasoning)
    print("Final Answer:", math_reasoning.final_answer)

# IMPORTANT: 
# The model will always try to adhere to the provided schema, which can result in hallucinations 
# if the input is completely unrelated to the schema.




# ----------------------------------------------------------------
# ----------------------- Streaming Output Example ---------------
# ----------------------------------------------------------------
# This example shows how to use the OpenAI API to generate a streaming output.
class EntitiesModel(BaseModel):
    attributes: List[str]
    colors: List[str]
    animals: List[str]

with client.beta.chat.completions.stream(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": "Extract the entities from the user's message and return them in a structured format."
        },
        {
            "role": "user",
            "content": "The quick brown fox jumps over the lazy dog with piercing blue eyes. The sky is blue and the grass is green."
        }
    ],
    response_format=EntitiesModel  # Specify the format you want the response in
) as stream:
    for event in stream:
        if event.type == "content.delta":
            # Print the content of the message as it is streamed
            if event.parsed is not None:
                print("content.delta parsed: ", event.parsed)
        elif event.type == "content.done":
            print("content.done")
        elif event.type == "error":
            print("Error in stream:", event.error)

respones = stream.get_final_completion()
print("Final response:", respones.choices[0].message.content)





# -----------------------------------------------------------------
# ---------------- Stream with function call Example --------------
# -----------------------------------------------------------------
class GetWeather(BaseModel):
    city: str
    country: str

with client.beta.chat.completions.stream(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that provides weather information."
        },
        {
            "role": "user",
            "content": "What's the weather like in SF and London?"
        }
    ],
    tools = [
        openai.pydantic_function_tool(GetWeather, name="get_weather", description="Get the current weather for a city and country.")
    ]
) as stream:
    for event in stream:
        if event.type == "tool_calls.function.arguments.delta" or event.type == "tool_calls.function.arguments.done":
            # Print the content of the message as it is streamed
            print(event)
print("Final response:", type(stream.get_final_completion()), stream.get_final_completion())






# ========================================================
# ============== Function Calling ========================
# ========================================================
# Function calling is a feature that allows you to define functions that the model can call to get information or perform actions.
# This is useful for tasks like retrieving data from an API, performing calculations, or interacting with external systems.
# To use function calling, you need to define the functions you want the model to call and provide them in the API request.
# Function calling has two primary use cases:
# Fetching Data:	Retrieve up-to-date information to incorporate into the model's response (RAG). Useful for searching knowledge bases and retrieving specific data from APIs (e.g. current weather data).
# Taking Action:	Perform actions like submitting a form, calling APIs, modifying application state (UI/frontend or backend), or taking agentic workflow actions (like handing off the conversation).
# Function calling is available in the gpt-4o-2024-08-06 and later models.
# Here's an example of how to use function calling with the OpenAI API:

# Step1: Call model with functions defined – along with your system and user messages.

def get_weather(latitude, longitude) -> str:
    response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m')
    data = response.json()
    return data['current']['temperature_2m']

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temprature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False
            }
        },
        "strict": True  # This ensures the model adheres to the function's schema
    }
]
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Paris, France?"
    }
]
response = client.chat.completions.create(
    model="gpt-4.1",
    messages = messages,
    tools = tools,
    tool_choice = "auto",  # This allows the model to choose which tool to call
    # stream = True  # Set to True if you want to stream the response
)

# Step 2: Check if the model called a function. Model decides to call function(s) – model returns the name and input arguments.
print("Model response:", response.choices[0].message.tool_calls)
# Streaming option with tools 
# for chunk in response:
#     delta = chunk.choices[0].delta
#     print(delta.tool_calls)

# Sample Response: 
# [{
#     "id": "call_12345xyz",
#     "type": "function",
#     "function": {
#       "name": "get_weather",
#       "arguments": "{\"latitude\":48.8566,\"longitude\":2.3522}"
#     }
# }]



#Step 3: Call the function with the arguments provided by the model. - Execute function code – parse the model's response and handle function calls.
def call_function(name, **kwargs): # Function to route each function call to the appropriate function. 
    if name == "get_weather":
        latitude = kwargs.get("latitude")
        longitude = kwargs.get("longitude")
        return get_weather(latitude, longitude)
    else:
        raise ValueError(f"Unknown function: {name}")

messages.append(response.choices[0].message)  # Add the function call message
for tool_call in response.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = call_function(name, **args)  # Call the function with the provided arguments
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result) # Add the result of the function call to the messages
    })
# Note: 
# The function call is executed with the arguments provided by the model.
# The result of the function call is then added to the conversation history as a tool message.
# This allows the model to incorporate the result into its final response.



#Step 4: Supply model with results – so it can incorporate them into its final response.
response2 = client.chat.completions.create(
    model = "gpt-4.1",
    messages = messages,
    tools = tools,
)



# Step 5: Model responds – incorporating the result in its output.
print(response2.choices[0].message.content)


# Best practises for function calling:
# 1. Define clear function schemas: Specify the function's name, description, and parameters. Explicitly describe the purpose of the function and each parameter (and its format), and what the output represents.
# 2. Use the system prompt to describe when (and when not) to use each function. Generally, tell the model exactly what to do.
# 3. Use strict mode: Set "strict": True to ensure the model adheres to the function's schema.
# 4. Handle function calls gracefully: Check if the model called a function and execute it with the provided arguments.
# 5. Incorporate results into the conversation: Add the function call message and the result to the conversation history before making the final API call.
# 6. Test and iterate: Experiment with different function definitions and prompts to improve the model's performance.
# 7. Use function calling for specific tasks: Function calling is best suited for tasks that require structured data retrieval or actions, such as fetching weather data, performing calculations, or interacting with APIs.
# 8. Monitor and log function calls: Keep track of function calls and their results to analyze the model's performance and improve the function definitions over time.
# 9. Use function calling for complex tasks: Function calling is particularly useful for tasks that require multiple steps or interactions with external systems, such as booking a flight, ordering food, or managing a calendar.
# 10. Use function calling for data retrieval: Function calling is ideal for tasks that require retrieving up-to-date information, such as current weather, stock prices, or news articles.
# 11. Use function calling for action-oriented tasks: Function calling is also useful for tasks that require taking action, such as submitting a form, making a reservation, or sending an email.
# 12. Use function calling for agentic workflows: Function calling can be used to create agentic workflows, where the model can take actions based on user input and external data.
# 13. Use function calling for RAG (Retrieval-Augmented Generation): Function calling can be used to retrieve relevant information from external sources and incorporate it into the model's response, enhancing the quality and relevance of the output.







# ==========================================================
# ============== Conversation State ========================
# ==========================================================


# ------------------------------------------------------
# ------ Manually managing conversation state ----------
# ------------------------------------------------------
# This is the simplest way to manage conversation state. You keep track of the messages yourself.
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "knock knock"},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange"},
    ]
)
print(response.choices[0].message.content)
# This is a simple example of how to manage conversation state manually.
# You keep track of the messages in a list and pass them to the API each time you want to get a response.
# By using alternating user and assistant messages, you capture the previous state of a conversation in one request to the model.


history = [
    {"role": "user", "content": "Tell me a joke."}
]
response = client.chat.completions.create(
    model="gpt-4",
    messages=history
)
print(response.choices[0].message)
history.append(response.choices[0].message)
history.append({
    "role": "user", 
    "content": "Tell me another"
})
response = client.chat.completions.create(
    model="gpt-4",
    messages=history
)
print(response.choices[0].message.content)




# -------------------------------------------------------------------------------
# ----------Managing Converstion State with Response API ------------------------
# -------------------------------------------------------------------------------
response = client.responses.create(
    model="gpt-4o-mini",
    input="tell me a joke",
    # store = False # setting this to False will not store response on openAI. By default it's True and saves responses for 30 days. 
)
print(response.output_text)

response2 = client.responses.create(
    model="gpt-4o-mini",
    previous_response_id=response.id,
    input="explain why is this funny"
)
print(response2.output_text)





# ===============================================================
# ============== Reasoning Examples =============================
# ===============================================================
response = client.responses.create(
    model="o4-mini",
    reasoning={
        "effort": "medium",  # low, medium, high
        # "summary": "auto" # may need to verify your organization for this. 
    },
    input = [
        {
            "role": "user",
            "content": """
                    Write a bash script that takes a matrix represented as a string with 
                    format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.
                """
        }
    ],
    max_output_tokens=3000,
)

if response.status == 'incomplete' and response.incomplete_details.reason == 'max_output_tokens':
    print("ran out pf tokens")
    if response.output_text:
        print("partial output: ", response.output_text)
    else:
        print("Ran out of tokens during reasoning")

print(response.output_text)
# A reasoning model is like a senior co-worker—you can give them a goal to achieve and trust them to work out the details.
# A GPT model is like a junior coworker—they'll perform best with explicit instructions to create a specific output.


# ----------------------------------------------------------------
# ------------- Coding / Refactoring Example ---------------------
# ----------------------------------------------------------------
prompt = """
Instructions:
- Given the React component below, change it so that nonfiction books have red
  text. 
- Return only the code in your reply
- Do not include any additional formatting, such as markdown code blocks
- For formatting, use four space tabs, and do not allow any lines of code to 
  exceed 80 columns

const books = [
  { title: 'Dune', category: 'fiction', id: 1 },
  { title: 'Frankenstein', category: 'fiction', id: 2 },
  { title: 'Moneyball', category: 'nonfiction', id: 3 },
];

export default function BookList() {
  const listItems = books.map(book =>
    <li>
      {book.title}
    </li>
  );

  return (
    <ul>{listItems}</ul>
  );
}
"""
response = client.responses.create(
    model="o4-mini",
    input=[
        {
            "role": "user",
            "content": prompt,
        }
    ]
)
print(response.output_text)



# ----------------------------------------------------------------
# ------------- Coding / Planning Example ------------------------
# ----------------------------------------------------------------
prompt = """
I want to build a Python app that takes user questions and looks 
them up in a database where they are mapped to answers. If there 
is close match, it retrieves the matched answer. If there isn't, 
it asks the user to provide an answer and stores the 
question/answer pair in the database. Make a plan for the directory 
structure you'll need, then return each file in full. Only supply 
your reasoning at the beginning and end, not throughout the code.
"""
response = client.responses.create(
    model="o4-mini",
    input=[
        {
            "role": "user",
            "content": prompt,
        }
    ]
)
print(response.output_text)




# ------------------------------------------------------------------------
# ------------------ STEM Research Example -------------------------------
# ------------------------------------------------------------------------
prompt = """
What are three compounds we should consider investigating to 
advance research into new antibiotics? Why should we consider 
them?
"""
response = client.responses.create(
    model="o4-mini",
    input=[
        {
            "role": "user", 
            "content": prompt
        }
    ]
)
print(response.output_text)