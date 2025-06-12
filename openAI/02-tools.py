from io import BytesIO
import openai
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional
import requests
import json
import base64
import time

client = OpenAI()




# ===============================================
# ============== Tools and Functions ============
# ===============================================
# Tools and functions are a way to extend the capabilities of the model by providing it with external tools and functions.
# This is useful for tasks like calling external APIs, accessing databases, or performing calculations.
# To use tools and functions, you need to provide the tool or function as part of the input.
# Here's an example of how to use tools and functions with the OpenAI API:

response = client.responses.create(
    model="gpt-4.1",
    input="What's the weather in Tokyo today?",
    tools=[
        {
            "type": "web_search_preview",
        }
    ]
)
print(response.output_text)

# Available Tools:
# Function calling: Call custom code to give the model access to additional data and capabilities.
# Web search: Include data from the Internet in model response generation.
# Remote MCP servers: Give the model access to new capabilities via Model Context Protocol (MCP) servers.
# File search: Search the contents of uploaded files for context when generating a response.
# Image Generation: Generate or edit images using GPT Image. 
# Code interpreter: Allow the model to execute code in a secure container.
# Computer use: Create agentic workflows that enable a model to control a computer interface.




# ========================================================
# ================== Remote MCP ==========================
# ========================================================

# Model Context Protocol (MCP) is an open protocol that standardizes how applications provide tools and context to LLMs. 
# The MCP tool in the Responses API allows developers to give the model access to tools hosted on Remote MCP servers. 
# These are MCP servers maintained by developers and organizations across the internet that expose these tools to MCP clients, 
# like the Responses API.

# The MCP tool works only in the Responses API, and is available across all new models (gpt-4o, gpt-4.1, and reasoning models). 
# When you're using the MCP tool, you only pay for tokens used when importing tool definitions 
# or making tool calls—there are no additional fees involved.


# Step 1: Getting the list of tools from the MCP server: 
# The first thing the Responses API does when you attach a remote MCP server to the tools array, is attempt to get a 
# list of tools from the server. The Responses API supports remote MCP servers that support 
# either the Streamable HTTP or the HTTP/SSE transport protocol.
response = client.responses.create(
    model="gpt-4.1",
    tools = [
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            # As long as the mcp_list_tools item is present in the context of the model, we will not attempt to pull a 
            # refreshed list of tools from an MCP server. We recommend you keep this item in the model's context 
            # as part of every conversation or workflow execu
            "allowed_tools": ["ask_question"], # This is a list of tools we want to import from the MCP server.
            "require_approval": "never",
        }
    ],
    input = "What transport protocols are supported in the 2025-03-26 version of the MCP spec?"
)
print(response.output_text)



# Step 2: Calling the tools to generate a response
# Once the model has access to these tool definitions, it may choose to call them depending on what's in the model's context. 
# When the model decides to call an MCP tool, we make an request to the remote MCP server to call the tool, take it's output 
# and put that into the model's context. This creates an mcp_call item which looks like this:
# {
#  "id": "mcp_682d437d90a88191bf88cd03aae0c3e503937d5f622d7a90",
#  "type": "mcp_call",
#  "approval_request_id": null,
#  "arguments": "{\"repoName\":\"modelcontextprotocol/modelcontextprotocol\",\"question\":\"What transport protocols does the 2025-03-26 version of the MCP spec support?\"}",
#  "error": null,
#  "name": "ask_question",
#  "output": "The 2025-03-26 version of the Model Context Protocol (MCP) specification supports two standard transport mechanisms: `stdio` and `Streamable HTTP` ...",
#  "server_label": "deepwiki"
# }

# By default, OpenAI will request your approval before any data is shared with a remote MCP server.
# The model may also choose to call the tool multiple times, and the tool may also return an error.

# A request for an approval to make an MCP tool call creates a mcp_approval_request item in the Response's output that looks like this:
# {
#  "id": "mcpr_682d498e3bd4819196a0ce1664f8e77b04ad1e533afccbfa",
#  "type": "mcp_approval_request",
#  "arguments": "{\"repoName\":\"modelcontextprotocol/modelcontextprotocol\",\"question\":\"What transport protocols are supported in the 2025-03-26 version of the MCP spec?\"}",
#  "name": "ask_question",
#  "server_label": "deepwiki"
# }


# You can then respond to this by creating a new Response object and appending an mcp_approval_response item to it.
response = client.responses.create(
    model="gpt-4.1",
    tools=[{
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
    }],
    input="What transport protocols are supported in the 2025-03-26 version of the MCP spec?"
)

# Extract the approval request ID from the response
approval_request_id = None
for item in response.output:
    if hasattr(item, 'type') and item.type == "mcp_approval_request":
        approval_request_id = item.id
        break

if approval_request_id:
    # Now create the approval response with the correct ID
    response2 = client.responses.create(
        model="gpt-4.1",
        tools=[{
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
        }],
        previous_response_id=response.id,
        input=[{
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": approval_request_id
        }],
    )
    print(response2.output_text)
else:
    print("No approval request found in the response")


# The MCP tool in the Responses API gives you the ability to flexibly specify headers that should be included 
# in any request made to a remote MCP server. This is useful for things like authentication, or for passing 
# information about the request to the MCP server.
# To do this, you can add a headers property to the MCP tool definition.
# The headers property is an object with the following properties:
# - Authorization: The authorization header to include in the request.
# - Content-Type: The content type of the request.
# - Other headers: Any other headers you want to include in the request.
response = client.responses.create(
    model="gpt-4.1",
    tools = [
        {
            "type": "mcp",
            "server_label": "stripe",
            "server_url": "https://mcp.stripe.com",
            "headers": {
                "Authorization": "Bearer sk_test_1234567890",
                "Content-Type": "application/json"
            }
        }
    ],
    input = "What is the balance of my account?"
)
print(response.output_text)






# -------------- Web Search --------------
# Using the Responses API, you can enable web search by configuring it in the tools array in an API request to generate content. 
# Like any other tool, the model can choose to search the web or not based on the content of the input prompt.
# When you're using the web search tool, you only pay for tokens used when making web search requests—there are no additional fees involved.
# The web search tool is available in all new models (gpt-4o, gpt-4.1, and reasoning models).


# To enable web search, you need to add a web_search_preview tool to the tools array in an API request to generate content.
# The web_search_preview tool is available in all new models (gpt-4o, gpt-4.1, and reasoning models).
response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "web_search_preview",
        }
    ],
    input="What's the top trending news from today?"
)
print(response.output_text)
# Once the model has access to the web search tool, it may choose to search the web or not based on the content of the input prompt.
# The model may also choose to call the tool multiple times, and the tool may also return an error.

# You can also force the use of the web_search_preview tool by using the tool_choice parameter, 
# and setting it to {type: "web_search_preview"} - this can help ensure lower latency and more consistent results.
response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "web_search_preview",
        }
    ],
    tool_choice={
        "type": "web_search_preview",
    },
    input="What's the top trending news from today?"
)
print(response.output_text)


# User location
# To refine search results based on geography, you can specify an approximate user location using country, city, region, and/or timezone.
response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "web_search_preview",
            "user_location": {
                "type": "approximate",
                "country": "US",
                "city": "New York",
                "region": "New York",
                "timezone": "America/New_York"
            }
        }
    ],
    input="What's the top trending news from today?"
)
print(response.output_text)


# Search Context Size
# When using this tool, the search_context_size parameter controls how much context is retrieved from the web 
# to help the tool formulate a response. 
response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "web_search_preview",
            "search_context_size": "low"
        }
    ],
    input="What's the top trending news from today?"
)
print(response.output_text)




# ========================================================
# ================== File Search =========================
# ========================================================
# Allow models to search your files for relevant information before generating a response.
# This is a hosted tool managed by OpenAI, meaning you don't have to implement code on your end to handle its execution. 
# When the model decides to use it, it will automatically call the tool, retrieve information from your files, and return an output.

# 1. Upload file to API 
def create_file(client, file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Download the file content from the URL
        response = requests.get(file_path)
        file_content = BytesIO(response.content)
        file_name = file_path.split("/")[-1]
        file_tuple = (file_name, file_content)
        result = client.files.create(
            file=file_tuple,
            purpose="assistants"
        )
    else:
        # Handle local file path
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="assistants"
            )
    print(result.id)
    return result.id

file_id = create_file(client, "https://cdn.openai.com/API/docs/deep_research_blog.pdf")

# 2. Create Vector Store
vector_store = client.vector_stores.create(
    name="my_knowledge_base",
)
print(vector_store.id)

# 3. Add file to vector store
client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file_id,
)

# 4. Run this code until the file is ready to be used (i.e., when the status is completed).
result = client.vector_stores.files.list(
    vector_store_id=vector_store.id,
)
while result.data[0].status != "completed":
    time.sleep(1)
    result = client.vector_stores.files.list(
        vector_store_id=vector_store.id,
    )

print(result.data[0].status)


# 5. Once your knowledge base is set up, you can include the file_search tool in the list of tools available to the model, 
#    along with the list of vector stores in which to search.
response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": [vector_store.id],
            "max_num_results": 2, # customize the number of results you wnat to retrieve from the vector stores. 
            # You can filter the search results based on the metadata of the files. 
            "filters": {
                "type": "eq",
                "key": "type",
                "value": "blog"
            }
        }
    ],
    # Once your knowledge base is set up, you can include the file_search tool in the list of tools 
    # available to the model, along with the list of vector stores in which to search.
    include=["file_search_call.results"], 
    input="What is the main idea of the document?",
)
print(response.output_text)








# ========================================================
# ================== Image Generation ====================
# ========================================================

# The image generation tool allows you to generate images using GPT Image. 
# This tool is available in all new models (gpt-4o, gpt-4.1, and reasoning models).
# To use the image generation tool, you need to provide the image generation tool as part of the input.
# Here's an example of how to use the image generation tool with the OpenAI API:
response = client.responses.create(
    model="gpt-4.1",
    input="Generate an image of a cat",
    stream=False, # Set to True to stream the image generation process.
    tools=[
        {
            "type": "image_generation",
            "size": "1024x1024",
            "quality": "auto",
            "background": "auto",
        }
    ]
)
# Save the image to a fle
image_data = [
    output.result for output in response.output
    if output.type == "image_generation_call"
]

if image_data:
    image_base64 = image_data[0]
    with open("cat.png", "wb") as f:
        f.write(base64.b64decode(image_base64))
else:
    print("No image data found")

# Image generation works best when you use terms like "draw" or "edit" in your prompt.
# For example, if you want to combine images, instead of saying "combine" or "merge", you can say something like 
# "edit the first image by adding this element from the second image".

# You can iteratively edit images by referencing previous response or image IDs. This allows you to refine images 
# across multiple turns in a conversation.








# ========================================================
# ================== Computer Use - CUA ==================
# ========================================================
# These are the high-level steps you need to follow to integrate the computer use tool in your application:
# 1. Send a request to the model: Include the computer tool as part of the available tools, specifying the display size and environment. You can also include in the first request a screenshot of the initial state of the environment.
# 2. Receive a response from the model: Check if the response has any computer_call items. This tool call contains a suggested action to take to progress towards the specified goal. These actions could be clicking at a given position, typing in text, scrolling, or even waiting.
# 3. Execute the requested action: Execute through code the corresponding action on your computer or browser environment.
# 4. Capture the updated state: After executing the action, capture the updated state of the environment as a screenshot.
# 5. Repeat: Send a new request with the updated state as a computer_call_output, and repeat this loop until the model stops requesting actions or you decide to stop.
