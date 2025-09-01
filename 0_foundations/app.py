from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import pymupdf4llm
import gradio as gr

load_dotenv(override=True)

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = f"https://api.pushover.net/1/messages.json"

def push(message):
    requests.post(pushover_url, json={"message": message, "user": pushover_user, "token": pushover_token})

def record_user_details(user_email, user_name="Unknown", user_notes="Unknown"):
    push(f"Recording interest from {user_name} with email: {user_email} and notes: {user_notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording unknown question: {question} that I couldn't answer")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "user_email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "user_name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "user_notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["user_email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:
    def __init__(self):
        self.deepseek_client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_name = "deepseek/deepseek-chat-v3.1:free"
        self.name = "Shubham Lad"
        self.resume_md = pymupdf4llm.to_markdown("./me/Shubham_Lad_Software_Engineer_II.pdf")
        with open("./me/summary.txt", "r") as f:
            self.summary = f.read()
            f.close()

    def handle_tool_calls(self, tool_calls):
        results = []
        available_functions = {
            "record_user_details": record_user_details,
            "record_unknown_question": record_unknown_question
        }
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            print(f"Calling function {function_name} with args {function_args}", flush=True)

            function_response = function_to_call(**function_args)
            results.append({"tool_call_id": tool_call.id, "role": "tool", "content": json.dumps(function_response)})
        
        return results
        
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
        particularly questions related to {self.name}'s career, background, skills and experience. \
        Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
        You are given a summary of {self.name}'s background and Resume which you can use to answer questions. \
        Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
        If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## Resume:\n{self.resume_md}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."

        return system_prompt

    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False

        while not done:
            # We are calling the LLM with tool description passed to it
            response = self.deepseek_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools
            )
            finish_reason = response.choices[0].finish_reason

            # If the LLM returns a tool call, we need to handle it
            if finish_reason == "tool_calls":
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                results = self.handle_tool_calls(tool_calls)
                messages.append(response_message)
                messages.extend(results)
            else:
                done = True

        return response.choices[0].message.content
    
if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()