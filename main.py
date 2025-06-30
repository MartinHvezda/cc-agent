import json

from ollama import chat

def get_identity_id(email):
    print("Getting identity by email: ", email)
    return {
        "type": "success",
        "content": {
            "identityId": "111111"
        }
    }

def send_reset_password_email(identity_id):
    print("Sending reset password email for identity: ", identity_id)
    return {
        "type": "success",
        "content": "Email sent"
    }

def create_jira_ticket(description, queue):
    print("Creating jira ticket")
    print(description)
    print(queue)
    return {
        "type": "success",
        "content": "Jira ticket created successfully"
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_identity_id",
            "description": "Retrieve a user's identity_id based on their email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email address of the user."
                    }
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_reset_password_email",
            "description": "Send a password reset email to the user based on their identity ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identity_id": {
                        "type": "string",
                        "description": "The identity ID of the user."
                    }
                },
                "required": ["identity_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_jira_ticket",
            "description": "Create a Jira ticket for a customer issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the user issue and context."
                    },
                    "queue": {
                        "type": "string",
                        "description": "The queue or team name the ticket should be assigned to."
                    }
                },
                "required": ["description", "queue"]
            }
        }
    }
]

tools_to_function_map = {
    "get_identity_id": get_identity_id,
    "send_reset_password_email": send_reset_password_email,
    "create_jira_ticket": create_jira_ticket,
}

team_queues = {
    "IDM": "Identity management",
    "IT": "Information technology",
    "HR": "Human resources",
    "MARKETING": "Marketing",
    "FINANCE": "Finance",
    "PRODUCTION": "Production",
    "DEVELOPMENT": "Development",
    "OPERATIONS": "Operations",
    "OTHER": "Other team queue"
}

def chat_and_append(messages, tools):
    response = chat(
        model='PetrosStav/gemma3-tools:12b',
        tools=tools,
        messages=messages
    )
    messages.append(response.message)

    return response

def execute_tools(messages, response, iterations_to_end):
    if response.message.tool_calls or iterations_to_end == 0:
        for tool_call in response.message.tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            print(f"Executing tool: {function_name}({function_args})")
            tool_response = tools_to_function_map[function_name](**function_args)

            print(f"Tool result: {tool_response}")

            messages.append({
                "role": "tool",
                "name": function_name,
                "content": json.dumps(tool_response),
            })
            tool_response = chat_and_append(messages, tools)

        return execute_tools(messages, tool_response, iterations_to_end - 1)
    else:
        return response

def main():
    print("Start")
    # issue = "email: test@email.com; Hi, I have problem with my account I cannot login." # cannot_login
    # issue = "email: test@email.com; Hi, I forgot my password. What should I do?" # forgot_pass
    issue = "email: test@email.com; Hi, I don't see my credit card in my account. Can you help me?" # card_problem
    messages = [
        {"role": "system",
         "content": "You are a customer care agent. A user will describe an issue including their email address. Investigate the issue and try to resolve it using only predefined tools or by providing general, factual advice. "
                    "You must first retrieve the user's identity_id to perform any actions. "
                    "Do not speculate, assume causes, or invent functionality. Do not hallucinate any tools or actions. "
                    "If you can resolve the issue with advice or available tools, inform the user clearly and confirm if their issue is resolved. "
                    "If the issue persists or cannot be resolved with available tools or advice, notify the user that the issue will be escalated, then create a Jira ticket with relevant details and assign it to the appropriate team queue. "
                    f"List of team queues and description {json.dumps(team_queues)}"},
        {"role": "user", "content": issue}
    ]
    response = chat_and_append(messages, tools)

    first_phase = execute_tools(messages, response, 3)
    print(first_phase.message.content)

    user_followup = input("\nYou: ")
    final_response = chat_and_append(messages + [{"role": "user", "content": user_followup}], tools)
    execute_tools(messages, final_response, 3)
    print(final_response)

    print(f"\nFinal content: {final_response.message.content}")
    return final_response



if __name__ == "__main__":
    main()
