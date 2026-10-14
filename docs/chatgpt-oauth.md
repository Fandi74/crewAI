# Use a ChatGPT plan with CrewAI

CrewAI can use the documented Sign in with ChatGPT flow for local, open-source applications. The user approves ChatGPT plan usage in the browser; CrewAI stores the returned OAuth credentials locally and refreshes access tokens as needed.

This flow uses the public OpenAI Responses API. It does not read ChatGPT conversations or use the private ChatGPT web backend.

## Connect an account

Run this on the machine where the crew will run:

```shell
crewai auth chatgpt login --app-name "My local crew"
crewai auth chatgpt status
```

On first sign-in, the browser asks the user to approve the app and the requested plan-usage permission. The callback uses a loopback address on 127.0.0.1. Credentials are written atomically to the current user's CrewAI configuration directory with owner-only file permissions on Unix. The stable host identifier is kept separately for future sign-ins.

To remove the local credentials, run:

```shell
crewai auth chatgpt logout
```

Logout attempts to revoke the renewable ChatGPT session before clearing local tokens. If the network prevents confirmation, the CLI says so; the user can disconnect the app in ChatGPT settings.

## Configure a CrewAI LLM

Set the chatgpt_oauth option to true on the OpenAI LLM:

```python
from crewai import LLM

llm = LLM(
    model="openai/gpt-6.1-sol",
    chatgpt_oauth=True,
    reasoning_effort="low",
)
```

Use this LLM for an agent or pass it anywhere CrewAI accepts an LLM. The OAuth option selects the Responses API, forces streaming, disables response storage, and sends requests only to https://api.openai.com/v1. It does not use OPENAI_API_KEY when enabled.

## Current route limits

ChatGPT plan usage follows the current Responses API limitations for this OAuth route:

- Requests are stateless, streamed, and sent with store=False.
- CrewAI function tools are supported. The built-in web search tool may be available according to the model and account policy.
- Other hosted Responses tools, including file search, Code Interpreter, and computer use, are not supported on this route.
- Unsupported request fields such as temperature, top_p, and output-token limits are omitted.
- Model availability and usage limits depend on the signed-in ChatGPT account and workspace.

Use a regular OpenAI API key for other OpenAI endpoints, unsupported tools, or workloads that need those request fields. Sign in with ChatGPT plan usage is intended here for local or open-source applications; paid or remotely hosted integrations require separate OpenAI approval.

For the upstream protocol and current route limitations, see the official [Sign in with ChatGPT guide](https://developers.openai.com/siwc/token-sharing-open-source) and [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).
