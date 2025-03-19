import streamlit as st
# from openai import OpenAI
from groq import Groq
import subprocess
# Show title and description.
st.title("📄 AI Software Manager")
# st.write(
#     "Upload a document below and ask a question about it – GPT will answer! "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
# )

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
#openai_api_key = st.text_input("OpenAI API Key", type="password")

    # Create an OpenAI client.
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
PAT = st.secrets["GitPAT"]
# Use the specified file instead of file uploader
file_path = "/workspaces/document-qa/PJTmain/codeV2.py"
with open(file_path, "r") as file:
    document = file.read()

# Ask the user for a prompt via `st.text_area`.
prompt = st.text_area(
    "What can I assist you with?",
    placeholder="Change the button color to...., Shift position of the image...",
    disabled=not document,
)

if document and prompt:
    system_instruction = (
        "Modify the given Python code based on the user's instruction. Ensure that all necessary changes are made and provide appropriate comments on the modified lines. Return the entire modified code as plain text without any additional explanations or omissions."
    )
    # Process the uploaded file and prompt.
    messages = [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": f"Original Code: {document} \n\n---\n\n User Request:{prompt}"
        }
    ]

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=True,
    )

    # Write the response to a test.py file.
    with open("/workspaces/document-qa/PJTmain/test.py", "w") as test_file:
        for chunk in stream:
            content = chunk.choices[0].delta.content
            try:
                if content.strip() not in ["```python","python", "```"]:
                    test_file.write(content)
            except:pass

    # Push the test.py file to the GitHub repository.
    repo_url = f"https://{PAT}@github.com/Mukundh0007/PJTmain.git"
    # subprocess.run(["git", "init", "/workspaces/document-qa/PJTmain"])
    # subprocess.run(["git", "-C", "/workspaces/document-qa/PJTmain", "remote", "set-url", "origin", repo_url])
    # subprocess.run(["git", "-C", "/workspaces/document-qa/PJTmain", "config", "pull.rebase", "false"])  # Set pull strategy to merge
    # subprocess.run(["git", "-C", "/workspaces/document-qa/PJTmain", "pull", "origin", "main"])  # Pull changes from the remote repository
    # subprocess.run(["git", "-C", "/workspaces/document-qa/PJTmain", "add", "test.py"])
    # subprocess.run(["git", "-C", "/workspaces/document-qa/PJTmain", "commit", "-m", "Add test.py"])
    # subprocess.run(["git", "-C", "/workspaces/document-qa/PJTmain", "push", "-u", "origin", "main", "--force"])  # Force push to the remote repository
    
    # Send a curl request to the specified URL.
    # subprocess.run([
    #     "curl", "-X", "POST", "http://98.70.35.30:5000/update"
    # ])