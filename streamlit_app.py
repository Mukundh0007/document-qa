import streamlit as st
# from openai import OpenAI
from groq import Groq
import subprocess
import os
import time

def remove_pjt():
    if os.path.exists(f"{cwd}/PJTmain"):
        subprocess.run(["rm", "-rf", f"{cwd}/PJTmain"])
    return
# Get the current working directory
cwd = os.getcwd()

# Show title and description.
st.title("📄 AI Software Manager")

# Create an AI client.
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY) # Replace with your OpenAI API key
PAT = st.secrets["GitPAT"]
repo_url = f"https://{PAT}@github.com/Mukundh0007/PJTmain.git"

subprocess.run(["git", "clone", repo_url, f"{cwd}/PJTmain"])

# Use the specified file instead of file uploader
file_path = f"{cwd}/PJTmain/codeV2.py"
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

    # Generate an answer using the API.
    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True,
        )

        # Write the response to a test.py file.
        with open(f"{cwd}/PJTmain/test.py", "w") as test_file:
            for chunk in stream:
                content = chunk.choices[0].delta.content
                try:
                    if content.strip() not in ["```python","python", "```"]:
                        test_file.write(content)
                except:pass
    except Exception as e:
        with open(f"{cwd}/PJTmain/test.py", "w") as test_file:
            try:
                test_file.write(document)
            except:pass
    # Push the test.py file to the GitHub repository.

    subprocess.run(["git", "init", f"{cwd}/PJTmain"])
    subprocess.run(["git", "-C", f"{cwd}/PJTmain", "remote", "set-url", "origin", repo_url])
    subprocess.run(["git", "-C", f"{cwd}/PJTmain", "config", "pull.rebase", "false"])  # Set pull strategy to merge
    subprocess.run(["git", "-C", f"{cwd}/PJTmain", "pull", "origin", "main"])  # Pull changes from the remote repository
    subprocess.run(["git", "-C", f"{cwd}/PJTmain", "add", "test.py"])
    subprocess.run(["git", "-C", f"{cwd}/PJTmain", "commit", "-m", "Add test.py"])
    subprocess.run(["git", "-C", f"{cwd}/PJTmain", "push", "-u", "origin", "main", "--force"])  # Force push to the remote repository
    
    # Send a curl request to the specified URL.
    subprocess.run([ "curl", "http://98.70.35.30:5000/update"])
    st.write("The code has been modified based on your request. Please check the server.")

    remove_pjt()
    # Add OK and NG buttons
    # col1, col2 = st.columns(2)

    # with col1:
    #     if st.button("OK"):
    #         # Copy the content of test.py to codeV2.py
    #         test_file_path = f"{cwd}/PJTmain/test.py"
    #         with open(test_file_path, "r") as test_file:
    #             modified_code = test_file.read()
    #         with open(file_path, "w") as code_file:
    #             code_file.write(modified_code)
    #         st.write("The code has been updated successfully.")

    # with col2:
    #     if st.button("NG"):
    #         # Restart the code by re-running the script
    #         st.experimental_rerun()