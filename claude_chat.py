import os
import requests
import json
import subprocess
from github import Github

# Constants
API_KEY = os.getenv('CLAUDE_API_KEY')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
API_URL = "https://api.anthropic.com/v1/messages"
GITHUB_REPO_URL = "https://github.com/olympusmons1256/polyglot-v5"
GITHUB_BRANCH = "claud-simplified"
CLAUDE_MODEL = "claude-3-opus-20240229"

if not API_KEY:
    raise ValueError("CLAUDE_API_KEY environment variable is not set.")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set.")

class GitHubRepo:
    def __init__(self, repo_url, branch):
        self.g = Github(GITHUB_TOKEN)
        self.repo = self.g.get_repo(repo_url.split('github.com/')[-1])
        self.branch = branch

    def get_file_structure(self):
        def traverse(contents, path=''):
            structure = []
            for content in contents:
                if content.type == 'dir':
                    structure.append(f"{path}{content.name}/")
                    structure.extend(traverse(self.repo.get_contents(content.path, ref=self.branch), f"{path}{content.name}/"))
                else:
                    structure.append(f"{path}{content.name}")
            return structure
        
        return traverse(self.repo.get_contents("", ref=self.branch))

    def get_file_content(self, file_path):
        try:
            content = self.repo.get_contents(file_path, ref=self.branch)
            return content.decoded_content.decode()
        except Exception as e:
            return f"Error fetching file content: {str(e)}"

def ask_claude(prompt: str, repo_structure: str):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    messages = [
        {"role": "system", "content": "You are an AI assistant working on a software project. For each query, you'll be provided with the current repository structure. Use this information to provide context-aware responses."},
        {"role": "user", "content": f"Current repository structure:\n{repo_structure}\n\nUser query: {prompt}"}
    ]
    
    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1000,
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        content = response.json()
        if 'content' in content and len(content) > 0:
            return content[0]['text']
        else:
            return "No content received from Claude."
    except requests.RequestException as e:
        return f"Error in API request: {e}"

def git_push(file_path: str, commit_message: str, branch: str) -> str:
    try:
        subprocess.run(['git', 'add', file_path], check=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        subprocess.run(['git', 'push', 'origin', branch], check=True)
        return f"Changes pushed to repository successfully on branch {branch}."
    except subprocess.CalledProcessError as e:
        return f"Error pushing to repository: {str(e)}"

def update_file(file_path: str, new_content: str) -> None:
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'w') as file:
        file.write(new_content)
    print(f"File updated: {file_path}")

def main():
    github_repo = GitHubRepo(GITHUB_REPO_URL, GITHUB_BRANCH)
    
    print(f"Welcome to the simplified Claude Chat! (Branch: {GITHUB_BRANCH})")
    print("Type 'exit' to end the conversation.")
    print("To update a file, start your message with 'UPDATE: filename'")
    
    while True:
        user_input = input("\nEnter your message: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            return
        
        if user_input.lower().startswith('update:'):
            parts = user_input.split(':', 1)
            if len(parts) == 2:
                file_path = parts[1].strip()
                print(f"Enter the new content for {file_path} (type '###' on a new line to finish):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == '###':
                        break
                    lines.append(line)
                new_content = '\n'.join(lines)
                update_file(file_path, new_content)
                commit_message = input("Enter commit message: ")
                result = git_push(file_path, commit_message, GITHUB_BRANCH)
                print(result)
            continue
        
        repo_structure = "\n".join(github_repo.get_file_structure())
        response = ask_claude(user_input, repo_structure)
        print("\nClaude:", response)

if __name__ == "__main__":
    main()