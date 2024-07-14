import os
import json
import requests
import sqlite3
import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict, Tuple
import subprocess
import re
import shutil

# Constants
API_KEY = os.getenv('CLAUDE_API_KEY')
API_URL = "https://api.anthropic.com/v1/messages"
GITHUB_REPO_URL = "https://github.com/olympusmons1256/polyglot-v4"
DB_PATH = "conversation_notebook.db"
CONTEXT_CHECK_INTERVAL = 3  # Number of turns before reminding Claude to check history

print(f"API Key found: {'Yes' if API_KEY else 'No'}")

if not API_KEY:
    raise ValueError("CLAUDE_API_KEY environment variable is not set. Please set it and try again.")

class VectorNotebook:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS entries
                              (id INTEGER PRIMARY KEY, timestamp TEXT, content TEXT, tags TEXT, vector BLOB)''')
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.load_vectors()

    def load_vectors(self):
        self.cursor.execute("SELECT content FROM entries")
        contents = [row[0] for row in self.cursor.fetchall()]
        if contents:
            self.vectors = self.vectorizer.fit_transform(contents)
        else:
            self.vectors = None

    def add_entry(self, content: str, tags: List[str]):
        timestamp = datetime.now().isoformat()
        if self.vectors is None:
            self.vectors = self.vectorizer.fit_transform([content])
        else:
            new_vector = self.vectorizer.transform([content])
            self.vectors = vstack([self.vectors, new_vector])
        
        vector = self.vectors[-1].toarray()[0]
        tags_str = ",".join(tags)
        self.cursor.execute("INSERT INTO entries (timestamp, content, tags, vector) VALUES (?, ?, ?, ?)",
                            (timestamp, content, tags_str, vector.tobytes()))
        self.conn.commit()

    def get_relevant_entries(self, query: str, n: int = 5) -> List[Dict]:
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        top_indices = similarities.argsort()[-n:][::-1]
        
        relevant_entries = []
        for index in top_indices:
            self.cursor.execute("SELECT id, timestamp, content, tags FROM entries LIMIT 1 OFFSET ?", (int(index),))
            entry = self.cursor.fetchone()
            if entry:
                relevant_entries.append({
                    "id": entry[0],
                    "timestamp": entry[1],
                    "content": entry[2],
                    "tags": entry[3].split(","),
                    "similarity": similarities[index]
                })
        return relevant_entries

    def close(self):
        self.conn.close()

def generate_instruction(vector_notebook: VectorNotebook, user_input: str, error_message: str = None) -> str:
    relevant_entries = vector_notebook.get_relevant_entries(user_input)
    instruction = f"Refer to the project at {GITHUB_REPO_URL}. "
    instruction += "Based on the following context and the current query, provide assistance:\n\n"
    for entry in relevant_entries:
        instruction += f"Previous interaction (Relevance: {entry['similarity']:.2f}):\n"
        instruction += f"Content: {entry['content']}\n"
        instruction += f"Tags: {', '.join(entry['tags'])}\n\n"
    instruction += f"Current query: {user_input}\n"
    if error_message:
        instruction += f"\nError message: {error_message}\n"
    instruction += "\nIf you suggest code changes, please format your response as follows:\n"
    instruction += "1. Explain the changes.\n"
    instruction += "2. Specify the file path like this: File: `path/to/file.ext`\n"
    instruction += "3. Provide the complete updated code wrapped in triple backticks.\n"
    instruction += "4. Suggest a commit message prefixed with 'Commit: '\n"
    instruction += "\nIf you suggest deleting files or folders, please format your response as follows:\n"
    instruction += "1. Explain the deletions.\n"
    instruction += "2. List each file or folder to be deleted, prefixed with 'Delete: '\n"
    instruction += "3. Suggest a commit message prefixed with 'Commit: '\n"
    return instruction

def ask_claude(prompt: str, conversation_history: List[Dict[str, str]] = []) -> str:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 4096,
        "messages": conversation_history + [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        content = response.json()
        if 'content' in content and len(content['content']) > 0:
            return content['content'][0]['text']
        else:
            return "No content received from Claude."
    except requests.RequestException as e:
        return f"Error in API request: {e}"

def get_multi_line_input() -> str:
    print("Enter your message (type '###' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == '###':
            break
        lines.append(line)
    return '\n'.join(lines)

def git_push(file_path: str, commit_message: str) -> str:
    try:
        subprocess.run(['git', 'add', file_path], check=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        subprocess.run(['git', 'push'], check=True)
        return "Changes pushed to repository successfully."
    except subprocess.CalledProcessError as e:
        return f"Error pushing to repository: {str(e)}"

def update_file(file_path: str, new_content: str) -> None:
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    try:
        with open(file_path, 'w') as file:
            file.write(new_content)
        print(f"File updated successfully: {file_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")

def extract_code_and_file_path(response: str) -> List[Tuple[str, str, str]]:
    code_block_patterns = [
        r"```[\w]*\n([\s\S]*?)```",
        r"~~~[\w]*\n([\s\S]*?)~~~",
        r"<code>([\s\S]*?)</code>",
        r"((?:^|\n)[\w\s]+\([^)]*\)\s*{[^}]*})"
    ]
    
    file_path_patterns = [
        r"File:?\s*[`\"'](.*?)[`\"']",
        r"File path:?\s*[`\"'](.*?)[`\"']",
        r"In\s+file\s+[`\"'](.*?)[`\"']",
        r"Update\s+[`\"'](.*?)[`\"']",
        r"Modify\s+[`\"'](.*?)[`\"']"
    ]
    
    commit_message_pattern = r"Commit:\s*(.*)"
    
    code_blocks = []
    for pattern in code_block_patterns:
        code_blocks.extend(re.findall(pattern, response))
    
    file_paths = []
    for pattern in file_path_patterns:
        file_paths.extend(re.findall(pattern, response))
    
    commit_messages = re.findall(commit_message_pattern, response)
    
    if not code_blocks:
        indented_block_pattern = r"(?m)^( {4}|\t).*$"
        indented_blocks = re.findall(indented_block_pattern, response, re.MULTILINE)
        if indented_blocks:
            code_blocks.append("\n".join(indented_blocks))
    
    results = []
    for i, code in enumerate(code_blocks):
        file_path = file_paths[i] if i < len(file_paths) else None
        commit_message = commit_messages[i] if i < len(commit_messages) else None
        results.append((code.strip(), file_path, commit_message))
    
    return results

def extract_deletions(response: str) -> Tuple[List[str], str]:
    deletion_pattern = r"Delete:\s*([^\n]+)"
    commit_message_pattern = r"Commit:\s*(.*)"
    
    deletions = re.findall(deletion_pattern, response)
    commit_messages = re.findall(commit_message_pattern, response)
    
    commit_message = commit_messages[-1] if commit_messages else None
    
    return [d.strip() for d in deletions], commit_message

def extract_error_message(response: str) -> str:
    error_pattern = r"Error:|Failed to|Exception:"
    error_match = re.search(error_pattern, response, re.IGNORECASE)
    if error_match:
        error_start = error_match.start()
        error_end = response.find("\n\n", error_start)
        if error_end == -1:
            error_end = len(response)
        return response[error_start:error_end].strip()
    return None

def delete_file_or_folder(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"Deleted file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Deleted folder: {path}")
        else:
            print(f"Path not found: {path}")
    except Exception as e:
        print(f"Error deleting {path}: {str(e)}")

def verify_git_push(file_path: str) -> bool:
    try:
        subprocess.run(['git', 'fetch', 'origin'], check=True, capture_output=True, text=True)
        local_hash = subprocess.run(['git', 'log', '-n', '1', '--pretty=format:%H', '--', file_path], 
                                    check=True, capture_output=True, text=True).stdout.strip()
        result = subprocess.run(['git', 'branch', '-r', '--contains', local_hash], 
                                check=True, capture_output=True, text=True)
        return 'origin/' in result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error verifying git push: {e}")
        return False

def handle_deletions(deletions: List[str], delete_commit_message: str) -> None:
    print("\nClaude has suggested deleting the following files/folders:")
    for i, item in enumerate(deletions, 1):
        print(f"{i}. {item}")
    
    if delete_commit_message:
        print(f"\nSuggested commit message: {delete_commit_message}")
    
    confirm = input("\nDo you want to proceed with these deletions? (y/n): ").lower()
    if confirm == 'y':
        for item in deletions:
            confirm_item = input(f"Delete {item}? (y/n): ").lower()
            if confirm_item == 'y':
                delete_file_or_folder(item)
        
        if delete_commit_message:
            use_suggested_message = input("Use the suggested commit message? (y/n): ").lower()
            if use_suggested_message != 'y':
                delete_commit_message = input("Enter commit message: ")
        
        result = git_push('.', delete_commit_message)
        print(result)

def handle_code_changes(code: str, file_path: str, commit_message: str) -> None:
    print("\n" + "="*50)
    print("Code change detected!")
    if file_path:
        print(f"File: {file_path}")
    else:
        print("File path not detected. Please provide the file path:")
        file_path = input().strip()
    print("Suggested code:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    if commit_message:
        print(f"\nSuggested commit message: {commit_message}")
    print("="*50)
    
    user_choice = input("\nDo you want to apply these changes? (y/n): ").lower()
    if user_choice == 'y':
        try:
            update_file(file_path, code)
            print(f"Changes applied to {file_path}")
            
            push_choice = input("Do you want to push these changes to git? (y/n): ").lower()
            if push_choice == 'y':
                if commit_message:
                    use_suggested_message = input("Use the suggested commit message? (y/n): ").lower()
                    if use_suggested_message != 'y':
                        commit_message = input("Enter commit message: ")
                else:
                    commit_message = input("Enter commit message: ")
                
                result = git_push(file_path, commit_message)
                print(result)
                
                if verify_git_push(file_path):
                    print("Changes successfully pushed and verified on remote repository.")
                else:
                    print("Warning: Changes may not have been pushed to the remote repository.")
                    print("Please check your GitHub repository to confirm.")
            else:
                print("Changes applied locally but not pushed to git.")
        except Exception as e:
            print(f"Error applying changes: {e}")
            print("Changes were not applied.")
    else:
        print("Changes not applied.")

def claude_chat():
    print("Enter project name:")
    project_name = input().strip()
    vector_notebook = VectorNotebook()
    
    print(f"Welcome to Claude Chat for project: {project_name}")
    print("Type 'exit' to end the conversation.")
    print("To input multi-line messages or code blocks, use '###' on a new line to finish your input.")
    
    turn_counter = 0
    while True:
        user_input = get_multi_line_input()
        
        if user_input.lower().strip() == 'exit':
            print("Goodbye!")
            vector_notebook.close()
            return
        
        error_message = extract_error_message(user_input)
        instruction = generate_instruction(vector_notebook, user_input, error_message)
        
        if turn_counter % CONTEXT_CHECK_INTERVAL == 0:
            instruction += "\nReminder: Please check the provided conversation history for relevant context before responding."
        
        print("\nClaude: ")
        response = ask_claude(instruction)
        print(response)  # Print the full response

        vector_notebook.add_entry(user_input, ["user_query"])
        vector_notebook.add_entry(response, ["claude_response"])
        
        deletions, delete_commit_message = extract_deletions(response)
        if deletions:
            handle_deletions(deletions, delete_commit_message)
        
        code_file_pairs = extract_code_and_file_path(response)
        if code_file_pairs:
            for code, file_path, commit_message in code_file_pairs:
                handle_code_changes(code, file_path, commit_message)
        elif not deletions:
            print("\nNo specific code changes or deletions detected in Claude's response.")
        
        turn_counter += 1

if __name__ == "__main__":
    claude_chat()