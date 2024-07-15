import os
import json
import requests
import sqlite3
import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from datetime import datetime
from typing import List, Dict, Tuple
import subprocess
import re
import shutil
from github import Github
from enum import Enum
import math
import networkx as nx

# Global Variables and Constants
DEBUG = False
API_KEY = os.getenv('CLAUDE_API_KEY')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
API_URL = "https://api.anthropic.com/v1/messages"
GITHUB_REPO_URL = "https://github.com/olympusmons1256/polyglot-v5"
DB_PATH = "conversation_notebook.db"
CONTEXT_CHECK_INTERVAL = 3

print(f"API Key found: {'Yes' if API_KEY else 'No'}")
print(f"GitHub Token found: {'Yes' if GITHUB_TOKEN else 'No'}")

if not API_KEY:
    raise ValueError("CLAUDE_API_KEY environment variable is not set.")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set.")

class Shape(Enum):
    SPIRAL = "Spiral"
    FRACTAL = "Fractal"
    BRANCHING = "Branching"
    MESH = "Mesh"
    LAYERED = "Layered"

class PlantGrowthPattern(Enum):
    VINE = ("Vine-like", Shape.SPIRAL)
    TREE = ("Tree-like", Shape.BRANCHING)
    BUSH = ("Bush-like", Shape.MESH)
    SUCCULENT = ("Succulent-like", Shape.LAYERED)
    RHIZOME = ("Rhizome-like", Shape.FRACTAL)

    def __init__(self, label, shape):
        self.label = label
        self.shape = shape

class MathComplexity:
    def __init__(self):
        self.operations = {}
        self.total_complexity = 0

    def add_operation(self, operation, count):
        self.operations[operation] = self.operations.get(operation, 0) + count
        self.calculate_total_complexity()

    def calculate_total_complexity(self):
        self.total_complexity = sum(math.factorial(count) for count in self.operations.values())

class ProjectGrowthCharacteristics:
    def __init__(self):
        self.growth_rate = 0.01
        self.structural_complexity = 0.01
        self.interconnectedness = 0.01
        self.resource_efficiency = 0.01
        self.adaptability = 0.01
        self.math_complexity = MathComplexity()
        self.evolution_rate = 0.01

class PlantGrowthTaxonomy:
    def __init__(self):
        self.patterns = {pattern: ProjectGrowthCharacteristics() for pattern in PlantGrowthPattern}
        self._initialize_patterns()

    def _initialize_patterns(self):
        self.patterns[PlantGrowthPattern.VINE].growth_rate = 0.9
        self.patterns[PlantGrowthPattern.VINE].structural_complexity = 0.3
        self.patterns[PlantGrowthPattern.VINE].interconnectedness = 0.7
        self.patterns[PlantGrowthPattern.VINE].resource_efficiency = 0.5
        self.patterns[PlantGrowthPattern.VINE].adaptability = 0.8
        self.patterns[PlantGrowthPattern.VINE].evolution_rate = 0.7

        for pattern in PlantGrowthPattern:
            if pattern != PlantGrowthPattern.VINE:
                chars = self.patterns[pattern]
                chars.growth_rate = np.random.rand()
                chars.structural_complexity = np.random.rand()
                chars.interconnectedness = np.random.rand()
                chars.resource_efficiency = np.random.rand()
                chars.adaptability = np.random.rand()
                chars.evolution_rate = np.random.rand()

    def match_pattern(self, project_characteristics):
        similarities = {}
        for pattern, ideal_chars in self.patterns.items():
            similarity = self._calculate_similarity(project_characteristics, ideal_chars)
            similarities[pattern] = similarity
        return max(similarities, key=similarities.get)

    def _calculate_similarity(self, project_characteristics, ideal_chars):
        if DEBUG:
            print("Project characteristics:", vars(project_characteristics))
            print("Ideal characteristics:", vars(ideal_chars))
        project_vector = np.array([
            project_characteristics.growth_rate,
            project_characteristics.structural_complexity,
            project_characteristics.interconnectedness,
            project_characteristics.resource_efficiency,
            project_characteristics.adaptability,
            project_characteristics.evolution_rate
        ])
        ideal_vector = np.array([
            ideal_chars.growth_rate,
            ideal_chars.structural_complexity,
            ideal_chars.interconnectedness,
            ideal_chars.resource_efficiency,
            ideal_chars.adaptability,
            ideal_chars.evolution_rate
        ])
        if DEBUG:
            print("Project vector:", project_vector)
            print("Ideal vector:", ideal_vector)
        
        if np.array_equal(project_vector, ideal_vector):
            return 1.0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(project_vector, ideal_vector)
        
        if np.isnan(corr_matrix).all():
            return 0.0
        elif corr_matrix.size == 1:
            return corr_matrix.item()
        else:
            return corr_matrix[0, 1]

class GitHubRepo:
    def __init__(self, repo_url, update_threshold=10):
        self.g = Github(GITHUB_TOKEN)
        self.repo = self.g.get_repo(repo_url.split('github.com/')[-1])
        self.file_structure = None
        self.update_count = 0
        self.update_threshold = update_threshold

    def get_file_structure(self, force_update=False):
        if self.file_structure is None or force_update or self.update_count >= self.update_threshold:
            self.file_structure = self._fetch_file_structure()
            self.update_count = 0
        else:
            self.update_count += 1
        return self.file_structure

    def _fetch_file_structure(self):
        def traverse(contents, path=''):
            structure = []
            for content in contents:
                if content.type == 'dir':
                    structure.append(f"{path}{content.name}/")
                    structure.extend(traverse(self.repo.get_contents(content.path), f"{path}{content.name}/"))
                else:
                    structure.append(f"{path}{content.name}")
            return structure
        
        return traverse(self.repo.get_contents(""))

    def get_file_content(self, file_path):
        try:
            content = self.repo.get_contents(file_path)
            return content.decoded_content.decode()
        except Exception as e:
            return f"Error fetching file content: {str(e)}"

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

class ProjectLearningNode:
    def __init__(self, project_name, n_clusters=10):
        self.project_name = project_name
        self.vector_notebook = VectorNotebook()
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.loci = None
        self.growth_characteristics = ProjectGrowthCharacteristics()
        self.growth_pattern = None
        self.taxonomy = PlantGrowthTaxonomy()
        self.last_update_time = datetime.now()

    def add_interaction(self, content, tags):
        self.vector_notebook.add_entry(content, tags)
        self._update_loci()
        self._update_growth_characteristics(content, tags)
        self._update_growth_pattern()
        self._update_math_complexity(content)
        self._update_evolution_rate()

    def _update_loci(self):
        if self.vector_notebook.vectors is not None and self.vector_notebook.vectors.shape[0] % 10 == 0:
            self.loci = self.kmeans.fit_predict(self.vector_notebook.vectors.toarray())

    def _update_growth_characteristics(self, content, tags):
        self.growth_characteristics.growth_rate = max(0.01, self.growth_characteristics.growth_rate + (0.1 if "commit" in tags else 0))
        self.growth_characteristics.adaptability = max(0.01, self.growth_characteristics.adaptability + (0.1 if "refactor" in tags else 0))
        self.growth_characteristics.structural_complexity = max(0.01, self.growth_characteristics.structural_complexity + (0.1 if "class" in content.lower() else 0))
        self.growth_characteristics.interconnectedness = max(0.01, self.growth_characteristics.interconnectedness + (0.1 if "import" in content.lower() else 0))
        self.growth_characteristics.resource_efficiency = max(0.01, self.growth_characteristics.resource_efficiency + (0.1 if "optimize" in content.lower() else 0))

        for attr in vars(self.growth_characteristics):
            if isinstance(getattr(self.growth_characteristics, attr), (int, float)):
                setattr(self.growth_characteristics, attr, min(getattr(self.growth_characteristics, attr), 1))

    def _update_growth_pattern(self):
        self.growth_pattern = self.taxonomy.match_pattern(self.growth_characteristics)

    def _update_math_complexity(self, content):
        operations = {
            'addition': content.count('+'),
            'multiplication': content.count('*'),
            'exponentiation': content.count('**'),
        }
        for op, count in operations.items():
            self.growth_characteristics.math_complexity.add_operation(op, count)

    def _update_evolution_rate(self):
        current_time = datetime.now()
        time_diff = (current_time - self.last_update_time).total_seconds()
        self.growth_characteristics.evolution_rate = (
            self.growth_characteristics.math_complexity.total_complexity / time_diff
            if time_diff > 0 else 0
        )
        self.last_update_time = current_time

    def get_relevant_loci(self, query):
        if self.loci is None:
            return []
        query_vector = self.vector_notebook.vectorizer.transform([query])
        query_locus = self.kmeans.predict(query_vector.toarray())[0]
        return [query_locus]

    def compare_loci(self, other_loci):
        if self.loci is None or other_loci is None:
            return {}
        similarities = {}
        for i in range(self.kmeans.n_clusters):
            for j in range(len(set(other_loci))):
                locus1 = self.vector_notebook.vectors[self.loci == i].mean(axis=0)
                locus2 = self.vector_notebook.vectors[other_loci == j].mean(axis=0)
                similarity = cosine_similarity(locus1, locus2)[0][0]
                similarities[(i, j)] = similarity
        return similarities

class UserLearningNetwork:
    def __init__(self, username):
        self.username = username
        self.projects = {}
        self.cross_project_patterns = nx.Graph()

    def add_project(self, project_name):
        if project_name not in self.projects:
            self.projects[project_name] = ProjectLearningNode(project_name)

    def add_interaction(self, project_name, content, tags):
        if project_name not in self.projects:
            self.add_project(project_name)
        self.projects[project_name].add_interaction(content, tags)
        self._update_cross_project_patterns(project_name)

    def _update_cross_project_patterns(self, current_project):
        current_loci = self.projects[current_project].loci
        for project, node in self.projects.items():
            if project != current_project:
                similarities = node.compare_loci(current_loci)
                for (locus1, locus2), similarity in similarities.items():
                    if similarity > 0.7:
                        self.cross_project_patterns.add_edge(
                            (current_project, locus1),
                            (project, locus2),
                            weight=similarity
                        )

    def get_relevant_patterns(self, project_name, query):
        relevant_loci = self.projects[project_name].get_relevant_loci(query)
        relevant_patterns = []
        for locus in relevant_loci:
            node = (project_name, locus)
            if self.cross_project_patterns.has_node(node):
                relevant_patterns.extend(self.cross_project_patterns.neighbors(node))
        return relevant_patterns

    def get_project_growth_pattern(self, project_name):
        return self.projects[project_name].growth_pattern

    def get_project_shape(self, project_name):
        growth_pattern = self.projects[project_name].growth_pattern
        return growth_pattern.shape if growth_pattern else None

def generate_instruction(vector_notebook, user_input, github_repo, relevant_patterns, growth_pattern, shape, math_complexity, evolution_rate):
    instruction = f"Refer to the project at {GITHUB_REPO_URL}. "
    instruction += "Based on the following context and the current query, provide assistance:\n\n"
    
    relevant_entries = vector_notebook.get_relevant_entries(user_input)
    for entry in relevant_entries:
        instruction += f"Previous interaction (Relevance: {entry['similarity']:.2f}):\n"
        instruction += f"Content: {entry['content']}\n"
        instruction += f"Tags: {', '.join(entry['tags'])}\n\n"
    
    instruction += f"Current query: {user_input}\n"
    instruction += f"\nRepository structure:\n"
    instruction += "\n".join(github_repo.get_file_structure())
    instruction += f"\nCurrent project growth pattern: {growth_pattern.label if growth_pattern else 'Not determined'}\n"
    instruction += f"Project shape: {shape.value if shape else 'Not determined'}\n"
    instruction += f"Mathematical complexity: {math_complexity.total_complexity}\n"
    instruction += f"Evolution rate: {evolution_rate}\n"
    instruction += "Consider these factors when suggesting changes or improvements.\n"
    
    if relevant_patterns:
        instruction += "\nRelevant patterns from other projects:\n"
        for pattern in relevant_patterns:
            instruction += f"- {pattern[0]}: Locus {pattern[1]}\n"
    
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

def claude_chat(user_learning_network, project_name):
    global DEBUG
    project_node = user_learning_network.projects[project_name]
    vector_notebook = project_node.vector_notebook
    github_repo = GitHubRepo(GITHUB_REPO_URL)
    
    print(f"Welcome to Claude Chat for project: {project_name}")
    print("Type 'exit' to end the conversation.")
    print("Type 'debug on' to enable debug mode, 'debug off' to disable it.")
    print("To input multi-line messages or code blocks, use '###' on a new line to finish your input.")
    
    turn_counter = 0
    while True:
        user_input = get_multi_line_input()
        
        if user_input.lower().strip() == 'exit':
            print("Goodbye!")
            vector_notebook.close()
            return
        elif user_input.lower().strip() == 'debug on':
            DEBUG = True
            print("Debug mode enabled.")
            continue
        elif user_input.lower().strip() == 'debug off':
            DEBUG = False
            print("Debug mode disabled.")
            continue
        
        user_learning_network.add_interaction(project_name, user_input, ["user_query"])
        
        relevant_patterns = user_learning_network.get_relevant_patterns(project_name, user_input)
        growth_pattern = user_learning_network.get_project_growth_pattern(project_name)
        shape = user_learning_network.get_project_shape(project_name)
        math_complexity = project_node.growth_characteristics.math_complexity
        evolution_rate = project_node.growth_characteristics.evolution_rate
        
        if DEBUG:
            print(f"Growth Pattern: {growth_pattern}")
            print(f"Shape: {shape}")
            print(f"Math Complexity: {math_complexity.total_complexity}")
            print(f"Evolution Rate: {evolution_rate}")
        
        instruction = generate_instruction(
            vector_notebook, user_input, github_repo, relevant_patterns, 
            growth_pattern, shape, math_complexity, evolution_rate
        )
        
        if turn_counter % CONTEXT_CHECK_INTERVAL == 0:
            instruction += "\nReminder: Please check the provided conversation history, repository structure, and project characteristics for relevant context before responding."
        
        print("\nClaude: ")
        response = ask_claude(instruction)
        print(response)

        user_learning_network.add_interaction(project_name, response, ["claude_response"])
        
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

def main():
    try:
        user_network = UserLearningNetwork("example_user")
        project_name = input("Enter project name: ").strip()
        user_network.add_project(project_name)
        claude_chat(user_network, project_name)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing database connections and performing final cleanup...")
        for project in user_network.projects.values():
            project.vector_notebook.close()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()